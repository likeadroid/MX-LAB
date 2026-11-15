//! Commandes pour fabriquer et contrôler les paquets (.app, AppImage), sans ouvrir de fenêtre :
//! `mx-lab --icon-png <taille> <fichier>` écrit l'icône, `mx-lab --self-check` vérifie que
//! ffmpeg et libmpv livrés avec l'app fonctionnent.

use crate::icon::icon_rgba;
use crate::mpv::Player;
use eframe::egui;
use std::path::Path;
use std::time::{Duration, Instant};

/// Traite les commandes de paquet ; `None` si l'app doit démarrer normalement.
pub fn run(arguments: &[String]) -> Option<i32> {
    match arguments {
        [command, size, path] if command == "--icon-png" => {
            let size: u32 = size.parse().ok()?;
            Some(match std::fs::write(path, png(size, &icon_rgba(size))) {
                Ok(()) => 0,
                Err(error) => {
                    eprintln!("{path} : {error}");
                    1
                }
            })
        }
        [command] if command == "--self-check" => Some(self_check()),
        _ => None,
    }
}

fn self_check() -> i32 {
    let mut failures = 0;
    match mx_lab::ffmpeg::path() {
        Ok(path) => println!("ffmpeg : {}", path.display()),
        Err(error) => println!("ffmpeg : {error}"),
    }
    match mx_lab::ffmpeg::run(&["-hide_banner".as_ref(), "-version".as_ref()]) {
        Ok((_, true)) => println!("ffmpeg : OK"),
        outcome => {
            println!("ffmpeg : ÉCHEC {outcome:?}");
            failures += 1;
        }
    }
    match video_frame() {
        Ok(()) => println!("libmpv : OK, une image rendue"),
        Err(error) => {
            println!("libmpv : ÉCHEC, {error}");
            failures += 1;
        }
    }
    (failures > 0) as i32
}

fn video_frame() -> Result<(), String> {
    let mut player = Player::new(&egui::Context::default())?;
    player.set_video_size([64, 36]);
    player.load(Path::new("av://lavfi:color=c=white:s=64x36"));
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut last = None;
    while Instant::now() < deadline {
        player.poll();
        if let Some(error) = &player.state.error {
            return Err(error.clone());
        }
        // Avant le chargement, mpv peut rendre une image noire : attendre la blanche
        if let Some(frame) = player.take_frame() {
            last = Some(frame.pixels[frame.pixels.len() / 2].to_array()[0]);
            if last.is_some_and(|red| red > 200) {
                return Ok(());
            }
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    Err(match last {
        Some(red) => format!("image fausse (rouge {red})"),
        None => "aucune image en 20 s".into(),
    })
}

/// PNG RGBA sans compression (blocs « stored ») : assez pour une icône, sans dépendance.
fn png(size: u32, rgba: &[u8]) -> Vec<u8> {
    let mut raw = Vec::with_capacity(rgba.len() + size as usize);
    for row in rgba.chunks_exact(size as usize * 4) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    let mut zlib = vec![0x78, 0x01];
    let mut blocks = raw.chunks(65535).peekable();
    while let Some(block) = blocks.next() {
        zlib.push(blocks.peek().is_none() as u8);
        let length = block.len() as u16;
        zlib.extend_from_slice(&length.to_le_bytes());
        zlib.extend_from_slice(&(!length).to_le_bytes());
        zlib.extend_from_slice(block);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in &raw {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    zlib.extend_from_slice(&((b << 16) | a).to_be_bytes());

    let mut header = Vec::new();
    header.extend_from_slice(&size.to_be_bytes());
    header.extend_from_slice(&size.to_be_bytes());
    header.extend_from_slice(&[8, 6, 0, 0, 0]);
    let mut file = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    for (kind, data) in [(b"IHDR", header.as_slice()), (b"IDAT", zlib.as_slice()), (b"IEND", &[][..])] {
        file.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let start = file.len();
        file.extend_from_slice(kind);
        file.extend_from_slice(data);
        let crc = crc32(&file[start..]);
        file.extend_from_slice(&crc.to_be_bytes());
    }
    file
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xedb8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_has_a_valid_signature_and_checksums() {
        assert_eq!(crc32(b"IEND"), 0xae42_6082);
        let file = png(16, &icon_rgba(16));
        assert_eq!(&file[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(&file[file.len() - 8..], b"IEND\xae\x42\x60\x82");
    }
}
