// Compresse ffmpeg et libmpv une fois pour toutes : les exe les embarquent et les
// extraient au 1er lancement.
use std::{env, fs, path::Path};

fn compress(source: &str, name: &str) {
    let source = Path::new(source);
    println!("cargo:rerun-if-changed={}", source.display());
    let target = Path::new(&env::var("OUT_DIR").unwrap()).join(name);
    let modified = |path: &Path| fs::metadata(path).and_then(|m| m.modified()).ok();
    if let (Some(from), Some(to)) = (modified(source), modified(&target)) {
        if to >= from {
            return;
        }
    }
    let data = fs::read(source).unwrap_or_else(|_| panic!("{} introuvable", source.display()));
    let mut encoder = zstd::Encoder::new(fs::File::create(&target).unwrap(), 19).unwrap();
    encoder.long_distance_matching(true).unwrap();
    encoder.window_log(27).unwrap();
    encoder.multithread(num_threads()).unwrap();
    std::io::copy(&mut data.as_slice(), &mut encoder).unwrap();
    encoder.finish().unwrap();
}

include!("src/bin/music-studio/icon.rs");

/// Écrit l'icône de Music Studio (.ico) et la donne à son exe, sans toucher aux autres.
fn studio_icon() {
    println!("cargo:rerun-if-changed=src/bin/music-studio/icon.rs");
    let folder = env::var("OUT_DIR").unwrap();
    let sizes = [16u32, 32, 48, 256];
    let images: Vec<Vec<u8>> = sizes
        .iter()
        .map(|&size| {
            let rgba = icon_rgba(size);
            let mut image = Vec::new();
            // En-tête BMP : hauteur doublée, avec le masque qui suit les pixels
            for value in [40u32, size, size * 2] {
                image.extend_from_slice(&value.to_le_bytes());
            }
            image.extend_from_slice(&1u16.to_le_bytes());
            image.extend_from_slice(&32u16.to_le_bytes());
            image.extend_from_slice(&[0u8; 24]);
            for row in (0..size as usize).rev() {
                for pixel in rgba[row * size as usize * 4..(row + 1) * size as usize * 4].chunks_exact(4) {
                    image.extend_from_slice(&[pixel[2], pixel[1], pixel[0], pixel[3]]);
                }
            }
            image.resize(image.len() + (size as usize).div_ceil(32) * 4 * size as usize, 0);
            image
        })
        .collect();
    let mut ico = Vec::new();
    ico.extend_from_slice(&[0, 0, 1, 0]);
    ico.extend_from_slice(&(sizes.len() as u16).to_le_bytes());
    let mut offset = 6 + 16 * sizes.len() as u32;
    for (size, image) in sizes.iter().zip(&images) {
        let side = if *size == 256 { 0 } else { *size as u8 };
        ico.extend_from_slice(&[side, side, 0, 0, 1, 0, 32, 0]);
        ico.extend_from_slice(&(image.len() as u32).to_le_bytes());
        ico.extend_from_slice(&offset.to_le_bytes());
        offset += image.len() as u32;
    }
    for image in &images {
        ico.extend_from_slice(image);
    }
    let ico_path = Path::new(&folder).join("music-studio.ico");
    fs::write(&ico_path, ico).unwrap();
    let rc_path = Path::new(&folder).join("music-studio.rc");
    let escaped = ico_path.display().to_string().replace('\\', "\\\\");
    fs::write(&rc_path, format!("1 ICON \"{escaped}\"\n")).unwrap();
    embed_resource::compile_for(&rc_path, ["music-studio"], embed_resource::NONE).manifest_optional().unwrap();
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    studio_icon();
    compress("../ffmpeg-min/ffmpeg.exe", "ffmpeg.exe.zst");
    compress("vendor/libmpv/libmpv-2.dll", "libmpv-2.dll.zst");
}

fn num_threads() -> u32 {
    std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(1)
}
