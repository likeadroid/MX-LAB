//! `mx-lab --self-check`, sans ouvrir de fenêtre : vérifie que ffmpeg, libmpv et le modèle MIDI livrés avec
//! l'app fonctionnent (contrôle des paquets .app et AppImage).

use crate::mpv::Player;
use eframe::egui;
use std::path::Path;
use std::time::{Duration, Instant};

/// Traite les commandes de paquet ; `None` si l'app doit démarrer normalement.
pub fn run(arguments: &[String]) -> Option<i32> {
    match arguments {
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
    // Le modèle de transcription MIDI : chargé et exécuté sur une seconde de la 4
    let a4: Vec<f32> = (0..mx_lab::transcribe::SAMPLE_RATE * 3)
        .map(|i| (0.5 * (std::f64::consts::TAU * 440.0 * i as f64 / mx_lab::transcribe::SAMPLE_RATE as f64).sin()) as f32)
        .collect();
    let transcription = mx_lab::transcribe::Transcriber::shared().and_then(|model| model.activations(&a4));
    match transcription.map(|activations| mx_lab::transcribe::notes(&activations)) {
        Ok(notes) if notes.iter().any(|n| n.pitch == 69) => println!("Basic Pitch : OK, la 4 transcrit"),
        outcome => {
            println!("Basic Pitch : ÉCHEC {outcome:?}");
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
