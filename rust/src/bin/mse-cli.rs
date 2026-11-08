//! Outil de validation sans interface.
//!   mse-cli trace <fenetres.f32> <dossier>   écrit les tenseurs intermédiaires
//!   mse-cli scores <video> <sortie.f32>      écrit P(musique), retourne la durée de calcul
//!   mse-cli segments <scores.f32> <duree> <sensibilite>
//!   mse-cli process <video> [sensibilite]    détection et découpe, comme l'interface
//!   mse-cli extract <video> <dossier>        détection et découpe vers un dossier choisi
//!   mse-cli classes <video> <projet.json> <sortie.csv>   classes moyennes de chaque séquence

use mse::detect::{self, Settings};
use mse::yamnet::{self, Yamnet, WINDOW};
use std::path::Path;
use std::time::Instant;

fn read_f32(path: &str) -> Vec<f32> {
    std::fs::read(path).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("trace") => {
            let model = Yamnet::load();
            let windows = read_f32(&args[1]);
            for (index, window) in windows.chunks_exact(WINDOW).enumerate() {
                let mut write = |tensor: i32, bytes: &[u8]| {
                    std::fs::write(format!("{}/{index}_{tensor}.bin", args[2]), bytes).unwrap();
                };
                model.run(window, Some(&mut write));
            }
        }
        Some("scores") => {
            let start = Instant::now();
            let model = Yamnet::load();
            let video = Path::new(&args[1]);
            let (layout, duration) = detect::probe_audio(video)?;
            let (mixes, reduced) = detect::choose_mixes(video, &layout)?;
            let scores = detect::detect_music_probabilities(&model, video, duration, &mixes, &|_| {})?;
            let bytes: Vec<u8> = scores.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&args[2], bytes).map_err(|e| e.to_string())?;
            println!("{layout} {duration} {} mix(es) attenue={reduced} {} scores {:.1}s", mixes.len(), scores.len(), start.elapsed().as_secs_f64());
        }
        Some("segments") => {
            let scores = read_f32(&args[1]);
            let duration: f64 = args[2].parse().unwrap();
            let sensitivity: f64 = args[3].parse().unwrap();
            for (a, b) in detect::scores_to_segments(&scores, sensitivity, 15.0, 6.0, 6.0, duration) {
                println!("{a:.6} {b:.6}");
            }
        }
        Some("process") => {
            let start = Instant::now();
            let model = Yamnet::load();
            let sensitivity = args.get(2).map_or(detect::DEFAULT_SENSITIVITY, |s| s.parse().unwrap());
            let settings = Settings { sensitivity, gap: 15.0, anticipation: 6.0, post: 6.0 };
            let summary = detect::process_video(&model, Path::new(&args[1]), &settings, None, &|s| eprintln!("{s}"))?;
            println!("{summary} ({:.1}s)", start.elapsed().as_secs_f64());
        }
        Some("extract") => {
            let start = Instant::now();
            let model = Yamnet::load();
            let settings = Settings { sensitivity: detect::DEFAULT_SENSITIVITY, gap: 15.0, anticipation: 6.0, post: 6.0 };
            let video = Path::new(&args[1]);
            let (segments, _) = detect::detect_music_segments(&model, video, &settings, &|_| {})?;
            let files = detect::extract_segments(video, &segments, Path::new(&args[2]))?;
            println!("{} clips ({:.1}s)", files.len(), start.elapsed().as_secs_f64());
        }
        Some("classes") => {
            let start = Instant::now();
            let model = Yamnet::load();
            let video = Path::new(&args[1]);
            let (layout, duration) = detect::probe_audio(video)?;
            let (mixes, _) = detect::choose_mixes(video, &layout)?;
            let labels = yamnet::class_labels();
            let classes: Vec<usize> = (0..labels.len()).collect();
            let rows = detect::class_logits(&model, video, &mixes[0], &classes, duration, &|_| {})?;
            eprintln!("{layout} {duration:.0} s, {} fenêtres, {:.1} s", rows.len(), start.elapsed().as_secs_f64());
            let text = std::fs::read_to_string(&args[2]).map_err(|e| e.to_string())?;
            let project: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
            let mut csv = String::from("sequence,start,end");
            for label in &labels {
                csv.push_str(&format!(",\"{label}\""));
            }
            csv.push('\n');
            for (index, segment) in project["segments"].as_array().ok_or("pas de séquences")?.iter().enumerate() {
                let (a, b) = (segment["start"].as_f64().unwrap(), segment["end"].as_f64().unwrap());
                let first = ((a / detect::STEP) as usize).min(rows.len());
                let last = ((b / detect::STEP).ceil() as usize).clamp(first, rows.len());
                csv.push_str(&format!("{},{a:.2},{b:.2}", index + 1));
                for class in 0..labels.len() {
                    let sum: f32 = rows[first..last].iter().map(|row| 1.0 / (1.0 + (-row[class]).exp())).sum();
                    csv.push_str(&format!(",{:.5}", sum / (last - first).max(1) as f32));
                }
                csv.push('\n');
            }
            std::fs::write(&args[3], csv).map_err(|e| e.to_string())?;
        }
        _ => return Err("commande inconnue".into()),
    }
    Ok(())
}
