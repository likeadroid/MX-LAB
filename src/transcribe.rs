//! Transcription des notes de la musique, avec le modèle Basic Pitch de Spotify
//! (Apache 2.0, `models/basic-pitch-nmp.onnx`), exécuté par tract.
//!
//! Portage fidèle de `basic_pitch/inference.py` (fenêtres de 2 s qui se chevauchent de
//! 30 images) et de `note_creation.py` (`output_to_notes_polyphonic`, réglages par défaut,
//! sans pitch bend).

use crate::ffmpeg;
use rayon::prelude::*;
use std::ffi::OsStr;
use std::io::Read;
use std::path::Path;
use std::sync::OnceLock;
use tract_onnx::prelude::*;

static MODEL: &[u8] = include_bytes!("../models/basic-pitch-nmp.onnx");

pub const SAMPLE_RATE: usize = 22050;
const FFT_HOP: usize = 256;
/// Échantillons d'une fenêtre : 2 s moins un pas.
const WINDOW: usize = SAMPLE_RATE * 2 - FFT_HOP;
const FRAMES_PER_WINDOW: usize = 172;
const OVERLAP_FRAMES: usize = 30;
const OVERLAP: usize = OVERLAP_FRAMES * FFT_HOP;
const HOP: usize = WINDOW - OVERLAP;
pub const PITCHES: usize = 88;
/// La touche la plus grave du piano.
pub const LOWEST_PITCH: u8 = 21;
const ONSET_THRESHOLD: f32 = 0.5;
const FRAME_THRESHOLD: f32 = 0.3;
/// 127,7 ms en images.
const MIN_NOTE_FRAMES: usize = 11;
const ENERGY_TOLERANCE: usize = 11;
const MAGIC_ALIGNMENT_OFFSET: f64 = 0.0018;

/// Une note transcrite, en secondes depuis le début de la vidéo.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Note {
    pub start: f64,
    pub end: f64,
    /// Numéro MIDI (60 : do 4).
    pub pitch: u8,
    /// 0 à 1 : l'activation moyenne du modèle sur la note.
    pub velocity: f32,
}

type Plan = TypedRunnableModel<TypedModel>;

pub struct Transcriber {
    plan: Plan,
    /// Place de « note » et « onset » parmi les sorties du modèle.
    note_output: usize,
    onset_output: usize,
}

/// Activations du modèle, une ligne de 88 hauteurs par image (256 échantillons).
pub struct Activations {
    pub frames: Vec<[f32; PITCHES]>,
    pub onsets: Vec<[f32; PITCHES]>,
}

impl Transcriber {
    fn load() -> Result<Self, String> {
        let error = |e: TractError| format!("modèle Basic Pitch : {e}");
        let model = tract_onnx::onnx().model_for_read(&mut &MODEL[..]).map_err(error)?;
        let names: Vec<String> = model
            .output_outlets()
            .map_err(error)?
            .iter()
            .map(|outlet| model.node(outlet.node).name.clone())
            .collect();
        let find = |name: &str| names.iter().position(|n| n == name).ok_or(format!("sortie {name} absente du modèle"));
        let (note_output, onset_output) = (find("StatefulPartitionedCall:1")?, find("StatefulPartitionedCall:2")?);
        let plan = model
            .with_input_fact(0, f32::fact([1, WINDOW, 1]).into())
            .and_then(|m| m.into_optimized())
            .and_then(|m| m.into_runnable())
            .map_err(error)?;
        Ok(Transcriber { plan, note_output, onset_output })
    }

    /// Le modèle, chargé une fois pour toute l'application.
    pub fn shared() -> Result<&'static Transcriber, String> {
        static SHARED: OnceLock<Result<Transcriber, String>> = OnceLock::new();
        SHARED.get_or_init(Transcriber::load).as_ref().map_err(Clone::clone)
    }

    /// Fait passer l'audio (mono, 22 050 Hz) dans le modèle, fenêtre par fenêtre.
    pub fn activations(&self, audio: &[f32]) -> Result<Activations, String> {
        // Une demi-recouvrement de silence devant, comme `get_audio_input`
        let mut padded = vec![0.0f32; OVERLAP / 2];
        padded.extend_from_slice(audio);
        let starts: Vec<usize> = (0..padded.len()).step_by(HOP).collect();
        let windows: Vec<Result<(Vec<[f32; PITCHES]>, Vec<[f32; PITCHES]>), String>> = starts
            .par_iter()
            .map(|&start| {
                let mut window = vec![0.0f32; WINDOW];
                let end = (start + WINDOW).min(padded.len());
                window[..end - start].copy_from_slice(&padded[start..end]);
                let input: Tensor = tract_ndarray::Array3::from_shape_vec((1, WINDOW, 1), window)
                    .map_err(|e| e.to_string())?
                    .into();
                let outputs = self.plan.run(tvec!(input.into())).map_err(|e| format!("Basic Pitch : {e}"))?;
                let rows = |index: usize| -> Result<Vec<[f32; PITCHES]>, String> {
                    let view = outputs[index].to_array_view::<f32>().map_err(|e| e.to_string())?;
                    let values: Vec<f32> = view.iter().copied().collect();
                    if values.len() != FRAMES_PER_WINDOW * PITCHES {
                        return Err(format!("sortie inattendue : {:?}", view.shape()));
                    }
                    // Sans la moitié du recouvrement de chaque côté, comme `unwrap_output`
                    let half = OVERLAP_FRAMES / 2;
                    Ok(values
                        .chunks_exact(PITCHES)
                        .skip(half)
                        .take(FRAMES_PER_WINDOW - OVERLAP_FRAMES)
                        .map(|row| row.try_into().unwrap())
                        .collect())
                };
                Ok((rows(self.note_output)?, rows(self.onset_output)?))
            })
            .collect();
        let mut frames = Vec::new();
        let mut onsets = Vec::new();
        for window in windows {
            let (f, o) = window?;
            frames.extend(f);
            onsets.extend(o);
        }
        let expected = (audio.len() as f64 / HOP as f64 * (FRAMES_PER_WINDOW - OVERLAP_FRAMES) as f64) as usize;
        frames.truncate(expected);
        onsets.truncate(expected);
        Ok(Activations { frames, onsets })
    }
}

/// Instant de chaque image, comme `model_frames_to_time` (avec sa petite correction par fenêtre).
fn frame_time(frame: usize) -> f64 {
    let hop = FFT_HOP as f64 / SAMPLE_RATE as f64;
    let window_offset = hop * (FRAMES_PER_WINDOW as f64 - WINDOW as f64 / FFT_HOP as f64) + MAGIC_ALIGNMENT_OFFSET;
    frame as f64 * hop - window_offset * (frame / FRAMES_PER_WINDOW) as f64
}

/// Attaques déduites des fortes montées d'énergie, en plus de celles du modèle.
fn inferred_onsets(onsets: &[[f32; PITCHES]], frames: &[[f32; PITCHES]]) -> Vec<[f32; PITCHES]> {
    const N_DIFF: usize = 2;
    let at = |t: isize, f: usize| if t < 0 { 0.0 } else { frames[t as usize][f] };
    let mut diff = vec![[0.0f32; PITCHES]; frames.len()];
    for t in N_DIFF..frames.len() {
        for f in 0..PITCHES {
            let smallest = (1..=N_DIFF).map(|n| at(t as isize, f) - at(t as isize - n as isize, f)).fold(f32::INFINITY, f32::min);
            diff[t][f] = smallest.max(0.0);
        }
    }
    let max_onset = onsets.iter().flatten().copied().fold(0.0, f32::max);
    let max_diff = diff.iter().flatten().copied().fold(0.0, f32::max);
    let scale = if max_diff > 0.0 { max_onset / max_diff } else { 0.0 };
    onsets
        .iter()
        .zip(&diff)
        .map(|(onset, diff)| std::array::from_fn(|f| onset[f].max(diff[f] * scale)))
        .collect()
}

/// Les notes, décodées comme `output_to_notes_polyphonic` (astuce « melodia » comprise).
pub fn notes(activations: &Activations) -> Vec<Note> {
    let frames = &activations.frames;
    let n = frames.len();
    if n < 3 {
        return Vec::new();
    }
    let onsets = inferred_onsets(&activations.onsets, frames);
    // Pics dans le temps au-dessus du seuil, parcourus du plus tard au plus tôt
    let mut peaks = Vec::new();
    for t in 1..n - 1 {
        for f in 0..PITCHES {
            let value = onsets[t][f];
            if value > onsets[t - 1][f] && value > onsets[t + 1][f] && value >= ONSET_THRESHOLD {
                peaks.push((t, f));
            }
        }
    }
    peaks.reverse();

    let mut remaining = frames.clone();
    let mut found: Vec<(usize, usize, usize, f32)> = Vec::new();
    let mean = |start: usize, end: usize, f: usize| frames[start..end].iter().map(|row| row[f]).sum::<f32>() / (end - start) as f32;
    let clear = |remaining: &mut Vec<[f32; PITCHES]>, t: usize, f: usize| {
        remaining[t][f] = 0.0;
        if f + 1 < PITCHES {
            remaining[t][f + 1] = 0.0;
        }
        if f > 0 {
            remaining[t][f - 1] = 0.0;
        }
    };
    for (start, f) in peaks {
        if start >= n - 1 {
            continue;
        }
        let mut i = start + 1;
        let mut k = 0;
        while i < n - 1 && k < ENERGY_TOLERANCE {
            if remaining[i][f] < FRAME_THRESHOLD {
                k += 1;
            } else {
                k = 0;
            }
            i += 1;
        }
        i -= k;
        if i - start <= MIN_NOTE_FRAMES {
            continue;
        }
        for t in start..i {
            clear(&mut remaining, t, f);
        }
        found.push((start, i, f, mean(start, i, f)));
    }

    // L'énergie restante forme des notes sans attaque nette
    loop {
        let (mut best, mut at) = (FRAME_THRESHOLD, None);
        for (t, row) in remaining.iter().enumerate() {
            for (f, &value) in row.iter().enumerate() {
                if value > best {
                    best = value;
                    at = Some((t, f));
                }
            }
        }
        let Some((middle, f)) = at else { break };
        remaining[middle][f] = 0.0;
        let mut i = middle + 1;
        let mut k = 0;
        while i < n - 1 && k < ENERGY_TOLERANCE {
            if remaining[i][f] < FRAME_THRESHOLD {
                k += 1;
            } else {
                k = 0;
            }
            clear(&mut remaining, i, f);
            i += 1;
        }
        let end = i - 1 - k;
        let mut i = middle as isize - 1;
        let mut k = 0;
        while i > 0 && k < ENERGY_TOLERANCE {
            if remaining[i as usize][f] < FRAME_THRESHOLD {
                k += 1;
            } else {
                k = 0;
            }
            clear(&mut remaining, i as usize, f);
            i -= 1;
        }
        let start = (i + 1 + k as isize).max(0) as usize;
        if end <= start || end - start <= MIN_NOTE_FRAMES {
            continue;
        }
        found.push((start, end, f, mean(start, end, f)));
    }

    let mut notes: Vec<Note> = found
        .into_iter()
        .map(|(start, end, f, amplitude)| Note {
            start: frame_time(start),
            end: frame_time(end),
            pitch: LOWEST_PITCH + f as u8,
            velocity: amplitude,
        })
        .collect();
    notes.sort_by(|a, b| a.start.total_cmp(&b.start).then(a.pitch.cmp(&b.pitch)));
    notes
}

/// Le mix transcrit : gauche et droite seulement dès qu'il y a plus de deux canaux (pas de
/// centre, donc moins de dialogues), sinon la piste ramenée en mono.
pub fn mix_filter(layout: &str) -> String {
    match layout {
        "mono" => "pan=mono|c0=c0".to_string(),
        "stereo" => "pan=mono|c0=0.5*c0+0.5*c1".to_string(),
        _ => "pan=mono|c0=0.5*c0+0.5*c1".to_string(),
    }
}

/// Transcrit un passage de la vidéo ; les notes sont datées depuis le début de la vidéo.
pub fn transcribe(video: &Path, track: usize, layout: &str, start: f64, end: f64) -> Result<Vec<Note>, String> {
    let transcriber = Transcriber::shared()?;
    let (start_text, length_text) = (format!("{:.3}", start.max(0.0)), format!("{:.3}", (end - start).max(0.0)));
    let (map, filter, rate) = (format!("0:a:{track}"), mix_filter(layout), SAMPLE_RATE.to_string());
    let args: Vec<&OsStr> = vec![
        OsStr::new("-v"),
        OsStr::new("error"),
        OsStr::new("-ss"),
        OsStr::new(&start_text),
        OsStr::new("-t"),
        OsStr::new(&length_text),
        OsStr::new("-i"),
        video.as_os_str(),
        OsStr::new("-vn"),
        OsStr::new("-map"),
        OsStr::new(&map),
        OsStr::new("-af"),
        OsStr::new(&filter),
        OsStr::new("-ac"),
        OsStr::new("1"),
        OsStr::new("-ar"),
        OsStr::new(&rate),
        OsStr::new("-f"),
        OsStr::new("f32le"),
        OsStr::new("-"),
    ];
    let mut stream = ffmpeg::Stream::spawn(&args)?;
    let mut bytes = Vec::new();
    stream.stdout().read_to_end(&mut bytes).map_err(|e| e.to_string())?;
    stream.finish()?;
    let audio: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    let activations = transcriber.activations(&audio)?;
    let mut notes = notes(&activations);
    for note in &mut notes {
        note.start += start;
        note.end += start;
    }
    Ok(notes)
}

/// Nom d'une note : C4 pour le do du milieu (MIDI 60).
pub fn note_name(pitch: u8) -> String {
    const NAMES: [&str; 12] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    format!("{}{}", NAMES[pitch as usize % 12], pitch as i32 / 12 - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(frequency: f64, seconds: f64) -> Vec<f32> {
        (0..(seconds * SAMPLE_RATE as f64) as usize)
            .map(|i| (0.5 * (std::f64::consts::TAU * frequency * i as f64 / SAMPLE_RATE as f64).sin()) as f32)
            .collect()
    }

    #[test]
    fn a_held_a4_is_transcribed_as_a_single_a4() {
        let mut audio = vec![0.0; SAMPLE_RATE / 2];
        audio.extend(sine(440.0, 3.0));
        audio.extend(vec![0.0; SAMPLE_RATE]);
        let notes = notes(&Transcriber::shared().unwrap().activations(&audio).unwrap());
        let a4: Vec<&Note> = notes.iter().filter(|n| n.pitch == 69).collect();
        assert!(!a4.is_empty(), "{notes:?}");
        let longest = a4.iter().max_by(|a, b| (a.end - a.start).total_cmp(&(b.end - b.start))).unwrap();
        assert!((longest.start - 0.5).abs() < 0.15, "{longest:?}");
        assert!((longest.end - 3.5).abs() < 0.3, "{longest:?}");
        // Rien de franc ailleurs qu'au la (et à ses harmoniques)
        assert!(notes.iter().all(|n| [69, 81, 88].contains(&n.pitch) || n.velocity < 0.5), "{notes:?}");
    }

    /// Sur une vraie vidéo : `MX_LAB_TRANSCRIBE=video.mkv cargo test --lib real_video -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn real_video_transcription() {
        let path = std::path::PathBuf::from(std::env::var("MX_LAB_TRANSCRIBE").expect("MX_LAB_TRANSCRIBE"));
        let (layout, duration) = crate::detect::probe_audio(&path).unwrap();
        let started = std::time::Instant::now();
        let notes = transcribe(&path, 0, &layout, 0.0, duration).unwrap();
        println!("{layout}, {duration:.0} s, {} notes en {:?}", notes.len(), started.elapsed());
        let mut counts = [0usize; 128];
        for note in &notes {
            counts[note.pitch as usize] += 1;
        }
        let mut common: Vec<(usize, u8)> = counts.iter().enumerate().map(|(p, &c)| (c, p as u8)).filter(|c| c.0 > 0).collect();
        common.sort_by(|a, b| b.cmp(a));
        let names: Vec<String> = common.iter().take(12).map(|(c, p)| format!("{} ×{c}", note_name(*p))).collect();
        println!("hauteurs fréquentes : {}", names.join(", "));
        for note in notes.iter().take(15) {
            println!("{:7.2} → {:7.2}  {:4}  {:.2}", note.start, note.end, note_name(note.pitch), note.velocity);
        }
        assert!(notes.iter().all(|n| n.start >= -0.05 && n.end <= duration + 0.5 && n.end > n.start));
    }

    #[test]
    fn note_names_follow_the_midi_convention() {
        assert_eq!(note_name(60), "C4");
        assert_eq!(note_name(69), "A4");
        assert_eq!(note_name(21), "A0");
    }

    #[test]
    fn frames_line_up_with_windows() {
        assert_eq!(WINDOW, 43844);
        assert_eq!(HOP, 36164);
        assert!(frame_time(0).abs() < 1e-9);
        assert!((frame_time(171) - 171.0 * 256.0 / 22050.0).abs() < 1e-9);
    }
}
