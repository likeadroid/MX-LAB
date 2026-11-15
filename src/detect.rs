use crate::ffmpeg;
use crate::yamnet::{Yamnet, HOP, WINDOW};
use rayon::prelude::*;
use std::ffi::OsStr;
use std::io::Read;
use std::path::{Path, PathBuf};

pub const SAMPLE_RATE: usize = 16000;
pub const STEP: f64 = HOP as f64 / SAMPLE_RATE as f64;

const SMOOTHING_S: f64 = 3.0;
const MIN_DURATION_S: f64 = 5.0;
const HIGH_THRESHOLD: f64 = 0.05; // un passage net déclenche la détection...
const LOW_THRESHOLD: f64 = 0.05; // ...qui se prolonge tant que la musique continue
pub const DEFAULT_SENSITIVITY: f64 = 50.0;
const MAX_SIDE_ATTENUATION_DB: f64 = 40.0; // au-delà, gauche - droite est considéré comme vide
const BATCH: usize = 512; // fenêtres analysées en parallèle

/// Canaux à garder : le central porte généralement les dialogues, qui masquent
/// la musique pour le modèle. Ordre ffmpeg : 5.1 = FL FR FC LFE BL BR.
fn channels_without_center(layout: &str) -> Option<&'static [u32]> {
    match layout {
        "5.1" | "5.1(side)" => Some(&[0, 1, 4, 5]),
        "5.0" | "5.0(side)" => Some(&[0, 1, 3, 4]),
        "7.1" | "7.1(wide)" => Some(&[0, 1, 4, 5, 6, 7]),
        _ => None,
    }
}

/// Retourne (disposition des canaux, durée en s) de la 1re piste audio.
/// ffprobe n'étant pas embarqué, on lit le résumé affiché par `ffmpeg -i`.
pub fn probe_audio(video: &Path) -> Result<(String, f64), String> {
    probe_audio_track(video, 0)
}

/// Comme `probe_audio`, pour la piste audio demandée (0 : la première).
pub fn probe_audio_track(video: &Path, track: usize) -> Result<(String, f64), String> {
    let (info, _) = ffmpeg::run(&[OsStr::new("-hide_banner"), OsStr::new("-i"), video.as_os_str()])?;
    let layout = info
        .lines()
        .filter_map(|line| line.split_once("Audio:").map(|(_, rest)| rest))
        .filter_map(|rest| {
            let (before, after) = rest.split_once(" Hz, ")?;
            before.chars().last().filter(char::is_ascii_digit)?;
            let (layout, _) = after.split_once(',')?;
            Some(layout.trim().to_string())
        })
        .nth(track)
        .ok_or("aucune piste audio trouvée")?;
    let duration = info.lines().find_map(|line| {
        let rest = line.split_once("Duration: ")?.1;
        let clock = rest.split(',').next()?;
        let mut parts = clock.split(':');
        let hours: f64 = parts.next()?.trim().parse().ok()?;
        let minutes: f64 = parts.next()?.parse().ok()?;
        let seconds: f64 = parts.next()?.parse().ok()?;
        Some(hours * 3600.0 + minutes * 60.0 + seconds)
    });
    Ok((layout, duration.unwrap_or(0.0)))
}

/// Niveau moyen en dB d'un mix de la 1re piste audio.
fn mean_volume(video: &Path, track: usize, filter: &str) -> Result<f64, String> {
    let filter = format!("{filter},volumedetect");
    let map = format!("0:a:{track}");
    let (info, _) = ffmpeg::run(&[
        OsStr::new("-hide_banner"),
        OsStr::new("-nostats"),
        OsStr::new("-i"),
        video.as_os_str(),
        OsStr::new("-map"),
        OsStr::new(&map),
        OsStr::new("-vn"),
        OsStr::new("-af"),
        OsStr::new(&filter),
        OsStr::new("-f"),
        OsStr::new("null"),
        OsStr::new("-"),
    ])?;
    Ok(info
        .lines()
        .find_map(|line| line.split_once("mean_volume: ")?.1.split(" dB").next()?.parse().ok())
        .unwrap_or(f64::NEG_INFINITY))
}

/// Retourne (mixes à analyser, dialogues atténués ?).
///
/// Le mix atténué fait ressortir la musique sous les voix : sans canal central en
/// 5.1, gauche moins droite en stéréo (muet sur un faux stéréo). Il perd la musique
/// placée au centre, d'où l'analyse complémentaire du mix mono.
pub fn choose_mixes(video: &Path, layout: &str) -> Result<(Vec<Vec<String>>, bool), String> {
    choose_mixes_on(video, 0, layout)
}

/// Comme `choose_mixes`, sur la piste audio demandée : chaque mix porte alors son `-map`.
pub fn choose_mixes_on(video: &Path, track: usize, layout: &str) -> Result<(Vec<Vec<String>>, bool), String> {
    let (mut mixes, reduced) = mixes_of_track(video, track, layout)?;
    if track > 0 {
        for mix in &mut mixes {
            mix.splice(0..0, ["-map".to_string(), format!("0:a:{track}")]);
        }
    }
    Ok((mixes, reduced))
}

fn mixes_of_track(video: &Path, track: usize, layout: &str) -> Result<(Vec<Vec<String>>, bool), String> {
    let args = |a: &str, b: &str| vec![a.to_string(), b.to_string()];
    if let Some(channels) = channels_without_center(layout) {
        let weight = 1.0 / channels.len() as f64;
        let mix: Vec<String> = channels.iter().map(|c| format!("{weight}*c{c}")).collect();
        let filter = format!("pan=mono|c0={}", mix.join("+"));
        return Ok((vec![args("-af", &filter), args("-ac", "1")], true));
    }
    if layout == "stereo" {
        let (side, mono) = ("pan=mono|c0=0.5*c0-0.5*c1", "pan=mono|c0=0.5*c0+0.5*c1");
        if mean_volume(video, track, side)? >= mean_volume(video, track, mono)? - MAX_SIDE_ATTENUATION_DB {
            return Ok((vec![args("-af", side), args("-af", mono)], true));
        }
        return Ok((vec![args("-af", mono)], false));
    }
    Ok((vec![args("-ac", "1")], false))
}

/// Calcule P(musique) toutes les 0,48 s, sans charger tout l'audio en mémoire.
pub fn music_probabilities(
    model: &Yamnet,
    video: &Path,
    mix: &[String],
    duration: f64,
    progress: &(dyn Fn(u32) + Sync),
) -> Result<Vec<f32>, String> {
    for_each_window(video, mix, duration, progress, |_| true, |window| model.music_probability(window))
}

/// Logits des classes demandées toutes les 0,48 s : une ligne par fenêtre.
pub fn class_logits(
    model: &Yamnet,
    video: &Path,
    mix: &[String],
    classes: &[usize],
    duration: f64,
    progress: &(dyn Fn(u32) + Sync),
) -> Result<Vec<Vec<f32>>, String> {
    for_each_window(video, mix, duration, progress, |_| true, |window| model.class_logits(window, classes))
}

/// Applique un calcul à chaque fenêtre de l'audio, en parallèle et dans l'ordre.
fn for_each_window<T: Send>(
    video: &Path,
    mix: &[String],
    duration: f64,
    progress: &(dyn Fn(u32) + Sync),
    mut on_batch: impl FnMut(&[T]) -> bool,
    compute: impl Fn(&[f32]) -> T + Sync,
) -> Result<Vec<T>, String> {
    let rate = SAMPLE_RATE.to_string();
    let mut args: Vec<&OsStr> = vec![
        OsStr::new("-v"),
        OsStr::new("error"),
        OsStr::new("-i"),
        video.as_os_str(),
        OsStr::new("-vn"),
    ];
    // Un mix peut désigner sa piste audio ; sinon, la première
    if !mix.iter().any(|arg| arg == "-map") {
        args.extend([OsStr::new("-map"), OsStr::new("0:a:0")]);
    }
    args.extend(mix.iter().map(OsStr::new));
    args.extend([OsStr::new("-ar"), OsStr::new(&rate), OsStr::new("-f"), OsStr::new("f32le"), OsStr::new("-")]);
    let mut stream = ffmpeg::Stream::spawn(&args)?;

    let mut scores = Vec::new();
    let span = WINDOW + (BATCH - 1) * HOP;
    let mut samples: Vec<f32> = Vec::with_capacity(span + HOP);
    let mut raw = vec![0u8; (span + HOP) * 4];
    let mut pending = 0; // octets lus mais pas encore assemblés en échantillons
    let mut ended = false;
    while !ended {
        let room = (span - samples.len()) * 4;
        while pending < room {
            let read = stream.stdout().read(&mut raw[pending..room]).map_err(|e| e.to_string())?;
            if read == 0 {
                ended = true;
                break;
            }
            pending += read;
        }
        let whole = pending / 4 * 4;
        samples.extend(raw[..whole].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())));
        raw.copy_within(whole..pending, 0);
        pending -= whole;
        if samples.len() < WINDOW {
            break;
        }
        let count = (samples.len() - WINDOW) / HOP + 1;
        let batch: Vec<T> = (0..count)
            .into_par_iter()
            .map(|i| compute(&samples[i * HOP..i * HOP + WINDOW]))
            .collect();
        scores.extend(batch);
        if !on_batch(&scores) {
            stream.kill();
            return Err(CANCELLED.to_string());
        }
        samples.drain(..count * HOP);
        if duration > 0.0 {
            progress(((scores.len() as f64 * STEP / duration * 100.0) as u32).min(99));
        }
    }
    drop(samples);
    stream.finish()?;
    Ok(scores)
}

#[cfg(test)]
mod partial_tests {
    use super::partial_scores;

    #[test]
    fn partial_scores_keep_the_previous_mix_beyond_what_is_read() {
        assert_eq!(partial_scores(&[], &[0.1, 0.2]), [0.1, 0.2]);
        assert_eq!(partial_scores(&[0.5, 0.0, 0.3], &[0.1, 0.4]), [0.5, 0.4, 0.3]);
    }
}

/// Bornes du curseur de sensibilité.
pub const MIN_SENSITIVITY: f64 = 10.0;
pub const MAX_SENSITIVITY: f64 = 100.0;

/// Curseur 10-100 : haut = plus sensible = seuils plus bas (défaut 50).
pub fn sensitivity_thresholds(sensitivity: f64) -> (f64, f64) {
    let factor = 2f64.powf((DEFAULT_SENSITIVITY - sensitivity) / 25.0);
    ((HIGH_THRESHOLD * factor).min(0.95), LOW_THRESHOLD * factor)
}

/// Moyenne glissante alignée comme `np.convolve(..., mode='same')`.
fn smooth(scores: &[f32], width: usize) -> Vec<f64> {
    let n = scores.len();
    let coefficient = 1.0 / width as f64;
    let length = n.max(width);
    let offset = (n.min(width) - 1) / 2;
    (0..length)
        .map(|i| {
            let k = i + offset; // indice dans la convolution complète
            let first = (k + 1).saturating_sub(width);
            let last = k.min(n - 1);
            (first..=last).map(|j| scores[j] as f64 * coefficient).sum()
        })
        .collect()
}

/// P(musique) lissée comme pour la détection : c'est elle que le seuil compare.
pub fn smoothed_scores(scores: &[f32]) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }
    smooth(scores, ((SMOOTHING_S / STEP) as usize).max(1))
}

/// Transforme P(musique) en segments (début, fin) en secondes.
///
/// Le gap mesure le vrai temps sans musique : anticipation et post ne
/// s'appliquent qu'après cette fusion, pour ne pas ponter davantage que le gap.
pub fn scores_to_segments(
    scores: &[f32],
    sensitivity: f64,
    gap: f64,
    anticipation: f64,
    post: f64,
    duration: f64,
) -> Vec<(f64, f64)> {
    if scores.is_empty() {
        return Vec::new();
    }
    let (high, low) = sensitivity_thresholds(sensitivity);
    let width = ((SMOOTHING_S / STEP) as usize).max(1);
    let smooth = smooth(scores, width);

    let mut merged: Vec<(usize, usize)> = Vec::new();
    let mut start = None;
    for i in 0..=smooth.len() {
        let above = i < smooth.len() && smooth[i] > low;
        match (above, start) {
            (true, None) => start = Some(i),
            (false, Some(a)) => {
                start = None;
                if smooth[a..i].iter().cloned().fold(f64::MIN, f64::max) <= high {
                    continue;
                }
                match merged.last_mut() {
                    Some(last) if (a - last.1) as f64 * STEP <= gap => last.1 = i,
                    _ => merged.push((a, i)),
                }
            }
            _ => {}
        }
    }

    let limit = if duration > 0.0 { duration } else { scores.len() as f64 * STEP };
    let mut segments: Vec<(f64, f64)> = Vec::new();
    for (a, b) in merged {
        if ((b - a) as f64 * STEP) < MIN_DURATION_S {
            continue;
        }
        let begin = (a as f64 * STEP - anticipation).max(0.0);
        let finish = (b as f64 * STEP + post).min(limit);
        match segments.last_mut() {
            Some(last) if begin <= last.1 => last.1 = finish,
            _ => segments.push((begin, finish)),
        }
    }
    segments
}

/// P(musique) maximale entre les mixes analysés.
pub fn detect_music_probabilities(
    model: &Yamnet,
    video: &Path,
    duration: f64,
    mixes: &[Vec<String>],
    progress: &(dyn Fn(u32) + Sync),
) -> Result<Vec<f32>, String> {
    detect_music_probabilities_live(model, video, duration, mixes, progress, &mut |_| true)
}

/// Ce qu'on sait à un instant du calcul : le maximum de ce qui est lu pour le mix en cours et
/// du résultat des mix précédents, qui vaut seul au-delà.
fn partial_scores(previous: &[f32], current: &[f32]) -> Vec<f32> {
    let mut merged = previous.to_vec();
    merged.resize(merged.len().max(current.len()), 0.0);
    for (value, new) in merged.iter_mut().zip(current) {
        *value = value.max(*new);
    }
    merged
}

/// Comme `detect_music_probabilities`, en livrant les scores connus après chaque lot.
pub fn detect_music_probabilities_live(
    model: &Yamnet,
    video: &Path,
    duration: f64,
    mixes: &[Vec<String>],
    progress: &(dyn Fn(u32) + Sync),
    partial: &mut dyn FnMut(Vec<f32>) -> bool,
) -> Result<Vec<f32>, String> {
    let mut scores: Option<Vec<f32>> = None;
    for (index, mix) in mixes.iter().enumerate() {
        let report = |p: u32| progress((index as u32 * 100 + p) / mixes.len() as u32);
        let previous = scores.clone().unwrap_or_default();
        let current = for_each_window(
            video,
            mix,
            duration,
            &report,
            |so_far: &[f32]| partial(partial_scores(&previous, so_far)),
            |window| model.music_probability(window),
        )?;
        scores = Some(match scores {
            None => current,
            Some(previous) => previous.iter().zip(&current).map(|(a, b)| a.max(*b)).collect(),
        });
    }
    Ok(scores.unwrap_or_default())
}

/// Erreur rendue quand `partial` demande l'arrêt de la détection.
pub const CANCELLED: &str = "détection interrompue";

pub struct Settings {
    pub sensitivity: f64,
    pub gap: f64,
    pub anticipation: f64,
    pub post: f64,
}

/// Retourne (segments, dialogues atténués ?) pour une vidéo.
pub fn detect_music_segments(
    model: &Yamnet,
    video: &Path,
    settings: &Settings,
    progress: &(dyn Fn(u32) + Sync),
) -> Result<(Vec<(f64, f64)>, bool), String> {
    let (layout, duration) = probe_audio(video)?;
    let (mixes, reduced) = choose_mixes(video, &layout)?;
    let scores = detect_music_probabilities(model, video, duration, &mixes, progress)?;
    let segments = scores_to_segments(
        &scores,
        settings.sensitivity,
        settings.gap,
        settings.anticipation,
        settings.post,
        duration,
    );
    Ok((segments, reduced))
}

fn format_ts(ts: f64) -> String {
    let total = ts as u64;
    format!("{:02}-{:02}-{:02}", total / 3600, total % 3600 / 60, total % 60)
}

pub fn extract_segments(video: &Path, segments: &[(f64, f64)], output_dir: &Path) -> Result<Vec<PathBuf>, String> {
    std::fs::create_dir_all(output_dir).map_err(|e| e.to_string())?;
    // Container de sortie identique à la source : garantit la compatibilité
    // du flux copié (-c copy). Copier vers .mp4 depuis un .mkv/.avi peut échouer
    // selon les codecs (audio DTS, sous-titres...). On retombe sur .mp4 si besoin.
    let extension = video
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .filter(|e| ["mp4", "mkv", "avi", "mov", "webm"].contains(&e.as_str()))
        .unwrap_or_else(|| "mp4".to_string());
    let mut files = Vec::new();
    for (index, &(start, end)) in segments.iter().enumerate() {
        let name = format!("segment_{:03}_{}_{}.{}", index + 1, format_ts(start), format_ts(end), extension);
        let file = output_dir.join(name);
        let (start_text, length_text) = (start.to_string(), (end - start).to_string());
        // -ss en option d'ENTREE : ffmpeg cale automatiquement le début sur la
        // keyframe précédente (« keyframe cut » façon LosslessCut) → image dès la
        // 1re frame et A/V synchro. Ne PAS ajouter de -ss en sortie (tronque la
        // vidéo et provoque le démarrage « audio sans image »).
        let (log, ok) = ffmpeg::run(&[
            OsStr::new("-hide_banner"),
            OsStr::new("-y"),
            OsStr::new("-ss"),
            OsStr::new(&start_text),
            OsStr::new("-i"),
            video.as_os_str(),
            OsStr::new("-t"),
            OsStr::new(&length_text),
            OsStr::new("-c"),
            OsStr::new("copy"),
            OsStr::new("-avoid_negative_ts"),
            OsStr::new("make_zero"),
            file.as_os_str(),
        ])?;
        if !ok {
            return Err(format!("ffmpeg a échoué : {}", log.lines().last().unwrap_or("").trim()));
        }
        files.push(file);
    }
    Ok(files)
}

/// Premier nom de dossier d'export libre.
pub fn export_folder(video: &Path, destination: Option<&Path>) -> PathBuf {
    let (parent, name) = match destination {
        Some(folder) => {
            let stem = video.file_stem().unwrap_or_default().to_string_lossy();
            (folder, format!("music-sequences - {stem}"))
        }
        None => (video.parent().unwrap_or(Path::new(".")), "music-sequences".to_string()),
    };
    let mut export = parent.join(&name);
    let mut count = 1;
    while export.exists() {
        export = parent.join(format!("{name} {count}"));
        count += 1;
    }
    export
}

/// Traite une vidéo : détection puis découpe, à côté d'elle ou dans `destination`.
/// Retourne un court résumé (ex. « 12 clips dans music-sequences »).
/// Avec `destination`, les clips vont dans un sous-dossier de celle-ci, au nom de la
/// vidéo pour ne pas mélanger les films d'un même lot.
pub fn process_video(
    model: &Yamnet,
    video: &Path,
    settings: &Settings,
    destination: Option<&Path>,
    status: &(dyn Fn(String) + Sync),
) -> Result<String, String> {
    let (segments, reduced) = detect_music_segments(model, video, settings, &|p| {
        status(format!("détection des séquences musicales… {p} %"))
    })?;
    let note = if reduced { "" } else { " (voix non séparables : musique sous dialogues parfois manquée)" };
    if segments.is_empty() {
        return Ok(format!("aucune séquence détectée{note}"));
    }
    status(format!("{} séquences, découpage…", segments.len()));
    let export = export_folder(video, destination);
    let files = extract_segments(video, &segments, &export)?;
    let folder = export.file_name().unwrap_or_default().to_string_lossy();
    Ok(format!("{} clips dans {folder}{note}", files.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_folder_follows_the_chosen_destination() {
        let video = Path::new("Z://absent//Mon film.mkv");
        assert_eq!(export_folder(video, None), Path::new("Z://absent//music-sequences"));
        assert_eq!(
            export_folder(video, Some(Path::new("Z://clips"))),
            Path::new("Z://clips//music-sequences - Mon film")
        );
    }

    #[test]
    fn smoothing_matches_numpy_same_mode() {
        // np.convolve([1,2,3,4,5,6,7,8], np.ones(6)/6, mode='same') * 6
        let got = smooth(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 6);
        let expected = [6.0, 10.0, 15.0, 21.0, 27.0, 33.0, 30.0, 26.0];
        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(expected) {
            assert!((g * 6.0 - e).abs() < 1e-9, "{g} {e}");
        }
    }
}
