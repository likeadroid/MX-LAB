//! Mood, ensembles et instruments d'une séquence, d'après les classes de YAMNet.
//! Ces classes sont très discrètes sur une musique de film (souvent moins de 1 %) :
//! on les lit comme des indices, et on préfère ne rien dire plutôt qu'inventer.

use mx_lab::detect;
use mx_lab::yamnet::{self, Yamnet};
use std::path::{Path, PathBuf};

/// (classes du modèle, nom affiché). Plusieurs classes peuvent former un même nom.
type Tag = (&'static [&'static str], &'static str);

const MOODS: [Tag; 6] = [
    (&["Happy music"], "Joyeux"),
    (&["Sad music"], "Triste"),
    (&["Tender music"], "Tendre"),
    (&["Exciting music"], "Exaltant"),
    (&["Angry music"], "Colère"),
    (&["Scary music"], "Inquiétant"),
];

const ENSEMBLES: [Tag; 6] = [
    (&["Orchestra"], "Orchestre"),
    (&["String section", "Bowed string instrument"], "Cordes"),
    (&["Brass instrument"], "Cuivres"),
    (&["Wind instrument, woodwind instrument"], "Bois"),
    (&["Choir"], "Chœur"),
    (&["Percussion"], "Percussions"),
];

const INSTRUMENTS: [Tag; 28] = [
    (&["Piano", "Electric piano"], "Piano"),
    (&["Synthesizer"], "Synthé"),
    (&["Organ", "Electronic organ", "Hammond organ"], "Orgue"),
    (&["Harpsichord"], "Clavecin"),
    (&["Guitar", "Acoustic guitar"], "Guitare"),
    (&["Electric guitar"], "Guitare élec."),
    (&["Bass guitar"], "Basse"),
    (&["Harp"], "Harpe"),
    (&["Violin, fiddle"], "Violon"),
    (&["Cello"], "Violoncelle"),
    (&["Double bass"], "Contrebasse"),
    (&["Pizzicato"], "Pizzicato"),
    (&["French horn"], "Cor"),
    (&["Trumpet"], "Trompette"),
    (&["Trombone"], "Trombone"),
    (&["Flute"], "Flûte"),
    (&["Saxophone"], "Saxophone"),
    (&["Clarinet"], "Clarinette"),
    (&["Timpani"], "Timbales"),
    (&["Drum kit"], "Batterie"),
    (&["Drum", "Snare drum", "Bass drum"], "Tambour"),
    (&["Cymbal"], "Cymbales"),
    (&["Glockenspiel"], "Glockenspiel"),
    (&["Marimba, xylophone"], "Marimba"),
    (&["Vibraphone"], "Vibraphone"),
    (&["Bell", "Tubular bells", "Chime"], "Cloches"),
    (&["Gong"], "Gong"),
    (&["Theremin"], "Thérémine"),
];

/// Un mood doit se détacher du reste de la vidéo...
const MOOD_MIN_DEVIATION: f32 = 1.0;
/// ...et atteindre un niveau minimal : sinon, un écart énorme sur presque rien passerait.
const MOOD_MIN_LEVEL: f32 = 0.001;
const ENSEMBLE_MIN_LEVEL: f32 = 0.005;
const INSTRUMENT_MIN_LEVEL: f32 = 0.003;
const MAX_ENSEMBLES: usize = 2;
const MAX_INSTRUMENTS: usize = 3;
/// Blocs de comparaison de la vidéo (environ 30 s), indépendants du découpage en séquences.
const BASELINE_WINDOWS: usize = 62;
const CACHE_VERSION: u32 = 1;

fn all_tags() -> impl Iterator<Item = &'static Tag> {
    MOODS.iter().chain(&ENSEMBLES).chain(&INSTRUMENTS)
}

/// Les classes à lire dans le modèle, dans l'ordre des noms.
fn class_names() -> Vec<&'static str> {
    all_tags().flat_map(|tag| tag.0.iter().copied()).collect()
}

fn model_classes() -> Result<Vec<usize>, String> {
    let labels = yamnet::class_labels();
    class_names()
        .into_iter()
        .map(|name| labels.iter().position(|l| *l == name).ok_or(format!("classe absente du modèle : {name}")))
        .collect()
}

/// Ce que le modèle voit de la vidéo : une probabilité par étiquette et par fenêtre de 0,48 s.
pub struct ClassTrack {
    tags: usize,
    values: Vec<f32>,
    /// Moyenne et écart de chaque mood sur la vidéo.
    baseline: Vec<(f32, f32)>,
}

/// Ce qu'on affiche pour une séquence. Vide : rien d'assez net.
#[derive(Clone, Default, PartialEq, Debug)]
pub struct SequenceTags {
    pub mood: String,
    pub ensembles: String,
    pub instruments: String,
}

fn sigmoid(logit: f32) -> f32 {
    1.0 / (1.0 + (-logit).exp())
}

impl ClassTrack {
    /// `rows` : une ligne par fenêtre, un logit par classe de `class_names`.
    fn from_logits(rows: &[Vec<f32>]) -> Self {
        let counts: Vec<usize> = all_tags().map(|tag| tag.0.len()).collect();
        let mut values = Vec::with_capacity(rows.len() * counts.len());
        for row in rows {
            let mut at = 0;
            for &count in &counts {
                // Plusieurs classes pour un nom : la plus présente des deux
                let best = row[at..at + count].iter().map(|l| sigmoid(*l)).fold(0.0, f32::max);
                values.push(best);
                at += count;
            }
        }
        Self::new(values)
    }

    fn new(values: Vec<f32>) -> Self {
        let tags = all_tags().count();
        let mut track = ClassTrack { tags, values, baseline: Vec::new() };
        let windows = track.values.len() / tags;
        let blocks: Vec<usize> = (0..windows).step_by(BASELINE_WINDOWS).collect();
        track.baseline = (0..MOODS.len())
            .map(|mood| {
                let levels: Vec<f32> = blocks
                    .iter()
                    .map(|&first| track.mean(mood, first, (first + BASELINE_WINDOWS).min(windows)))
                    .collect();
                let mean = levels.iter().sum::<f32>() / levels.len().max(1) as f32;
                let variance = levels.iter().map(|l| (l - mean).powi(2)).sum::<f32>() / levels.len().max(1) as f32;
                (mean, variance.sqrt())
            })
            .collect();
        track
    }

    fn windows(&self) -> usize {
        self.values.len() / self.tags
    }

    fn mean(&self, tag: usize, first: usize, last: usize) -> f32 {
        if first >= last {
            return 0.0;
        }
        (first..last).map(|window| self.values[window * self.tags + tag]).sum::<f32>() / (last - first) as f32
    }

    pub fn describe(&self, start: f64, end: f64) -> SequenceTags {
        let first = ((start.max(0.0) / detect::STEP) as usize).min(self.windows());
        let last = ((end / detect::STEP).ceil() as usize).clamp(first, self.windows());
        let strongest = |offset: usize, tags: &[Tag], floor: f32, limit: usize| {
            let mut found: Vec<(f32, &str)> = tags
                .iter()
                .enumerate()
                .map(|(index, tag)| (self.mean(offset + index, first, last), tag.1))
                .filter(|(level, _)| *level >= floor)
                .collect();
            found.sort_by(|a, b| b.0.total_cmp(&a.0));
            found.iter().take(limit).map(|f| f.1).collect::<Vec<_>>().join(", ")
        };
        // Le mood qui se détache le plus de la vidéo : « inquiétant » sort partout en valeur brute
        let mood = MOODS
            .iter()
            .enumerate()
            .filter_map(|(index, tag)| {
                let level = self.mean(index, first, last);
                let (mean, deviation) = self.baseline[index];
                let distance = (level - mean) / deviation.max(1e-6);
                (level >= MOOD_MIN_LEVEL && distance >= MOOD_MIN_DEVIATION).then_some((distance, tag.1))
            })
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .map_or(String::new(), |found| found.1.to_string());
        SequenceTags {
            mood,
            ensembles: strongest(MOODS.len(), &ENSEMBLES, ENSEMBLE_MIN_LEVEL, MAX_ENSEMBLES),
            instruments: strongest(MOODS.len() + ENSEMBLES.len(), &INSTRUMENTS, INSTRUMENT_MIN_LEVEL, MAX_INSTRUMENTS),
        }
    }

    fn save(&self, path: &Path) -> Result<(), String> {
        let mut bytes = Vec::with_capacity(self.values.len() * 4 + 8);
        bytes.extend_from_slice(&CACHE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(self.tags as u32).to_le_bytes());
        bytes.extend(self.values.iter().flat_map(|v| v.to_le_bytes()));
        let partial = path.with_extension(format!("{}.tmp", std::process::id()));
        let compressed = zstd::encode_all(bytes.as_slice(), 3).map_err(|e| e.to_string())?;
        std::fs::write(&partial, compressed).map_err(|e| e.to_string())?;
        std::fs::rename(&partial, path).map_err(|e| e.to_string())
    }

    fn load(path: &Path) -> Option<Self> {
        let bytes = zstd::decode_all(std::fs::File::open(path).ok()?).ok()?;
        let version = u32::from_le_bytes(bytes.get(0..4)?.try_into().ok()?);
        let tags = u32::from_le_bytes(bytes.get(4..8)?.try_into().ok()?) as usize;
        if version != CACHE_VERSION || tags != all_tags().count() {
            return None;
        }
        Some(Self::new(bytes[8..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()))
    }
}

fn cache_path(video: &Path, track: usize) -> Option<PathBuf> {
    crate::signals::cache_path(video, &crate::signals::track_cache_kind("tags", track))
}

/// Lit le cache, ou analyse la vidéo. Le premier mix est celui qui atténue les dialogues.
pub fn analyze(video: &Path, track: usize, layout: &str, duration: f64, progress: &(dyn Fn(u32) + Sync)) -> Result<ClassTrack, String> {
    let cache = cache_path(video, track);
    if let Some(track) = cache.as_deref().and_then(ClassTrack::load) {
        return Ok(track);
    }
    let classes = model_classes()?;
    let model = std::panic::catch_unwind(Yamnet::load).map_err(|_| "chargement du modèle impossible".to_string())?;
    let (mixes, _) = detect::choose_mixes_on(video, track, layout)?;
    let mix = mixes.first().ok_or("aucun mix à analyser")?;
    let rows = detect::class_logits(&model, video, mix, &classes, duration, progress)?;
    let track = ClassTrack::from_logits(&rows);
    if let Some(cache) = &cache {
        let _ = track.save(cache);
    }
    Ok(track)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index_of(wanted: &str) -> usize {
        all_tags().position(|tag| tag.1 == wanted).unwrap()
    }

    /// Film de `windows` fenêtres : `level` partout, et les valeurs données sur [20, 40[.
    fn track(windows: usize, level: f32, inside: &[(&str, f32)]) -> ClassTrack {
        let tags = all_tags().count();
        let mut values = vec![level; windows * tags];
        for window in 20..40 {
            for (name, value) in inside {
                values[window * tags + index_of(name)] = *value;
            }
        }
        ClassTrack::new(values)
    }

    fn inside() -> (f64, f64) {
        (20.0 * detect::STEP, 40.0 * detect::STEP)
    }

    #[test]
    fn every_tag_exists_in_the_model() {
        let classes = model_classes().unwrap();
        assert_eq!(classes.len(), class_names().len());
        let labels = yamnet::class_labels();
        assert_eq!(labels.len(), 521);
        assert_eq!(labels[yamnet::MUSIC_CLASS], "Music");
    }

    #[test]
    fn nothing_clear_stays_empty() {
        let (a, b) = inside();
        assert_eq!(track(2000, 0.0001, &[]).describe(a, b), SequenceTags::default());
    }

    #[test]
    fn a_mood_must_stand_out_from_the_film() {
        let (a, b) = inside();
        let standing_out = track(2000, 0.0005, &[("Triste", 0.01)]);
        assert_eq!(standing_out.describe(a, b).mood, "Triste");
        // Présent partout de la même façon : ça ne dit rien de cette séquence
        assert_eq!(track(2000, 0.01, &[]).describe(a, b).mood, "");
    }

    #[test]
    fn a_huge_deviation_on_almost_nothing_is_not_a_mood() {
        let (a, b) = inside();
        assert_eq!(track(2000, 0.0, &[("Colère", 0.0005)]).describe(a, b).mood, "");
    }

    #[test]
    fn the_mood_furthest_from_the_film_wins_over_the_highest_value() {
        let tags = all_tags().count();
        let mut values = vec![0.0f32; 2000 * tags];
        for window in 0..2000 {
            // « Inquiétant » est haut et varie partout ; « Tendre » est bas et stable
            values[window * tags + index_of("Inquiétant")] = 0.003 + 0.002 * ((window / 62) % 2) as f32;
            values[window * tags + index_of("Tendre")] = 0.001;
        }
        for window in 20..40 {
            values[window * tags + index_of("Inquiétant")] = 0.006;
            values[window * tags + index_of("Tendre")] = 0.004;
        }
        let (a, b) = inside();
        assert_eq!(ClassTrack::new(values).describe(a, b).mood, "Tendre");
    }

    #[test]
    fn ensembles_and_instruments_keep_the_strongest_above_their_floor() {
        let (a, b) = inside();
        let tags = track(
            2000,
            0.0,
            &[
                ("Orchestre", 0.012),
                ("Chœur", 0.006),
                ("Cuivres", 0.004),
                ("Cordes", 0.008),
                ("Violon", 0.01),
                ("Timbales", 0.004),
                ("Piano", 0.0035),
                ("Harpe", 0.0032),
                ("Flûte", 0.002),
            ],
        )
        .describe(a, b);
        assert_eq!(tags.ensembles, "Orchestre, Cordes");
        assert_eq!(tags.instruments, "Violon, Timbales, Piano");
    }

    #[test]
    fn several_classes_can_form_one_tag() {
        // Logits : 2 classes pour « Cordes », on garde la plus présente
        let names = class_names();
        let mut row = vec![-20.0f32; names.len()];
        row[names.iter().position(|n| *n == "Bowed string instrument").unwrap()] = -3.0;
        let track = ClassTrack::from_logits(&vec![row; 100]);
        let tags = track.describe(0.0, 100.0 * detect::STEP);
        assert_eq!(tags.ensembles, "Cordes");
    }

    #[test]
    fn track_survives_the_cache() {
        let folder = std::env::temp_dir().join(format!("music-tags-{}", std::process::id()));
        std::fs::create_dir_all(&folder).unwrap();
        let path = folder.join("tags.zst");
        let original = track(500, 0.0005, &[("Triste", 0.01), ("Violon", 0.02)]);
        original.save(&path).unwrap();
        let loaded = ClassTrack::load(&path).unwrap();
        std::fs::remove_dir_all(&folder).unwrap();
        let (a, b) = inside();
        assert_eq!(loaded.describe(a, b), original.describe(a, b));
        assert_eq!(loaded.describe(a, b).instruments, "Violon");
    }

    #[test]
    fn portion_outside_the_film_is_empty() {
        let tags = track(100, 0.02, &[]).describe(1000.0, 2000.0);
        assert_eq!(tags, SequenceTags::default());
    }
}
