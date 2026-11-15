//! Le projet d'une vidéo : un fichier lisible, créé à côté de la vidéo dès l'import
//! et sauvegardé automatiquement, comme le .llc de LosslessCut.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const PROJECT_VERSION: u32 = 1;
pub const MIN_SEGMENT_S: f64 = 0.2;

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug, Default)]
pub enum Origin {
    /// Vient de la détection : un changement de réglage peut le remplacer.
    #[default]
    Detected,
    /// Détecté puis retouché : conservé.
    Edited,
    /// Ajouté à la main : conservé.
    Manual,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug, Default)]
pub enum MusicKind {
    #[default]
    Unknown,
    NonDiegetic,
    Diegetic,
    Mixed,
}

impl MusicKind {
    pub const ALL: [MusicKind; 4] =
        [MusicKind::Unknown, MusicKind::NonDiegetic, MusicKind::Diegetic, MusicKind::Mixed];

    pub fn label(self) -> &'static str {
        match self {
            MusicKind::Unknown => "Non renseigné",
            MusicKind::NonDiegetic => "Extradiégétique",
            MusicKind::Diegetic => "Diégétique",
            MusicKind::Mixed => "Mixte",
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Default)]
pub struct Segment {
    pub id: u64,
    pub start: f64,
    pub end: f64,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub kind: MusicKind,
    #[serde(default)]
    pub notes: String,
    #[serde(default)]
    pub origin: Origin,
    /// Valeurs du tableau corrigées à la main, à la place de ce qui est calculé.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub overrides: BTreeMap<String, String>,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct DetectionSettings {
    pub sensitivity: f64,
    pub gap: f64,
    pub anticipation: f64,
    pub post: f64,
}

impl Default for DetectionSettings {
    /// Sensibilité et gap de l'utilitaire d'extraction, avec des marges plus courtes.
    fn default() -> Self {
        DetectionSettings { sensitivity: 50.0, gap: 15.0, anticipation: 2.0, post: 2.0 }
    }
}

/// Cibles de loudness de la vidéo, comme dans un compteur de mixage.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub struct LoudnessTarget {
    pub true_peak: f32,
    pub integrated: f32,
    pub tolerance: f32,
}

impl Default for LoudnessTarget {
    fn default() -> Self {
        LoudnessTarget { true_peak: -2.0, integrated: -24.0, tolerance: 2.0 }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Default)]
pub struct Project {
    pub version: u32,
    pub media_file_name: String,
    #[serde(default)]
    pub duration: f64,
    #[serde(default)]
    pub settings: DetectionSettings,
    #[serde(default)]
    pub scores: Vec<f32>,
    #[serde(default)]
    pub segments: Vec<Segment>,
    #[serde(default)]
    pub next_id: u64,
    #[serde(default)]
    pub film_notes: String,
    #[serde(default)]
    pub loudness_target: LoudnessTarget,
    /// Piste audio étudiée (0 : la première du fichier).
    #[serde(default)]
    pub audio_track: usize,
    /// Faders du mixeur, en dB ; seuls ceux qui ne sont pas à 0 dB y figurent.
    #[serde(default)]
    pub channel_gains: BTreeMap<usize, f32>,
    /// Fader du mix, en dB.
    #[serde(default)]
    pub master_gain_db: f32,
    /// Détection coupée : plus de séquences automatiques, ni de calcul.
    #[serde(default)]
    pub detection_off: bool,
    /// Les séquences détectées, mises de côté avec leurs noms et notes pendant que la détection est coupée.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub set_aside: Vec<Segment>,
}

pub fn project_path(video: &Path) -> PathBuf {
    let stem = video.file_stem().unwrap_or_default().to_string_lossy();
    let path = video.with_file_name(format!("{stem}.mx-lab.json"));
    // Projet créé sous l'ancien nom (Music Studio) : on le reprend
    let old = video.with_file_name(format!("{stem}.music-studio.json"));
    if !path.exists() && old.is_file() {
        let _ = std::fs::rename(&old, &path);
    }
    path
}

impl Project {
    pub fn new(video: &Path) -> Self {
        Project {
            version: PROJECT_VERSION,
            media_file_name: video.file_name().unwrap_or_default().to_string_lossy().into_owned(),
            ..Default::default()
        }
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&text).map_err(|e| e.to_string())
    }

    /// Écriture atomique : une coupure ne laisse jamais un projet tronqué.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        let partial = path.with_extension(format!("json.{}.tmp", std::process::id()));
        std::fs::write(&partial, text).map_err(|e| e.to_string())?;
        std::fs::rename(&partial, path).map_err(|e| {
            let _ = std::fs::remove_file(&partial);
            e.to_string()
        })
    }

    pub fn new_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    pub fn segment(&self, id: u64) -> Option<&Segment> {
        self.segments.iter().find(|s| s.id == id)
    }

    pub fn segment_mut(&mut self, id: u64) -> Option<&mut Segment> {
        self.segments.iter_mut().find(|s| s.id == id)
    }

    /// Remplace les séquences détectées, sans toucher à ce qui a été retouché ou ajouté.
    /// Une séquence qui recouvre une ancienne en reprend l'identité : la détection progresse
    /// sans faire perdre la sélection ni ce qui a déjà été saisi.
    /// Coupe ou rétablit la détection. Coupée, ses séquences sont mises de côté ; rétablie,
    /// elles reviennent, puis `apply_detection` les recale sur les scores.
    pub fn set_detection(&mut self, on: bool) {
        if on != self.detection_off {
            return;
        }
        self.detection_off = !on;
        if on {
            let mut back = std::mem::take(&mut self.set_aside);
            back.retain(|s| !self.segments.iter().any(|kept| s.start < kept.end && kept.start < s.end));
            self.segments.extend(back);
            self.segments.sort_by(|a, b| a.start.total_cmp(&b.start));
        } else {
            let (detected, kept): (Vec<Segment>, Vec<Segment>) =
                std::mem::take(&mut self.segments).into_iter().partition(|s| s.origin == Origin::Detected);
            self.segments = kept;
            self.set_aside = detected;
        }
    }

    pub fn apply_detection(&mut self, detected: &[(f64, f64)]) {
        let (mut previous, kept): (Vec<Segment>, Vec<Segment>) =
            std::mem::take(&mut self.segments).into_iter().partition(|s| s.origin == Origin::Detected);
        self.segments = kept;
        for &(start, end) in detected {
            let free = !self.segments.iter().any(|s| s.origin != Origin::Detected && start < s.end && s.start < end);
            if free {
                let overlap = |s: &Segment| end.min(s.end) - start.max(s.start);
                let closest = (0..previous.len())
                    .filter(|&index| overlap(&previous[index]) > 0.0)
                    .max_by(|&a, &b| overlap(&previous[a]).total_cmp(&overlap(&previous[b])));
                let segment = match closest {
                    Some(index) => Segment { start, end, ..previous.swap_remove(index) },
                    None => Segment { id: self.new_id(), start, end, origin: Origin::Detected, ..Default::default() },
                };
                self.segments.push(segment);
            }
        }
        self.segments.sort_by(|a, b| a.start.total_cmp(&b.start));
    }

    /// Limites libres autour d'une séquence : la fin de la précédente, le début de la suivante.
    fn neighbours(&self, id: u64) -> (f64, f64) {
        let Some(current) = self.segment(id) else { return (0.0, self.limit()) };
        let previous = self
            .segments
            .iter()
            .filter(|s| s.id != id && s.start < current.start)
            .map(|s| s.end)
            .fold(0.0, f64::max);
        let next = self
            .segments
            .iter()
            .filter(|s| s.id != id && s.start >= current.start)
            .map(|s| s.start)
            .fold(self.limit(), f64::min);
        (previous, next)
    }

    fn limit(&self) -> f64 {
        if self.duration > 0.0 {
            self.duration
        } else {
            f64::MAX
        }
    }

    fn mark_edited(&mut self, id: u64) {
        if let Some(segment) = self.segment_mut(id) {
            if segment.origin == Origin::Detected {
                segment.origin = Origin::Edited;
            }
        }
    }

    pub fn set_start(&mut self, id: u64, time: f64) {
        let (previous, _) = self.neighbours(id);
        self.mark_edited(id);
        if let Some(segment) = self.segment_mut(id) {
            segment.start = time.max(previous).min(segment.end - MIN_SEGMENT_S);
        }
    }

    pub fn set_end(&mut self, id: u64, time: f64) {
        let (_, next) = self.neighbours(id);
        self.mark_edited(id);
        if let Some(segment) = self.segment_mut(id) {
            segment.end = time.min(next).max(segment.start + MIN_SEGMENT_S);
        }
    }

    /// Déplace toute une séquence sans changer sa durée, entre ses voisines.
    pub fn move_to(&mut self, id: u64, start: f64) {
        let (previous, next) = self.neighbours(id);
        self.mark_edited(id);
        if let Some(segment) = self.segment_mut(id) {
            let length = segment.end - segment.start;
            let start = start.max(previous).min((next - length).max(previous));
            segment.start = start;
            segment.end = start + length;
        }
    }

    /// Ajoute une séquence si la place est libre. Retourne son identifiant.
    pub fn add_manual(&mut self, start: f64, wanted_end: f64) -> Option<u64> {
        let limit = self.limit();
        if start >= limit || self.segments.iter().any(|s| s.start <= start && start < s.end) {
            return None;
        }
        let next = self.segments.iter().filter(|s| s.start > start).map(|s| s.start).fold(limit, f64::min);
        let end = wanted_end.min(next);
        if end - start < MIN_SEGMENT_S {
            return None;
        }
        let id = self.new_id();
        self.segments.push(Segment { id, start, end, origin: Origin::Manual, ..Default::default() });
        self.segments.sort_by(|a, b| a.start.total_cmp(&b.start));
        Some(id)
    }

    /// Fusionne une séquence avec sa voisine si le bord donné la touche, à `tolerance` près.
    /// La séquence garde son identifiant ; noms et notes s'ajoutent.
    pub fn merge_at_edge(&mut self, id: u64, end_edge: bool, tolerance: f64) -> bool {
        let Some(current) = self.segment(id).cloned() else { return false };
        let other = self
            .segments
            .iter()
            .filter(|s| s.id != id)
            .find(|s| if end_edge { (s.start - current.end).abs() <= tolerance } else { (s.end - current.start).abs() <= tolerance })
            .cloned();
        let Some(other) = other else { return false };
        self.segments.retain(|s| s.id != other.id);
        let join = |a: &str, b: &str| match (a.is_empty(), b.is_empty()) {
            (true, _) => b.to_string(),
            (_, true) => a.to_string(),
            _ if a == b => a.to_string(),
            _ => format!("{a} / {b}"),
        };
        let (first, second) = if end_edge { (&current, &other) } else { (&other, &current) };
        let (name, notes) = (join(&first.name, &second.name), join(&first.notes, &second.notes));
        if let Some(segment) = self.segment_mut(id) {
            segment.start = first.start;
            segment.end = second.end;
            segment.name = name;
            segment.notes = notes;
            if segment.kind == MusicKind::Unknown {
                segment.kind = other.kind;
            }
            // Ce qui avait été saisi décrivait un seul des deux morceaux
            segment.overrides.clear();
            if segment.origin == Origin::Detected {
                segment.origin = Origin::Edited;
            }
        }
        true
    }

    pub fn total_music(&self) -> f64 {
        // Partir de 0 explicitement : la somme d'une liste vide de f64 vaut -0, affiché « -0 % »
        self.segments.iter().fold(0.0, |total, s| total + (s.end - s.start))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn touching_sequences_merge_and_keep_their_texts() {
        let mut project = Project::new(Path::new("C://films//film.mkv"));
        project.duration = 100.0;
        project.apply_detection(&[(10.0, 20.0), (30.0, 40.0), (60.0, 70.0)]);
        let (a, b) = (project.segments[0].id, project.segments[1].id);
        project.segment_mut(a).unwrap().name = "Un".into();
        project.segment_mut(b).unwrap().name = "Deux".into();
        assert!(!project.merge_at_edge(a, true, 0.5), "encore loin : rien ne change");
        project.set_end(a, 35.0); // arrêtée par la voisine
        assert!(project.merge_at_edge(a, true, 0.0));
        assert_eq!(project.segments.len(), 2);
        let merged = project.segment(a).unwrap();
        assert_eq!((merged.start, merged.end, merged.name.as_str()), (10.0, 40.0, "Un / Deux"));
        // Par le début, avec une petite tolérance
        let c = project.segments[1].id;
        project.set_start(c, 40.2);
        assert!(project.merge_at_edge(c, false, 0.5));
        let merged = project.segment(c).unwrap();
        assert_eq!((merged.start, merged.end), (10.0, 70.0));
    }

    fn project() -> Project {
        let mut project = Project::new(Path::new("C://films//film.mkv"));
        project.duration = 100.0;
        project.apply_detection(&[(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]);
        project
    }

    #[test]
    fn project_file_sits_next_to_the_video() {
        assert_eq!(
            project_path(Path::new("C://films//Mon.Film.2024.mkv")),
            Path::new("C://films//Mon.Film.2024.mx-lab.json")
        );
    }

    #[test]
    fn handles_stop_at_the_neighbours() {
        let mut project = project();
        let middle = project.segments[1].id;
        project.set_start(middle, 5.0);
        assert_eq!(project.segment(middle).unwrap().start, 20.0);
        project.set_end(middle, 95.0);
        assert_eq!(project.segment(middle).unwrap().end, 50.0);
    }

    #[test]
    fn handles_cannot_cross_each_other() {
        let mut project = project();
        let first = project.segments[0].id;
        project.set_start(first, 25.0);
        assert!((project.segment(first).unwrap().start - (20.0 - MIN_SEGMENT_S)).abs() < 1e-9);
        project.set_end(first, 0.0);
        assert!(project.segment(first).unwrap().end > project.segment(first).unwrap().start);
    }

    #[test]
    fn last_handle_stops_at_the_end_of_the_film() {
        let mut project = project();
        let last = project.segments[2].id;
        project.set_end(last, 500.0);
        assert_eq!(project.segment(last).unwrap().end, 100.0);
    }

    #[test]
    fn a_growing_detection_keeps_the_identity_of_its_sequences() {
        let mut project = Project::default();
        project.apply_detection(&[(10.0, 20.0), (40.0, 50.0)]);
        let (first, second) = (project.segments[0].id, project.segments[1].id);
        project.segment_mut(second).unwrap().notes = "Thème".into();
        project.apply_detection(&[(10.0, 25.0), (40.0, 52.0), (80.0, 90.0)]);
        assert_eq!(project.segments[0].id, first);
        assert_eq!(project.segments[1].id, second);
        assert_eq!(project.segments[1].end, 52.0);
        assert_eq!(project.segments[1].notes, "Thème");
        assert!(project.segments[2].id != first && project.segments[2].id != second);
    }

    #[test]
    fn redetection_keeps_edited_and_manual_sequences() {
        let mut project = project();
        let edited = project.segments[0].id;
        project.set_end(edited, 22.0);
        let manual = project.add_manual(70.0, 75.0).unwrap();
        // Une détection qui chevauche la séquence retouchée ne l'écrase pas
        project.apply_detection(&[(15.0, 25.0), (45.0, 55.0)]);
        assert_eq!(project.segments.len(), 3);
        assert_eq!(project.segment(edited).unwrap().end, 22.0);
        assert_eq!(project.segment(edited).unwrap().origin, Origin::Edited);
        assert!(project.segment(manual).is_some());
        assert_eq!((project.segments[1].start, project.segments[1].end), (45.0, 55.0));
    }

    #[test]
    fn manual_sequence_stops_before_the_next_one() {
        let mut project = project();
        let id = project.add_manual(25.0, 60.0).unwrap();
        assert_eq!(project.segment(id).unwrap().end, 30.0);
        assert_eq!(project.segments[1].id, id);
        // Pas de séquence dans une séquence
        assert_eq!(project.add_manual(35.0, 37.0), None);
    }

    #[test]
    fn moving_keeps_the_length_and_stays_between_neighbours() {
        let mut project = project();
        let middle = project.segments[1].id;
        project.move_to(middle, 42.0);
        assert_eq!((project.segment(middle).unwrap().start, project.segment(middle).unwrap().end), (40.0, 50.0));
        project.move_to(middle, 0.0);
        assert_eq!((project.segment(middle).unwrap().start, project.segment(middle).unwrap().end), (20.0, 30.0));
    }

    #[test]
    fn empty_project_has_no_negative_zero_of_music() {
        let project = Project::new(Path::new("film.mkv"));
        assert_eq!(format!("{:.0}", project.total_music() / 100.0 * 100.0), "0");
    }

    /// Les projets créés quand la fiche venait de TMDB doivent toujours s'ouvrir, notes comprises.
    #[test]
    fn older_project_with_a_film_sheet_still_opens() {
        let text = r#"{
            "version": 1,
            "media_file_name": "film.mkv",
            "film": { "tmdb_id": 12, "title": "Film", "year": "2020", "directors": [], "composers": [], "overview": "" },
            "film_notes": "Mes notes",
            "segments": [{ "id": 1, "start": 1.0, "end": 2.0, "notes": "Thème" }]
        }"#;
        let project: Project = serde_json::from_str(text).unwrap();
        assert_eq!(project.film_notes, "Mes notes");
        assert_eq!(project.segments[0].notes, "Thème");
    }

    #[test]
    fn project_survives_the_file() {
        let folder = std::env::temp_dir().join(format!("music-studio-project-{}", std::process::id()));
        std::fs::create_dir_all(&folder).unwrap();
        let path = folder.join("film.mx-lab.json");
        let mut project = project();
        project.scores = vec![0.125, 0.5];
        project.segments[0].name = "Thème d'ouverture".into();
        project.segments[0].kind = MusicKind::NonDiegetic;
        project.film_notes = "Notes".into();
        project.save(&path).unwrap();
        let loaded = Project::load(&path).unwrap();
        std::fs::remove_dir_all(&folder).unwrap();
        assert_eq!(loaded, project);
    }
}
