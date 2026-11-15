#![cfg_attr(not(test), windows_subsystem = "windows")]

//! MX-LAB : étudier la musique à l'image.
//! Timeline de la vidéo, détection de la musique, onde, spectre, loudness, lecteur précis,
//! correction et export des séquences.

mod monitor;
mod mpv;
mod music_tags;
mod packaging;
mod project;
mod recent;
mod release_name;
mod renderer;
mod signals;
mod timeline;

use eframe::egui::{self, Color32, RichText};
use mx_lab::detect;
use monitor::Monitor;
use mx_lab::yamnet::Yamnet;
use project::{MusicKind, Origin, Project, Segment};
use release_name::ReleaseInfo;
use signals::{Detail, Signals, Strip};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};
use timeline::{AudioView, Drag, Tracks, View};

const VIDEO_EXTENSIONS: &[&str] = &["mp4", "mkv", "avi", "mov", "webm", "m4v", "ts", "mts", "m2ts", "mpg", "mpeg", "wmv"];
const TIMELINE_SHARE: f32 = 0.62;
/// Largeur que la barre de lecture demande sous la vidéo.
const MIN_VIDEO_WIDTH: f32 = 680.0;
const LEFT_PANEL_MIN: f32 = 230.0;
const RIGHT_PANEL_MIN: f32 = 300.0;
const RIGHT_PANEL_SHARE: f32 = 0.42;
const MIN_VIDEO_HEIGHT: f32 = 180.0;
const UNDO_DEPTH: usize = 200;
const NEW_SEQUENCE_S: f64 = 30.0;
pub const ATLAS_COLUMNS: usize = 40;
const SPEEDS: [f64; 7] = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0];
const SHUTTLE: [f64; 5] = [1.0, 2.0, 4.0, 8.0, 16.0];

const GAP_HELP: &str = "Interruption de la musique tolérée dans une séquence.";
const ANTICIPATION_HELP: &str = "Marge ajoutée avant chaque séquence détectée.";
const POST_HELP: &str = "Marge ajoutée après chaque séquence détectée.";
pub const SENSITIVITY_HELP: &str = "Plus elle est haute, plus la détection retient de passages : 50 par défaut, 70 pour des nappes ou drones discrets. Les séquences retouchées ou ajoutées à la main sont conservées.";
const SHORTCUTS: &str = "Espace : lecture / pause\n\
J / K / L : arrière / stop / avant (répéter pour accélérer)\n\
Flèches gauche / droite : image précédente / suivante\n\
Maj + flèches : 1 seconde\n\
Ctrl + flèches : 10 secondes\n\
Flèches haut / bas : séquence précédente / suivante\n\
Début / Fin : début / fin de la vidéo\n\
I / O : poser le début / la fin de la séquence choisie\n\
N : nouvelle séquence à la tête de lecture\n\
Suppr : supprimer la séquence choisie\n\
B : boucler sur la séquence choisie\n\
V : spectre ou onde\n\
Z : zoomer sur la séquence choisie\n\
F : toute la vidéo\n\
+ / − : zoomer / dézoomer\n\
Ctrl + Z / Ctrl + Y : annuler / rétablir\n\n\
Timeline :\n\
Molette : zoom, Maj + molette : défilement\n\
Clic : aller à cet instant\n\
Glisser sur les pistes : parcourir la vidéo\n\
Glisser une poignée : corriger un début ou une fin\n\
Maj en glissant : se caler sur les keyframes\n\
Glisser sur une zone vide des séquences : en créer une";

/// Résultats des travaux en arrière-plan.
pub enum Job {
    Probed(Result<(String, f64), String>),
    Chapters(Vec<signals::Chapter>),
    Info(signals::MediaInfo),
    TagsProgress(u32),
    Tags(Result<music_tags::ClassTrack, String>),
    DetectProgress(u32),
    DetectPartial(Vec<f32>),
    Detected(Result<Vec<f32>, String>),
    SignalsProgress(u32),
    Signals(Result<Signals, String>),
    StripProgress(u32),
    Strip(Result<Strip, String>),
    Detail(usize, Result<Detail, String>),
    SpectrumPart(usize, usize, Vec<u8>),
    SpectrumDone(usize),
    Exported(Result<(usize, PathBuf), String>),
}

pub struct Media {
    path: PathBuf,
    release: ReleaseInfo,
    project_path: PathBuf,
    project: Project,
    layout: Option<String>,
    smoothed: Vec<f64>,
    signals: Option<Arc<Signals>>,
    strip: Option<Arc<Strip>>,
    strip_texture: Option<egui::TextureHandle>,
    detect_progress: Option<u32>,
    /// Levé pour interrompre la détection en cours.
    detect_cancel: Arc<AtomicBool>,
    /// Détection rétablie avant que la piste audio soit connue : à lancer dès qu'elle l'est.
    detect_waiting: bool,
    signals_progress: Option<u32>,
    strip_progress: Option<u32>,
    undo: Vec<Vec<Segment>>,
    redo: Vec<Vec<Segment>>,
    dirty: bool,
    changed_at: Instant,
    /// Spectrogrammes prêts à afficher, par canal.
    overviews: HashMap<usize, (timeline::OverviewKey, egui::TextureHandle)>,
    /// Le spectre de chaque piste déjà demandée, complet ou en cours.
    spectra: HashMap<usize, SpectrumState>,
    details: HashMap<usize, (Detail, egui::TextureHandle)>,
    detail_pending: HashSet<usize>,
    chapters: Vec<signals::Chapter>,
    info: signals::MediaInfo,
    /// Ce que le modèle reconnaît au fil de la vidéo, et sa lecture pour chaque séquence.
    class_track: Option<Arc<music_tags::ClassTrack>>,
    tags_progress: Option<u32>,
    sequence_tags: HashMap<u64, ((f64, f64), music_tags::SequenceTags)>,
    /// Loudness de la vidéo sur la timeline, par canal.
    film_loudness: HashMap<usize, FilmLoudness>,
    /// Loudness du programme par séquence, pour le tableau, avec les bornes mesurées.
    sequence_loudness: HashMap<u64, ((f64, f64), signals::Measure)>,
}

#[derive(Default)]
pub struct SpectrumState {
    pub values: Vec<u8>,
    pub finished: bool,
    pub revision: u32,
}

pub struct FilmLoudness {
    pub measure: signals::Measure,
    pub running: Vec<f32>,
}

impl Media {
    pub fn mark_changed(&mut self) {
        self.dirty = true;
        self.changed_at = Instant::now();
    }

    pub fn detail_covers(&self, channel: usize, view: View) -> bool {
        self.details.get(&channel).is_some_and(|(d, _)| {
            d.start <= view.start && d.end >= view.end() && d.end - d.start <= view.span * 2.5
        })
    }
}

pub struct App {
    player: Option<mpv::Player>,
    player_error: Option<String>,
    recent: recent::RecentProjects,
    /// Moteur choisi au menu, et celui de ce lancement.
    render_choice: renderer::RenderBackend,
    render_running: renderer::RenderBackend,
    video_texture: Option<egui::TextureHandle>,
    video_size: [i32; 2],
    media: Option<Media>,
    generation: u64,
    jobs_sender: mpsc::Sender<(u64, Job)>,
    jobs: mpsc::Receiver<(u64, Job)>,
    picker: Option<mpsc::Receiver<Option<PathBuf>>>,
    fake_position: f64,
    pending_seek: Option<(f64, bool)>,
    view: View,
    view_changed_at: Instant,
    selected: Option<u64>,
    scroll_list: bool,
    listed_playing: Option<u64>,
    drag: Drag,
    timeline_body: egui::Rect,
    /// Une séquence choisie remplit la timeline.
    auto_zoom: bool,
    last_scrub: Option<(f64, bool)>,
    resume_after_drag: bool,
    #[cfg(test)]
    seek_requests: u32,
    requested_view: Option<View>,
    tracks: Tracks,
    waveform_normalized: bool,
    monitor: Monitor,
    /// Valeurs affichées par le mixeur, en dB : pistes, puis master gauche et droite.
    mixer_display: Vec<f32>,
    /// Short-term et momentary affichés par chaque barre.
    mixer_loudness_display: Vec<(f32, f32)>,
    applied_monitor: Monitor,
    /// Disposition pour laquelle la chaîne audio a été posée ; `None` : à poser.
    applied_layout: Option<Option<String>>,
    /// Routage en place : tant qu'il ne change pas, la chaîne audio n'est pas relancée.
    applied_filter: Option<String>,
    show_help: bool,
    loop_on: bool,
    applied_loop: Option<(f64, f64)>,
    reverse_speed: f64,
    reverse_last: Instant,
    shuttle: usize,
    settings_grabbed: bool,
    table_edit: Option<TableEdit>,
    /// Hauteur laissée sous le tableau, une fois réglée à la main.
    sheet_height: Option<f32>,
    sheet_height_now: f32,
    /// Ce que la fiche occupe vraiment : sans séquence choisie, puis avec.
    sheet_measured: [f32; 2],
    /// Le champ de la fiche de séquence en cours de saisie.
    field_edit: Option<TableEdit>,
    /// Début ou fin en train d'être glissé : colonne, valeur de départ, décalage en secondes.
    bound_drag: Option<(Column, f64, f64)>,
    /// Ce qui attend la réponse à « enregistrer les modifications ? ».
    after_save_prompt: Option<AfterPrompt>,
    allow_close: bool,
    exporting: bool,
    last_export: Option<PathBuf>,
    status: String,
    title: String,
}

/// Timecode réel de la vidéo : heures:minutes:secondes:images.
pub fn tc(time: f64, fps: f64) -> String {
    let time = time.max(0.0) + 1e-6;
    let total = time as u64;
    let frame = (time.fract() * fps) as u64;
    format!("{:02}:{:02}:{:02}:{:02}", total / 3600, total / 60 % 60, total % 60, frame)
}

fn short_duration(seconds: f64) -> String {
    let total = seconds.round() as u64;
    if total >= 3600 {
        format!("{} h {:02} min", total / 3600, total / 60 % 60)
    } else {
        format!("{}:{:02}", total / 60, total % 60)
    }
}

/// Durée d'une ligne du tableau : toujours en minutes, pour garder la colonne alignée.
fn table_duration(seconds: f64) -> String {
    let total = seconds.round() as u64;
    format!("{}:{:02}", total / 60, total % 60)
}

/// Colonnes fixes du tableau des séquences. La police à chasse fixe les garde alignées.
fn table_columns(number: &str, marker: &str, start: &str, end: &str, duration: &str) -> String {
    format!("{number:>3} {marker} {start:<11}  {end:<11}  {duration:>6}")
}

/// Une colonne modifiable du tableau.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Column {
    Start,
    End,
    TruePeak,
    Integrated,
    ShortTerm,
    Momentary,
    Range,
    Mood,
    Ensembles,
    Instruments,
    Name,
    Note,
}

impl Column {
    /// Clé d'une valeur corrigée à la main ; `None` si la colonne a son propre champ.
    fn override_key(self) -> Option<&'static str> {
        match self {
            Column::TruePeak => Some("tp"),
            Column::Integrated => Some("i"),
            Column::ShortTerm => Some("s"),
            Column::Momentary => Some("m"),
            Column::Range => Some("lra"),
            Column::Mood => Some("mood"),
            Column::Ensembles => Some("ensemble"),
            Column::Instruments => Some("instruments"),
            _ => None,
        }
    }
}

/// Colonnes à chasse fixe : (colonne, premier caractère, dernier caractère exclu).
const FIXED_CELLS: [(Column, usize, usize); 2] = [(Column::Start, 6, 17), (Column::End, 19, 30)];
const FIXED_CHARS: usize = 38;
/// Les mesures de loudness, calées sur le bord droit : les toutes dernières colonnes.
const LOUDNESS_CELLS: [(Column, usize, usize); 5] = [
    (Column::TruePeak, 0, 5),
    (Column::Integrated, 6, 11),
    (Column::ShortTerm, 12, 17),
    (Column::Momentary, 18, 23),
    (Column::Range, 24, 29),
];
const LOUDNESS_CHARS: usize = 29;
/// Une case en cours d'édition reste assez large pour lire ce qu'on tape.
const TABLE_EDIT_MIN_WIDTH: f32 = 96.0;
/// Hauteur minimale sous le tableau : les boutons d'export restent visibles.
const MIN_SHEET_HEIGHT: f32 = 50.0;
const SHEET_BOTTOM_MARGIN: f32 = 6.0;
const TABLE_OVERRIDE_COLOR: Color32 = Color32::from_rgb(245, 190, 90);

/// Ce qu'on voulait faire quand des modifications n'étaient pas enregistrées.
enum AfterPrompt {
    Quit,
    Open(PathBuf),
}

/// La valeur du tableau qu'on est en train de taper.
struct TableEdit {
    segment: u64,
    column: Column,
    initial: String,
    text: String,
    needs_focus: bool,
}

/// Ce que le tableau montre pour une colonne, valeur saisie à la main comprise.
fn table_value(
    segment: &Segment,
    column: Column,
    loudness: Option<&signals::Measure>,
    tags: &music_tags::SequenceTags,
    fps: f64,
) -> String {
    if let Some(written) = column.override_key().and_then(|key| segment.overrides.get(key)) {
        return written.clone();
    }
    match column {
        Column::Start => tc(segment.start, fps),
        Column::End => tc(segment.end, fps),
        Column::TruePeak => loudness.map_or(String::new(), |m| timeline::loudness_text(m.true_peak)),
        Column::Integrated => loudness.map_or(String::new(), |m| timeline::loudness_text(m.integrated)),
        Column::ShortTerm => loudness.map_or(String::new(), |m| timeline::loudness_text(m.max_short_term)),
        Column::Momentary => loudness.map_or(String::new(), |m| timeline::loudness_text(m.max_momentary)),
        Column::Range => loudness.map_or(String::new(), |m| format!("{:.1}", m.range)),
        Column::Mood => tags.mood.clone(),
        Column::Ensembles => tags.ensembles.clone(),
        Column::Instruments => tags.instruments.clone(),
        Column::Name => segment.name.clone(),
        Column::Note => segment.notes.clone(),
    }
}

fn table_char_width(ctx: &egui::Context) -> f32 {
    ctx.fonts_mut(|fonts| {
        fonts.layout_no_wrap("0".to_string(), egui::FontId::monospace(TABLE_FONT_SIZE), Color32::WHITE).size().x
    })
}

/// Où commencent les mesures de loudness : calées à droite s'il y a la place. Sinon elles
/// passent après le texte et sortent par la droite : ce sont elles qui cèdent.
fn loudness_left(rect: egui::Rect, char_width: f32) -> f32 {
    let text_end = rect.left()
        + TABLE_PADDING
        + FIXED_CHARS as f32 * char_width
        + TABLE_NOTE_GAP
        + TABLE_MOOD_WIDTH
        + TABLE_ENSEMBLES_WIDTH
        + TABLE_INSTRUMENTS_WIDTH
        + TABLE_TEXT_MIN_WIDTH
        + TABLE_NOTE_GAP;
    (rect.right() - TABLE_PADDING - LOUDNESS_CHARS as f32 * char_width).max(text_end)
}

/// La place de chaque colonne dans une ligne du tableau.
fn table_cells(rect: egui::Rect, char_width: f32) -> Vec<(Column, egui::Rect)> {
    let base = rect.left() + TABLE_PADDING;
    let cell = |left: f32, right: f32| {
        egui::Rect::from_min_max(egui::pos2(left, rect.top()), egui::pos2(right, rect.bottom()))
    };
    let mut cells: Vec<(Column, egui::Rect)> = FIXED_CELLS
        .iter()
        .map(|&(column, first, last)| (column, cell(base + first as f32 * char_width, base + last as f32 * char_width)))
        .collect();
    let loudness_left = loudness_left(rect, char_width);
    cells.extend(LOUDNESS_CELLS.iter().map(|&(column, first, last)| {
        (column, cell(loudness_left + first as f32 * char_width, loudness_left + last as f32 * char_width))
    }));
    let mut x = base + FIXED_CHARS as f32 * char_width + TABLE_NOTE_GAP;
    for (column, width) in [
        (Column::Mood, TABLE_MOOD_WIDTH),
        (Column::Ensembles, TABLE_ENSEMBLES_WIDTH),
        (Column::Instruments, TABLE_INSTRUMENTS_WIDTH),
    ] {
        cells.push((column, cell(x, x + width - 6.0)));
        x += width;
    }
    let (name, note) = text_column_widths(loudness_left - TABLE_NOTE_GAP - x);
    cells.push((Column::Name, cell(x, x + name)));
    cells.push((Column::Note, cell(x + name + TABLE_NOTE_GAP, x + name + TABLE_NOTE_GAP + note)));
    cells
}

/// Une durée ou un instant saisi : secondes, m:s, h:m:s ou h:m:s:images.
fn parse_time(text: &str, fps: f64) -> Option<f64> {
    let cleaned = text.trim().replace(',', ".");
    let parts: Vec<f64> = cleaned.split(':').map(|part| part.trim().parse::<f64>().ok().filter(|v| *v >= 0.0)).collect::<Option<_>>()?;
    let value = match parts.as_slice() {
        [s] => *s,
        [m, s] => m * 60.0 + s,
        [h, m, s] => h * 3600.0 + m * 60.0 + s,
        [h, m, s, f] => h * 3600.0 + m * 60.0 + s + f / fps,
        _ => return None,
    };
    value.is_finite().then_some(value)
}

/// Colonnes de loudness du tableau, alignées sur leur en-tête. Une valeur corrigée à la main
/// laisse sa place vide : elle se dessine à part, dans sa propre couleur.
fn loudness_columns(measure: Option<&signals::Measure>, overrides: &BTreeMap<String, String>) -> String {
    let (tp, i, s, m, lra) = match measure {
        Some(measure) => (
            timeline::loudness_text(measure.true_peak),
            timeline::loudness_text(measure.integrated),
            timeline::loudness_text(measure.max_short_term),
            timeline::loudness_text(measure.max_momentary),
            format!("{:.1}", measure.range),
        ),
        None => Default::default(),
    };
    let shown = |column: Column, value: String| {
        if column.override_key().is_some_and(|key| overrides.contains_key(key)) {
            String::new()
        } else {
            value
        }
    };
    loudness_header(
        &shown(Column::TruePeak, tp),
        &shown(Column::Integrated, i),
        &shown(Column::ShortTerm, s),
        &shown(Column::Momentary, m),
        &shown(Column::Range, lra),
    )
}

/// Les colonnes de texte du tableau, à gauche du nom et de la note.
const TABLE_MOOD_WIDTH: f32 = 62.0;
const TABLE_ENSEMBLES_WIDTH: f32 = 112.0;
const TABLE_INSTRUMENTS_WIDTH: f32 = 150.0;
/// Place gardée au nom et à la note avant de laisser celle des mesures.
const TABLE_TEXT_MIN_WIDTH: f32 = 240.0;

fn loudness_header(tp: &str, i: &str, s: &str, m: &str, lra: &str) -> String {
    format!("{tp:>5} {i:>5} {s:>5} {m:>5} {lra:>5}")
}

/// Une note tient sur une ligne dans le tableau, quelle que soit sa mise en forme.
fn note_preview(notes: &str) -> String {
    notes.split_whitespace().collect::<Vec<_>>().join(" ")
}

const TABLE_FONT_SIZE: f32 = 11.5;
const TABLE_ROW_HEIGHT: f32 = 18.0;
const TABLE_PADDING: f32 = 4.0;
const TABLE_NOTE_GAP: f32 = 12.0;
const TABLE_NAME_SHARE: f32 = 0.38;
const TABLE_NAME_MAX: f32 = 170.0;

/// Largeurs des colonnes Nom et Note dans la place restante. Le nom est court par nature :
/// il garde une largeur raisonnable, la note prend tout le reste.
fn text_column_widths(room: f32) -> (f32, f32) {
    let name = (room * TABLE_NAME_SHARE).clamp(0.0, TABLE_NAME_MAX);
    (name, (room - name - TABLE_NOTE_GAP).max(0.0))
}

/// Un texte sur une ligne, coupé par « … » s'il dépasse sa colonne.
fn paint_cell(painter: &egui::Painter, x: f32, center: f32, width: f32, text: &str, color: Color32) {
    if width < 12.0 || text.is_empty() {
        return;
    }
    let mut job = egui::text::LayoutJob::simple_singleline(
        text.to_string(),
        egui::FontId::proportional(TABLE_FONT_SIZE),
        color,
    );
    job.wrap = egui::text::TextWrapping::truncate_at_width(width);
    let galley = painter.layout_job(job);
    painter.galley(egui::pos2(x, center - galley.size().y / 2.0), galley, color);
}

/// Une ligne du tableau : colonnes fixes à gauche, le texte, puis les mesures à droite.
#[expect(clippy::too_many_arguments)]
fn paint_table_row(
    ui: &egui::Ui,
    rect: egui::Rect,
    columns: &str,
    loudness: &str,
    tags: &music_tags::SequenceTags,
    overrides: &BTreeMap<String, String>,
    name: &str,
    note: &str,
    text_color: Color32,
) {
    let painter = ui.painter_at(rect);
    let center = rect.center().y;
    let fixed = painter.layout_no_wrap(columns.to_string(), egui::FontId::monospace(TABLE_FONT_SIZE), text_color);
    painter.galley(egui::pos2(rect.left() + TABLE_PADDING, center - fixed.size().y / 2.0), fixed, text_color);
    let char_width = table_char_width(ui.ctx());
    let measures = painter.layout_no_wrap(loudness.to_string(), egui::FontId::monospace(TABLE_FONT_SIZE), text_color);
    painter.galley(egui::pos2(loudness_left(rect, char_width), center - measures.size().y / 2.0), measures, text_color);
    for (column, cell) in table_cells(rect, char_width) {
        let written = column.override_key().and_then(|key| overrides.get(key));
        let color = if written.is_some() { TABLE_OVERRIDE_COLOR } else { text_color };
        match column {
            Column::TruePeak | Column::Integrated | Column::ShortTerm | Column::Momentary | Column::Range => {
                if let Some(value) = written {
                    let clipped = painter.with_clip_rect(cell.intersect(rect));
                    clipped.text(
                        egui::pos2(cell.right(), center),
                        egui::Align2::RIGHT_CENTER,
                        value,
                        egui::FontId::monospace(TABLE_FONT_SIZE),
                        color,
                    );
                }
            }
            Column::Mood => paint_cell(&painter, cell.left(), center, cell.width(), written.unwrap_or(&tags.mood), color),
            Column::Ensembles => paint_cell(&painter, cell.left(), center, cell.width(), written.unwrap_or(&tags.ensembles), color),
            Column::Instruments => paint_cell(&painter, cell.left(), center, cell.width(), written.unwrap_or(&tags.instruments), color),
            Column::Name => paint_cell(&painter, cell.left(), center, cell.width(), name, text_color),
            Column::Note => paint_cell(&painter, cell.left(), center, cell.width(), note, text_color),
            Column::Start | Column::End => {}
        }
    }
}

/// Applique une détection sans perdre la séquence choisie : si elle a fusionné avec une
/// voisine, la sélection passe à celle qui recouvre désormais son milieu.
/// Détection de la musique, sur son propre fil ; `cancel` l'arrête entre deux lots.
fn run_detection(path: &Path, track: usize, layout: &str, duration: f64, cancel: Arc<AtomicBool>, send: &(impl Fn(Job) + Clone + Send + Sync + 'static)) {
    let result = guarded(|| {
        let model = std::panic::catch_unwind(Yamnet::load).map_err(|_| "chargement du modèle impossible".to_string())?;
        let (mixes, _) = detect::choose_mixes_on(path, track, layout)?;
        let progress = {
            let send = send.clone();
            move |p: u32| send(Job::DetectProgress(p))
        };
        detect::detect_music_probabilities_live(&model, path, duration, &mixes, &progress, &mut |scores| {
            if cancel.load(Ordering::Relaxed) {
                return false;
            }
            send(Job::DetectPartial(scores));
            true
        })
    });
    send(Job::Detected(result));
}

fn apply_detection_keeping_selection(project: &mut Project, selected: &mut Option<u64>, detected: &[(f64, f64)]) {
    let middle = selected.and_then(|id| project.segment(id)).map(|s| (s.start + s.end) / 2.0);
    project.apply_detection(detected);
    if !project.segments.iter().any(|s| Some(s.id) == *selected) {
        *selected = middle.and_then(|time| project.segments.iter().find(|s| s.start <= time && time < s.end)).map(|s| s.id);
    }
}

fn is_video(path: &Path) -> bool {
    path.extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .is_some_and(|e| VIDEO_EXTENSIONS.contains(&e.as_str()))
}

impl App {
    fn new(player: Option<mpv::Player>, player_error: Option<String>) -> Self {
        let (jobs_sender, jobs) = mpsc::channel();
        App {
            player,
            player_error,
            recent: recent::RecentProjects::default(),
            render_choice: renderer::RenderBackend::default(),
            render_running: renderer::RenderBackend::default(),
            video_texture: None,
            video_size: [0, 0],
            media: None,
            generation: 0,
            jobs_sender,
            jobs,
            picker: None,
            fake_position: 0.0,
            pending_seek: None,
            view: View { start: 0.0, span: 60.0 },
            view_changed_at: Instant::now(),
            selected: None,
            scroll_list: false,
            listed_playing: None,
            drag: Drag::None,
            timeline_body: egui::Rect::NOTHING,
            auto_zoom: false,
            last_scrub: None,
            resume_after_drag: false,
            #[cfg(test)]
            seek_requests: 0,
            requested_view: None,
            tracks: Tracks::default(),
            waveform_normalized: true,
            monitor: Monitor::default(),
            mixer_display: Vec::new(),
            mixer_loudness_display: Vec::new(),
            applied_monitor: Monitor::default(),
            applied_layout: None,
            applied_filter: None,
            show_help: false,
            loop_on: false,
            applied_loop: None,
            reverse_speed: 0.0,
            reverse_last: Instant::now(),
            shuttle: 0,
            settings_grabbed: false,
            table_edit: None,
            field_edit: None,
            sheet_height: None,
            sheet_height_now: 70.0,
            sheet_measured: [70.0, 175.0],
            bound_drag: None,
            after_save_prompt: None,
            allow_close: false,
            exporting: false,
            last_export: None,
            status: String::new(),
            title: String::new(),
        }
    }

    fn from_creation(cc: &eframe::CreationContext) -> Self {
        let (player, error) = match mpv::Player::new(&cc.egui_ctx) {
            Ok(player) => (Some(player), None),
            Err(error) => (None, Some(error)),
        };
        style(&cc.egui_ctx);
        let mut app = App::new(player, error);
        app.recent = recent::RecentProjects::load();
        app.render_choice = renderer::RenderBackend::load();
        app.render_running = app.render_choice;
        let mut arguments = std::env::args_os().skip(1);
        let mut seek = None;
        let mut sequence = None;
        let number = |arguments: &mut std::iter::Skip<std::env::ArgsOs>| {
            arguments.next().and_then(|v| v.to_string_lossy().parse::<f64>().ok())
        };
        while let Some(argument) = arguments.next() {
            if argument == "--seek" {
                seek = number(&mut arguments);
            } else if argument == "--sequence" {
                sequence = number(&mut arguments);
            } else if argument == "--canaux" {
                app.tracks.audio_channels = timeline::Channels::tracks_only();
            } else if argument == "--onde" {
                app.tracks.view = AudioView::Waveform;
            } else if argument == "--view" {
                if let (Some(start), Some(span)) = (number(&mut arguments), number(&mut arguments)) {
                    app.requested_view = Some(View { start, span });
                }
            } else {
                app.open(&cc.egui_ctx, PathBuf::from(argument));
            }
        }
        if let Some(time) = seek {
            app.seek_to(time, true);
        }
        if let Some(number) = sequence {
            app.selected = app.media.as_ref().and_then(|m| m.project.segments.get((number as usize).checked_sub(1)?)).map(|s| s.id);
        }
        app
    }

    pub fn fps(&self) -> f64 {
        self.player.as_ref().map(|p| p.state.fps).filter(|f| *f > 0.0).unwrap_or(25.0)
    }

    pub fn duration(&self) -> f64 {
        let project = self.media.as_ref().map_or(0.0, |m| m.project.duration);
        let player = self.player.as_ref().map_or(0.0, |p| p.state.duration);
        if project > 0.0 {
            project
        } else {
            player
        }
    }

    pub fn position(&self) -> f64 {
        if let Some((time, _)) = self.pending_seek {
            return time;
        }
        self.player.as_ref().map_or(self.fake_position, |p| p.state.position)
    }

    fn paused(&self) -> bool {
        self.player.as_ref().is_none_or(|p| p.state.paused)
    }

    pub fn seek_to(&mut self, time: f64, exact: bool) {
        let duration = self.duration();
        let time = if duration > 0.0 { time.clamp(0.0, duration) } else { time.max(0.0) };
        self.pending_seek = Some((time, exact));
        self.fake_position = time;
        #[cfg(test)]
        {
            self.seek_requests += 1;
        }
    }

    /// Début d'un glissé sur la timeline : la lecture attend, sinon l'image repartirait
    /// à chaque déplacement.
    pub fn begin_scrub(&mut self) {
        self.last_scrub = None;
        self.stop_shuttle();
        if !self.paused() {
            self.resume_after_drag = true;
            self.set_paused(true);
        }
    }

    /// Suit la souris sans redemander une position déjà demandée : bouton maintenu
    /// sans bouger, l'image reste fixe.
    pub fn scrub_to(&mut self, time: f64, exact: bool) {
        if self.last_scrub.is_some_and(|(last, was_exact)| last == time && (was_exact || !exact)) {
            return;
        }
        self.last_scrub = Some((time, exact));
        self.seek_to(time, exact);
    }

    /// Fin d'un glissé : position exacte, puis reprise de la lecture si elle tournait.
    pub fn end_scrub(&mut self) {
        if let Some((time, false)) = self.last_scrub {
            self.seek_to(time, true);
        }
        self.last_scrub = None;
        if std::mem::take(&mut self.resume_after_drag) {
            self.set_paused(false);
        }
    }

    fn stop_shuttle(&mut self) {
        self.reverse_speed = 0.0;
        self.shuttle = 0;
    }

    fn set_paused(&mut self, paused: bool) {
        if let Some(player) = &mut self.player {
            player.set_paused(paused);
        }
    }

    fn toggle_play(&mut self) {
        let was_reversing = self.reverse_speed > 0.0;
        let paused = self.paused();
        self.stop_shuttle();
        if let Some(player) = &mut self.player {
            player.set_speed(1.0);
        }
        // Relancer depuis la fin de la vidéo repart du début
        if paused && !was_reversing && self.duration() > 0.0 && self.position() >= self.duration() - 0.1 {
            self.seek_to(0.0, true);
        }
        self.set_paused(!paused || was_reversing);
    }

    fn shuttle_forward(&mut self) {
        let speed = if self.reverse_speed > 0.0 || self.paused() {
            self.shuttle = 0;
            SHUTTLE[0]
        } else {
            self.shuttle = (self.shuttle + 1).min(SHUTTLE.len() - 1);
            SHUTTLE[self.shuttle]
        };
        self.reverse_speed = 0.0;
        if let Some(player) = &mut self.player {
            player.set_speed(speed);
            player.set_paused(false);
        }
    }

    fn shuttle_backward(&mut self) {
        if self.reverse_speed > 0.0 {
            self.shuttle = (self.shuttle + 1).min(SHUTTLE.len() - 1);
        } else {
            self.shuttle = 0;
            self.reverse_last = Instant::now();
        }
        self.reverse_speed = SHUTTLE[self.shuttle];
        self.set_paused(true);
    }

    fn stop(&mut self) {
        self.stop_shuttle();
        if let Some(player) = &mut self.player {
            player.set_speed(1.0);
        }
        self.set_paused(true);
    }

    fn step_frame(&mut self, forward: bool) {
        self.stop_shuttle();
        match &self.player {
            Some(player) => player.step_frame(forward),
            None => {
                let frame = 1.0 / self.fps();
                self.seek_to(self.fake_position + if forward { frame } else { -frame }, true);
            }
        }
    }

    fn make_visible(&mut self, time: f64) {
        if time < self.view.start || time > self.view.end() {
            self.view.start = time - self.view.span * 0.1;
            self.view.clamp(self.duration());
            self.view_changed_at = Instant::now();
        }
    }

    fn go_to(&mut self, time: f64) {
        self.seek_to(time, true);
        self.make_visible(time);
    }

    fn select(&mut self, id: u64) {
        self.selected = Some(id);
        self.scroll_list = true;
        // Zoomer d'abord : la tête de lecture arrive ensuite dans une vue déjà cadrée
        if self.auto_zoom {
            self.zoom_on_selected();
        }
        if let Some(start) = self.media.as_ref().and_then(|m| m.project.segment(id)).map(|s| s.start) {
            self.go_to(start);
        }
    }

    fn select_neighbour(&mut self, forward: bool) {
        let position = self.position();
        let Some(media) = &self.media else { return };
        let segments = &media.project.segments;
        let target = if forward {
            segments.iter().find(|s| s.start > position + 0.01)
        } else {
            segments.iter().rev().find(|s| s.start < position - 0.01)
        };
        if let Some(id) = target.map(|s| s.id) {
            self.select(id);
        }
    }

    /// Mémorise l'état des séquences avant une modification, pour pouvoir l'annuler.
    pub fn remember(&mut self) {
        if let Some(media) = &mut self.media {
            media.undo.push(media.project.segments.clone());
            if media.undo.len() > UNDO_DEPTH {
                media.undo.remove(0);
            }
            media.redo.clear();
        }
    }

    fn undo(&mut self, forward: bool) {
        let Some(media) = &mut self.media else { return };
        let (from, to) = if forward { (&mut media.redo, &mut media.undo) } else { (&mut media.undo, &mut media.redo) };
        if let Some(state) = from.pop() {
            to.push(std::mem::replace(&mut media.project.segments, state));
            media.mark_changed();
            if !media.project.segments.iter().any(|s| Some(s.id) == self.selected) {
                self.selected = None;
            }
        }
    }

    fn set_edge_at_playhead(&mut self, start: bool) {
        let position = self.position();
        let Some(id) = self.selected else {
            self.status = "Choisissez d'abord une séquence.".into();
            return;
        };
        self.remember();
        if let Some(media) = &mut self.media {
            if start {
                media.project.set_start(id, position);
            } else {
                media.project.set_end(id, position);
            }
            media.mark_changed();
        }
    }

    fn add_at_playhead(&mut self) {
        let position = self.position();
        self.remember();
        let id = self.media.as_mut().and_then(|media| {
            let id = media.project.add_manual(position, position + NEW_SEQUENCE_S)?;
            media.mark_changed();
            Some(id)
        });
        match id {
            Some(id) => {
                self.selected = Some(id);
                self.scroll_list = true;
            }
            None => {
                if let Some(media) = &mut self.media {
                    media.undo.pop();
                }
                self.status = "Pas de place ici : la tête de lecture est déjà dans une séquence.".into();
            }
        }
    }

    fn delete_selected(&mut self) {
        let Some(id) = self.selected else { return };
        self.remember();
        if let Some(media) = &mut self.media {
            media.project.segments.retain(|s| s.id != id);
            media.mark_changed();
        }
        self.selected = None;
    }

    pub fn zoom_on_selected(&mut self) {
        let segment = self.selected.and_then(|id| self.media.as_ref()?.project.segment(id).cloned());
        if let Some(segment) = segment {
            // Une petite marge : les deux poignées restent visibles et attrapables
            let margin = (segment.end - segment.start) * 0.04 + 0.5;
            self.view = View { start: segment.start - margin, span: segment.end - segment.start + margin * 2.0 };
            self.view.clamp(self.duration());
            self.view_changed_at = Instant::now();
        }
    }

    // --- Ouverture et travaux en arrière-plan ---------------------------------

    fn save_now(&mut self) {
        if let Some(media) = &mut self.media {
            match media.project.save(&media.project_path) {
                Ok(()) => media.dirty = false,
                Err(error) => {
                    self.status = format!("Sauvegarde impossible : {error}");
                }
            }
        }
    }

    /// Passe à une autre piste audio : tout ce qui en dépend est recalculé.
    fn change_audio_track(&mut self, ctx: &egui::Context, track: usize) {
        let position = self.position();
        let Some(media) = &mut self.media else { return };
        media.project.audio_track = track;
        // Les faders décrivaient les canaux de l'autre piste
        media.project.channel_gains.clear();
        media.project.scores.clear();
        let path = media.path.clone();
        self.save_now();
        if self.media.as_ref().is_some_and(|m| m.dirty) {
            return;
        }
        self.open(ctx, path);
        self.seek_to(position, true);
    }

    fn help_dialog(&mut self, ctx: &egui::Context) {
        if !self.show_help {
            return;
        }
        let mut close = false;
        let modal = egui::Modal::new(egui::Id::unique("help")).show(ctx, |ui| {
            ui.set_width(460.0);
            ui.heading("Raccourcis clavier");
            ui.add_space(6.0);
            egui::ScrollArea::vertical().max_height(ctx.content_rect().height() - 160.0).show(ui, |ui| {
                ui.label(SHORTCUTS);
            });
            ui.add_space(10.0);
            close = ui.button("Fermer").clicked();
        });
        if close || modal.should_close() {
            self.show_help = false;
        }
    }

    /// Demande quoi faire des modifications, tant qu'une action attend.
    fn save_prompt(&mut self, ctx: &egui::Context) {
        if self.after_save_prompt.is_none() {
            return;
        }
        #[derive(PartialEq)]
        enum Choice {
            Save,
            Discard,
            Cancel,
        }
        let mut choice = None;
        egui::Modal::new(egui::Id::unique("save-prompt")).show(ctx, |ui| {
            ui.set_width(380.0);
            ui.heading("Enregistrer les modifications ?");
            ui.add_space(6.0);
            ui.label("Le projet de cette vidéo a des modifications non enregistrées.");
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                if ui.button("Enregistrer").clicked() {
                    choice = Some(Choice::Save);
                }
                if ui.button("Ne pas enregistrer").clicked() {
                    choice = Some(Choice::Discard);
                }
                if ui.button("Annuler").clicked() {
                    choice = Some(Choice::Cancel);
                }
            });
        });
        let Some(choice) = choice else { return };
        if choice == Choice::Cancel {
            self.after_save_prompt = None;
            return;
        }
        if choice == Choice::Save {
            self.save_now();
            // Échec : rien n'est perdu, l'erreur s'affiche et on reste sur la vidéo
            if self.media.as_ref().is_some_and(|m| m.dirty) {
                self.after_save_prompt = None;
                return;
            }
        }
        if let Some(media) = &mut self.media {
            media.dirty = false;
        }
        match self.after_save_prompt.take() {
            Some(AfterPrompt::Quit) => {
                self.allow_close = true;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            Some(AfterPrompt::Open(path)) => self.open(ctx, path),
            None => {}
        }
    }

    pub fn open(&mut self, ctx: &egui::Context, path: PathBuf) {
        if self.media.as_ref().is_some_and(|m| m.dirty) {
            self.after_save_prompt = Some(AfterPrompt::Open(path));
            return;
        }
        self.recent.add(&path);
        self.generation += 1;
        // La détection de la vidéo précédente n'a plus de raison de tourner
        if let Some(media) = &self.media {
            media.detect_cancel.store(true, Ordering::Relaxed);
        }
        let project_path = project::project_path(&path);
        let mut status = String::new();
        let project = if project_path.is_file() {
            Project::load(&project_path).unwrap_or_else(|error| {
                // Ne pas écraser un fichier illisible : il contient peut-être des heures de travail
                let backup = project_path.with_extension("json.illisible");
                let _ = std::fs::rename(&project_path, &backup);
                status = format!("Projet illisible ({error}), mis de côté dans {}", backup.display());
                Project::new(&path)
            })
        } else {
            Project::new(&path)
        };
        let needs_detection = project.scores.is_empty() && !project.detection_off;
        let detect_cancel = Arc::new(AtomicBool::new(false));
        let audio_track = project.audio_track;
        let channel_gains = project.channel_gains.clone();
        let master_gain_db = project.master_gain_db;
        let smoothed = detect::smoothed_scores(&project.scores);
        self.media = Some(Media {
            release: release_name::describe(&path),
            path: path.clone(),
            project_path,
            project,
            layout: None,
            smoothed,
            signals: None,
            strip: None,
            strip_texture: None,
            detect_progress: needs_detection.then_some(0),
            detect_cancel: detect_cancel.clone(),
            detect_waiting: false,
            signals_progress: Some(0),
            strip_progress: Some(0),
            undo: Vec::new(),
            redo: Vec::new(),
            dirty: false,
            changed_at: Instant::now(),
            overviews: HashMap::new(),
            spectra: HashMap::new(),
            details: HashMap::new(),
            detail_pending: HashSet::new(),
            chapters: Vec::new(),
            info: Default::default(),
            class_track: None,
            tags_progress: Some(0),
            sequence_tags: HashMap::new(),
            film_loudness: HashMap::new(),
            sequence_loudness: HashMap::new(),
        });
        // Le fichier projet existe dès l'import, comme avec LosslessCut
        self.save_now();
        if !status.is_empty() {
            self.status = status;
        }
        self.selected = None;
        self.monitor.clear();
        self.monitor.restore_gains(&channel_gains);
        self.monitor.set_master_db(master_gain_db);
        self.applied_layout = None;
        self.loop_on = false;
        self.pending_seek = None;
        self.fake_position = 0.0;
        self.view = View { start: 0.0, span: 60.0 };
        self.view_changed_at = Instant::now();
        self.drag = Drag::None;
        self.stop_shuttle();
        self.last_export = None;
        if let Some(player) = &mut self.player {
            player.set_audio_track(audio_track);
            player.load(&path);
            player.set_speed(1.0);
        }
        self.start_jobs(ctx, path, audio_track, needs_detection.then_some(detect_cancel));
    }

    fn start_jobs(&mut self, ctx: &egui::Context, path: PathBuf, track: usize, detection: Option<Arc<AtomicBool>>) {
        let generation = self.generation;
        let sender = self.jobs_sender.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let send = move |job: Job| {
                let _ = sender.send((generation, job));
                ctx.request_repaint();
            };
            send(Job::Info(signals::read_media_info(&path)));
            let probe = detect::probe_audio_track(&path, track);
            send(Job::Probed(probe.clone()));
            send(Job::Chapters(signals::read_chapters(&path)));
            let duration = probe.as_ref().map_or(0.0, |p| p.1);

            // Vignettes et keyframes : indépendants de la présence d'une piste audio
            {
                let (path, send) = (path.clone(), send.clone());
                std::thread::spawn(move || {
                    let cache = signals::cache_path(&path, "strip");
                    let cached = cache.as_deref().and_then(Strip::load);
                    let strip = cached.map(Ok).unwrap_or_else(|| {
                        let progress = {
                            let send = send.clone();
                            move |p: u32| send(Job::StripProgress(p))
                        };
                        let built = guarded(|| signals::build_strip(&path, duration, &progress));
                        if let (Ok(strip), Some(cache)) = (&built, &cache) {
                            let _ = strip.save(cache);
                        }
                        built
                    });
                    send(Job::Strip(strip));
                });
            }

            let Ok((layout, duration)) = probe else {
                send(Job::Tags(Err("pas de piste audio".to_string())));
                return;
            };

            // Mood et instruments : en arrière-plan, le tableau se complète quand c'est prêt
            {
                let (path, layout, send) = (path.clone(), layout.clone(), send.clone());
                std::thread::spawn(move || {
                    let progress = {
                        let send = send.clone();
                        move |p: u32| send(Job::TagsProgress(p))
                    };
                    let result = guarded(|| music_tags::analyze(&path, track, &layout, duration, &progress));
                    send(Job::Tags(result));
                });
            }

            // Coupée entre-temps : la détection ne démarre pas
            if let Some(cancel) = detection.filter(|cancel| !cancel.load(Ordering::Relaxed)) {
                let (path, layout, send) = (path.clone(), layout.clone(), send.clone());
                std::thread::spawn(move || run_detection(&path, track, &layout, duration, cancel, &send));
            }

            let cache = signals::cache_path(&path, &signals::track_cache_kind("signals", track));
            let cached = cache.as_deref().and_then(Signals::load);
            let result = cached.map(Ok).unwrap_or_else(|| {
                let progress = {
                    let send = send.clone();
                    move |p: u32| send(Job::SignalsProgress(p))
                };
                let built = guarded(|| signals::analyze(&path, track, &layout, duration, &progress));
                if let (Ok(signals), Some(cache)) = (&built, &cache) {
                    let _ = signals.save(cache);
                }
                built
            });
            send(Job::Signals(result));
        });
    }

    /// Lance la détection seule, quand on la rétablit sans scores.
    fn spawn_detection(&mut self, ctx: &egui::Context) {
        let generation = self.generation;
        let Some(media) = &mut self.media else { return };
        let Some(layout) = media.layout.clone() else {
            media.detect_waiting = true;
            return;
        };
        let cancel = Arc::new(AtomicBool::new(false));
        media.detect_cancel = cancel.clone();
        media.detect_waiting = false;
        media.detect_progress = Some(0);
        let (path, track, duration) = (media.path.clone(), media.project.audio_track, media.project.duration);
        let (sender, ctx) = (self.jobs_sender.clone(), ctx.clone());
        std::thread::spawn(move || {
            let send = move |job: Job| {
                let _ = sender.send((generation, job));
                ctx.request_repaint();
            };
            run_detection(&path, track, &layout, duration, cancel, &send);
        });
    }

    /// L'interrupteur « Détection » : coupée, plus de séquences automatiques et le calcul s'arrête.
    pub fn set_detection(&mut self, ctx: &egui::Context, on: bool) {
        let Some(media) = &mut self.media else { return };
        if on != media.project.detection_off {
            return;
        }
        media.project.set_detection(on);
        if !on {
            media.detect_cancel.store(true, Ordering::Relaxed);
            media.detect_progress = None;
            media.detect_waiting = false;
            if self.selected.is_some_and(|id| media.project.segment(id).is_none()) {
                self.selected = None;
            }
            media.mark_changed();
        } else if media.project.scores.is_empty() {
            media.mark_changed();
            self.spawn_detection(ctx);
        } else {
            self.apply_detection();
        }
    }

    fn apply_detection(&mut self) {
        let Some(media) = &mut self.media else { return };
        if media.project.detection_off {
            return;
        }
        let settings = media.project.settings.clone();
        let detected = detect::scores_to_segments(
            &media.project.scores,
            settings.sensitivity,
            settings.gap,
            settings.anticipation,
            settings.post,
            media.project.duration,
        );
        apply_detection_keeping_selection(&mut media.project, &mut self.selected, &detected);
        media.mark_changed();
    }

    fn poll_jobs(&mut self, ctx: &egui::Context) {
        while let Ok((generation, job)) = self.jobs.try_recv() {
            match job {
                // Un export se termine, même si un autre vidéo a été ouvert entre-temps
                Job::Exported(result) => {
                    self.exporting = false;
                    match result {
                        Ok((count, folder)) => {
                            self.status = format!("{count} séquence(s) exportée(s) dans {}", folder.display());
                            self.last_export = Some(folder);
                        }
                        Err(error) => self.status = format!("Export impossible : {error}"),
                    }
                    continue;
                }
                _ if generation != self.generation => continue,
                _ => {}
            }
            let Some(media) = &mut self.media else { continue };
            match job {
                Job::Probed(Ok((layout, duration))) => {
                    media.layout = Some(layout);
                    let waiting = media.detect_waiting;
                    if duration > 0.0 {
                        media.project.duration = duration;
                        // Un résultat de calcul, pas une modification de l'utilisateur
                        if !media.dirty {
                            let _ = media.project.save(&media.project_path);
                        }
                    }
                    self.view = self.requested_view.take().unwrap_or(View { start: 0.0, span: self.duration().max(1.0) });
                    self.view_changed_at = Instant::now();
                    if waiting {
                        self.spawn_detection(ctx);
                    }
                }
                Job::Chapters(chapters) => media.chapters = chapters,
                Job::Info(info) => media.info = info,
                Job::TagsProgress(progress) => media.tags_progress = Some(progress),
                Job::Tags(result) => {
                    media.tags_progress = None;
                    // Sans résultat, le tableau reste simplement sans ces colonnes remplies
                    media.class_track = result.ok().map(Arc::new);
                    media.sequence_tags.clear();
                }
                Job::Probed(Err(error)) => {
                    media.detect_progress = None;
                    media.signals_progress = None;
                    self.status = format!("Analyse audio impossible : {error}");
                }
                // Détection coupée : ce qui arrive encore du calcul est ignoré
                Job::DetectProgress(_) | Job::DetectPartial(_) if media.project.detection_off => {}
                Job::DetectProgress(progress) => media.detect_progress = Some(progress),
                // Un aperçu : ni enregistré, ni compté comme une modification
                Job::DetectPartial(scores) => {
                    media.smoothed = detect::smoothed_scores(&scores);
                    let settings = media.project.settings.clone();
                    let detected = detect::scores_to_segments(
                        &scores,
                        settings.sensitivity,
                        settings.gap,
                        settings.anticipation,
                        settings.post,
                        media.project.duration,
                    );
                    apply_detection_keeping_selection(&mut media.project, &mut self.selected, &detected);
                }
                Job::Detected(Ok(scores)) => {
                    media.detect_progress = None;
                    media.smoothed = detect::smoothed_scores(&scores);
                    media.project.scores = scores;
                    let clean = !media.dirty;
                    self.apply_detection();
                    // La détection n'a pas à faire demander « enregistrer ? » : elle se garde seule
                    if clean {
                        self.save_now();
                    }
                }
                Job::Detected(Err(error)) => {
                    media.detect_progress = None;
                    if error != detect::CANCELLED {
                        self.status = format!("Détection impossible : {error}");
                    }
                }
                Job::SignalsProgress(progress) => media.signals_progress = Some(progress),
                Job::Signals(Ok(signals)) => {
                    media.signals_progress = None;
                    media.signals = Some(Arc::new(signals));
                    media.overviews.clear();
                    media.film_loudness.clear();
                    media.sequence_loudness.clear();
                }
                Job::Signals(Err(error)) => {
                    media.signals_progress = None;
                    self.status = format!("Analyse audio impossible : {error}");
                }
                Job::StripProgress(progress) => media.strip_progress = Some(progress),
                Job::Strip(Ok(strip)) => {
                    media.strip_progress = None;
                    if strip.count > 0 {
                        let rows = strip.count.div_ceil(ATLAS_COLUMNS);
                        let size = [strip.width * ATLAS_COLUMNS, strip.height * rows];
                        let mut rgb = vec![0u8; size[0] * size[1] * 3];
                        for index in 0..strip.count {
                            let (column, row) = (index % ATLAS_COLUMNS, index / ATLAS_COLUMNS);
                            for line in 0..strip.height {
                                let from = (index * strip.height + line) * strip.width * 3;
                                let to = ((row * strip.height + line) * size[0] + column * strip.width) * 3;
                                rgb[to..to + strip.width * 3].copy_from_slice(&strip.pixels[from..from + strip.width * 3]);
                            }
                        }
                        let image = egui::ColorImage::from_rgb(size, &rgb);
                        media.strip_texture = Some(ctx.load_texture("vignettes", image, egui::TextureOptions::LINEAR));
                    }
                    media.strip = Some(Arc::new(strip));
                }
                Job::Strip(Err(error)) => {
                    media.strip_progress = None;
                    self.status = format!("Vignettes impossibles : {error}");
                }
                Job::SpectrumPart(channel, first, values) => {
                    if let Some(state) = media.spectra.get_mut(&channel) {
                        let start = first * signals::OVERVIEW_BINS;
                        let end = start + values.len();
                        if state.values.len() < end {
                            state.values.resize(end, 0);
                        }
                        state.values[start..end].copy_from_slice(&values);
                        state.revision += 1;
                    }
                }
                Job::SpectrumDone(channel) => {
                    if let Some(state) = media.spectra.get_mut(&channel) {
                        state.finished = true;
                        state.revision += 1;
                    }
                }
                Job::Detail(channel, result) => {
                    media.detail_pending.remove(&channel);
                    if let Ok(detail) = result {
                        let image = timeline::detail_image(&detail);
                        match media.details.get_mut(&channel) {
                            Some(slot) => {
                                slot.1.set(image, egui::TextureOptions::LINEAR);
                                slot.0 = detail;
                            }
                            None => {
                                let name = format!("spectre-détail-{channel}");
                                let texture = ctx.load_texture(name, image, egui::TextureOptions::LINEAR);
                                media.details.insert(channel, (detail, texture));
                            }
                        }
                    }
                }
                Job::Exported(_) => {}
            }
        }
    }

    fn export(&mut self, ctx: &egui::Context, only: Option<u64>) {
        let Some(media) = &self.media else { return };
        let segments: Vec<(f64, f64)> = media
            .project
            .segments
            .iter()
            .filter(|s| only.is_none_or(|id| s.id == id))
            .map(|s| (s.start, s.end))
            .collect();
        if segments.is_empty() {
            return;
        }
        self.exporting = true;
        self.status = format!("Export de {} séquence(s)…", segments.len());
        let path = media.path.clone();
        let sender = self.jobs_sender.clone();
        let generation = self.generation;
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let folder = detect::export_folder(&path, None);
            let result = guarded(|| detect::extract_segments(&path, &segments, &folder))
                .map(|files| (files.len(), folder));
            let _ = sender.send((generation, Job::Exported(result)));
            ctx.request_repaint();
        });
    }

    fn pick_file(&mut self, ctx: &egui::Context) {
        let (sender, receiver) = mpsc::channel();
        self.picker = Some(receiver);
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let file = rfd::FileDialog::new()
                .set_title("Ouvrir une vidéo")
                .add_filter("Vidéo", VIDEO_EXTENSIONS)
                .add_filter("Tous les fichiers", &["*"])
                .pick_file();
            let _ = sender.send(file);
            ctx.request_repaint();
        });
    }

    // --- Boucle d'animation ---------------------------------------------------

    /// Tout ce qui doit avancer à chaque image, avant le dessin.
    fn tick(&mut self, ctx: &egui::Context) {
        if let Some(player) = &mut self.player {
            player.poll();
            if let Some(error) = player.state.error.take() {
                self.status = error;
            }
        }

        // Si la durée n'est connue que par le lecteur (fichier sans audio), cadrer dessus
        let duration = self.duration();
        if let Some(media) = &mut self.media {
            if media.project.duration <= 0.0 && duration > 0.0 {
                media.project.duration = duration;
                media.mark_changed();
                self.view = View { start: 0.0, span: duration };
            }
        }

        // Lecture arrière simulée : une image à la fois, au rythme de ce que le décodeur suit
        if self.reverse_speed > 0.0 {
            let elapsed = self.reverse_last.elapsed().as_secs_f64();
            let ready = self.pending_seek.is_none() && self.player.as_ref().is_none_or(|p| p.can_seek());
            if ready && elapsed >= 0.04 {
                self.reverse_last = Instant::now();
                let target = self.position() - elapsed * self.reverse_speed;
                self.seek_to(target, self.reverse_speed <= 2.0);
                if target <= 0.0 {
                    self.stop_shuttle();
                }
            }
            ctx.request_repaint();
        }

        if let Some(player) = &mut self.player {
            if let Some((time, exact)) = self.pending_seek {
                if player.can_seek() {
                    player.seek(time, exact);
                    self.pending_seek = None;
                }
            }
            let layout = self.media.as_ref().and_then(|m| m.layout.clone());
            if self.monitor != self.applied_monitor || self.applied_layout.as_ref() != Some(&layout) {
                if self.monitor.master_db() != self.applied_monitor.master_db() {
                    player.set_master_gain(self.monitor.master_gain());
                }
                let new_layout = self.applied_layout.as_ref() != Some(&layout);
                let graph = layout.as_deref().and_then(|l| self.monitor.routing_graph(l));
                if graph.is_some() {
                    let names: Vec<String> =
                        layout.as_deref().and_then(signals::channel_names).unwrap_or_default().iter().map(|n| n.to_string()).collect();
                    let matrix = self.monitor.matrix(&names);
                    if new_layout {
                        // Une seule fois par vidéo : c'est le seul moment où le son est relancé
                        player.set_audio_chain(graph.as_deref(), None);
                    } else {
                        // Solo, mute ou fader : de simples commandes, la lecture continue
                        let before = self.applied_monitor.matrix(&names);
                        for (channel, (now, old)) in matrix.iter().zip(&before).enumerate() {
                            if now.0 != old.0 {
                                player.set_route('l', channel, now.0);
                            }
                            if now.1 != old.1 {
                                player.set_route('r', channel, now.1);
                            }
                        }
                        player.remember_routing(graph.as_deref());
                    }
                    self.applied_filter = None;
                } else {
                    // Mono ou disposition inconnue : le routage par filtre
                    let names: Vec<String> = self
                        .media
                        .as_ref()
                        .and_then(|m| m.signals.as_ref())
                        .map(|s| s.channels.iter().skip(1).map(|c| c.name.clone()).collect())
                        .unwrap_or_default();
                    let filter = self.monitor.filter(&names);
                    if new_layout || filter != self.applied_filter {
                        player.set_audio_chain(None, filter.as_deref());
                        self.applied_filter = filter;
                    }
                }
                self.applied_layout = Some(layout);
                self.applied_monitor = self.monitor.clone();
            }
        } else {
            self.pending_seek = None;
        }

        // La boucle suit la séquence choisie et ses poignées
        let wanted = if self.loop_on {
            self.selected.and_then(|id| self.media.as_ref()?.project.segment(id)).map(|s| (s.start, s.end))
        } else {
            None
        };
        if wanted != self.applied_loop {
            if let Some(player) = &self.player {
                player.set_loop(wanted);
            }
            self.applied_loop = wanted;
        }

        // La vue suit la lecture, page par page
        if !self.paused() && self.drag == Drag::None {
            let position = self.position();
            if position > self.view.end() || position < self.view.start {
                self.view.start = position - self.view.span * 0.02;
                self.view.clamp(self.duration());
                self.view_changed_at = Instant::now();
            }
            ctx.request_repaint();
        }

        // Attend la fin d'un calcul lancé pour le détail du spectrogramme
        if self.view.span <= timeline::DETAIL_MAX_SPAN && self.view_changed_at.elapsed() < Duration::from_millis(400) {
            ctx.request_repaint_after(Duration::from_millis(100));
        }

        if self.pending_seek.is_some() {
            ctx.request_repaint();
        }

        if let Some(receiver) = &self.picker {
            match receiver.try_recv() {
                Ok(Some(path)) => {
                    self.picker = None;
                    self.open(ctx, path);
                }
                Ok(None) | Err(mpsc::TryRecvError::Disconnected) => self.picker = None,
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }

        let dropped = ctx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).find(|p| is_video(p)));
        if let Some(path) = dropped {
            self.open(ctx, path);
        }

        let title = self.media.as_ref().map_or("MX-LAB".to_string(), |m| {
            format!("{}{} — MX-LAB", if m.dirty { "* " } else { "" }, m.path.file_name().unwrap_or_default().to_string_lossy())
        });
        if title != self.title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.title = title;
        }
    }

    fn keyboard(&mut self, ctx: &egui::Context) {
        // Aussi pendant la saisie d'un texte
        let save = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::S);
        if ctx.input_mut(|i| i.consume_shortcut(&save)) && self.media.is_some() {
            self.save_now();
            if self.media.as_ref().is_some_and(|m| !m.dirty) {
                self.status = "Projet enregistré".to_string();
            }
        }
        if ctx.egui_wants_keyboard_input() || self.media.is_none() {
            return;
        }
        use egui::Key;
        let duration = self.duration();
        let pressed: Vec<(Key, egui::Modifiers)> = ctx.input(|i| {
            i.events
                .iter()
                .filter_map(|event| match event {
                    egui::Event::Key { key, pressed: true, modifiers, .. } => Some((*key, *modifiers)),
                    _ => None,
                })
                .collect()
        });
        for (key, modifiers) in pressed {
            let position = self.position();
            match key {
                Key::Space => self.toggle_play(),
                Key::J => self.shuttle_backward(),
                Key::K => self.stop(),
                Key::L => self.shuttle_forward(),
                Key::ArrowLeft | Key::ArrowRight => {
                    let forward = key == Key::ArrowRight;
                    let jump = if modifiers.command { Some(10.0) } else if modifiers.shift { Some(1.0) } else { None };
                    match jump {
                        Some(seconds) => self.go_to(position + if forward { seconds } else { -seconds }),
                        None => self.step_frame(forward),
                    }
                }
                Key::ArrowUp => self.select_neighbour(false),
                Key::ArrowDown => self.select_neighbour(true),
                Key::Home => self.go_to(0.0),
                Key::End => self.go_to(duration),
                Key::I => self.set_edge_at_playhead(true),
                Key::O => self.set_edge_at_playhead(false),
                Key::N => self.add_at_playhead(),
                Key::Delete => self.delete_selected(),
                Key::B => self.loop_on = !self.loop_on,
                Key::V => {
                    self.tracks.view = match self.tracks.view {
                        AudioView::Spectrum => AudioView::Waveform,
                        AudioView::Waveform => AudioView::Spectrum,
                    };
                }
                Key::Z if modifiers.command && modifiers.shift => self.undo(true),
                Key::Z if modifiers.command => self.undo(false),
                Key::Y if modifiers.command => self.undo(true),
                Key::Z => self.zoom_on_selected(),
                Key::F => {
                    self.view = View { start: 0.0, span: duration };
                    self.view_changed_at = Instant::now();
                }
                Key::Plus | Key::Equals => {
                    self.view.zoom_at(position, 1.0 / 1.5, duration);
                    self.view_changed_at = Instant::now();
                }
                Key::Minus => {
                    self.view.zoom_at(position, 1.5, duration);
                    self.view_changed_at = Instant::now();
                }
                _ => {}
            }
        }
    }

    // --- Dessin ---------------------------------------------------------------

    fn draw(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.poll_jobs(&ctx);
        self.tick(&ctx);
        self.keyboard(&ctx);
        if ctx.input(|i| i.viewport().close_requested()) && !self.allow_close && self.media.as_ref().is_some_and(|m| m.dirty) {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.after_save_prompt = Some(AfterPrompt::Quit);
        }
        self.save_prompt(&ctx);
        self.help_dialog(&ctx);

        egui::Panel::top("toolbar").show(ui, |ui| self.toolbar(ui, &ctx));
        // Les pistes passent avant la vidéo : c'est là qu'on travaille
        let window_height = ctx.content_rect().height();
        egui::Panel::bottom("timeline")
            .resizable(true)
            .default_size(window_height * TIMELINE_SHARE)
            .min_size(200.0)
            .max_size((window_height - MIN_VIDEO_HEIGHT).max(200.0))
            .frame(egui::Frame::NONE)
            .show(ui, |ui| self.timeline(ui));
        // La vidéo et sa barre de lecture gardent leur place, même dans une petite fenêtre
        let window_width = ctx.content_rect().width();
        egui::Panel::left("film")
            .resizable(true)
            .default_size((window_width * 0.18).clamp(LEFT_PANEL_MIN, 300.0))
            .min_size(LEFT_PANEL_MIN)
            .max_size((window_width - RIGHT_PANEL_MIN - MIN_VIDEO_WIDTH).max(LEFT_PANEL_MIN))
            .show(ui, |ui| self.left_panel(ui));
        // Sans plafond : le tableau et ses notes profitent des grands écrans
        egui::Panel::right("sequences")
            .resizable(true)
            .default_size((window_width * RIGHT_PANEL_SHARE).max(RIGHT_PANEL_MIN))
            .min_size(RIGHT_PANEL_MIN)
            .max_size((window_width - LEFT_PANEL_MIN - MIN_VIDEO_WIDTH).max(RIGHT_PANEL_MIN))
            .show(ui, |ui| self.side_panel(ui, &ctx));
        egui::CentralPanel::default().frame(egui::Frame::NONE.fill(Color32::BLACK)).show(ui, |ui| self.video_panel(ui));
    }

    fn recent_menu(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let mut chosen = None;
        for path in self.recent.paths() {
            let row = menu_row(ui, true, None, &recent::label(path), None).on_hover_text(path.display().to_string());
            if row.clicked() {
                chosen = Some(path.clone());
                ui.close();
            }
        }
        if self.recent.paths().is_empty() {
            menu_row(ui, false, None, "Aucun projet récent", None);
        }
        ui.separator();
        if menu_action(ui, !self.recent.paths().is_empty(), "Effacer la liste des projets récents", None) {
            self.recent.clear();
        }
        if let Some(path) = chosen {
            if path.is_file() {
                self.open(ctx, path);
            } else {
                self.status = format!("Vidéo introuvable : {}", path.display());
                self.recent.remove(&path);
            }
        }
    }

    fn toolbar(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let menus = egui::containers::menu::MenuConfig::new().style(classic_menu_style);
        egui::MenuBar::new().style(classic_menu_style).config(menus).ui(ui, |ui| {
            ui.menu_button("Fichier", |ui| {
                if menu_action(ui, true, "Ouvrir une vidéo…", None) && self.picker.is_none() {
                    self.pick_file(ctx);
                }
                // La marge reprend celle des autres lignes du menu, réservée à la coche
                ui.spacing_mut().button_padding = egui::vec2(24.0, 5.0);
                ui.menu_button("Ouvrir les projets récents", |ui| self.recent_menu(ui, ctx));
                ui.separator();
                let dirty = self.media.as_ref().is_some_and(|m| m.dirty);
                if menu_action(ui, dirty, "Enregistrer", Some("Ctrl+S")) {
                    self.save_now();
                    if self.media.as_ref().is_some_and(|m| !m.dirty) {
                        self.status = "Projet enregistré".to_string();
                    }
                }
            });
            let (can_undo, can_redo) = self.media.as_ref().map_or((false, false), |m| (!m.undo.is_empty(), !m.redo.is_empty()));
            ui.menu_button("Édition", |ui| {
                if menu_action(ui, can_undo, "Annuler", Some("Ctrl+Z")) {
                    self.undo(false);
                }
                if menu_action(ui, can_redo, "Rétablir", Some("Ctrl+Y")) {
                    self.undo(true);
                }
            });
            ui.menu_button("Affichage", |ui| {
                ui.spacing_mut().button_padding = egui::vec2(12.0, 5.0);
                ui.menu_button("Pistes", |ui| {
                    if menu_check(ui, self.tracks.strip, "Image").clicked() {
                        self.tracks.strip = !self.tracks.strip;
                    }
                    if menu_check(ui, self.tracks.audio, "Audio").clicked() {
                        self.tracks.audio = !self.tracks.audio;
                    }
                });
                ui.menu_button("Rendu", |ui| {
                    for &backend in renderer::RenderBackend::ALL {
                        if menu_check(ui, self.render_choice == backend, backend.label()).clicked() && self.render_choice != backend {
                            self.render_choice = backend;
                            backend.save();
                        }
                    }
                    if self.render_choice != self.render_running {
                        ui.separator();
                        ui.label(RichText::new("Appliqué au prochain lancement").size(11.0));
                    }
                });
            });
            if ui.button("Aide").clicked() {
                self.show_help = true;
            }
            ui.separator();
            if let Some(error) = &self.player_error {
                ui.colored_label(Color32::from_rgb(240, 120, 110), format!("Lecteur indisponible : {error}"));
            }
            if !self.status.is_empty() {
                ui.add(egui::Label::new(self.status.clone()).truncate()).on_hover_text(self.status.clone());
            }
            if let Some(folder) = self.last_export.clone() {
                if ui.small_button("Ouvrir le dossier").clicked() {
                    mx_lab::paths::reveal(&folder);
                }
            }
        });
    }

    fn video_panel(&mut self, ui: &mut egui::Ui) {
        let full = ui.available_rect_before_wrap();
        let transport_height = 40.0;
        let video = egui::Rect::from_min_max(full.min, egui::pos2(full.right(), full.bottom() - transport_height));
        let response = ui.interact(video, ui.scope_id().with("video-surface"), egui::Sense::click());
        let scale = ui.ctx().pixels_per_point();
        self.video_size = [(video.width() * scale) as i32, (video.height() * scale) as i32];
        if self.media.is_none() {
            let hovering = ui.input(|i| !i.raw.hovered_files.is_empty());
            let text = if hovering { "Déposez la vidéo" } else { "Glissez une vidéo ici, ou cliquez pour l'ouvrir" };
            ui.painter().text(
                video.center(),
                egui::Align2::CENTER_CENTER,
                text,
                egui::FontId::proportional(18.0),
                Color32::from_gray(170),
            );
            if response.clicked() && self.picker.is_none() {
                self.pick_file(&ui.ctx().clone());
            }
        } else {
            if let Some(texture) = &self.video_texture {
                let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                ui.painter().image(texture.id(), video, uv, Color32::WHITE);
            }
            if self.player.as_ref().is_some_and(|p| !p.state.loaded && !p.state.failed) {
                let time = ui.input(|i| i.time);
                timeline::paint_loading(ui.painter(), video, "Ouverture de la vidéo", None, time);
                ui.ctx().request_repaint_after(timeline::LOADING_REFRESH);
            }
            if response.clicked() {
                self.toggle_play();
            }
        }
        let bar = egui::Rect::from_min_max(egui::pos2(full.left(), video.bottom()), full.max);
        ui.painter().rect_filled(bar, 0.0, Color32::from_rgb(28, 30, 35));
        let mut child = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(bar.shrink2(egui::vec2(8.0, 4.0)))
                .layout(egui::Layout::left_to_right(egui::Align::Center)),
        );
        self.transport(&mut child);
    }

    fn transport(&mut self, ui: &mut egui::Ui) {
        let fps = self.fps();
        let position = self.position();
        let duration = self.duration();
        ui.label(RichText::new(tc(position, fps)).monospace().size(17.0).color(Color32::WHITE));
        ui.label(RichText::new(format!("/ {}", tc(duration, fps))).monospace().size(11.0).weak());
        ui.separator();
        ui.spacing_mut().item_spacing.x = 4.0;
        if icon_button(ui, Icon::Previous, "Séquence précédente (flèche haut)").clicked() {
            self.select_neighbour(false);
        }
        if icon_button(ui, Icon::Reverse, "Lecture arrière, plus vite à chaque appui (J)").clicked() {
            self.shuttle_backward();
        }
        if icon_button(ui, Icon::StepBack, "Image précédente (flèche gauche)").clicked() {
            self.step_frame(false);
        }
        let playing = !self.paused() || self.reverse_speed > 0.0;
        let icon = if playing { Icon::Pause } else { Icon::Play };
        if icon_button(ui, icon, "Lecture / pause (Espace)").clicked() {
            self.toggle_play();
        }
        if icon_button(ui, Icon::StepForward, "Image suivante (flèche droite)").clicked() {
            self.step_frame(true);
        }
        if icon_button(ui, Icon::Forward, "Lecture avant, plus vite à chaque appui (L)").clicked() {
            self.shuttle_forward();
        }
        if icon_button(ui, Icon::Next, "Séquence suivante (flèche bas)").clicked() {
            self.select_neighbour(true);
        }
        ui.separator();
        if icon_toggle(ui, Icon::Loop, "Boucler sur la séquence choisie (B)", self.loop_on).clicked() {
            self.loop_on = !self.loop_on;
        }

        let speed = if self.reverse_speed > 0.0 {
            -self.reverse_speed
        } else {
            self.player.as_ref().map_or(1.0, |p| p.state.speed)
        };
        let mut chosen = speed;
        egui::ComboBox::from_id_salt("speed")
            .width(58.0)
            .selected_text(format!("{speed}×"))
            .show_ui(ui, |ui| {
                for value in SPEEDS {
                    ui.selectable_value(&mut chosen, value, format!("{value}×"));
                }
            })
            .response
            .on_hover_text("Vitesse de lecture");
        if chosen != speed {
            self.reverse_speed = 0.0;
            if let Some(player) = &mut self.player {
                player.set_speed(chosen);
            }
        }

    }

    fn left_panel(&mut self, ui: &mut egui::Ui) {
        ui.set_min_width(ui.available_width());
        if self.media.is_none() {
            ui.add_space(12.0);
            ui.weak("La fiche de la vidéo et sa loudness apparaîtront ici.");
            return;
        }
        egui::ScrollArea::vertical().id_salt("left-panel").auto_shrink([false, false]).show(ui, |ui| {
            self.film_section(ui);
            ui.add_space(6.0);
            ui.separator();
            self.loudness_section(ui);
        });
    }

    /// La poignée entre le tableau et la fiche : la glisser partage la hauteur du panneau.
    fn table_splitter(&mut self, ui: &mut egui::Ui) {
        let (rect, response) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 9.0), egui::Sense::drag());
        response.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Slider, true, "Partage tableau et fiche"));
        let active = response.hovered() || response.dragged();
        if active {
            ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
        }
        let color = if active { Color32::from_gray(150) } else { ui.visuals().widgets.noninteractive.bg_stroke.color };
        ui.painter().line_segment(
            [egui::pos2(rect.left(), rect.center().y), egui::pos2(rect.right(), rect.center().y)],
            egui::Stroke::new(if active { 2.0 } else { 1.0 }, color),
        );
        if response.dragged() {
            self.sheet_height = Some((self.sheet_height_now - response.drag_delta().y).max(MIN_SHEET_HEIGHT));
        }
    }

    fn side_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        // Un panneau prend la largeur de son contenu : vide, il se replierait pour de bon
        ui.set_min_width(ui.available_width());
        if self.media.is_none() {
            ui.add_space(12.0);
            ui.weak("Les séquences de la vidéo apparaîtront ici.");
            return;
        }
        self.sequence_list(ui);
        let top = ui.cursor().top();
        self.export_row(ui, ctx);
        self.table_splitter(ui);
        self.selected_section(ui);
        // Le tableau prendra tout le reste à l'image suivante : pas de vide sous la fiche
        self.sheet_measured[self.selected.is_some() as usize] = ui.cursor().top() - top + SHEET_BOTTOM_MARGIN;
    }

    fn film_section(&mut self, ui: &mut egui::Ui) {
        let mut new_track = None;
        egui::CollapsingHeader::new(RichText::new("Fiche de la vidéo").strong()).default_open(true).show(ui, |ui| {
            let Some(media) = &mut self.media else { return };
            egui::Grid::new("film-sheet").num_columns(2).spacing([8.0, 3.0]).show(ui, |ui| {
                let mut rows = media.release.rows();
                if !media.info.resolution.is_empty() {
                    rows.push(("Résolution", media.info.resolution.clone()));
                }
                if !media.info.video.is_empty() {
                    rows.push(("Vidéo", media.info.video.clone()));
                }
                for (label, value) in rows {
                    ui.weak(label);
                    ui.add(egui::Label::new(value).wrap());
                    ui.end_row();
                }
                let tracks = &media.info.audio;
                let current = media.project.audio_track;
                if tracks.len() == 1 {
                    ui.weak("Audio");
                    ui.add(egui::Label::new(tracks[0].label.clone()).wrap());
                    ui.end_row();
                } else if tracks.len() > 1 {
                    ui.weak("Audio");
                    let shown = tracks.get(current).map_or(String::new(), |t| t.label.clone());
                    let mut chosen = current;
                    egui::ComboBox::from_id_salt("audio-track")
                        .truncate()
                        .width(ui.available_width().max(100.0))
                        .selected_text(shown)
                        .show_ui(ui, |ui| {
                            for (index, track) in tracks.iter().enumerate() {
                                ui.selectable_value(&mut chosen, index, track.label.clone());
                            }
                        })
                        .response
                        .on_hover_text("Piste écoutée, analysée et mesurée. En changer relance les analyses de la vidéo.");
                    ui.end_row();
                    if chosen != current {
                        new_track = Some(chosen);
                    }
                }
            });
            ui.add_space(4.0);
            let label = ui.label("Notes sur la vidéo");
            let notes = ui
                .add(egui::TextEdit::multiline(&mut media.project.film_notes).desired_rows(2).desired_width(f32::INFINITY))
                .labelled_by(label.id);
            if notes.changed() {
                media.mark_changed();
            }
        });
        if let Some(track) = new_track {
            let ctx = ui.ctx().clone();
            self.change_audio_track(&ctx, track);
        }
    }

    /// Gap, anticipation et post, dans la fenêtre de l'engrenage. La sensibilité est dans l'en-tête.
    fn detection_controls(&mut self, ui: &mut egui::Ui) {
        let Some(media) = &self.media else { return };
        let mut settings = media.project.settings.clone();
        let has_scores = !media.project.scores.is_empty();
        let mut changed = false;
        let mut released = false;
        ui.set_min_width(250.0);
        ui.add_enabled_ui(has_scores, |ui| {
            ui.spacing_mut().interact_size = egui::vec2(timeline::VALUE_WIDTH, 18.0);
            let mut row = |label: &str, help: &str, value: &mut f64, range: std::ops::RangeInclusive<f64>| {
                ui.horizontal(|ui| {
                    let label = timeline::left_label(ui, 100.0, RichText::new(label)).on_hover_text(help);
                    // Mêmes dimensions sur toutes les lignes, quelle que soit la valeur
                    ui.spacing_mut().slider_width = (ui.available_width() - timeline::VALUE_WIDTH - ui.spacing().item_spacing.x).max(30.0);
                    let slider = ui.add(egui::Slider::new(value, range.clone()).step_by(1.0).show_value(false));
                    let drag = egui::DragValue::new(value).range(range).speed(0.25).fixed_decimals(0);
                    let drag = ui.add_sized([timeline::VALUE_WIDTH, 18.0], drag);
                    for response in [slider.labelled_by(label.id).on_hover_text(help), drag.labelled_by(label.id).on_hover_text(help)] {
                        changed |= response.changed();
                        released |= response.drag_stopped() || (response.changed() && !response.dragged());
                    }
                });
            };
            row("Gap (s)", GAP_HELP, &mut settings.gap, 1.0..=60.0);
            row("Anticipation (s)", ANTICIPATION_HELP, &mut settings.anticipation, 0.0..=20.0);
            row("Post (s)", POST_HELP, &mut settings.post, 0.0..=20.0);
        });
        self.apply_detection_settings(settings, changed, released);
    }

    /// Applique des réglages de détection modifiés ; une seule étape d'annulation par geste.
    fn apply_detection_settings(&mut self, settings: project::DetectionSettings, changed: bool, released: bool) {
        if changed {
            if !self.settings_grabbed {
                self.remember();
                self.settings_grabbed = true;
            }
            if let Some(media) = &mut self.media {
                media.project.settings = settings;
            }
            self.apply_detection();
        }
        if released {
            self.settings_grabbed = false;
        }
    }

    fn sequence_list(&mut self, ui: &mut egui::Ui) {
        let fps = self.fps();
        // La lecture entre dans une autre séquence : la liste la garde visible. Seulement à ce
        // moment-là, pour ne pas reprendre la main à qui fait défiler la liste.
        let playing = self.segment_at(self.position());
        let follow = playing.is_some() && playing != self.listed_playing && !self.paused();
        self.listed_playing = playing;
        let Some(media) = &self.media else { return };
        let count = media.project.segments.len();
        let total = media.project.total_music();
        let duration = self.duration();
        let share = if duration > 0.0 { total / duration * 100.0 } else { 0.0 };
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("Séquences ({count})")).strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.weak(format!("{} de musique · {share:.0} %", short_duration(total)));
                if let Some(progress) = media.tags_progress {
                    ui.weak(format!("Mood et instruments : {progress} %  ·"));
                }
            });
        });
        let header_columns = table_columns("#", " ", "Début", "Fin", "Durée");
        let header_loudness = loudness_header("TP", "I", "S", "M", "LRA");
        let (header_rect, header) =
            ui.allocate_exact_size(egui::vec2(ui.available_width(), TABLE_ROW_HEIGHT - 2.0), egui::Sense::hover());
        header.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Label, true, format!("{header_columns} Mood Ensemble Instruments Nom Note {header_loudness}")));
        let weak = ui.visuals().weak_text_color();
        let header_tags = music_tags::SequenceTags {
            mood: "Mood".into(),
            ensembles: "Ensemble".into(),
            instruments: "Instruments".into(),
        };
        paint_table_row(ui, header_rect, &header_columns, &header_loudness, &header_tags, &BTreeMap::new(), "Nom", "Note", weak);
        header.on_hover_text(
            "Loudness du mix complet sur chaque séquence.\nTP : true peak, en dBTP.\nI : intégrée, en LKFS.\n\
             S : short-term max, en LKFS.\nM : momentary max, en LKFS.\nLRA : plage de loudness, en LU.\n\n\
             Mood, ensemble et instruments : indices du modèle de détection, à vérifier à l'écoute. \
             Vide : rien d'assez net.\n\n\
             Double-clic sur une valeur : la modifier au clavier (Entrée : valider, Échap : annuler). \
             Une valeur saisie apparaît en orange ; la vider rend la valeur calculée.",
        );
        let mut clicked = None;
        let mut begin_edit = None;
        let mut finished_edit = None;
        let char_width = table_char_width(ui.ctx());
        let scroll = std::mem::take(&mut self.scroll_list);
        let selected = self.selected;
        // La liste prend tout ce que la fiche de la séquence choisie laisse libre
        let reserved = self.sheet_height.unwrap_or(self.sheet_measured[selected.is_some() as usize]);
        self.sheet_height_now = reserved;
        let room = (ui.available_height() - reserved).max(90.0);
        egui::ScrollArea::vertical().id_salt("sequence-list").max_height(room).auto_shrink([false, false]).show(ui, |ui| {
            for (index, segment) in media.project.segments.iter().enumerate() {
                let marker = match segment.origin {
                    Origin::Detected => " ",
                    Origin::Edited => "*",
                    Origin::Manual => "+",
                };
                let columns = table_columns(
                    &(index + 1).to_string(),
                    marker,
                    &tc(segment.start, fps),
                    &tc(segment.end, fps),
                    &table_duration(segment.end - segment.start),
                );
                let loudness = media.sequence_loudness.get(&segment.id).map(|(_, measure)| measure);
                let measures = loudness_columns(loudness, &segment.overrides);
                let empty = music_tags::SequenceTags::default();
                let tags = media.sequence_tags.get(&segment.id).map_or(&empty, |(_, tags)| tags);
                let name = note_preview(&segment.name);
                let note = note_preview(&segment.notes);
                let is_selected = selected == Some(segment.id);
                let is_playing = playing == Some(segment.id);
                // Toute la ligne est cliquable, note comprise
                let (rect, row) = ui.allocate_exact_size(egui::vec2(ui.available_width(), TABLE_ROW_HEIGHT), egui::Sense::click());
                row.widget_info(|| {
                    let state = if is_playing { " (en lecture)" } else { "" };
                    egui::WidgetInfo::selected(egui::Role::Button, true, is_selected, format!("{columns} {} {} {} {name} {note} {measures}{state}", tags.mood, tags.ensembles, tags.instruments))
                });
                let visuals = ui.visuals();
                if is_selected {
                    ui.painter().rect_filled(rect, 3.0, visuals.selection.bg_fill);
                } else if row.hovered() {
                    ui.painter().rect_filled(rect, 3.0, visuals.widgets.hovered.weak_bg_fill);
                }
                // Le liseré reprend la couleur de la tête de lecture, même sur une ligne sélectionnée
                if is_playing {
                    let (left, middle) = (rect.left() + 1.0, rect.center().y);
                    ui.painter().add(egui::Shape::convex_polygon(
                        vec![egui::pos2(left, middle - 5.0), egui::pos2(left + 8.0, middle), egui::pos2(left, middle + 5.0)],
                        timeline::PLAYHEAD,
                        egui::Stroke::NONE,
                    ));
                }
                let text_color = if is_selected {
                    visuals.selection.stroke.color
                } else {
                    visuals.text_color()
                };
                paint_table_row(ui, rect, &columns, &measures, tags, &segment.overrides, &name, &note, text_color);

                // Double-clic : la valeur sous la souris devient modifiable
                if row.double_clicked() {
                    let cell = row.interact_pointer_pos().and_then(|pos| {
                        table_cells(rect, char_width).into_iter().find(|(_, cell)| pos.x >= cell.left() && pos.x < cell.right() + 3.0)
                    });
                    if let Some((column, _)) = cell {
                        let text = table_value(segment, column, loudness, tags, fps);
                        begin_edit = Some(TableEdit { segment: segment.id, column, initial: text.clone(), text, needs_focus: true });
                    }
                }
                if let Some(edit) = self.table_edit.as_mut().filter(|edit| edit.segment == segment.id) {
                    let cell = table_cells(rect, char_width).into_iter().find(|(column, _)| *column == edit.column).map(|c| c.1);
                    if let Some(cell) = cell {
                        // Les dernières colonnes s'ouvrent vers la gauche, pour rester dans la ligne
                        let width = cell.width().max(TABLE_EDIT_MIN_WIDTH);
                        let left = cell.left().min(rect.right() - width).max(rect.left());
                        let area = egui::Rect::from_min_max(egui::pos2(left, cell.top()), egui::pos2((left + width).min(rect.right()), cell.bottom()));
                        let input = egui::TextEdit::singleline(&mut edit.text).margin(egui::vec2(2.0, 0.0)).desired_width(area.width());
                        let response = ui.put(area, input).labelled_by(row.id);
                        if edit.needs_focus {
                            edit.needs_focus = false;
                            response.request_focus();
                            // Tout sélectionné : on tape directement la nouvelle valeur
                            if let Some(mut state) = egui::TextEdit::load_state(ui.ctx(), response.id) {
                                let length = edit.text.chars().count();
                                state.cursor.set_char_range(Some(egui::text::CCursorRange::two(
                                    egui::text::CCursor::new(0),
                                    egui::text::CCursor::new(length),
                                )));
                                state.store(ui.ctx(), response.id);
                            }
                        } else if response.lost_focus() {
                            finished_edit = Some(!ui.input(|i| i.key_pressed(egui::Key::Escape)));
                        }
                    }
                }
                // Au survol, le nom et la note entiers, sans coupure
                let tooltip = match (segment.name.is_empty(), segment.notes.trim().is_empty()) {
                    (true, true) => None,
                    (false, true) => Some(segment.name.clone()),
                    (true, false) => Some(segment.notes.clone()),
                    (false, false) => Some(format!("{}\n\n{}", segment.name, segment.notes)),
                };
                let row = match tooltip {
                    Some(text) => row.on_hover_text(text),
                    None => row,
                };
                if row.clicked() {
                    clicked = Some(segment.id);
                }
                if (is_selected && scroll) || (is_playing && follow) {
                    row.scroll_to_me(Some(egui::Align::Center));
                }
            }
            if count == 0 {
                let text = if media.detect_progress.is_some() {
                    "Détection en cours…"
                } else {
                    "Aucune séquence. Glissez sur la piste Séquences pour en créer une."
                };
                ui.weak(text);
            }
        });
        if let Some(id) = clicked {
            self.select(id);
        }
        if let Some(keep) = finished_edit {
            if let Some(edit) = self.table_edit.take() {
                if keep {
                    self.commit_table_edit(edit);
                }
            }
        }
        if let Some(edit) = begin_edit {
            self.table_edit = Some(edit);
        }
    }

    /// Applique ce qui a été tapé dans le tableau. Annulable comme toute autre correction.
    fn commit_table_edit(&mut self, edit: TableEdit) {
        if edit.text == edit.initial {
            return;
        }
        let fps = self.fps();
        if self.media.as_ref().and_then(|m| m.project.segment(edit.segment)).is_none() {
            return;
        }
        let time = match edit.column {
            Column::Start | Column::End => match parse_time(&edit.text, fps) {
                Some(time) => Some(time),
                None => {
                    self.status = format!("Valeur de temps non comprise : {}", edit.text);
                    return;
                }
            },
            _ => None,
        };
        self.remember();
        let Some(media) = &mut self.media else { return };
        let mut limited = false;
        match (edit.column, time) {
            (Column::Start, Some(time)) => {
                media.project.set_start(edit.segment, time);
                limited = media.project.segment(edit.segment).is_some_and(|s| (s.start - time).abs() > 0.001);
            }
            (Column::End, Some(time)) => {
                media.project.set_end(edit.segment, time);
                limited = media.project.segment(edit.segment).is_some_and(|s| (s.end - time).abs() > 0.001);
            }
            _ => {
                if let Some(target) = media.project.segment_mut(edit.segment) {
                    match edit.column {
                        Column::Name => target.name = edit.text.clone(),
                        Column::Note => target.notes = edit.text.clone(),
                        column => {
                            if let Some(key) = column.override_key() {
                                if edit.text.trim().is_empty() {
                                    target.overrides.remove(key);
                                } else {
                                    target.overrides.insert(key.to_string(), edit.text.trim().to_string());
                                }
                            }
                        }
                    }
                    // Une séquence corrigée ne doit pas disparaître au prochain réglage de détection
                    if target.origin == Origin::Detected {
                        target.origin = Origin::Edited;
                    }
                }
            }
        }
        media.mark_changed();
        if limited {
            self.status = "Valeur limitée par la séquence voisine ou la durée minimale.".to_string();
        }
    }

    fn export_row(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let count = self.media.as_ref().map_or(0, |m| m.project.segments.len());
        ui.horizontal(|ui| {
            let available = !self.exporting;
            let hint = "Export sans réencodage, dans un dossier « music-sequences » à côté de la vidéo";
            if ui
                .add_enabled(available && self.selected.is_some(), egui::Button::new("Exporter"))
                .on_hover_text(hint)
                .clicked()
            {
                self.export(ctx, self.selected);
            }
            if ui
                .add_enabled(available && count > 0, egui::Button::new(format!("Tout exporter ({count})")))
                .on_hover_text(hint)
                .clicked()
            {
                self.export(ctx, None);
            }
            if ui
                .add_enabled(self.selected.is_some(), egui::Button::new("Supprimer"))
                .on_hover_text("Supprimer la séquence choisie (Suppr)")
                .clicked()
            {
                self.delete_selected();
            }
            if self.exporting {
                ui.spinner();
            }
        });
    }

    /// Début ou fin : on glisse pour déplacer la borne, on double-clique pour la taper.
    #[allow(clippy::too_many_arguments)]
    fn bound_control(&mut self, ui: &mut egui::Ui, id: u64, column: Column, name: &str, width: f32, value: String, segment: &Segment, fps: f64) {
        let (rect, response) = ui.allocate_exact_size(egui::vec2(width, ui.spacing().interact_size.y), egui::Sense::click_and_drag());
        response.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Slider, true, name));
        let visuals = ui.visuals();
        let (fill, color) = (visuals.extreme_bg_color, visuals.text_color());
        ui.painter().rect_filled(rect, rect.height() / 2.0, fill);
        ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, &value, egui::FontId::monospace(12.0), color);
        if response.hovered() || response.dragged() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
        }
        let current = if column == Column::Start { segment.start } else { segment.end };
        if response.drag_started() {
            self.remember();
            self.bound_drag = Some((column, current, 0.0));
        }
        if response.dragged() {
            // Une image par pixel ; une seconde par pixel avec Maj
            let step = if ui.input(|i| i.modifiers.shift) { 1.0 } else { 1.0 / fps };
            let delta = response.drag_delta().x as f64 * step;
            if let (Some((dragged, origin, offset)), Some(media)) = (&mut self.bound_drag, &mut self.media) {
                if *dragged == column {
                    *offset += delta;
                    let time = (*origin + *offset).max(0.0);
                    if column == Column::Start {
                        media.project.set_start(id, time);
                    } else {
                        media.project.set_end(id, time);
                    }
                    media.mark_changed();
                }
            }
        }
        if response.drag_stopped() {
            self.bound_drag = None;
        }
        if response.double_clicked() {
            self.field_edit = Some(TableEdit { segment: id, column, initial: value.clone(), text: value, needs_focus: true });
        }
    }

    /// Les valeurs du tableau, en champs : mêmes règles que le double-clic dans le tableau.
    fn sequence_value_fields(&mut self, ui: &mut egui::Ui, id: u64) {
        let fps = self.fps();
        let Some(media) = &self.media else { return };
        let Some(segment) = media.project.segment(id).cloned() else { return };
        let loudness = media.sequence_loudness.get(&id).map(|(_, measure)| measure.clone());
        let tags = media.sequence_tags.get(&id).map_or_else(Default::default, |(_, tags)| tags.clone());
        const ROWS: [&[(Column, &str, f32)]; 2] = [
            &[
                (Column::Start, "Début", 90.0),
                (Column::End, "Fin", 90.0),
            ],
            &[(Column::Mood, "Mood", 100.0), (Column::Ensembles, "Ensemble", 170.0), (Column::Instruments, "Instruments", 240.0)],
        ];
        let mut finished = None;
        for (row_index, row) in ROWS.iter().enumerate() {
            ui.horizontal(|ui| {
                // Les champs se resserrent dans un panneau étroit : sinon il s'élargirait tout seul
                let scale = ((ui.available_width() - 420.0) / 270.0).clamp(0.3, 1.0);
                if row_index == 1 {
                    self.type_field(ui, id);
                }
                for &(column, name, width) in row.iter() {
                    let value = table_value(&segment, column, loudness.as_ref(), &tags, fps);
                    let editing = self.field_edit.as_ref().is_some_and(|e| e.segment == id && e.column == column);
                    if matches!(column, Column::Start | Column::End) && !editing {
                        ui.label(name);
                        self.bound_control(ui, id, column, name, width, value, &segment, fps);
                        continue;
                    }
                    let written = column.override_key().is_some_and(|key| segment.overrides.contains_key(key));
                    let active = self.field_edit.as_ref().filter(|e| e.segment == id && e.column == column);
                    let mut text = active.map_or_else(|| value.clone(), |e| e.text.clone());
                    let label = ui.label(name);
                    let width = if column == Column::Instruments { f32::INFINITY } else { width * scale };
                    let mut input = egui::TextEdit::singleline(&mut text).desired_width(width);
                    if written {
                        input = input.text_color(TABLE_OVERRIDE_COLOR);
                    }
                    let response = ui.add(input).labelled_by(label.id);
                    if self.field_edit.as_ref().is_some_and(|e| e.needs_focus && e.segment == id && e.column == column) {
                        response.request_focus();
                        if let Some(mut state) = egui::TextEdit::load_state(ui.ctx(), response.id) {
                            state.cursor.set_char_range(Some(egui::text::CCursorRange::two(
                                egui::text::CCursor::new(0),
                                egui::text::CCursor::new(text.chars().count()),
                            )));
                            state.store(ui.ctx(), response.id);
                        }
                        if let Some(edit) = &mut self.field_edit {
                            edit.needs_focus = false;
                        }
                    } else if response.has_focus() {
                        match self.field_edit.as_mut().filter(|e| e.segment == id && e.column == column) {
                            Some(edit) => edit.text = text,
                            None => {
                                self.field_edit = Some(TableEdit { segment: id, column, initial: value, text, needs_focus: false });
                            }
                        }
                    } else if response.lost_focus() {
                        if let Some(edit) = self.field_edit.as_mut().filter(|e| e.segment == id && e.column == column) {
                            edit.text = text;
                        }
                        finished = Some(!ui.input(|i| i.key_pressed(egui::Key::Escape)));
                    }
                }
                if row_index == 0 {
                    self.name_field(ui, id);
                }
            });
        }
        if let Some(keep) = finished {
            if let Some(edit) = self.field_edit.take() {
                if keep {
                    self.commit_table_edit(edit);
                }
            }
        }
    }

    fn name_field(&mut self, ui: &mut egui::Ui, id: u64) {
        let Some(media) = &mut self.media else { return };
        let Some(segment) = media.project.segment_mut(id) else { return };
        let label = ui.label("Nom");
        let changed = ui.add(egui::TextEdit::singleline(&mut segment.name).desired_width(f32::INFINITY)).labelled_by(label.id).changed();
        if changed {
            if segment.origin == Origin::Detected {
                segment.origin = Origin::Edited;
            }
            media.mark_changed();
        }
    }

    fn type_field(&mut self, ui: &mut egui::Ui, id: u64) {
        let Some(media) = &mut self.media else { return };
        let Some(segment) = media.project.segment_mut(id) else { return };
        let label = ui.label("Type");
        let mut kind = segment.kind;
        egui::ComboBox::from_id_salt("music-kind")
            .selected_text(kind.label())
            .show_ui(ui, |ui| {
                for option in MusicKind::ALL {
                    ui.selectable_value(&mut kind, option, option.label());
                }
            })
            .response
            .labelled_by(label.id);
        if kind != segment.kind {
            segment.kind = kind;
            if segment.origin == Origin::Detected {
                segment.origin = Origin::Edited;
            }
            media.mark_changed();
        }
    }

    fn selected_section(&mut self, ui: &mut egui::Ui) {
        let Some(id) = self.selected else {
            ui.weak("Choisissez une séquence pour l'annoter.");
            return;
        };
        self.sequence_value_fields(ui, id);
        if let Some(media) = &mut self.media {
            let mut touched = false;
            if let Some(segment) = media.project.segment_mut(id) {
                ui.horizontal_top(|ui| {
                    let label = ui.label("Notes");
                    touched |= ui
                        .add(
                            egui::TextEdit::multiline(&mut segment.notes)
                                .hint_text("Thème, instrumentation, rôle dans la scène…")
                                .desired_rows(2)
                                .desired_width(f32::INFINITY),
                        )
                        .labelled_by(label.id)
                        .changed();
                });
                // Une séquence annotée ne doit pas disparaître au prochain réglage de détection
                if touched && segment.origin == Origin::Detected {
                    segment.origin = Origin::Edited;
                }
            }
            if touched {
                media.mark_changed();
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Icon {
    Previous,
    Reverse,
    StepBack,
    Play,
    Pause,
    StepForward,
    Forward,
    Next,
    Loop,
}

/// Bouton de transport. Dessiné plutôt qu'écrit : la police n'a pas ces symboles.
fn icon_button(ui: &mut egui::Ui, icon: Icon, label: &str) -> egui::Response {
    icon_toggle(ui, icon, label, false)
}

/// Même bouton, avec un état allumé ou éteint.
fn icon_toggle(ui: &mut egui::Ui, icon: Icon, label: &str, on: bool) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(30.0, 26.0), egui::Sense::click());
    response.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, label));
    let visuals = ui.style().interact(&response);
    let painter = ui.painter();
    let selection = ui.visuals().selection;
    painter.rect_filled(rect, 3.0, if on { selection.bg_fill } else { visuals.weak_bg_fill });
    let color = if on { selection.stroke.color } else { visuals.fg_stroke.color };
    let center = rect.center();
    let triangle = |x: f32, right: bool, size: f32| {
        let tip = if right { size } else { -size };
        painter.add(egui::Shape::convex_polygon(
            vec![
                egui::pos2(center.x + x - tip, center.y - size),
                egui::pos2(center.x + x + tip, center.y),
                egui::pos2(center.x + x - tip, center.y + size),
            ],
            color,
            egui::Stroke::NONE,
        ));
    };
    let bar = |x: f32, half_height: f32| {
        let size = egui::vec2(2.5, half_height * 2.0);
        painter.rect_filled(egui::Rect::from_center_size(center + egui::vec2(x, 0.0), size), 0.0, color);
    };
    match icon {
        Icon::Previous => {
            bar(-7.0, 6.0);
            triangle(0.0, false, 6.0);
        }
        Icon::Reverse => {
            triangle(-5.0, false, 5.0);
            triangle(5.0, false, 5.0);
        }
        Icon::StepBack => {
            triangle(-2.0, false, 4.0);
            bar(5.0, 5.0);
        }
        Icon::Play => triangle(1.0, true, 7.0),
        Icon::Pause => {
            bar(-3.5, 6.5);
            bar(3.5, 6.5);
        }
        Icon::StepForward => {
            bar(-5.0, 5.0);
            triangle(2.0, true, 4.0);
        }
        Icon::Forward => {
            triangle(-5.0, true, 5.0);
            triangle(5.0, true, 5.0);
        }
        Icon::Next => {
            triangle(0.0, true, 6.0);
            bar(7.0, 6.0);
        }
        Icon::Loop => {
            // Un cercle ouvert en haut à droite, terminé par une pointe de flèche
            let radius = 6.5f32;
            let point = |degrees: f32| {
                let angle = degrees.to_radians();
                center + radius * egui::vec2(angle.cos(), angle.sin())
            };
            let arc: Vec<egui::Pos2> = (0..=28).map(|step| point(-30.0 - step as f32 * 10.0)).collect();
            let end = *arc.last().unwrap();
            painter.add(egui::Shape::line(arc, egui::Stroke::new(1.8, color)));
            painter.add(egui::Shape::convex_polygon(
                vec![end + egui::vec2(-1.0, -4.5), end + egui::vec2(5.0, -0.5), end + egui::vec2(-1.5, 3.5)],
                color,
                egui::Stroke::NONE,
            ));
        }
    }
    response.on_hover_text(label)
}

/// Une erreur interne d'un travail en arrière-plan ne doit pas emporter l'application.
fn guarded<T>(work: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(work))
        .unwrap_or_else(|_| Err("échec interne de l'analyse".to_string()))
}

/// Inter d'abord ; les polices d'egui restent derrière pour les symboles qu'elle n'a pas.
fn interface_fonts() -> egui::FontDefinitions {
    // Inter dessine plus large que la police d'egui : les colonnes denses doivent tenir
    const INTER_SCALE: f32 = 0.94;
    let mut fonts = egui::FontDefinitions::default();
    let regular: &[u8] = include_bytes!("../../assets/fonts/Inter-Regular.ttf");
    let medium: &[u8] = include_bytes!("../../assets/fonts/Inter-Medium.ttf");
    for (name, data) in [("Inter", regular), ("Inter Medium", medium)] {
        let tweak = egui::FontTweak { scale: INTER_SCALE, ..Default::default() };
        fonts.font_data.insert(name.to_owned(), Arc::new(egui::FontData::from_static(data).tweak(tweak)));
    }
    let proportional = fonts.families.entry(egui::FontFamily::Proportional).or_default();
    proportional.insert(0, "Inter".to_owned());
    let mut heavier = proportional.clone();
    heavier[0] = "Inter Medium".to_owned();
    fonts.families.insert(egui::FontFamily::Name("medium".into()), heavier);
    fonts
}

fn style(ctx: &egui::Context) {
    ctx.set_fonts(interface_fonts());
    ctx.global_style_mut(|style| {
        style.visuals = egui::Visuals::dark();
        // Menus déroulants : arrondis, ombre douce, fin liseré, un peu d'air autour des lignes
        style.visuals.menu_corner_radius = egui::CornerRadius::same(8);
        style.visuals.popup_shadow =
            egui::Shadow { offset: [0, 6], blur: 20, spread: 0, color: Color32::from_black_alpha(110) };
        style.visuals.window_fill = Color32::from_gray(36);
        style.visuals.window_stroke = egui::Stroke::new(1.0, Color32::from_gray(62));
        style.spacing.menu_margin = egui::Margin::same(6);
        style.text_styles.insert(
            egui::TextStyle::Heading,
            egui::FontId::new(18.0, egui::FontFamily::Name("medium".into())),
        );
    });
    ctx.options_mut(|options| options.zoom_with_keyboard = false);
}

/// Le texte des menus ne change jamais de couleur : seul le fond réagit au survol.
const MENU_TEXT: Color32 = Color32::from_gray(215);

fn classic_menu_style(style: &mut egui::Style) {
    egui::containers::menu::menu_style(style);
    let widgets = &mut style.visuals.widgets;
    for state in [&mut widgets.inactive, &mut widgets.hovered, &mut widgets.active, &mut widgets.open] {
        state.fg_stroke = egui::Stroke::new(1.0, MENU_TEXT);
    }
}

/// Ligne de menu classique : place pour une coche à gauche, raccourci à droite,
/// et toute la largeur réagit au survol. `checked` à `None` : une simple commande.
fn menu_row(ui: &mut egui::Ui, enabled: bool, checked: Option<bool>, label: &str, shortcut: Option<&str>) -> egui::Response {
    const GUTTER: f32 = 24.0;
    const RIGHT_MARGIN: f32 = 18.0;
    const SHORTCUT_GAP: f32 = 32.0;
    const ROW_HEIGHT: f32 = 26.0;
    const MIN_WIDTH: f32 = 150.0;
    let font = egui::TextStyle::Button.resolve(ui.style());
    let weak = ui.visuals().weak_text_color();
    let text_color = if enabled { MENU_TEXT } else { weak };
    let galley = ui.painter().layout_no_wrap(label.to_string(), font.clone(), text_color);
    let shortcut = shortcut.map(|text| ui.painter().layout_no_wrap(text.to_string(), font, weak));
    let shortcut_width = shortcut.as_ref().map_or(0.0, |g| g.size().x + SHORTCUT_GAP);
    let width = (GUTTER + galley.size().x + shortcut_width + RIGHT_MARGIN).max(MIN_WIDTH).max(ui.available_width());
    let sense = if enabled { egui::Sense::click() } else { egui::Sense::hover() };
    let (rect, response) = ui.allocate_exact_size(egui::vec2(width, ROW_HEIGHT), sense);
    response.widget_info(|| match checked {
        Some(checked) => egui::WidgetInfo::selected(egui::Role::CheckBox, enabled, checked, label),
        None => egui::WidgetInfo::labeled(egui::Role::Button, enabled, label),
    });
    if enabled && response.hovered() {
        ui.painter().rect_filled(rect, 4.0, ui.visuals().widgets.hovered.weak_bg_fill);
    }
    if checked == Some(true) {
        let center = egui::pos2(rect.left() + GUTTER / 2.0, rect.center().y);
        let mark = [center + egui::vec2(-4.0, 0.0), center + egui::vec2(-1.0, 3.5), center + egui::vec2(4.5, -4.0)];
        ui.painter().line(mark.to_vec(), egui::Stroke::new(1.6, text_color));
    }
    let position = egui::pos2(rect.left() + GUTTER, rect.center().y - galley.size().y / 2.0);
    ui.painter().galley(position, galley, text_color);
    if let Some(shortcut) = shortcut {
        let position = egui::pos2(rect.right() - RIGHT_MARGIN - shortcut.size().x, rect.center().y - shortcut.size().y / 2.0);
        ui.painter().galley(position, shortcut, weak);
    }
    response
}

fn menu_check(ui: &mut egui::Ui, checked: bool, label: &str) -> egui::Response {
    menu_row(ui, true, Some(checked), label, None)
}

/// Vrai si la commande est choisie ; le menu se referme alors.
fn menu_action(ui: &mut egui::Ui, enabled: bool, label: &str, shortcut: Option<&str>) -> bool {
    let clicked = menu_row(ui, enabled, None, label, shortcut).clicked();
    if clicked {
        ui.close();
    }
    clicked
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if let Some(frame) = self.player.as_ref().and_then(|player| player.take_frame()) {
            let image = egui::ColorImage::new(frame.size, frame.pixels);
            match &mut self.video_texture {
                Some(texture) => texture.set(image, egui::TextureOptions::LINEAR),
                None => self.video_texture = Some(ui.ctx().load_texture("video", image, egui::TextureOptions::LINEAR)),
            }
        }
        self.draw(ui);
        // Après le dessin : la taille de la zone vidéo est désormais connue
        if let Some(player) = &self.player {
            player.set_video_size(self.video_size);
        }
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.player = None;
    }
}


fn main() -> eframe::Result {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if let Some(code) = packaging::run(&arguments) {
        std::process::exit(code);
    }
    let backend = renderer::RenderBackend::load();
    let options = eframe::NativeOptions {
        renderer: backend.apply(),
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1500.0, 950.0])
            .with_maximized(true)
            .with_min_inner_size([LEFT_PANEL_MIN + MIN_VIDEO_WIDTH + RIGHT_PANEL_MIN + 16.0, 650.0])
            .with_icon(egui::IconData {
                rgba: include_bytes!(concat!(env!("OUT_DIR"), "/window-icon.rgba")).to_vec(),
                width: 128,
                height: 128,
            })
            .with_drag_and_drop(true),
        ..Default::default()
    };
    let result = eframe::run_native("MX-LAB", options, Box::new(|cc| Ok(Box::new(App::from_creation(cc)))));
    // Un moteur qui ne démarre pas ne doit pas empêcher le prochain lancement
    if result.is_err() {
        renderer::RenderBackend::default().save();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui_kittest::kittest::{NodeT, Queryable};
    use egui_kittest::Harness;

    /// Application sans lecteur ni vidéo réelle, avec un projet en mémoire.
    fn app() -> App {
        let folder = std::env::temp_dir().join(format!("music-studio-ui-{}", std::process::id()));
        std::fs::create_dir_all(&folder).unwrap();
        let path = folder.join(format!("film-{:?}.mkv", std::thread::current().id()));
        let mut project = Project::new(&path);
        project.duration = 600.0;
        project.apply_detection(&[(60.0, 120.0), (200.0, 260.0), (400.0, 500.0)]);
        let mut app = App::new(None, None);
        app.view = View { start: 0.0, span: 600.0 };
        app.media = Some(Media {
            release: release_name::describe(Path::new("Dune.2021.2160p.UHD.BluRay.x265.TrueHD.7.1-GRP.mkv")),
            project_path: project::project_path(&path),
            path,
            project,
            layout: Some("stereo".into()),
            smoothed: Vec::new(),
            signals: None,
            strip: None,
            strip_texture: None,
            detect_progress: None,
            detect_cancel: Arc::new(AtomicBool::new(false)),
            detect_waiting: false,
            signals_progress: None,
            strip_progress: None,
            undo: Vec::new(),
            redo: Vec::new(),
            dirty: false,
            changed_at: Instant::now(),
            overviews: HashMap::new(),
            spectra: HashMap::new(),
            details: HashMap::new(),
            detail_pending: HashSet::new(),
            chapters: Vec::new(),
            info: Default::default(),
            class_track: None,
            tags_progress: None,
            sequence_tags: HashMap::new(),
            film_loudness: HashMap::new(),
            sequence_loudness: HashMap::new(),
        });
        app
    }

    fn harness(app: App) -> Harness<'static, App> {
        Harness::builder()
            .with_size(egui::vec2(1500.0, 950.0))
            .build_ui_state(|ui, app: &mut App| app.draw(ui), app)
    }

    /// Image de l'interface, pour la comparer à une maquette :
    /// `MX_LAB_SNAPSHOT=fichier.png cargo test --profile studio-dev snapshot_of_the_window -- --ignored`
    #[test]
    #[ignore]
    fn snapshot_of_the_window() {
        let mut app = app_with_surround();
        app.media.as_mut().unwrap().project.scores = vec![0.0; 100];
        // Une courbe qui balaie de 0 à 1, pour voir l'échelle de la piste des séquences
        let steps = (600.0 / detect::STEP) as usize;
        app.media.as_mut().unwrap().smoothed = (0..steps).map(|i| 10f64.powf(-4.0 + 4.0 * i as f64 / steps as f64)).collect();
        let mut harness = Harness::builder()
            .with_size(egui::vec2(1500.0, 950.0))
            .with_pixels_per_point(2.0)
            .build_ui_state(
                |ui, app: &mut App| {
                    style(ui.ctx());
                    app.draw(ui)
                },
                app,
            );
        for _ in 0..4 {
            harness.step();
        }
        let path = std::env::var("MX_LAB_SNAPSHOT").unwrap_or_else(|_| "mx-lab-snapshot.png".into());
        harness.render().expect("rendu wgpu").save(&path).expect("écriture de l'image");
    }

    #[test]
    fn recent_menu_lists_clears_and_drops_missing_videos() {
        let mut harness = harness(app());
        harness.step();
        harness.get_by_label("Fichier").click();
        harness.step();
        harness.step();
        harness.get_by_label("Ouvrir les projets récents").hover();
        harness.step();
        harness.step();
        harness.step();
        harness.get_by_label("Aucun projet récent");
        harness.state_mut().recent.add(Path::new("Z://absent//Disparu.mp4"));
        harness.step();
        harness.step();
        assert_eq!(harness.query_all_by_label("Aucun projet récent").count(), 0);
        harness.get_by_label("Disparu").click();
        harness.step();
        harness.step();
        harness.step();
        assert!(harness.state().recent.paths().is_empty(), "une vidéo disparue quitte la liste");
        assert!(harness.state().status.contains("introuvable"));
        assert_eq!(harness.query_all_by_label("Ouvrir une vidéo…").count(), 0, "le menu se referme");
    }

    #[test]
    fn menu_text_keeps_one_color_whatever_the_state() {
        let mut style = egui::Style::default();
        classic_menu_style(&mut style);
        let widgets = &style.visuals.widgets;
        for state in [&widgets.inactive, &widgets.hovered, &widgets.active, &widgets.open] {
            assert_eq!(state.fg_stroke.color, MENU_TEXT);
        }
    }

    #[test]
    fn selection_survives_a_detection_in_progress() {
        let mut project = Project::default();
        project.apply_detection(&[(10.0, 20.0), (40.0, 50.0)]);
        let mut selected = Some(project.segments[1].id);
        let chosen = selected;
        // La séquence s'allonge : même identité
        apply_detection_keeping_selection(&mut project, &mut selected, &[(10.0, 20.0), (40.0, 60.0)]);
        assert_eq!(selected, chosen);
        // Elle fusionne avec sa voisine : la sélection suit
        project.segment_mut(project.segments[0].id).unwrap().end = 20.0;
        apply_detection_keeping_selection(&mut project, &mut selected, &[(5.0, 70.0)]);
        assert_eq!(selected, Some(project.segments[0].id));
        // Elle disparaît vraiment : plus rien de choisi
        apply_detection_keeping_selection(&mut project, &mut selected, &[]);
        assert_eq!(selected, None);
    }

    #[test]
    fn inter_is_the_interface_font_and_loads() {
        assert_eq!(interface_fonts().families[&egui::FontFamily::Proportional][0], "Inter");
        let mut harness = Harness::builder().build_ui(|ui| {
            style(ui.ctx());
            ui.label("Énergie, intégrée : -24,9 LUFS");
            ui.heading("Séquences");
        });
        harness.step();
        harness.step();
    }

    #[test]
    fn classic_menu_line_toggles_and_exposes_its_state() {
        let mut harness = Harness::builder().build_ui_state(
            |ui, on: &mut bool| {
                if menu_check(ui, *on, "Audio").clicked() {
                    *on = !*on;
                }
            },
            true,
        );
        harness.step();
        harness.get_by_label("Audio").click();
        harness.step();
        harness.step();
        assert!(!*harness.state());
    }

    #[test]
    fn display_menu_holds_tracks_and_rendering() {
        let mut harness = harness(app());
        harness.step();
        harness.get_by_label("Affichage").click();
        harness.step();
        harness.step();
        harness.get_by_label("Pistes");
        harness.get_by_label("Rendu");
    }

    fn press(harness: &mut Harness<'static, App>, key: egui::Key, modifiers: egui::Modifiers) {
        harness.key_press_modifiers(modifiers, key);
        harness.step();
        harness.step();
    }

    fn starts(app: &App) -> Vec<f64> {
        app.media.as_ref().unwrap().project.segments.iter().map(|s| s.start).collect()
    }

    #[test]
    fn table_shows_the_name_next_to_the_note() {
        let mut app = app();
        {
            let segment = &mut app.media.as_mut().unwrap().project.segments[1];
            segment.name = "Générique\nde début".into();
            segment.notes = "Thème de cordes".into();
        }
        let mut harness = harness(app);
        harness.step();
        harness.step();
        harness.get_by_label_contains("Instruments Nom Note");
        // Le nom d'abord, la note ensuite, sur une seule ligne
        harness.get_by_label_contains("Générique de début Thème de cordes").click();
        harness.step();
        harness.step();
        assert_eq!(harness.state().position(), 200.0);
    }

    #[test]
    fn name_column_leaves_most_of_the_room_to_the_note() {
        let (name, note) = text_column_widths(200.0);
        assert!(name > 50.0 && note > name, "{name} {note}");
        // Sur un grand écran, le nom plafonne et la note profite de tout le reste
        let (name, note) = text_column_widths(1000.0);
        assert_eq!(name, TABLE_NAME_MAX);
        assert_eq!(note, 1000.0 - TABLE_NAME_MAX - TABLE_NOTE_GAP);
        // Panneau très étroit : pas de largeur négative
        let (name, note) = text_column_widths(-20.0);
        assert_eq!((name, note), (0.0, 0.0));
    }

    #[test]
    fn undo_and_redo_live_in_the_edit_menu() {
        let mut harness = harness(app());
        harness.step();
        assert_eq!(harness.query_all_by_label("Annuler").count(), 0);
        harness.get_by_label("Édition").click();
        harness.step();
        harness.step();
        harness.get_by_label_contains("Annuler");
        harness.get_by_label_contains("Rétablir");
    }

    /// Le cas réel : l'application démarre vide, la vidéo arrive ensuite.
    #[test]
    fn right_panel_keeps_its_width_when_the_app_starts_empty() {
        let mut filled = app();
        let media = filled.media.take();
        let mut harness = Harness::builder()
            .with_size(egui::vec2(2575.0, 1400.0))
            .build_ui_state(|ui, app: &mut App| app.draw(ui), filled);
        for _ in 0..3 {
            harness.step();
        }
        harness.state_mut().media = media;
        for _ in 0..3 {
            harness.step();
        }
        let width = harness.ctx.content_rect().right() - harness.get_by_label_contains("Séquences (3)").rect().left();
        assert!((width - 2575.0 * RIGHT_PANEL_SHARE).abs() < 30.0, "{width}");
    }

    #[test]
    fn sequence_sheet_has_a_field_for_each_table_value() {
        let mut app = app();
        let first = app.media.as_ref().unwrap().project.segments[0].id;
        app.selected = Some(first);
        let mut harness = wide_harness(app);
        harness.step();
        harness.step();
        assert_eq!(harness.query_all_by_label("Durée").count(), 0, "calculée, donc pas modifiable");
        for name in ["Début", "Fin", "Mood", "Ensemble", "Instruments"] {
            assert!(harness.query_all_by_label(name).next().is_some(), "{name}");
        }
        // Le nom désigne son champ
        let label = harness.get_by_label("Mood").rect();
        let pos = label.center();
        harness.input_mut().events.push(egui::Event::PointerMoved(pos));
        harness.step();
        for pressed in [true, false] {
            harness.input_mut().events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::NONE,
            });
            harness.step();
        }
        harness.step();
        harness.input_mut().events.push(egui::Event::Text("Héroïque".to_string()));
        harness.step();
        harness.input_mut().events.push(egui::Event::Key {
            key: egui::Key::Enter,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        });
        harness.step();
        harness.step();
        assert_eq!(first_segment(&harness).overrides["mood"], "Héroïque");
        assert!(harness.state().field_edit.is_none());
    }

    #[test]
    fn help_opens_as_a_dialog_on_click() {
        let mut harness = harness(app());
        harness.step();
        assert_eq!(harness.query_all_by_label("Raccourcis clavier").count(), 0);
        harness.get_by_label("Aide").click();
        harness.step();
        harness.step();
        harness.step();
        harness.step();
        harness.get_by_label("Raccourcis clavier");
        harness.get_by_label("Fermer").click();
        harness.step();
        harness.step();
        assert!(!harness.state().show_help);
    }

    #[test]
    fn sequence_sheet_is_laid_out_on_three_lines() {
        let mut app = app();
        app.selected = app.media.as_ref().unwrap().project.segments.first().map(|s| s.id);
        let harness = wide_harness(app);
        let center = |name: &str| harness.get_all_by_label(name).last().unwrap().rect().center().y;
        let first = center("Début");
        assert!((center("Fin") - first).abs() < 4.0 && (center("Nom") - first).abs() < 4.0);
        let second = center("Mood");
        for name in ["Type", "Ensemble", "Instruments"] {
            assert!((center(name) - second).abs() < 4.0, "{name}");
        }
        assert!(second > first && center("Notes") > second);
    }

    #[test]
    fn bounds_move_by_dragging_and_become_text_on_double_click() {
        let mut app = app();
        let id = app.media.as_ref().unwrap().project.segments[0].id;
        app.selected = Some(id);
        let mut harness = wide_harness(app);
        harness.step();
        harness.step();
        let before = first_segment(&harness).start;
        let start = harness.get_all_by_label("Début").map(|node| node.rect()).max_by(|a, b| a.left().total_cmp(&b.left())).unwrap().center();
        let press = |harness: &mut Harness<'static, App>, pos: egui::Pos2, pressed: bool| {
            harness.input_mut().events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::NONE,
            });
        };
        harness.input_mut().events.push(egui::Event::PointerMoved(start));
        harness.step();
        press(&mut harness, start, true);
        harness.step();
        for step in 1..=5 {
            harness.input_mut().events.push(egui::Event::PointerMoved(start + egui::vec2(step as f32 * 10.0, 0.0)));
            harness.step();
        }
        press(&mut harness, start + egui::vec2(50.0, 0.0), false);
        harness.step();
        assert!(first_segment(&harness).start > before, "vers la droite, la valeur grandit");
        assert!(harness.state().bound_drag.is_none());

        // Double-clic : le champ devient modifiable au clavier
        harness.input_mut().events.push(egui::Event::PointerMoved(start));
        harness.step();
        for _ in 0..2 {
            press(&mut harness, start, true);
            press(&mut harness, start, false);
            harness.step();
        }
        harness.step();
        harness.step();
        assert!(harness.state().field_edit.as_ref().is_some_and(|e| e.column == Column::Start));
        harness.input_mut().events.push(egui::Event::Text("70".to_string()));
        harness.step();
        harness.input_mut().events.push(egui::Event::Key {
            key: egui::Key::Enter,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        });
        harness.step();
        harness.step();
        assert_eq!(first_segment(&harness).start, 70.0);
    }

    #[test]
    fn sheet_takes_only_the_height_it_needs_by_default() {
        let mut app = app();
        app.selected = app.media.as_ref().unwrap().project.segments.first().map(|s| s.id);
        let mut harness = wide_harness(app);
        for _ in 0..4 {
            harness.step();
        }
        let reserved = harness.state().sheet_height_now;
        let splitter = harness.get_by_label("Partage tableau et fiche").rect().top();
        let notes = harness.query_all_by_label("Notes").map(|node| node.rect().bottom()).fold(0.0, f32::max);
        let below_splitter = notes - splitter;
        assert!(reserved >= below_splitter, "la fiche tient : {reserved} {below_splitter}");
        assert!(reserved - below_splitter < 50.0, "pas de grand vide : {reserved} {below_splitter}");
    }

    #[test]
    fn splitter_shares_the_panel_between_table_and_sheet() {
        let mut app = app();
        app.selected = app.media.as_ref().unwrap().project.segments.first().map(|s| s.id);
        let mut harness = wide_harness(app);
        harness.step();
        harness.step();
        let start = harness.get_by_label("Partage tableau et fiche").rect().center();
        let before = harness.state().sheet_height_now;
        harness.input_mut().events.push(egui::Event::PointerMoved(start));
        harness.step();
        harness.input_mut().events.push(egui::Event::PointerButton {
            pos: start,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        });
        harness.step();
        for step in 1..=5 {
            harness.input_mut().events.push(egui::Event::PointerMoved(start - egui::vec2(0.0, step as f32 * 12.0)));
            harness.step();
        }
        harness.input_mut().events.push(egui::Event::PointerButton {
            pos: start - egui::vec2(0.0, 60.0),
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        });
        harness.step();
        let after = harness.state().sheet_height.unwrap();
        assert!(after > before + 30.0, "vers le haut, la fiche grandit : {before} {after}");
    }

    #[test]
    fn note_fits_on_one_line_in_the_table() {
        assert_eq!(note_preview("  Thème de cordes\n\nreprise\t au piano "), "Thème de cordes reprise au piano");
        assert_eq!(note_preview("   \n "), "");
    }

    #[test]
    fn table_duration_keeps_its_column_aligned() {
        assert_eq!(table_duration(72.0), "1:12");
        assert_eq!(table_duration(3725.0), "62:05");
        let short = table_columns("1", " ", "00:00:00:00", "00:01:12:00", &table_duration(72.0));
        let long = table_columns("100", "*", "01:00:00:00", "02:02:05:00", &table_duration(3725.0));
        assert_eq!(short.chars().count(), long.chars().count());
    }

    #[test]
    fn table_shows_the_note_next_to_the_duration() {
        let mut app = app();
        app.media.as_mut().unwrap().project.segments[1].notes = "Thème de cordes\nreprise au piano".into();
        let mut harness = harness(app);
        harness.step();
        harness.step();
        harness.get_by_label_contains("Instruments Nom Note");
        let row = harness.get_by_label_contains("Thème de cordes reprise au piano");
        // La ligne porte les colonnes et la note, et un clic mène à la séquence
        row.click();
        harness.step();
        harness.step();
        assert_eq!(harness.state().position(), 200.0);
        let state = harness.state();
        assert_eq!(state.selected, state.media.as_ref().unwrap().project.segments.get(1).map(|s| s.id));
    }

    #[test]
    fn list_highlights_the_sequence_under_the_playhead() {
        let mut harness = harness(app());
        harness.step();
        harness.step();
        assert_eq!(harness.query_all_by_label_contains("(en lecture)").count(), 0, "tête hors de toute séquence");

        harness.state_mut().seek_to(230.0, true);
        harness.step();
        harness.step();
        let row = harness.get_by_label_contains("(en lecture)");
        assert!(row.accesskit_node().label().unwrap_or_default().contains("00:03:20:00"));
        assert_eq!(harness.state().selected, None, "la lecture ne change pas la sélection");

        // La séquence suivante prend le relais
        harness.state_mut().seek_to(450.0, true);
        harness.step();
        harness.step();
        let row = harness.get_by_label_contains("(en lecture)");
        assert!(row.accesskit_node().label().unwrap_or_default().contains("00:06:40:00"));

        // Entre deux séquences, plus rien
        harness.state_mut().seek_to(550.0, true);
        harness.step();
        harness.step();
        assert_eq!(harness.query_all_by_label_contains("(en lecture)").count(), 0);
    }

    #[test]
    fn playing_and_selected_are_independent() {
        let mut harness = harness(app());
        harness.step();
        press(&mut harness, egui::Key::ArrowDown, egui::Modifiers::NONE);
        let first = harness.state().selected;
        assert!(first.is_some());
        // La tête part dans une autre séquence : la sélection reste, le surlignage suit
        harness.state_mut().seek_to(230.0, true);
        harness.step();
        harness.step();
        assert_eq!(harness.state().selected, first);
        let row = harness.get_by_label_contains("(en lecture)");
        assert!(row.accesskit_node().label().unwrap_or_default().contains("00:03:20:00"));
    }

    #[test]
    fn film_sheet_comes_from_the_file_name_only() {
        let harness = harness(app());
        for text in ["Dune", "2021", "Blu-ray UHD"] {
            harness.get_by_label(text);
        }
        // Le reste vient des métadonnées, pas du nom
        for text in ["2160p", "x265", "Dolby TrueHD 7.1", "GRP"] {
            assert_eq!(harness.query_all_by_label(text).count(), 0, "{text}");
        }
        assert_eq!(harness.query_all_by_label_contains("TMDB").count(), 0, "plus aucune trace de TMDB");
    }

    #[test]
    fn loop_is_an_icon_that_toggles() {
        let mut harness = harness(app());
        harness.step();
        harness.step();
        assert_eq!(harness.query_all_by_label("Boucle").count(), 0, "plus de texte");
        assert!(!harness.state().loop_on);
        harness.get_by_label("Boucler sur la séquence choisie (B)").click();
        harness.step();
        harness.step();
        assert!(harness.state().loop_on);
        harness.get_by_label("Boucler sur la séquence choisie (B)").click();
        harness.step();
        harness.step();
        assert!(!harness.state().loop_on);
    }

    #[test]
    fn without_auto_zoom_choosing_a_sequence_keeps_the_view() {
        let mut harness = harness(app());
        harness.step();
        harness.step();
        harness.get_by_label_contains("00:03:20:00").click();
        harness.step();
        harness.step();
        assert_eq!(harness.state().view.span, 600.0);
    }

    /// La séquence remplit la vue, à une petite marge près.
    fn fills_the_view(app: &App, start: f64, end: f64) {
        let view = app.view;
        assert!(view.start <= start && view.end() >= end, "{view:?}");
        assert!(view.span < (end - start) * 1.2, "{view:?}");
    }

    #[test]
    fn auto_zoom_fits_the_sequence_chosen_in_the_list() {
        let mut harness = harness(app());
        harness.step();
        harness.step();
        harness.get_by_label("Auto").click();
        harness.step();
        harness.step();
        assert!(harness.state().auto_zoom);
        assert_eq!(harness.state().view.span, 600.0, "rien de choisi : rien à cadrer");
        harness.get_by_label_contains("00:03:20:00").click();
        harness.step();
        harness.step();
        fills_the_view(harness.state(), 200.0, 260.0);
        assert_eq!(harness.state().position(), 200.0);
        // Et de séquence en séquence, au clavier
        press(&mut harness, egui::Key::ArrowDown, egui::Modifiers::NONE);
        fills_the_view(harness.state(), 400.0, 500.0);
    }

    #[test]
    fn auto_zoom_fits_the_sequence_clicked_on_the_timeline() {
        let mut app = app();
        app.auto_zoom = true;
        let mut harness = harness(app);
        harness.step();
        click_on_row(&mut harness, timeline::Row::Sequences, 230.0);
        fills_the_view(harness.state(), 200.0, 260.0);
        // Cliquer dans le vide ne bouge pas la vue
        let before = harness.state().view;
        click_on_row(&mut harness, timeline::Row::Sequences, before.end() - 0.2);
        assert_eq!(harness.state().view, before);
    }

    #[test]
    fn turning_auto_zoom_on_fits_the_current_sequence_at_once() {
        let mut harness = harness(app());
        harness.step();
        press(&mut harness, egui::Key::ArrowDown, egui::Modifiers::NONE);
        assert_eq!(harness.state().view.span, 600.0);
        harness.get_by_label("Auto").click();
        harness.step();
        harness.step();
        fills_the_view(harness.state(), 60.0, 120.0);
    }

    #[test]
    fn dragging_a_handle_does_not_move_the_view_in_auto_zoom() {
        let mut app = app();
        app.auto_zoom = true;
        let mut harness = harness(app);
        harness.step();
        drag_on_row(&mut harness, timeline::Row::Sequences, 60.0, 80.0, false);
        assert_eq!(harness.state().view.span, 600.0, "corriger une poignée ne doit pas recadrer sous la souris");
    }

    #[test]
    fn timecode_counts_frames_of_real_time() {
        assert_eq!(tc(3661.5, 24.0), "01:01:01:12");
        assert_eq!(tc(0.0, 25.0), "00:00:00:00");
        assert_eq!(tc(59.999, 25.0), "00:00:59:24");
    }

    #[test]
    fn clicking_a_timecode_goes_to_the_sequence() {
        let mut harness = harness(app());
        harness.step();
        harness.get_by_label_contains("00:03:20:00").click();
        harness.step();
        harness.step();
        let app = harness.state();
        assert_eq!(app.position(), 200.0);
        assert_eq!(app.selected, app.media.as_ref().unwrap().project.segments.get(1).map(|s| s.id));
    }

    #[test]
    fn arrows_walk_through_the_sequences() {
        let mut harness = harness(app());
        harness.step();
        press(&mut harness, egui::Key::ArrowDown, egui::Modifiers::NONE);
        assert_eq!(harness.state().position(), 60.0);
        press(&mut harness, egui::Key::ArrowDown, egui::Modifiers::NONE);
        assert_eq!(harness.state().position(), 200.0);
        press(&mut harness, egui::Key::ArrowUp, egui::Modifiers::NONE);
        assert_eq!(harness.state().position(), 60.0);
    }

    #[test]
    fn i_and_o_move_the_handles_to_the_playhead() {
        let mut harness = harness(app());
        harness.step();
        press(&mut harness, egui::Key::ArrowDown, egui::Modifiers::NONE);
        harness.state_mut().seek_to(70.0, true);
        harness.step();
        press(&mut harness, egui::Key::I, egui::Modifiers::NONE);
        assert_eq!(starts(harness.state())[0], 70.0);
        harness.state_mut().seek_to(100.0, true);
        harness.step();
        press(&mut harness, egui::Key::O, egui::Modifiers::NONE);
        let app = harness.state();
        let first = &app.media.as_ref().unwrap().project.segments[0];
        assert_eq!((first.start, first.end), (70.0, 100.0));
        assert_eq!(first.origin, Origin::Edited);
    }

    #[test]
    fn new_sequence_is_added_at_the_playhead_then_undone() {
        let mut harness = harness(app());
        harness.step();
        harness.state_mut().seek_to(300.0, true);
        harness.step();
        press(&mut harness, egui::Key::N, egui::Modifiers::NONE);
        assert_eq!(starts(harness.state()), [60.0, 200.0, 300.0, 400.0]);
        let added = harness.state().media.as_ref().unwrap().project.segments[2].clone();
        assert_eq!((added.end, added.origin), (330.0, Origin::Manual));
        press(&mut harness, egui::Key::Z, egui::Modifiers::COMMAND);
        assert_eq!(starts(harness.state()), [60.0, 200.0, 400.0]);
        press(&mut harness, egui::Key::Y, egui::Modifiers::COMMAND);
        assert_eq!(starts(harness.state()), [60.0, 200.0, 300.0, 400.0]);
    }

    #[test]
    fn delete_removes_the_selected_sequence() {
        let mut harness = harness(app());
        harness.step();
        press(&mut harness, egui::Key::ArrowDown, egui::Modifiers::NONE);
        press(&mut harness, egui::Key::Delete, egui::Modifiers::NONE);
        assert_eq!(starts(harness.state()), [200.0, 400.0]);
        assert_eq!(harness.state().selected, None);
    }

    /// Glisse à la souris d'un temps à un autre sur la piste voulue.
    fn drag_on_row(harness: &mut Harness<'static, App>, row: timeline::Row, from: f64, to: f64, shift: bool) {
        let body = harness.state().timeline_body;
        let rows = timeline::layout_rows(body.top(), body.height(), harness.state().tracks);
        let (_, y0, y1) = *rows.iter().find(|r| r.0 == row).unwrap();
        let y = (y0 + y1) / 2.0;
        let (left, width) = (body.left(), body.width());
        let view = harness.state().view;
        let start = egui::pos2(view.x(left, width, from), y);
        let end = egui::pos2(view.x(left, width, to), y);
        let modifiers = if shift { egui::Modifiers::SHIFT } else { egui::Modifiers::NONE };
        let input = harness.input_mut();
        input.events.push(egui::Event::ModifiersChanged(modifiers));
        input.events.push(egui::Event::PointerMoved(start));
        input.events.push(egui::Event::PointerButton {
            pos: start,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers,
        });
        harness.step();
        for step in 1..=8 {
            let pos = start + (end - start) * (step as f32 / 8.0);
            harness.input_mut().events.push(egui::Event::PointerMoved(pos));
            harness.step();
        }
        harness.input_mut().events.push(egui::Event::PointerButton {
            pos: end,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers,
        });
        harness.step();
        harness.step();
    }

    /// Un vrai clic, bref, à un instant de la vidéo sur la piste voulue.
    fn click_on_row(harness: &mut Harness<'static, App>, row: timeline::Row, time: f64) {
        let body = harness.state().timeline_body;
        let rows = timeline::layout_rows(body.top(), body.height(), harness.state().tracks);
        let (_, y0, y1) = *rows.iter().find(|r| r.0 == row).unwrap();
        let pos = egui::pos2(harness.state().view.x(body.left(), body.width(), time), (y0 + y1) / 2.0);
        harness.input_mut().events.push(egui::Event::PointerMoved(pos));
        harness.step();
        for pressed in [true, false] {
            harness.input_mut().events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::NONE,
            });
            harness.step();
        }
        harness.step();
    }

    #[test]
    fn dragging_a_handle_corrects_the_sequence() {
        let mut harness = harness(app());
        harness.step();
        drag_on_row(&mut harness, timeline::Row::Sequences, 60.0, 80.0, false);
        let app = harness.state();
        let first = &app.media.as_ref().unwrap().project.segments[0];
        assert!((first.start - 80.0).abs() < 1.0, "{}", first.start);
        assert_eq!(first.end, 120.0);
        assert_eq!(first.origin, Origin::Edited);
        assert_eq!(app.selected, Some(first.id));
    }

    #[test]
    fn dragging_on_empty_space_creates_a_sequence() {
        let mut harness = harness(app());
        harness.step();
        drag_on_row(&mut harness, timeline::Row::Sequences, 300.0, 340.0, false);
        let app = harness.state();
        let added = &app.media.as_ref().unwrap().project.segments[2];
        assert_eq!(added.origin, Origin::Manual);
        assert!((added.start - 300.0).abs() < 1.0, "{}", added.start);
        assert!((added.end - 340.0).abs() < 1.0, "{}", added.end);
    }

    #[test]
    fn dragging_a_sequence_body_moves_it() {
        let mut harness = harness(app());
        harness.step();
        drag_on_row(&mut harness, timeline::Row::Sequences, 90.0, 110.0, false);
        let app = harness.state();
        let first = &app.media.as_ref().unwrap().project.segments[0];
        assert!((first.start - 80.0).abs() < 1.0, "{}", first.start);
        assert!((first.end - first.start - 60.0).abs() < 1e-9);
    }

    #[test]
    fn shift_snaps_a_handle_to_a_keyframe() {
        let mut app = app();
        app.media.as_mut().unwrap().strip = Some(Arc::new(Strip {
            width: 0,
            height: 0,
            interval: 10.0,
            count: 0,
            pixels: Vec::new(),
            keyframes: vec![0.0, 75.0, 90.0],
        }));
        let mut harness = harness(app);
        harness.step();
        drag_on_row(&mut harness, timeline::Row::Sequences, 60.0, 77.0, true);
        let app = harness.state();
        assert_eq!(app.media.as_ref().unwrap().project.segments[0].start, 75.0);
    }

    #[test]
    fn holding_the_mouse_still_keeps_the_same_image() {
        let mut harness = harness(app());
        harness.step();
        let body = harness.state().timeline_body;
        let rows = timeline::layout_rows(body.top(), body.height(), harness.state().tracks);
        let (_, y0, y1) = *rows.iter().find(|r| r.0 == timeline::Row::Audio).unwrap();
        let pos = egui::pos2(body.center().x, (y0 + y1) / 2.0);
        harness.input_mut().events.push(egui::Event::PointerMoved(pos));
        harness.input_mut().events.push(egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        });
        // Assez longtemps pour qu'egui considère l'appui comme un glissé
        for _ in 0..80 {
            harness.step();
        }
        assert_eq!(harness.state().drag, Drag::Scrub);
        assert_eq!(harness.state().seek_requests, 1, "un seul déplacement tant que la souris ne bouge pas");
        harness.input_mut().events.push(egui::Event::PointerMoved(pos + egui::vec2(40.0, 0.0)));
        harness.step();
        assert_eq!(harness.state().seek_requests, 2, "la souris bouge : l'image suit");
    }

    fn app_with_surround() -> App {
        let mut app = app();
        let mut analysis = signals::Analysis::new(vec!["L", "R", "C", "LFE", "Ls", "Rs"]);
        analysis.feed(&vec![0.1f32; 48000 * 6 * 3]);
        app.media.as_mut().unwrap().signals = Some(Arc::new(analysis.finish()));
        app
    }

    #[test]
    fn timeline_measures_the_film_and_the_table_each_sequence() {
        let mut app = app();
        // 10 s calmes, puis 10 s fortes : assez pour distinguer une séquence de la vidéo
        let quiet: Vec<f32> = (0..48000 * 10).map(|i| (i as f32 * 0.13).sin() * 0.01).collect();
        let loud: Vec<f32> = (0..48000 * 10).map(|i| (i as f32 * 0.13).sin() * 0.5).collect();
        let mut analysis = signals::Analysis::new(vec![]);
        analysis.feed(&quiet);
        analysis.feed(&loud);
        let media = app.media.as_mut().unwrap();
        media.signals = Some(Arc::new(analysis.finish()));
        media.project.segments.clear();
        let quiet_id = media.project.add_manual(1.0, 9.0).unwrap();
        let loud_id = media.project.add_manual(11.0, 19.0).unwrap();
        let mut harness = harness(app);
        harness.step();

        let media = harness.state().media.as_ref().unwrap();
        let film_before = media.film_loudness[&0].measure.integrated;
        let quiet_level = media.sequence_loudness[&quiet_id].1.integrated;
        let loud_level = media.sequence_loudness[&loud_id].1.integrated;
        assert!(loud_level - quiet_level > 30.0, "{quiet_level} {loud_level}");

        // Choisir une séquence ne change rien à la timeline
        harness.state_mut().selected = Some(quiet_id);
        harness.step();
        assert_eq!(harness.state().media.as_ref().unwrap().film_loudness[&0].measure.integrated, film_before);

        // Corriger une séquence la remesure
        let media = harness.state_mut().media.as_mut().unwrap();
        media.project.set_end(loud_id, 15.0);
        harness.step();
        let bounds = harness.state().media.as_ref().unwrap().sequence_loudness[&loud_id].0;
        assert_eq!(bounds, (11.0, 15.0));

        // Une séquence supprimée ne laisse pas de mesure derrière elle
        let media = harness.state_mut().media.as_mut().unwrap();
        media.project.segments.retain(|s| s.id != quiet_id);
        harness.step();
        assert!(!harness.state().media.as_ref().unwrap().sequence_loudness.contains_key(&quiet_id));
    }

    #[test]
    fn loudness_columns_stay_aligned_with_their_header() {
        let header = loudness_header("TP", "I", "S", "M", "LRA");
        let none = BTreeMap::new();
        assert_eq!(loudness_columns(None, &none).chars().count(), header.chars().count());
        let measure = signals::Measure { true_peak: -11.1, integrated: -24.9, range: 19.5, max_short_term: -20.1, max_momentary: -18.8 };
        assert_eq!(loudness_columns(Some(&measure), &none), "-11.1 -24.9 -20.1 -18.8  19.5");
        let written = BTreeMap::from([("i".to_string(), "-30".to_string())]);
        assert_eq!(loudness_columns(Some(&measure), &written), "-11.1       -20.1 -18.8  19.5", "la valeur saisie se dessine à part");
    }

    #[test]
    fn typed_times_are_understood() {
        assert_eq!(parse_time("90", 25.0), Some(90.0));
        assert_eq!(parse_time("1:30", 25.0), Some(90.0));
        assert_eq!(parse_time("01:00:30", 25.0), Some(3630.0));
        assert_eq!(parse_time("00:00:10:12", 25.0), Some(10.48));
        assert_eq!(parse_time(" 2,5 ", 25.0), Some(2.5));
        for wrong in ["", "abc", "1:2:3:4:5", "-3", "1::2"] {
            assert_eq!(parse_time(wrong, 25.0), None, "{wrong}");
        }
    }

    #[test]
    fn table_columns_match_the_fixed_text() {
        let line = table_columns("1", " ", "00:00:01:00", "00:00:02:00", "0:01");
        assert_eq!(line.chars().count(), FIXED_CHARS);
        let slice = |text: &str, first: usize, last: usize| text.chars().skip(first).take(last - first).collect::<String>();
        assert_eq!(slice(&line, 6, 17), "00:00:01:00");
        assert_eq!(slice(&line, 19, 30), "00:00:02:00");
        let loudness = loudness_header("TP", "I", "S", "M", "LRA");
        assert_eq!(loudness.chars().count(), LOUDNESS_CHARS);
        let (_, first, last) = LOUDNESS_CELLS[0];
        assert_eq!(slice(&loudness, first, last).trim(), "TP");
        let (_, first, last) = LOUDNESS_CELLS[4];
        assert_eq!(slice(&loudness, first, last).trim(), "LRA");
    }

    #[test]
    fn loudness_measures_are_the_last_columns() {
        let char_width = 7.0;
        let cell = |rect: egui::Rect, wanted: Column| table_cells(rect, char_width).into_iter().find(|c| c.0 == wanted).unwrap().1;
        let wide = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1400.0, 18.0));
        assert_eq!(cell(wide, Column::Range).right(), 1400.0 - TABLE_PADDING);
        assert!(cell(wide, Column::Note).right() <= cell(wide, Column::TruePeak).left());
        assert!(cell(wide, Column::Instruments).right() <= cell(wide, Column::Name).left());
        // Panneau étroit : le texte garde sa place, les mesures sortent par la droite
        let narrow = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(620.0, 18.0));
        assert!(cell(narrow, Column::Range).right() > 620.0);
        assert_eq!(cell(narrow, Column::Instruments).width(), TABLE_INSTRUMENTS_WIDTH - 6.0);
        assert!(cell(narrow, Column::Name).width() > 80.0 && cell(narrow, Column::Note).width() > 100.0);
        assert!(cell(narrow, Column::Note).right() <= cell(narrow, Column::TruePeak).left());
    }

    /// Double-clic dans une colonne de la ligne voulue, puis frappe au clavier.
    fn edit_cell(harness: &mut Harness<'static, App>, row: usize, column: Column, typed: &str, finish: egui::Key) {
        // Les lignes du tableau : ce qui porte un timecode et occupe la largeur du panneau
        let line = harness
            .query_all_by_label_contains(":")
            .map(|node| node.rect())
            .filter(|rect| rect.width() > 300.0 && (rect.height() - TABLE_ROW_HEIGHT).abs() < 0.5)
            .nth(row)
            .unwrap();
        let cell = table_cells(line, table_char_width(&harness.ctx)).into_iter().find(|c| c.0 == column).unwrap().1;
        let pos = cell.left_center() + egui::vec2(3.0, 0.0);
        harness.input_mut().events.push(egui::Event::PointerMoved(pos));
        harness.step();
        // Le simulateur avance par pas de 0,25 s : chaque clic tient dans une image, sinon les
        // deux clics seraient trop espacés pour faire un double-clic
        for _ in 0..2 {
            for pressed in [true, false] {
                harness.input_mut().events.push(egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                });
            }
            harness.step();
        }
        harness.step();
        harness.step();
        assert!(harness.state().table_edit.as_ref().is_some_and(|e| e.column == column), "{column:?} : édition non ouverte");
        harness.input_mut().events.push(egui::Event::Text(typed.to_string()));
        harness.step();
        harness.input_mut().events.push(egui::Event::Key {
            key: finish,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        });
        harness.step();
        harness.step();
        assert!(harness.state().table_edit.is_none());
    }

    fn first_segment(harness: &Harness<'static, App>) -> Segment {
        harness.state().media.as_ref().unwrap().project.segments[0].clone()
    }

    /// Aussi large que l'écran de l'utilisateur : toutes les colonnes tiennent dans le panneau.
    fn wide_harness(app: App) -> Harness<'static, App> {
        Harness::builder().with_size(egui::vec2(2575.0, 1400.0)).build_ui_state(|ui, app: &mut App| app.draw(ui), app)
    }

    #[test]
    fn double_click_edits_text_values() {
        let mut harness = wide_harness(app_with_surround());
        harness.step();
        edit_cell(&mut harness, 0, Column::Name, "Générique", egui::Key::Enter);
        assert_eq!(first_segment(&harness).name, "Générique");
        assert_eq!(first_segment(&harness).origin, Origin::Edited, "une correction survit à la détection");
        edit_cell(&mut harness, 0, Column::Note, "Thème", egui::Key::Enter);
        assert_eq!(first_segment(&harness).notes, "Thème");
        edit_cell(&mut harness, 0, Column::Mood, "Héroïque", egui::Key::Enter);
        assert_eq!(first_segment(&harness).overrides["mood"], "Héroïque");
        edit_cell(&mut harness, 0, Column::Instruments, "Cuivres", egui::Key::Enter);
        assert_eq!(first_segment(&harness).overrides["instruments"], "Cuivres");
        edit_cell(&mut harness, 0, Column::Integrated, "-30", egui::Key::Enter);
        assert_eq!(first_segment(&harness).overrides["i"], "-30");
    }

    #[test]
    fn double_click_edits_the_bounds_and_undo_restores_them() {
        let mut harness = harness(app());
        harness.step();
        edit_cell(&mut harness, 0, Column::Start, "50", egui::Key::Enter);
        assert_eq!(first_segment(&harness).start, 50.0);
        edit_cell(&mut harness, 0, Column::End, "2:10", egui::Key::Enter);
        assert_eq!(first_segment(&harness).end, 130.0);
        harness.state_mut().undo(false);
        assert_eq!(first_segment(&harness).end, 120.0);
    }

    #[test]
    fn a_wrong_time_changes_nothing_and_says_so() {
        let mut harness = harness(app());
        harness.step();
        let before = first_segment(&harness);
        edit_cell(&mut harness, 0, Column::Start, "abc", egui::Key::Enter);
        assert_eq!(first_segment(&harness), before);
        assert!(harness.state().status.contains("abc"), "{}", harness.state().status);
    }

    #[test]
    fn escape_cancels_and_typing_does_not_trigger_shortcuts() {
        let mut harness = wide_harness(app());
        harness.step();
        let before = first_segment(&harness);
        // « n » créerait une séquence, « z » zoomerait : ici, ce ne sont que des lettres
        let count = harness.state().media.as_ref().unwrap().project.segments.len();
        edit_cell(&mut harness, 0, Column::Name, "nz", egui::Key::Escape);
        assert_eq!(first_segment(&harness), before);
        assert_eq!(harness.state().media.as_ref().unwrap().project.segments.len(), count);
    }

    #[test]
    fn clearing_a_written_value_brings_back_the_computed_one() {
        let mut harness = harness(app());
        harness.step();
        edit_cell(&mut harness, 0, Column::Mood, "Héroïque", egui::Key::Enter);
        assert!(first_segment(&harness).overrides.contains_key("mood"));
        let id = first_segment(&harness).id;
        harness.state_mut().commit_table_edit(TableEdit {
            segment: id,
            column: Column::Mood,
            initial: "Héroïque".into(),
            text: String::new(),
            needs_focus: false,
        });
        assert!(first_segment(&harness).overrides.is_empty());
    }

    /// Un ancien projet n'a pas de cibles : il prend celles par défaut, et les garde ensuite.
    #[test]
    fn loudness_target_survives_the_project_file() {
        let old: Project = serde_json::from_str(r#"{"version":1,"media_file_name":"a.mkv"}"#).unwrap();
        assert_eq!(old.loudness_target, project::LoudnessTarget::default());
        let mut project = old;
        project.loudness_target.integrated = -27.0;
        let again: Project = serde_json::from_str(&serde_json::to_string(&project).unwrap()).unwrap();
        assert_eq!(again.loudness_target.integrated, -27.0);
    }

    /// Glisse sur la barre de zoom, positions données en temps de la vidéo.
    fn drag_zoom_bar(harness: &mut Harness<'static, App>, from: f64, to: f64) {
        let body = harness.state().timeline_body;
        let duration = harness.state().duration();
        let rows = timeline::layout_rows(body.top(), body.height(), harness.state().tracks);
        let (_, y0, y1) = *rows.iter().find(|r| r.0 == timeline::Row::Overview).unwrap();
        let y = (y0 + y1) / 2.0;
        let (left, width) = timeline::zoom_span(body.left(), body.width());
        let at = |t: f64| egui::pos2(left + (t / duration) as f32 * width, y);
        let (start, end) = (at(from), at(to));
        harness.input_mut().events.push(egui::Event::PointerMoved(start));
        harness.input_mut().events.push(egui::Event::PointerButton {
            pos: start,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        });
        harness.step();
        for step in 1..=8 {
            harness.input_mut().events.push(egui::Event::PointerMoved(start + (end - start) * (step as f32 / 8.0)));
            harness.step();
        }
        harness.input_mut().events.push(egui::Event::PointerButton {
            pos: end,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        });
        harness.step();
        harness.step();
    }

    #[test]
    fn zoom_bar_handles_change_the_view_and_the_bar_moves_it() {
        let mut app = app();
        app.view = View { start: 100.0, span: 200.0 };
        let mut harness = harness(app);
        harness.step();
        // Poignée gauche : le début bouge, la fin reste
        drag_zoom_bar(&mut harness, 100.0, 200.0);
        let view = harness.state().view;
        assert!((view.start - 200.0).abs() < 1.0, "{view:?}");
        assert!((view.end() - 300.0).abs() < 1e-6, "{view:?}");
        // Poignée droite : la fin bouge, le début reste
        drag_zoom_bar(&mut harness, 300.0, 500.0);
        let view = harness.state().view;
        assert!((view.start - 200.0).abs() < 1.0, "{view:?}");
        assert!((view.end() - 500.0).abs() < 1.0, "{view:?}");
        // La barre elle-même : tout se déplace, la largeur reste
        let span = view.span;
        drag_zoom_bar(&mut harness, 350.0, 250.0);
        let view = harness.state().view;
        assert!((view.span - span).abs() < 1e-6, "{view:?}");
        assert!((view.start - 100.0).abs() < 1.0, "{view:?}");
        // La barre ne touche jamais à la tête de lecture
        assert_eq!(harness.state().position(), 0.0);
    }

    #[test]
    fn normalize_button_sits_just_above_the_view_selector() {
        let mut app = app_with_surround();
        app.tracks.view = AudioView::Waveform;
        let mut harness = harness(app);
        harness.step();
        harness.step();
        let button = harness.get_by_label("Normaliser").rect();
        let last_track = harness.get_by_label("Right Surround").rect();
        let bottom = harness.state().timeline_body.bottom();
        let selector = harness.get_by_label("Onde").rect();
        assert!(button.top() > last_track.bottom() + 50.0, "{button:?} {last_track:?}");
        assert!(button.bottom() <= selector.top() && selector.top() - button.bottom() < 6.0, "{button:?} {selector:?}");
        assert!(bottom - selector.bottom() < 14.0, "{selector:?} {bottom}");
    }

    #[test]
    fn waveform_normalize_button_is_named() {
        let mut app = app_with_surround();
        app.tracks.view = AudioView::Waveform;
        let mut harness = harness(app);
        harness.step();
        harness.step();
        assert!(harness.state().waveform_normalized);
        harness.get_by_label("Normaliser").click();
        harness.step();
        harness.step();
        assert!(!harness.state().waveform_normalized);
    }

    #[test]
    fn each_track_says_what_it_is_waiting_for() {
        use timeline::{Loading, Row};
        let mut app = app();
        assert_eq!(app.pending_work(), [], "rien en cours");
        {
            let media = app.media.as_mut().unwrap();
            media.strip_progress = Some(40);
            media.detect_progress = Some(0);
            media.signals_progress = Some(75);
        }
        let work = app.pending_work();
        assert_eq!(work.len(), 3);
        assert_eq!(work[0], Loading { row: Row::Strip, label: "Vignettes et keyframes", progress: Some(40) });
        // À 0 %, l'avancement n'est pas connu : barre animée
        assert_eq!(work[1], Loading { row: Row::Sequences, label: "Détection de la musique", progress: None });
        assert_eq!(work[2], Loading { row: Row::Audio, label: "Analyse audio", progress: Some(75) });
        app.media.as_mut().unwrap().signals_progress = None;
        assert!(app.pending_work().iter().all(|w| w.row != Row::Audio), "fini : la barre disparaît");
    }

    #[test]
    fn loading_tracks_draw_without_any_result_yet() {
        let mut app = app();
        {
            let media = app.media.as_mut().unwrap();
            media.strip_progress = Some(0);
            media.detect_progress = Some(50);
            media.signals_progress = Some(99);
            media.detail_pending.insert(0);
        }
        let mut harness = harness(app);
        // Dessine les barres, y compris l'animation, sans planter
        for _ in 0..10 {
            harness.step();
        }
        assert_eq!(harness.state().pending_work().len(), 3);
    }

    #[test]
    fn analysis_failure_ends_the_loading_state() {
        let mut app = app();
        app.media.as_mut().unwrap().signals_progress = Some(30);
        app.media.as_mut().unwrap().detect_progress = Some(30);
        let generation = app.generation;
        app.jobs_sender.send((generation, Job::Signals(Err("illisible".into())))).unwrap();
        app.jobs_sender.send((generation, Job::Detected(Err("illisible".into())))).unwrap();
        let mut harness = harness(app);
        harness.step();
        harness.step();
        assert_eq!(harness.state().pending_work(), [], "une erreur ne laisse pas une barre tourner sans fin");
        assert!(harness.state().status.contains("impossible"));
    }

    #[test]
    fn audio_track_shows_spectrum_or_waveform() {
        let mut harness = harness(app_with_surround());
        harness.step();
        harness.step();
        assert_eq!(harness.state().tracks.view, AudioView::Spectrum);
        assert!(harness.state().media.as_ref().unwrap().overviews.contains_key(&0));
        harness.get_by_label("Onde").click();
        harness.step();
        harness.step();
        assert_eq!(harness.state().tracks.view, AudioView::Waveform);
        press(&mut harness, egui::Key::V, egui::Modifiers::NONE);
        assert_eq!(harness.state().tracks.view, AudioView::Spectrum);
    }

    #[test]
    fn each_track_has_its_own_checkbox() {
        let mut harness = harness(app_with_surround());
        harness.step();
        harness.step();
        let lanes = |harness: &Harness<'static, App>| {
            let state = harness.state();
            state.lanes(state.tracks.audio_channels)
        };
        assert_eq!(lanes(&harness), [0], "le mix seul, au départ");
        // Toutes les pistes sont listées, l'une sous l'autre
        let rows: Vec<egui::Rect> = ["Mix", "Left", "Right", "Center", "LFE", "Left Surround", "Right Surround"].iter().map(|n| harness.get_by_label(n).rect()).collect();
        for pair in rows.windows(2) {
            assert_eq!(pair[0].left(), pair[1].left());
            assert!(pair[1].top() >= pair[0].bottom() - 0.5, "{pair:?}");
        }
        assert!(rows[6].right() < harness.state().timeline_body.left());

        // On compose ce qu'on veut, dans l'ordre de la liste
        harness.get_by_label("LFE").click();
        harness.step();
        harness.step();
        harness.get_by_label("Left").click();
        harness.step();
        harness.step();
        assert_eq!(lanes(&harness), [0, 1, 4]);
        let media = harness.state().media.as_ref().unwrap();
        assert!(media.overviews.contains_key(&1) && media.overviews.contains_key(&4));

        harness.query_all_by_label("Mix").next().unwrap().click();
        harness.step();
        harness.step();
        assert_eq!(lanes(&harness), [1, 4]);
    }

    #[test]
    fn the_last_visible_track_cannot_be_hidden() {
        let mut harness = harness(app_with_surround());
        harness.step();
        harness.step();
        harness.get_by_label("Mix").click();
        harness.step();
        harness.step();
        let state = harness.state();
        assert_eq!(state.lanes(state.tracks.audio_channels), [0]);
    }

    #[test]
    fn each_channel_has_solo_and_mute_buttons() {
        let mut app = app_with_surround();
        app.tracks.audio_channels = timeline::Channels::tracks_only();
        let mut harness = harness(app);
        harness.step();
        harness.step();
        assert_eq!(harness.get_all_by_label("S").count(), 6);
        assert_eq!(harness.get_all_by_label("M").count(), 6);
        harness.get_all_by_label("S").nth(2).unwrap().click();
        harness.step();
        harness.step();
        assert!(harness.state().monitor.is_solo(2));
        assert!(harness.state().monitor.audible(2) && !harness.state().monitor.audible(0));
        harness.get_all_by_label("M").nth(0).unwrap().click();
        harness.step();
        harness.step();
        assert!(harness.state().monitor.is_muted(0));
        // Pas de bouton de retour à la normale : chaque S et M se recoupe lui-même
        assert_eq!(harness.query_all_by_label("Tout écouter").count(), 0);
        harness.get_all_by_label("S").nth(2).unwrap().click();
        harness.step();
        harness.step();
        harness.get_all_by_label("M").nth(0).unwrap().click();
        harness.step();
        harness.step();
        assert!(!harness.state().monitor.is_active());
    }

    #[test]
    fn transport_stays_reachable_in_the_smallest_window() {
        let mut harness = Harness::builder()
            .with_size(egui::vec2(LEFT_PANEL_MIN + MIN_VIDEO_WIDTH + RIGHT_PANEL_MIN + 16.0, 650.0))
            .build_ui_state(|ui, app: &mut App| app.draw(ui), app_with_surround());
        harness.step();
        harness.step();
        let window = harness.ctx.content_rect();
        let play = harness.get_by_label("Lecture / pause (Espace)").rect();
        let next = harness.get_by_label("Séquence suivante (flèche bas)").rect();
        let list = harness.get_by_label_contains("Instruments Nom Note").rect();
        assert!(next.right() < list.left(), "{next:?} {list:?}");
        assert!(window.contains_rect(play));
    }

    #[test]
    fn solo_and_mute_stay_in_the_table_whatever_is_shown() {
        let mut harness = harness(app_with_surround());
        harness.step();
        harness.step();
        // Le mix seul affiché : chaque canal garde ses S et M, le mix n'en a pas
        assert_eq!(harness.query_all_by_label("S").count(), 6);
        assert_eq!(harness.query_all_by_label("M").count(), 6);
        let solo = harness.get_all_by_label("S").next().unwrap().rect();
        let left = harness.get_by_label("Left").rect();
        assert!((solo.center().y - left.center().y).abs() < 1.0, "en face de son canal : {solo:?} {left:?}");
    }

    #[test]
    fn sensitivity_box_and_detection_switch() {
        let mut app = app();
        app.media.as_mut().unwrap().project.scores = vec![0.0; 100];
        let mut harness = harness(app);
        harness.step();
        let rect = harness.get_by_label("Sensibilité").rect();
        harness.input_mut().events.push(egui::Event::PointerMoved(egui::pos2(rect.right() - 2.0, rect.bottom() - 5.0)));
        harness.input_mut().events.push(egui::Event::PointerButton {
            pos: egui::pos2(rect.right() - 2.0, rect.bottom() - 5.0),
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        });
        harness.input_mut().events.push(egui::Event::PointerButton {
            pos: egui::pos2(rect.right() - 2.0, rect.bottom() - 5.0),
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        });
        harness.step();
        harness.step();
        assert_eq!(harness.state().media.as_ref().unwrap().project.settings.sensitivity, 100.0);
    }

    #[test]
    fn cancelled_detection_stops_at_once() {
        let folder = std::env::temp_dir().join(format!("mx-lab-cancel-{}", std::process::id()));
        std::fs::create_dir_all(&folder).unwrap();
        let source = folder.join("long.wav");
        let (log, ok) = mx_lab::ffmpeg::run(&[
            "-y".as_ref(),
            "-f".as_ref(),
            "lavfi".as_ref(),
            "-i".as_ref(),
            "sine=frequency=440:duration=1800".as_ref(),
            source.as_os_str(),
        ])
        .unwrap();
        assert!(ok, "{log}");
        let model = Yamnet::load();
        let started = Instant::now();
        let mut batches = 0;
        let result = detect::detect_music_probabilities_live(&model, &source, 1800.0, &[Vec::new()], &|_| {}, &mut |_| {
            batches += 1;
            false
        });
        assert_eq!(result, Err(detect::CANCELLED.to_string()));
        assert_eq!(batches, 1, "arrêt dès le premier lot");
        assert!(started.elapsed() < Duration::from_secs(10), "{:?}", started.elapsed());
        let _ = std::fs::remove_dir_all(folder);
    }

    #[test]
    fn detection_switch_sets_detected_sequences_aside() {
        let mut app = app();
        app.media.as_mut().unwrap().project.scores = vec![0.0; 100];
        let mut harness = harness(app);
        harness.step();
        // Coupée : les séquences détectées sont mises de côté, la manuelle reste
        harness.state_mut().media.as_mut().unwrap().project.segments[1].origin = Origin::Manual;
        harness.get_by_label("Détection").click();
        harness.step();
        harness.step();
        let project = &harness.state().media.as_ref().unwrap().project;
        assert!(project.detection_off);
        assert_eq!(project.segments.len(), 1);
        assert_eq!(project.segments[0].origin, Origin::Manual);
        assert_eq!(project.set_aside.len(), 2);
        // Rétablie : elles reviennent, recalées sur les scores
        harness.get_by_label("Détection").click();
        harness.step();
        harness.step();
        let project = &harness.state().media.as_ref().unwrap().project;
        assert!(!project.detection_off && project.set_aside.is_empty());
        assert!(project.segments.iter().any(|s| s.origin == Origin::Manual));
    }

    #[test]
    fn tracks_get_more_room_than_the_video() {
        let harness = harness(app());
        let body = harness.state().timeline_body;
        let window = harness.ctx.content_rect();
        assert!(body.height() > window.height() * 0.55, "{body:?}");
    }

    #[test]
    fn film_sheet_and_loudness_sit_left_of_the_video() {
        let harness = harness(app_with_surround());
        let film = harness.get_by_label("Fiche de la vidéo").rect();
        let loudness = harness.get_by_label("Loudness de la vidéo").rect();
        let sequences = harness.get_by_label_contains("Séquences (3)").rect();
        let transport = harness.get_by_label("Lecture / pause (Espace)").rect();
        assert!(film.right() < transport.left(), "{film:?} {transport:?}");
        assert!(loudness.right() < transport.left());
        assert!(loudness.top() > film.top());
        assert!(sequences.left() > transport.right());
    }

    /// Un grand écran : c'est là qu'un plafond de largeur empêchait le panneau de grandir.
    #[test]
    fn right_panel_takes_its_share_on_a_large_screen() {
        let mut harness = Harness::builder()
            .with_size(egui::vec2(2575.0, 1400.0))
            .build_ui_state(|ui, app: &mut App| app.draw(ui), app());
        harness.step();
        let width = harness.ctx.content_rect().right() - harness.get_by_label_contains("Séquences (3)").rect().left();
        assert!((width - 2575.0 * RIGHT_PANEL_SHARE).abs() < 30.0, "{width}");
    }

    #[test]
    fn chapters_show_on_the_image_track() {
        let mut app = app();
        app.media.as_mut().unwrap().chapters = vec![
            signals::Chapter { start: 0.0, end: 100.0, title: "Intro".into() },
            signals::Chapter { start: 100.0, end: 600.0, title: "Un chapitre au titre vraiment très long".into() },
        ];
        let mut harness = harness(app);
        harness.step();
        assert_eq!(harness.state().chapter_at(50.0).unwrap().title, "Intro");
        assert_eq!(harness.state().chapter_at(100.0).unwrap().start, 100.0);
        assert_eq!(harness.state().chapter_at(700.0), None);
        // Le survol de la piste Image passe par le titre : il ne doit rien casser
        let body = harness.state().timeline_body;
        let rows = timeline::layout_rows(body.top(), body.height(), harness.state().tracks);
        let (_, top, bottom) = *rows.iter().find(|r| r.0 == timeline::Row::Strip).unwrap();
        harness.input_mut().events.push(egui::Event::PointerMoved(egui::pos2(body.left() + 300.0, (top + bottom) / 2.0)));
        harness.step();
    }

    #[test]
    fn mixer_gives_each_track_a_bar_and_its_buttons() {
        let mut harness = harness(app_with_surround());
        harness.step();
        let window = harness.ctx.content_rect();
        assert!(harness.state().timeline_body.right() < window.right() - 150.0, "les pistes laissent sa place au mixeur");
        harness.state_mut().fake_position = 1.0;
        harness.step();
        let (tracks, master) = harness.state().mixer_levels().unwrap();
        assert_eq!(tracks.len(), 6);
        assert!((tracks[0] + 20.0).abs() < 0.1, "{}", tracks[0]);
        assert!(master[0] > -30.0 && master[0] == master[1], "{master:?}");

        harness.get_by_label("Solo L (mixeur)").click();
        harness.step();
        harness.step();
        assert!(harness.state().monitor.is_solo(0));
        let (tracks, master) = harness.state().mixer_levels().unwrap();
        assert!((tracks[0] + 20.0).abs() < 0.1, "{}", tracks[0]);
        assert_eq!(tracks[1], timeline::METER_FLOOR, "un canal qu'on n'entend pas ne monte pas");
        assert!((master[0] + 20.0).abs() < 0.1, "{master:?}");

        harness.get_by_label("Mute R (mixeur)").click();
        harness.step();
        harness.step();
        assert!(harness.state().monitor.is_muted(1));
    }

    #[test]
    fn view_selector_sits_at_the_bottom_and_switches_the_view() {
        let mut app = app_with_surround();
        app.tracks.audio_channels = timeline::Channels::tracks_only(); // Les S et M ne doivent pas gêner
        let mut harness = harness(app);
        harness.step();
        harness.step();
        let spectrum = harness.get_by_label("Spectre").rect();
        let wave = harness.get_by_label("Onde").rect();
        assert!(wave.left() >= spectrum.right() - 0.5, "{spectrum:?} {wave:?}");
        assert!(harness.state().timeline_body.bottom() - wave.bottom() < 14.0, "{wave:?}");
        assert_eq!(spectrum.top(), wave.top());
        assert_eq!(spectrum.width(), wave.width());
        assert!(wave.right() < harness.state().timeline_body.left());
        // Les S et M de la dernière ligne ne passent pas sous le sélecteur
        let solo = harness.get_all_by_label("S").last().unwrap().rect();
        assert!(solo.bottom() <= wave.top(), "{solo:?} {wave:?}");

        assert_eq!(harness.state().tracks.view, AudioView::Spectrum);
        harness.get_by_label("Onde").click();
        harness.step();
        harness.step();
        assert_eq!(harness.state().tracks.view, AudioView::Waveform);
        harness.get_by_label("Spectre").click();
        harness.step();
        harness.step();
        assert_eq!(harness.state().tracks.view, AudioView::Spectrum);
    }

    /// La normalisation n'écrase pas les écarts entre pistes : un seul gain, celui du mix.
    #[test]
    fn waveform_normalization_uses_the_gain_of_the_mix_for_every_track() {
        let frames = 48000 * 2;
        let mut samples = vec![0.0f32; frames * 2];
        for i in 0..frames {
            samples[i * 2] = 0.4; // L
            samples[i * 2 + 1] = 0.04; // R, 10 fois plus bas
        }
        let mut analysis = signals::Analysis::new(vec!["L", "R"]);
        analysis.feed(&samples);
        let signals = analysis.finish();
        let mix = signals.channels[0].peaks.max_abs();
        assert!((mix - 0.22).abs() < 0.01, "{mix}");
        let gain = timeline::waveform_gain(&signals, true);
        assert!((gain - 1.0 / mix).abs() < 0.01, "{gain}");
        assert_eq!(timeline::waveform_gain(&signals, false), 1.0);
        // Les deux pistes gardent leur rapport de 10
        let ratio = signals.channels[1].peaks.max_abs() * gain / (signals.channels[2].peaks.max_abs() * gain);
        assert!((ratio - 10.0).abs() < 0.1, "{ratio}");
    }

    #[test]
    fn mixer_ignores_the_waveform_normalization() {
        let mut harness = harness(app_with_surround());
        harness.step();
        harness.state_mut().fake_position = 1.0;
        let normalized = harness.state().mixer_levels().unwrap();
        harness.state_mut().waveform_normalized = false;
        assert_eq!(harness.state().mixer_levels().unwrap(), normalized);
        assert!((normalized.0[0] + 20.0).abs() < 0.1, "niveau réel : 0,1 = -20 dB");
    }

    #[test]
    fn mixer_fader_changes_the_level_of_its_track() {
        let mut harness = harness(app_with_surround());
        harness.step();
        harness.state_mut().fake_position = 1.0;
        harness.step();
        let area = harness.get_by_label("Volume L (mixeur)").rect();
        let pos = area.center();
        harness.input_mut().events.push(egui::Event::PointerMoved(area.center_top() + egui::vec2(0.0, 7.0)));
        harness.step();
        harness.input_mut().events.push(egui::Event::PointerButton {
            pos: area.center_top() + egui::vec2(0.0, 7.0),
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        });
        harness.step();
        for _ in 0..4 {
            harness.input_mut().events.push(egui::Event::PointerMoved(pos));
            harness.step();
        }
        harness.input_mut().events.push(egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        });
        harness.step();
        let gain = harness.state().monitor.gain_db(0);
        assert!((-37.0..=-27.0).contains(&gain), "milieu de la barre : {gain}");
        assert_eq!(harness.state().monitor.gain_db(1), 0.0, "les autres pistes ne bougent pas");
        let media = harness.state().media.as_ref().unwrap();
        assert_eq!(media.project.channel_gains[&0], gain, "le fader est gardé dans le projet");
        assert!(media.dirty);
        let (tracks, _) = harness.state().mixer_levels().unwrap();
        assert!((tracks[0] - (-20.0 + gain)).abs() < 0.2, "barre après le fader : {}", tracks[0]);
    }

    /// Sinus 1 kHz stéréo à -23 dBFS : -23 LUFS pour le mix, 3 de moins par canal.
    #[test]
    fn mixer_lines_follow_short_term_and_momentary() {
        let mut app = app();
        let amplitude = 10f32.powf(-23.0 / 20.0);
        let samples: Vec<f32> = (0..48000 * 8)
            .flat_map(|i| {
                let x = amplitude * (i as f32 * 1000.0 * std::f32::consts::TAU / 48000.0).sin();
                [x, x]
            })
            .collect();
        let mut analysis = signals::Analysis::new(vec!["L", "R"]);
        analysis.feed(&samples);
        app.media.as_mut().unwrap().signals = Some(Arc::new(analysis.finish()));
        app.fake_position = 6.0;
        let (tracks, mix) = app.mixer_loudness().unwrap();
        assert!((mix.0 + 23.0).abs() < 0.2 && (mix.1 + 23.0).abs() < 0.2, "{mix:?}");
        assert!((tracks[0].0 + 26.0).abs() < 0.2, "{:?}", tracks[0]);

        app.monitor.set_gain_db(0, -12.0);
        let (tracks, _) = app.mixer_loudness().unwrap();
        assert!((tracks[0].0 + 38.0).abs() < 0.2, "le fader agit : {:?}", tracks[0]);
        app.monitor.set_gain_db(0, 0.0);

        app.monitor.toggle_mute(0);
        let (tracks, mix) = app.mixer_loudness().unwrap();
        assert_eq!(tracks[0].0, timeline::METER_FLOOR, "coupé : rien");
        assert!((mix.0 + 26.0).abs() < 0.2, "le mix perd ce canal : {mix:?}");
        app.monitor.clear();

        app.monitor.set_master_db(6.0);
        let (_, mix) = app.mixer_loudness().unwrap();
        assert!((mix.0 + 17.0).abs() < 0.2, "le fader du mix agit : {mix:?}");
    }

    #[test]
    fn detection_shows_up_while_it_is_computed() {
        let mut harness = harness(app());
        harness.step();
        let generation = harness.state().generation;
        harness.state_mut().media.as_mut().unwrap().project.segments.clear();
        let sender = harness.state().jobs_sender.clone();
        sender.send((generation, Job::DetectPartial(vec![0.9; 200]))).unwrap();
        harness.step();
        let media = harness.state().media.as_ref().unwrap();
        assert!(!media.smoothed.is_empty(), "la courbe apparaît");
        assert_eq!(media.project.segments.len(), 1, "et la séquence aussi");
        assert!(media.project.scores.is_empty(), "le résultat final n'est pas encore là");
        assert!(!media.dirty, "un aperçu ne demande pas d'enregistrer");
    }

    /// Sur le vrai modèle : les lots arrivent, et le dernier est le résultat final.
    #[test]
    fn detection_reports_growing_scores_on_real_audio() {
        let folder = std::env::temp_dir().join(format!("music-studio-live-{}", std::process::id()));
        std::fs::create_dir_all(&folder).unwrap();
        let source = folder.join("source.wav");
        let (log, ok) = mx_lab::ffmpeg::run(&[
            std::ffi::OsStr::new("-y"),
            std::ffi::OsStr::new("-f"),
            std::ffi::OsStr::new("lavfi"),
            std::ffi::OsStr::new("-i"),
            std::ffi::OsStr::new("sine=frequency=440:duration=600:sample_rate=16000"),
            source.as_os_str(),
        ])
        .unwrap();
        assert!(ok, "{log}");
        let model = Yamnet::load();
        let mixes = vec![vec!["-ac".to_string(), "1".to_string()]];
        let mut lengths = Vec::new();
        let scores = detect::detect_music_probabilities_live(&model, &source, 600.0, &mixes, &|_| {}, &mut |partial| {
            lengths.push(partial.len());
            true
        }).unwrap();
        std::fs::remove_dir_all(&folder).unwrap();
        assert!(lengths.len() >= 2, "{lengths:?}");
        assert!(lengths.windows(2).all(|p| p[1] > p[0]), "{lengths:?}");
        assert_eq!(*lengths.last().unwrap(), scores.len());
    }

    #[test]
    fn mixer_and_track_list_do_not_wait_for_the_audio_analysis() {
        // La disposition est connue, l'analyse n'est pas finie
        let mut surround = app();
        surround.media.as_mut().unwrap().layout = Some("5.1".into());
        let mut harness = harness(surround);
        harness.step();
        harness.step();
        assert!(harness.state().mixer_levels().is_none());
        assert!(harness.state().timeline_body.right() < harness.ctx.content_rect().right() - 150.0, "le mixeur est déjà là");
        harness.get_by_label("Left Surround");
        harness.get_by_label("Solo Ls (mixeur)");
        harness.get_by_label("Volume Mix (mixeur)");
    }

    #[test]
    fn loudness_toggle_replaces_the_gate_setting() {
        let mut harness = harness(app_with_surround());
        harness.step();
        harness.step();
        assert_eq!(harness.query_all_by_label("Gate").count(), 0, "le gate reste actif, sans réglage");
        assert!(harness.state().tracks.loudness);
        harness.get_by_label("Loudness").click();
        harness.step();
        harness.step();
        harness.step();
        assert!(!harness.state().tracks.loudness);
        harness.get_by_label("Cible");
    }

    #[test]
    fn without_a_known_layout_there_is_no_mixer_yet() {
        let mut waiting = app();
        waiting.media.as_mut().unwrap().layout = None;
        let harness = harness(waiting);
        assert_eq!(harness.state().timeline_body.right(), harness.ctx.content_rect().right() - 8.0);
    }

    #[test]
    fn nothing_is_saved_without_asking() {
        let mut harness = harness(app());
        harness.state_mut().media.as_mut().unwrap().mark_changed();
        for _ in 0..30 {
            harness.step();
        }
        assert!(harness.state().media.as_ref().unwrap().dirty, "plus de sauvegarde automatique");
        harness.state_mut().after_save_prompt = Some(AfterPrompt::Quit);
        harness.step();
        harness.step();
        harness.get_all_by_label("Annuler").last().unwrap().click();
        harness.step();
        harness.step();
        assert!(harness.state().after_save_prompt.is_none());
        assert!(harness.state().media.as_ref().unwrap().dirty);
    }

    #[test]
    fn opening_another_film_asks_first_when_changes_are_pending() {
        let mut harness = harness(app());
        harness.state_mut().media.as_mut().unwrap().mark_changed();
        let before = harness.state().media.as_ref().unwrap().path.clone();
        let ctx = harness.ctx.clone();
        harness.state_mut().open(&ctx, PathBuf::from("autre.mkv"));
        assert!(matches!(harness.state().after_save_prompt, Some(AfterPrompt::Open(_))));
        assert_eq!(harness.state().media.as_ref().unwrap().path, before);
    }

    #[test]
    fn detection_settings_sit_left_of_the_sequences_track() {
        let harness = harness(app());
        let body = harness.state().timeline_body;
        let rows = timeline::layout_rows(body.top(), body.height(), harness.state().tracks);
        let (_, top, bottom) = *rows.iter().find(|r| r.0 == timeline::Row::Sequences).unwrap();
        for name in ["Sensibilité", "Gap (s)", "Anticipation (s)", "Post (s)"] {
            // Le texte, son curseur et sa valeur portent le même nom
            for node in harness.query_all_by_label(name) {
                let rect = node.rect();
                assert!(rect.right() < body.left(), "{name} : {rect:?}");
                assert!(rect.top() >= top - 1.0 && rect.bottom() <= bottom + 1.0, "{name} : {rect:?} {top} {bottom}");
            }
        }
    }

    #[test]
    fn detection_rows_are_left_aligned_and_share_their_dimensions() {
        let mut app = app();
        app.media.as_mut().unwrap().project.scores = vec![0.0; 100];
        let mut harness = harness(app);
        harness.step();
        // Gap, anticipation et post sont derrière l'engrenage
        assert_eq!(harness.query_all_by_label("Gap (s)").count(), 0);
        harness.get_by_label("Réglages de la détection").click();
        harness.step();
        harness.step();
        assert_eq!(harness.query_all_by_label("Gap (s)").count(), 2, "glissière et valeur");
        let rows: Vec<Vec<egui::Rect>> = ["Gap (s)", "Anticipation (s)", "Post (s)"]
            .iter()
            .map(|name| harness.query_all_by_label(name).map(|node| node.rect()).collect())
            .collect();
        for row in &rows {
            assert_eq!(row.len(), rows[0].len());
            for (rect, first) in row.iter().zip(&rows[0]) {
                assert_eq!(rect.left(), first.left(), "colonnes alignées");
                assert_eq!(rect.width(), first.width(), "mêmes dimensions");
            }
        }
    }

    #[test]
    fn new_project_has_short_margins() {
        let settings = project::DetectionSettings::default();
        assert_eq!((settings.anticipation, settings.post), (2.0, 2.0));
    }

    #[test]
    fn clicking_a_track_moves_the_playhead() {
        let mut harness = harness(app());
        harness.step();
        drag_on_row(&mut harness, timeline::Row::Audio, 300.0, 300.0, false);
        assert!((harness.state().position() - 300.0).abs() < 1.0, "{}", harness.state().position());
    }

    #[test]
    fn notes_keep_a_detected_sequence_through_redetection() {
        let mut app = app();
        app.media.as_mut().unwrap().project.segments[0].name = "Générique".into();
        app.media.as_mut().unwrap().project.segments[0].origin = Origin::Edited;
        app.media.as_mut().unwrap().project.scores = vec![0.0; 100];
        app.apply_detection();
        let segments = &app.media.as_ref().unwrap().project.segments;
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].name, "Générique");
    }

    /// Extrait fourni avec le projet, copié pour ne rien écrire à côté de l'original.
    fn real_clip(name: &str) -> Option<PathBuf> {
        let folder = Path::new(env!("CARGO_MANIFEST_DIR")).join("Severance");
        let source = std::fs::read_dir(folder).ok()?.flatten().find(|e| e.file_name().to_string_lossy().ends_with("seg04.mkv"))?;
        let target = std::env::temp_dir().join(format!("music-studio-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&target);
        std::fs::create_dir_all(&target).ok()?;
        let clip = target.join("clip.mkv");
        std::fs::copy(source.path(), &clip).ok()?;
        Some(clip)
    }

    /// De bout en bout, sur un vrai fichier : analyse, projet créé à côté, correction,
    /// export, réouverture sans recalcul.
    #[test]
    fn real_file_goes_from_import_to_export_and_back() {
        let Some(clip) = real_clip("flow") else { return };
        let mut harness = harness(App::new(None, None));
        let ctx = harness.ctx.clone();
        harness.state_mut().open(&ctx, clip.clone());
        let project_file = project::project_path(&clip);
        assert!(project_file.is_file(), "le projet doit exister dès l'import");

        let ready = |app: &App| {
            app.media.as_ref().is_some_and(|m| {
                m.detect_progress.is_none() && m.signals.is_some() && m.strip.is_some()
            })
        };
        for _ in 0..1200 {
            harness.step();
            if ready(harness.state()) {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        assert!(ready(harness.state()), "{}", harness.state().status);
        {
            let media = harness.state().media.as_ref().unwrap();
            assert!(!media.project.scores.is_empty());
            assert!(media.project.duration > 40.0, "{}", media.project.duration);
            assert_eq!(media.signals.as_ref().unwrap().channels.len(), 3, "mix, L, R");
            assert!(!media.strip.as_ref().unwrap().keyframes.is_empty());
        }

        // Corrige une séquence ajoutée à la main, l'annote, l'exporte
        harness.state_mut().media.as_mut().unwrap().project.segments.clear();
        harness.state_mut().seek_to(10.0, true);
        harness.step();
        harness.state_mut().add_at_playhead();
        let id = harness.state().selected.unwrap();
        {
            let media = harness.state_mut().media.as_mut().unwrap();
            media.project.set_end(id, 20.0);
            media.project.segment_mut(id).unwrap().name = "Thème".into();
            media.mark_changed();
        }
        harness.state_mut().export(&ctx, Some(id));
        for _ in 0..300 {
            harness.step();
            if !harness.state().exporting {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let folder = harness.state().last_export.clone().unwrap_or_else(|| panic!("{}", harness.state().status));
        let files: Vec<_> = std::fs::read_dir(&folder).unwrap().flatten().collect();
        assert_eq!(files.len(), 1);
        assert!(files[0].metadata().unwrap().len() > 10_000);

        harness.state_mut().save_now();
        let saved = Project::load(&project_file).unwrap();
        assert_eq!(saved.segments.len(), 1);
        assert_eq!(saved.segments[0].name, "Thème");

        // Réouverture : la correction revient, la détection n'est pas relancée
        harness.state_mut().open(&ctx, clip.clone());
        {
            let media = harness.state().media.as_ref().unwrap();
            assert_eq!(media.detect_progress, None);
            assert_eq!(media.project.segments[0].name, "Thème");
            assert_eq!(media.project.segments[0].end, 20.0);
        }
        let _ = std::fs::remove_dir_all(clip.parent().unwrap());
    }

    #[test]
    fn real_file_gives_a_fine_spectrogram_when_zooming() {
        let Some(clip) = real_clip("detail") else { return };
        let detail = signals::detail_spectrogram(&clip, "stereo", 1, 5.0, 15.0, 800, 0).unwrap();
        assert_eq!(detail.channel, 1);
        assert_eq!(detail.values.len(), 800 * signals::DETAIL_BINS);
        assert!(detail.values.iter().any(|v| *v > 100), "le spectre ne doit pas être vide");
        // Le canal mono mixé fonctionne aussi
        signals::detail_spectrogram(&clip, "stereo", 0, 0.0, 2.0, 100, 0).unwrap();
        let _ = std::fs::remove_dir_all(clip.parent().unwrap());
    }

    #[test]
    fn typing_in_notes_does_not_trigger_shortcuts() {
        let mut harness = harness(app());
        harness.step();
        press(&mut harness, egui::Key::ArrowDown, egui::Modifiers::NONE);
        let before = starts(harness.state());
        harness.get_by_label("Notes").click();
        harness.step();
        harness.step();
        press(&mut harness, egui::Key::Delete, egui::Modifiers::NONE);
        press(&mut harness, egui::Key::N, egui::Modifiers::NONE);
        assert_eq!(starts(harness.state()), before);
    }
}
