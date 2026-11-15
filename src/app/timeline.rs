//! La timeline : pistes synchronisées, poignées, zoom.

use crate::project::{LoudnessTarget, Origin};
use crate::signals::{self, Peaks, BLOCK_S, DETAIL_BINS, F_MAX, F_MIN, NO_LOUDNESS, OVERVIEW_BINS};
use crate::{tc, App, Job};
use eframe::egui::{self, Align2, Color32, FontId, Mesh, Pos2, Rect, Sense, Shape, Stroke, TextureOptions};
use mx_lab::detect;
use rayon::prelude::*;
use std::sync::OnceLock;
use std::time::Instant;

pub const HEADER_WIDTH: f32 = 192.0;
const MIN_SPAN: f64 = 0.5;
/// En pixels : à cette distance d'une voisine, une poignée lâchée s'y colle.
const MERGE_DISTANCE: f32 = 4.0;
const HANDLE_GRAB: f32 = 7.0;
/// Jusqu'où le spectre fin est calculé à la demande. Sur tous les canaux à la fois, plus court :
/// chaque canal lit sa propre portion de la vidéo.
pub const DETAIL_MAX_SPAN: f64 = 600.0;
const DETAIL_MAX_SPAN_SPLIT: f64 = 240.0;
const MAX_DETAIL_COLUMNS: usize = 8192;
const ZOOM_HANDLE_GRAB: f32 = 9.0;
const ZOOM_MIN_WIDTH: f32 = 18.0;
const ZOOM_MARGIN: f32 = 14.0;
const ZOOM_BLUE: Color32 = Color32::from_rgb(52, 132, 235);
pub const LOADING_REFRESH: std::time::Duration = std::time::Duration::from_millis(33);
const LOUDNESS_FLOOR: f32 = -60.0;
/// La piste des séquences loge dans son en-tête son titre et les 4 réglages de détection.
const SEQUENCES_HEIGHT: f32 = 92.0;
const SCALE_MIN_SPACING: f32 = 14.0;
pub const METER_FLOOR: f32 = crate::monitor::FADER_FLOOR_DB;
const METER_TOP: f32 = crate::monitor::FADER_TOP_DB;
/// Dégradé de la barre, de -Inf à 0 dB : les couleurs du spectrogramme, aux mêmes proportions.
const METER_GRADIENT: [(f32, [u8; 3]); 6] = [
    (METER_FLOOR, [4, 4, 10]),
    (METER_FLOOR * 0.72, [50, 16, 110]),
    (METER_FLOOR * 0.5, [150, 40, 120]),
    (METER_FLOOR * 0.3, [235, 90, 60]),
    (METER_FLOOR * 0.12, [252, 200, 90]),
    (0.0, [255, 255, 230]),
];
const METER_WINDOW_S: f64 = 0.05;
const SHORT_TERM_BLOCKS: f64 = 30.0;
const MOMENTARY_BLOCKS: f64 = 4.0;
const METER_MOMENTARY_LINE: Color32 = Color32::from_rgb(165, 170, 178);
const METER_FALL_DB_PER_S: f32 = 30.0;
/// Inertie des repères de loudness, à la montée et à la descente.
const METER_RISE_S: f32 = 0.06;
const METER_FALL_S: f32 = 0.18;
const MIXER_TRACK_WIDTH: f32 = 24.0;
const MIXER_TRACK_BAR: f32 = 12.0;
const MIXER_MASTER_BAR: f32 = 24.0;
const MIXER_SCALE_WIDTH: f32 = 26.0;
const MIXER_PADDING: f32 = 8.0;
const METER_BACK: Color32 = Color32::from_rgb(12, 13, 16);

// Colonne des en-têtes
const PAD: f32 = 11.0;
const FONT_SIZE: f32 = 11.5;
const CONTROL_HEIGHT: f32 = 24.0;
const CELL_HEIGHT: f32 = 22.0;
const TOGGLE_CELL: f32 = 24.0;
const SWITCH_HEIGHT: f32 = 20.0;
const SWITCH_GAP: f32 = 10.0;
const TARGET_LABEL_HEIGHT: f32 = 18.0;
const TARGET_ROW_HEIGHT: f32 = 21.0;
const TARGET_VALUE_WIDTH: f32 = 44.0;
const SECTION_LINE: Color32 = Color32::from_rgb(9, 9, 11);
const CELL_LINE: Color32 = Color32::from_rgb(50, 52, 58);
const CELL_FILL: Color32 = Color32::from_rgb(40, 41, 46);
const SELECTED_FILL: Color32 = Color32::from_rgb(60, 61, 67);
const TITLE: Color32 = Color32::from_rgb(240, 241, 244);
const NAME: Color32 = Color32::from_rgb(222, 224, 228);

const BG: Color32 = Color32::from_rgb(20, 22, 26);
const ROW_LINE: Color32 = Color32::from_rgb(42, 45, 52);
const TEXT: Color32 = Color32::from_rgb(190, 195, 205);
const WEAK: Color32 = Color32::from_rgb(120, 125, 135);
pub const PLAYHEAD: Color32 = Color32::from_rgb(255, 72, 72);
const DETECTED: Color32 = Color32::from_rgb(38, 150, 136);
const EDITED: Color32 = Color32::from_rgb(70, 130, 210);
const MANUAL: Color32 = Color32::from_rgb(222, 140, 50);
const WAVE: Color32 = Color32::from_rgb(92, 190, 170);
const LOUD_INTEGRATED: Color32 = Color32::from_rgb(70, 160, 255);
const LOUD_OFF_TARGET: Color32 = Color32::from_rgb(235, 40, 50);
const LOUD_SHORT: Color32 = Color32::from_rgb(245, 245, 245);
const LOUD_MOMENTARY: Color32 = Color32::from_rgb(165, 170, 178);
const SOLO: Color32 = Color32::from_rgb(245, 200, 70);
const MUTE: Color32 = Color32::from_rgb(225, 85, 80);

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Row {
    Overview,
    Ruler,
    Strip,
    Sequences,
    Audio,
}

/// Ce que la piste audio montre sous la loudness, toujours superposée.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum AudioView {
    #[default]
    Spectrum,
    Waveform,
}

/// Les pistes affichées dans l'analyse : 0 pour le mix, puis les canaux de la source.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Channels {
    visible: u32,
}

impl Default for Channels {
    fn default() -> Self {
        Channels { visible: 1 }
    }
}

impl Channels {
    /// Tous les canaux, sans le mix.
    pub fn tracks_only() -> Self {
        Channels { visible: !1 }
    }

    pub fn is_visible(self, track: usize) -> bool {
        track < 32 && self.visible & (1 << track) != 0
    }

    pub fn set(&mut self, track: usize, visible: bool) {
        if track < 32 {
            if visible {
                self.visible |= 1 << track;
            } else {
                self.visible &= !(1 << track);
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct Tracks {
    pub strip: bool,
    pub audio: bool,
    pub view: AudioView,
    pub audio_channels: Channels,
    /// Courbes de loudness sur le spectre et repères du mixeur.
    pub loudness: bool,
}

impl Default for Tracks {
    fn default() -> Self {
        Tracks {
            strip: true,
            audio: true,
            view: AudioView::default(),
            audio_channels: Channels::default(),
            loudness: true,
        }
    }
}

/// Répartit la hauteur entre les pistes : hauteur fixe pour les unes, le reste aux analyses.
pub fn layout_rows(top: f32, height: f32, tracks: Tracks) -> Vec<(Row, f32, f32)> {
    let mut items: Vec<(Row, f32, f32)> = vec![(Row::Overview, 20.0, 0.0), (Row::Ruler, 22.0, 0.0)];
    if tracks.strip {
        items.push((Row::Strip, 40.0, 0.0));
    }
    items.push((Row::Sequences, SEQUENCES_HEIGHT, 0.0));
    if tracks.audio {
        items.push((Row::Audio, 0.0, 1.0));
    }
    let fixed: f32 = items.iter().map(|i| i.1).sum();
    let weights: f32 = items.iter().map(|i| i.2).sum();
    let free = (height - fixed).max(0.0);
    let mut y = top;
    items
        .into_iter()
        .map(|(row, base, weight)| {
            let size = if weight > 0.0 { (free * weight / weights).max(30.0) } else { base };
            let span = (row, y, y + size);
            y += size;
            span
        })
        .collect()
}

/// Portion de la vidéo visible.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct View {
    pub start: f64,
    pub span: f64,
}

impl View {
    pub fn end(&self) -> f64 {
        self.start + self.span
    }

    pub fn x(&self, left: f32, width: f32, time: f64) -> f32 {
        left + ((time - self.start) / self.span) as f32 * width
    }

    pub fn time(&self, left: f32, width: f32, x: f32) -> f64 {
        self.start + (x - left) as f64 / width as f64 * self.span
    }

    pub fn clamp(&mut self, duration: f64) {
        if duration <= 0.0 {
            return;
        }
        self.span = self.span.clamp(MIN_SPAN.min(duration), duration);
        self.start = self.start.clamp(0.0, duration - self.span);
    }

    /// Déplace le début de la vue, sans bouger sa fin : c'est la poignée gauche de la barre de zoom.
    pub fn set_start(&mut self, time: f64, duration: f64) {
        let end = self.end().min(duration);
        let start = time.clamp(0.0, (end - MIN_SPAN).max(0.0));
        self.start = start;
        self.span = end - start;
    }

    /// Déplace la fin de la vue, sans bouger son début : c'est la poignée droite.
    pub fn set_end(&mut self, time: f64, duration: f64) {
        let end = time.clamp((self.start + MIN_SPAN).min(duration), duration);
        self.span = end - self.start;
    }

    /// Zoome en gardant `anchor` immobile à l'écran.
    pub fn zoom_at(&mut self, anchor: f64, factor: f64, duration: f64) {
        let ratio = (anchor - self.start) / self.span;
        self.span *= factor;
        self.clamp(duration);
        self.start = anchor - ratio * self.span;
        self.clamp(duration);
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Edge {
    Start,
    End,
}

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum Drag {
    #[default]
    None,
    ZoomEdge(Edge),
    ZoomMove(f64),
    Scrub,
    Handle(u64, Edge),
    Move(u64, f64),
    Create(f64),
    /// La vue attrapée au clic molette.
    Pan,
}

/// Un calcul en cours, à montrer là où son résultat va apparaître.
#[derive(Clone, PartialEq, Debug)]
pub struct Loading {
    pub row: Row,
    pub label: &'static str,
    /// En pourcents ; `None` quand l'avancement n'est pas encore connu.
    pub progress: Option<u32>,
}

/// À 0 %, on ne sait rien encore : mieux vaut une barre animée qu'une barre vide et figée.
fn known_progress(progress: u32) -> Option<u32> {
    (progress > 0).then_some(progress)
}

/// Barre animée quand l'avancement est inconnu.
fn progress_bar(painter: &egui::Painter, rect: Rect, progress: Option<u32>, time: f64) {
    painter.rect_filled(rect, rect.height() / 2.0, Color32::from_white_alpha(35));
    let fill = match progress {
        Some(percent) => Rect::from_min_size(
            rect.min,
            egui::vec2(rect.width() * (percent.min(100) as f32 / 100.0), rect.height()),
        ),
        None => {
            let width = rect.width() * 0.28;
            let position = (time * 0.7).fract() as f32 * (rect.width() + width) - width;
            let left = (rect.left() + position).max(rect.left());
            let right = (rect.left() + position + width).min(rect.right());
            Rect::from_min_max(Pos2::new(left, rect.top()), Pos2::new(right.max(left), rect.bottom()))
        }
    };
    painter.rect_filled(fill, rect.height() / 2.0, ZOOM_BLUE);
}

/// Texte et barre au centre de la zone qui attend son contenu.
pub fn paint_loading(painter: &egui::Painter, rect: Rect, label: &str, progress: Option<u32>, time: f64) {
    let text = match progress {
        Some(percent) => format!("{label} {percent} %"),
        None => label.to_string(),
    };
    let galley = painter.layout_no_wrap(text, FontId::proportional(12.0), Color32::WHITE);
    let bar_width = (rect.width() * 0.3).clamp(80.0, 320.0);
    let total = galley.size().x + 10.0 + bar_width;
    let top = rect.center().y - galley.size().y / 2.0;
    let left = rect.center().x - total / 2.0;
    let frame = Rect::from_min_size(Pos2::new(left, top), egui::vec2(total, galley.size().y)).expand2(egui::vec2(8.0, 3.0));
    painter.rect_filled(frame, 4.0, Color32::from_black_alpha(190));
    let bar = Rect::from_center_size(
        Pos2::new(left + galley.size().x + 10.0 + bar_width / 2.0, rect.center().y),
        egui::vec2(bar_width, 6.0),
    );
    painter.galley(Pos2::new(left, top), galley, Color32::WHITE);
    progress_bar(painter, bar, progress, time);
}

/// Le cadre d'un chargement, en bas à droite d'une piste. `lift` : pour en empiler deux.
fn loading_corner(rect: Rect, size: egui::Vec2, lift: f32) -> Rect {
    Rect::from_min_size(Pos2::new(rect.right() - 40.0 - size.x, rect.bottom() - 6.0 - lift - size.y), size)
}

/// Texte et barre en bas à droite de la piste qui attend son contenu.
pub fn paint_track_loading(painter: &egui::Painter, rect: Rect, label: &str, progress: Option<u32>, time: f64, lift: f32) {
    let text = match progress {
        Some(percent) => format!("{label} {percent} %"),
        None => label.to_string(),
    };
    const BAR_WIDTH: f32 = 90.0;
    let galley = painter.layout_no_wrap(text, FontId::proportional(11.0), Color32::WHITE);
    let size = egui::vec2(galley.size().x + 8.0 + BAR_WIDTH + 12.0, galley.size().y + 4.0);
    let frame = loading_corner(rect, size, lift);
    painter.rect_filled(frame, 4.0, Color32::from_black_alpha(190));
    let bar = Rect::from_min_size(Pos2::new(frame.right() - 6.0 - BAR_WIDTH, frame.center().y - 2.5), egui::vec2(BAR_WIDTH, 5.0));
    painter.galley(Pos2::new(frame.left() + 6.0, frame.top() + 2.0), galley, Color32::WHITE);
    progress_bar(painter, bar, progress, time);
}

/// Début et largeur de la barre de zoom : une marge de chaque côté garde ses poignées
/// entières et attrapables, loin du bord de la fenêtre.
pub fn zoom_span(left: f32, width: f32) -> (f32, f32) {
    (left + ZOOM_MARGIN, (width - ZOOM_MARGIN * 2.0).max(1.0))
}

/// Position à l'écran des deux poignées de la barre de zoom. Elles restent attrapables
/// séparément, même quand la vue ne représente qu'un point de la vidéo.
fn zoom_handles(view: View, left: f32, width: f32, duration: f64) -> (f32, f32) {
    let right = left + width;
    let x0 = left + (view.start / duration) as f32 * width;
    let x1 = left + (view.end() / duration) as f32 * width;
    let x1 = x1.max(x0 + ZOOM_MIN_WIDTH);
    if x1 > right {
        (right - (x1 - x0), right)
    } else {
        (x0, x1)
    }
}

/// Dégradé noir, violet, orange, jaune, blanc.
fn spectrum_color(value: u8) -> Color32 {
    static TABLE: OnceLock<Vec<Color32>> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        const STOPS: [(f32, [f32; 3]); 6] = [
            (0.0, [4.0, 4.0, 10.0]),
            (0.28, [50.0, 16.0, 110.0]),
            (0.5, [150.0, 40.0, 120.0]),
            (0.7, [235.0, 90.0, 60.0]),
            (0.88, [252.0, 200.0, 90.0]),
            (1.0, [255.0, 255.0, 230.0]),
        ];
        (0..256)
            .map(|i| {
                let t = i as f32 / 255.0;
                let upper = STOPS.iter().position(|s| s.0 >= t).unwrap_or(STOPS.len() - 1).max(1);
                let (a, b) = (STOPS[upper - 1], STOPS[upper]);
                let f = ((t - a.0) / (b.0 - a.0)).clamp(0.0, 1.0);
                let mix = |k: usize| (a.1[k] + (b.1[k] - a.1[k]) * f) as u8;
                Color32::from_rgb(mix(0), mix(1), mix(2))
            })
            .collect()
    });
    table[value as usize]
}

fn ruler_step(pixels_per_second: f64) -> f64 {
    const STEPS: [f64; 17] =
        [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 900.0, 1800.0, 3600.0, 7200.0];
    STEPS.iter().copied().find(|s| s * pixels_per_second >= 96.0).unwrap_or(7200.0)
}

fn ruler_label(time: f64, step: f64) -> String {
    let total = time.round() as u64;
    if step < 1.0 {
        let tenths = (time * 10.0).round() as u64;
        format!("{:02}:{:02}.{}", tenths / 600, tenths / 10 % 60, tenths % 10)
    } else {
        format!("{:02}:{:02}:{:02}", total / 3600, total / 60 % 60, total % 60)
    }
}

/// Une légende en haut à droite d'une piste, lue de droite à gauche.
fn paint_legend(painter: &egui::Painter, rect: Rect, entries: &[(&str, Color32)]) {
    let mut x = rect.right() - 8.0;
    for (name, color) in entries {
        let swatch = Rect::from_min_size(Pos2::new(x - 16.0, rect.top() + 6.0), egui::vec2(16.0, 3.0));
        painter.rect_filled(swatch.expand(1.0), 2.0, Color32::from_black_alpha(150));
        painter.rect_filled(swatch, 1.5, *color);
        let label = painter.layout_no_wrap(name.to_string(), FontId::proportional(10.5), Color32::WHITE);
        let at = Pos2::new(swatch.left() - 4.0 - label.size().x, rect.top() + 1.0);
        painter.rect_filled(Rect::from_min_size(at, label.size()).expand(1.0), 2.0, Color32::from_black_alpha(150));
        x = at.x - 10.0;
        painter.galley(at, label, Color32::WHITE);
    }
}

/// La même couleur, éclaircie vers le blanc (`amount` de 0 à 1).
fn lighter(color: Color32, amount: f32) -> Color32 {
    let mix = |value: u8| (value as f32 + (255.0 - value as f32) * amount) as u8;
    Color32::from_rgb(mix(color.r()), mix(color.g()), mix(color.b()))
}

/// Nom complet d'une piste dans la liste à cocher.
fn track_full_name(name: &str) -> &str {
    match name {
        "L" => "Left",
        "R" => "Right",
        "C" => "Center",
        "Ls" => "Left Surround",
        "Rs" => "Right Surround",
        "Lb" => "Left Back",
        "Rb" => "Right Back",
        "Cs" => "Center Surround",
        other => other,
    }
}

fn segment_color(origin: Origin) -> Color32 {
    match origin {
        Origin::Detected => DETECTED,
        Origin::Edited => EDITED,
        Origin::Manual => MANUAL,
    }
}

fn frequency_y(rect: Rect, hz: f64) -> f32 {
    let position = (hz / F_MIN).ln() / (F_MAX / F_MIN).ln();
    rect.bottom() - position as f32 * rect.height()
}

fn frequency_at(rect: Rect, y: f32) -> f64 {
    let position = ((rect.bottom() - y) / rect.height()).clamp(0.0, 1.0) as f64;
    F_MIN * (F_MAX / F_MIN).powf(position)
}

fn frequency_label(hz: f64) -> String {
    if hz >= 1000.0 {
        format!("{:.1} kHz", hz / 1000.0)
    } else {
        format!("{hz:.0} Hz")
    }
}

/// Lance le calcul du spectre d'une piste, livré morceau par morceau.
fn start_spectrum(media: &mut crate::Media, channel: usize, generation: u64, sender: &std::sync::mpsc::Sender<(u64, Job)>, ctx: &egui::Context) {
    let Some(layout) = media.layout.clone() else { return };
    media.spectra.insert(channel, crate::SpectrumState::default());
    let (path, track, sender, ctx) = (media.path.clone(), media.project.audio_track, sender.clone(), ctx.clone());
    std::thread::spawn(move || {
        let _ = signals::overview_spectrum(&path, &layout, channel, track, &mut |first, values| {
            let _ = sender.send((generation, Job::SpectrumPart(channel, first, values)));
            ctx.request_repaint();
        });
        let _ = sender.send((generation, Job::SpectrumDone(channel)));
        ctx.request_repaint();
    });
}

/// La prochaine piste à préparer en arrière-plan : aucune tant qu'un calcul tourne.
fn next_background_spectrum(spectra: &std::collections::HashMap<usize, crate::SpectrumState>, count: usize) -> Option<usize> {
    if spectra.values().any(|state| !state.finished) {
        return None;
    }
    (0..count).find(|channel| !spectra.contains_key(channel))
}

/// Espace minimal entre deux repères de fréquence.
const FREQUENCY_TICK_SPACING: f32 = 13.0;
/// Les repères, du plus important au moins important : les premiers restent quand la place manque.
const FREQUENCY_TICKS: [f64; 24] = [
    1000.0, 100.0, 10000.0, 5000.0, 500.0, 2000.0, 20000.0, 300.0, 3000.0, 50.0, 15000.0, 700.0, 7000.0, 1500.0, 200.0, 8000.0,
    6000.0, 12000.0, 2500.0, 3500.0, 1200.0, 4000.0, 400.0, 70.0,
];

/// Les repères qui tiennent dans cette ligne sans se toucher.
fn frequency_ticks(band: Rect) -> Vec<f64> {
    let mut kept: Vec<f64> = Vec::new();
    for hz in FREQUENCY_TICKS {
        let y = frequency_y(band, hz);
        let inside = y >= band.top() + 7.0 && y <= band.bottom() - 7.0;
        if inside && kept.iter().all(|other| (frequency_y(band, *other) - y).abs() >= FREQUENCY_TICK_SPACING) {
            kept.push(hz);
        }
    }
    kept
}

fn tick_label(hz: f64) -> String {
    if hz >= 1000.0 {
        format!("{}k", hz / 1000.0).replace('.', ",")
    } else {
        format!("{hz:.0}")
    }
}

/// Clé qui identifie l'image de spectrogramme déjà calculée.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct OverviewKey {
    start: f64,
    span: f64,
    width: usize,
    /// Change à chaque morceau de spectre reçu.
    revision: u32,
}

impl App {
    /// Les pistes qui attendent encore leur contenu.
    pub fn pending_work(&self) -> Vec<Loading> {
        let Some(media) = &self.media else { return Vec::new() };
        let mut work = Vec::new();
        let mut add = |row, label, progress: Option<u32>| {
            if let Some(progress) = progress {
                work.push(Loading { row, label, progress: known_progress(progress) });
            }
        };
        add(Row::Strip, "Vignettes et keyframes", media.strip_progress);
        add(Row::Sequences, "Détection de la musique", media.detect_progress);
        add(Row::Audio, "Analyse audio", media.signals_progress);
        work
    }

    fn hit_handle(&self, pos: Pos2, body: Rect, row: Row) -> Option<(u64, Edge)> {
        let media = self.media.as_ref()?;
        let (x_of, candidates): (_, Vec<_>) = (
            |t: f64| self.view.x(body.left(), body.width(), t),
            media
                .project
                .segments
                .iter()
                // Hors de la piste Séquences, seules les poignées de la séquence choisie répondent
                .filter(|s| row == Row::Sequences || Some(s.id) == self.selected)
                .collect(),
        );
        let mut best: Option<(u64, Edge, f32)> = None;
        for segment in candidates {
            for (edge, time) in [(Edge::Start, segment.start), (Edge::End, segment.end)] {
                let distance = (x_of(time) - pos.x).abs();
                if distance <= HANDLE_GRAB && best.is_none_or(|b| distance < b.2) {
                    best = Some((segment.id, edge, distance));
                }
            }
        }
        best.map(|b| (b.0, b.1))
    }

    pub fn segment_at(&self, time: f64) -> Option<u64> {
        let media = self.media.as_ref()?;
        media.project.segments.iter().find(|s| s.start <= time && time < s.end).map(|s| s.id)
    }

    fn snap(&self, time: f64, modifiers: egui::Modifiers) -> f64 {
        if modifiers.shift {
            if let Some(strip) = self.media.as_ref().and_then(|m| m.strip.as_ref()) {
                return strip.nearest_keyframe(time).unwrap_or(time);
            }
        }
        time
    }

    /// Glissés et clics sur la timeline.
    fn timeline_input(&mut self, ui: &egui::Ui, body: Rect, rows: &[(Row, f32, f32)], duration: f64) {
        let response = ui.interact(body, ui.scope_id().with("timeline-body"), Sense::click_and_drag());
        let (zoom_left, zoom_width) = zoom_span(body.left(), body.width());
        let row_at = |y: f32| rows.iter().find(|r| y >= r.1 && y < r.2).map(|r| r.0);
        let hover = response.hover_pos();
        let modifiers = ui.input(|i| i.modifiers);

        // Molette : zoom autour du curseur ; Maj + molette : défilement
        if let Some(pos) = hover {
            let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta, i.zoom_delta()));
            if pinch != 1.0 {
                let anchor = self.view.time(body.left(), body.width(), pos.x);
                self.view.zoom_at(anchor, 1.0 / pinch as f64, duration);
                self.view_changed_at = Instant::now();
            } else if scroll.x != 0.0 {
                self.view.start -= scroll.x as f64 / body.width() as f64 * self.view.span;
                self.view.clamp(duration);
                self.view_changed_at = Instant::now();
            } else if scroll.y != 0.0 {
                let anchor = self.view.time(body.left(), body.width(), pos.x);
                self.view.zoom_at(anchor, (-scroll.y as f64 * 0.003).exp(), duration);
                self.view_changed_at = Instant::now();
            }
            if row_at(pos.y) == Some(Row::Overview) {
                let (x0, x1) = zoom_handles(self.view, zoom_left, zoom_width, duration);
                if (pos.x - x0).abs() <= ZOOM_HANDLE_GRAB || (pos.x - x1).abs() <= ZOOM_HANDLE_GRAB {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
                } else if pos.x > x0 && pos.x < x1 {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                }
            } else if let Some(row) = row_at(pos.y) {
                if self.hit_handle(pos, body, row).is_some() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
                }
            }
        }

        // Clic molette : on attrape la vue et on la promène
        if response.dragged_by(egui::PointerButton::Middle) {
            self.view.start -= response.drag_delta().x as f64 / body.width() as f64 * self.view.span;
            self.view.clamp(duration);
            self.view_changed_at = Instant::now();
            self.drag = Drag::Pan;
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
        }
        if response.drag_stopped_by(egui::PointerButton::Middle) {
            self.drag = Drag::None;
        }

        if response.drag_started_by(egui::PointerButton::Primary) {
            let origin = ui.input(|i| i.pointer.press_origin()).unwrap_or(body.left_top());
            let time = self.view.time(body.left(), body.width(), origin.x).clamp(0.0, duration);
            let row = row_at(origin.y);
            self.drag = match row {
                Some(Row::Overview) => {
                    let (x0, x1) = zoom_handles(self.view, zoom_left, zoom_width, duration);
                    let (to_start, to_end) = ((origin.x - x0).abs(), (origin.x - x1).abs());
                    let on_bar = ((origin.x - zoom_left) / zoom_width).clamp(0.0, 1.0) as f64 * duration;
                    if to_start <= ZOOM_HANDLE_GRAB && to_start <= to_end {
                        Drag::ZoomEdge(Edge::Start)
                    } else if to_end <= ZOOM_HANDLE_GRAB {
                        Drag::ZoomEdge(Edge::End)
                    } else if origin.x > x0 && origin.x < x1 {
                        Drag::ZoomMove(on_bar - self.view.start)
                    } else {
                        // Hors de la vue : elle saute à cet endroit, et suit la souris
                        Drag::ZoomMove(self.view.span / 2.0)
                    }
                }
                Some(row) => match self.hit_handle(origin, body, row) {
                    Some((id, edge)) => {
                        self.selected = Some(id);
                        self.scroll_list = true;
                        self.remember();
                        Drag::Handle(id, edge)
                    }
                    None if row == Row::Sequences => match self.segment_at(time) {
                        Some(id) => {
                            self.selected = Some(id);
                            self.scroll_list = true;
                            self.remember();
                            let start = self.media.as_ref().and_then(|m| m.project.segment(id)).map_or(0.0, |s| s.start);
                            Drag::Move(id, time - start)
                        }
                        None => Drag::Create(time),
                    },
                    None => {
                        self.begin_scrub();
                        Drag::Scrub
                    }
                },
                None => Drag::None,
            };
        }

        if let (true, Some(pos)) = (response.dragged_by(egui::PointerButton::Primary), response.interact_pointer_pos()) {
            let time = self.view.time(body.left(), body.width(), pos.x).clamp(0.0, duration);
            match self.drag {
                Drag::ZoomEdge(edge) => {
                    let on_bar = ((pos.x - zoom_left) / zoom_width).clamp(0.0, 1.0) as f64 * duration;
                    match edge {
                        Edge::Start => self.view.set_start(on_bar, duration),
                        Edge::End => self.view.set_end(on_bar, duration),
                    }
                    self.view_changed_at = Instant::now();
                }
                Drag::ZoomMove(grab) => {
                    let on_bar = ((pos.x - zoom_left) / zoom_width).clamp(0.0, 1.0) as f64 * duration;
                    self.view.start = on_bar - grab;
                    self.view.clamp(duration);
                    self.view_changed_at = Instant::now();
                }
                Drag::Scrub => {
                    let exact = body.width() as f64 / self.view.span > 20.0;
                    self.scrub_to(time, exact);
                }
                Drag::Handle(id, edge) => {
                    let time = self.snap(time, modifiers);
                    if let Some(media) = &mut self.media {
                        match edge {
                            Edge::Start => media.project.set_start(id, time),
                            Edge::End => media.project.set_end(id, time),
                        }
                        media.mark_changed();
                    }
                }
                Drag::Move(id, grab) => {
                    if let Some(media) = &mut self.media {
                        media.project.move_to(id, time - grab);
                        media.mark_changed();
                    }
                }
                Drag::Create(anchor) => {
                    if (time - anchor).abs() * body.width() as f64 / self.view.span >= 4.0 {
                        let (low, high) = (anchor.min(time), anchor.max(time));
                        self.remember();
                        let id = self.media.as_mut().and_then(|m| {
                            let id = m.project.add_manual(low, (high).max(low + crate::project::MIN_SEGMENT_S))?;
                            m.mark_changed();
                            Some(id)
                        });
                        if let Some(id) = id {
                            self.selected = Some(id);
                            self.scroll_list = true;
                            let edge = if time >= anchor { Edge::End } else { Edge::Start };
                            self.drag = Drag::Handle(id, edge);
                        } else {
                            self.drag = Drag::None;
                        }
                    }
                }
                Drag::Pan | Drag::None => {}
            }
        }

        if response.drag_stopped_by(egui::PointerButton::Primary) {
            if self.drag == Drag::Scrub {
                self.end_scrub();
            }
            // Une poignée lâchée contre sa voisine : les deux séquences n'en font plus qu'une
            if let Drag::Handle(id, edge) = self.drag {
                let tolerance = MERGE_DISTANCE as f64 * self.view.span / body.width() as f64;
                if let Some(media) = &mut self.media {
                    if media.project.merge_at_edge(id, edge == Edge::End, tolerance) {
                        media.mark_changed();
                    }
                }
            }
            self.drag = Drag::None;
        }

        if response.double_clicked()
            && response.interact_pointer_pos().and_then(|p| row_at(p.y)) == Some(Row::Overview)
        {
            self.view = View { start: 0.0, span: duration };
            self.view_changed_at = Instant::now();
        }

        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                let time = self.view.time(body.left(), body.width(), pos.x).clamp(0.0, duration);
                match row_at(pos.y) {
                    Some(Row::Overview) => {
                        // Un clic sur la barre déplace la vue, pas la tête de lecture
                        let (x0, x1) = zoom_handles(self.view, zoom_left, zoom_width, duration);
                        if pos.x < x0 || pos.x > x1 {
                            let center = ((pos.x - zoom_left) / zoom_width) as f64 * duration;
                            self.view.start = center - self.view.span / 2.0;
                            self.view.clamp(duration);
                            self.view_changed_at = Instant::now();
                        }
                    }
                    Some(Row::Sequences) => {
                        self.selected = self.segment_at(time);
                        self.scroll_list = self.selected.is_some();
                        if self.auto_zoom {
                            self.zoom_on_selected();
                        }
                        self.seek_to(time, true);
                    }
                    Some(_) => self.seek_to(time, true),
                    None => {}
                }
            }
        }
    }

    /// Les canaux de la source : connus dès l'ouverture, sans attendre l'analyse.
    pub fn track_names(&self) -> Vec<String> {
        let Some(media) = &self.media else { return Vec::new() };
        if let Some(signals) = &media.signals {
            return signals.channels.iter().skip(1).map(|c| c.name.clone()).collect();
        }
        media.layout.as_deref().and_then(signals::channel_names).unwrap_or_default().iter().map(|n| n.to_string()).collect()
    }

    /// Pistes à afficher, de haut en bas. Toujours au moins une : le mix si rien d'autre n'existe.
    pub fn lanes(&self, channels: Channels) -> Vec<usize> {
        let count = self.track_names().len() + 1;
        (0..count).filter(|track| channels.is_visible(*track)).collect()
    }

    /// Une fois les pistes affichées servies, prépare les autres, une à la fois, pour qu'elles
    /// soient prêtes quand on les coche.
    fn update_background_spectra(&mut self, ctx: &egui::Context) {
        let generation = self.generation;
        let sender = self.jobs_sender.clone();
        let Some(media) = &mut self.media else { return };
        let Some(count) = media.signals.as_ref().map(|s| s.channels.len()) else { return };
        if let Some(channel) = next_background_spectrum(&media.spectra, count) {
            start_spectrum(media, channel, generation, &sender, ctx);
        }
    }

    /// Prépare les images de spectrogramme de la portion visible.
    fn update_spectrogram(&mut self, ctx: &egui::Context, width: usize) {
        if !self.tracks.audio || self.tracks.view != AudioView::Spectrum || width == 0 {
            return;
        }
        let view = self.view;
        let lanes = self.lanes(self.tracks.audio_channels);
        let limit = if lanes.len() > 1 { DETAIL_MAX_SPAN_SPLIT } else { DETAIL_MAX_SPAN };
        let wants_detail = view.span <= limit;
        let settled = self.view_changed_at.elapsed().as_millis() > 200;
        let generation = self.generation;
        let sender = self.jobs_sender.clone();
        let Some(media) = &mut self.media else { return };
        for channel in lanes {
            // Le spectre d'une piste ne se calcule que lorsqu'elle est affichée
            if !media.spectra.contains_key(&channel) {
                start_spectrum(media, channel, generation, &sender, ctx);
            }
            let Some(state) = media.spectra.get(&channel) else { continue };

            // Vue d'ensemble : recalculée quand la vue change, ou qu'un morceau arrive
            let key = OverviewKey { start: view.start, span: view.span, width, revision: state.revision };
            if media.overviews.get(&channel).is_none_or(|(known, _)| *known != key) {
                let image = overview_image(&state.values, view, width);
                match media.overviews.get_mut(&channel) {
                    Some(slot) => {
                        slot.1.set(image, TextureOptions::LINEAR);
                        slot.0 = key;
                    }
                    None => {
                        let texture = ctx.load_texture(format!("spectre-{channel}"), image, TextureOptions::LINEAR);
                        media.overviews.insert(channel, (key, texture));
                    }
                }
            }

            // Détail : calculé à la demande une fois la vue stabilisée
            if wants_detail && settled && !media.detail_covers(channel, view) && !media.detail_pending.contains(&channel) {
                media.detail_pending.insert(channel);
                let start = (view.start - view.span * 0.25).max(0.0);
                let end = view.end() + view.span * 0.25;
                let columns = ((width as f64 * 1.5) as usize).min(MAX_DETAIL_COLUMNS);
                let path = media.path.clone();
                let layout = media.layout.clone().unwrap_or_default();
                let track = media.project.audio_track;
                let sender = sender.clone();
                let ctx = ctx.clone();
                std::thread::spawn(move || {
                    let result = signals::detail_spectrogram(&path, &layout, channel, start, end, columns, track);
                    let _ = sender.send((generation, Job::Detail(channel, result)));
                    ctx.request_repaint();
                });
            }
        }
    }

    /// Mesure ce qui manque : la vidéo pour les canaux affichés sur la timeline, chaque séquence
    /// pour son tableau. Une séquence n'est remesurée que si ses bornes ont changé.
    fn update_measures(&mut self) {
        // Le mix toujours : la fiche de la vidéo l'affiche, quel que soit le canal à l'écran
        let mut lanes = self.lanes(self.tracks.audio_channels);
        lanes.push(0);
        let Some(media) = &mut self.media else { return };
        if let Some(track) = media.class_track.clone() {
            let segments = &media.project.segments;
            media.sequence_tags.retain(|id, _| segments.iter().any(|s| s.id == *id));
            for segment in segments {
                let bounds = (segment.start, segment.end);
                if media.sequence_tags.get(&segment.id).is_none_or(|(known, _)| *known != bounds) {
                    media.sequence_tags.insert(segment.id, (bounds, track.describe(segment.start, segment.end)));
                }
            }
        }
        let Some(signals) = media.signals.clone() else { return };
        for channel in lanes {
            let Some(data) = signals.channels.get(channel) else { continue };
            media.film_loudness.entry(channel).or_insert_with(|| crate::FilmLoudness {
                measure: data.measure(0.0, f64::INFINITY),
                running: data.running_integrated(),
            });
        }
        let Some(program) = signals.channels.first() else { return };
        let segments = &media.project.segments;
        media.sequence_loudness.retain(|id, _| segments.iter().any(|s| s.id == *id));
        for segment in segments {
            let bounds = (segment.start, segment.end);
            if media.sequence_loudness.get(&segment.id).is_none_or(|(known, _)| *known != bounds) {
                media.sequence_loudness.insert(segment.id, (bounds, program.measure(segment.start, segment.end)));
            }
        }
    }

    pub fn timeline(&mut self, ui: &mut egui::Ui) {
        let full = ui.available_rect_before_wrap();
        ui.allocate_rect(full, Sense::hover());
        let duration = self.duration();
        let time = ui.input(|i| i.time);
        if self.media.is_none() || duration <= 0.0 {
            ui.painter().rect_filled(full, 0.0, BG);
            if self.media.is_some() {
                paint_loading(ui.painter(), full, "Lecture de la vidéo", None, time);
                ui.ctx().request_repaint_after(LOADING_REFRESH);
            } else {
                ui.painter().text(
                    full.center(),
                    Align2::CENTER_CENTER,
                    "La timeline apparaîtra ici",
                    FontId::proportional(14.0),
                    WEAK,
                );
            }
            return;
        }
        self.view.clamp(duration);
        // Le mixeur prend sa colonne sur toute la hauteur : les pistes restent alignées dans le temps
        let mixer_width = self.mixer_width();
        let mixer = Rect::from_min_max(Pos2::new(full.right() - mixer_width, full.top()), full.max);
        let full = Rect::from_min_max(full.min, Pos2::new(full.right() - mixer_width, full.bottom()));
        if mixer_width > 0.0 {
            self.mixer(ui, mixer);
        }
        let rows = layout_rows(full.top(), full.height(), self.tracks);
        let last = rows.last().map_or(full.bottom(), |r| r.2).min(full.bottom());
        let body = Rect::from_min_max(Pos2::new(full.left() + HEADER_WIDTH, full.top()), Pos2::new(full.right(), last));

        self.timeline_body = Rect::from_min_max(body.min, Pos2::new(body.right(), full.bottom()));
        self.timeline_input(ui, body, &rows, duration);
        let pixels = ui.ctx().pixels_per_point();
        self.update_spectrogram(ui.ctx(), (body.width() * pixels) as usize);
        self.update_background_spectra(ui.ctx());
        self.update_measures();

        let painter = ui.painter_at(Rect::from_min_max(full.min, Pos2::new(full.right(), last)));
        painter.rect_filled(full, 0.0, BG);
        // Ce qui suit le temps s'arrête au bord des pistes : au zoom, une séquence commencée
        // avant la vue ne doit pas déborder sur les titres
        let tracks = painter.with_clip_rect(Rect::from_min_max(body.min, Pos2::new(body.right(), last)));
        let view = self.view;
        let row_rect = |top: f32, bottom: f32| {
            Rect::from_min_max(Pos2::new(body.left(), top), Pos2::new(body.right(), bottom))
        };
        let position = self.position();

        for &(row, top, bottom) in &rows {
            let rect = row_rect(top, bottom);
            match row {
                Row::Overview => self.paint_overview(&tracks, rect, duration, position),
                Row::Ruler => paint_ruler(&tracks, rect, view),
                Row::Strip => {
                    self.paint_strip(&tracks, rect);
                    self.paint_chapters(&tracks, rect);
                }
                Row::Sequences => self.paint_sequences(&tracks, rect),
                Row::Audio => self.paint_audio(&tracks, rect),
            }
            // La barre de zoom et la règle partagent leur en-tête : pas de trait entre elles
            let left = if row == Row::Overview { body.left() } else { full.left() };
            painter.line_segment([Pos2::new(left, bottom), Pos2::new(full.right(), bottom)], Stroke::new(1.0, ROW_LINE));
        }

        // Teinte des séquences et poignées de la séquence choisie, au travers des analyses
        let analysis_top = rows.iter().find(|r| !matches!(r.0, Row::Overview | Row::Ruler | Row::Strip | Row::Sequences)).map(|r| r.1);
        if let (Some(media), Some(top)) = (&self.media, analysis_top) {
            for segment in &media.project.segments {
                if segment.end < view.start || segment.start > view.end() {
                    continue;
                }
                let x0 = view.x(body.left(), body.width(), segment.start);
                let x1 = view.x(body.left(), body.width(), segment.end);
                let color = segment_color(segment.origin);
                if Some(segment.id) == self.selected {
                    for x in [x0, x1] {
                        tracks.line_segment([Pos2::new(x, top), Pos2::new(x, last)], Stroke::new(1.5, Color32::WHITE));
                    }
                }
                let band = Rect::from_min_max(Pos2::new(x0, top), Pos2::new(x1, last));
                tracks.rect_filled(band, 0.0, color.gamma_multiply(0.09));
            }
        }

        // Ce qui charge, à la place de ce qui va apparaître
        let work = self.pending_work();
        for loading in &work {
            if let Some(&(_, top, bottom)) = rows.iter().find(|r| r.0 == loading.row) {
                paint_track_loading(&tracks, row_rect(top, bottom), loading.label, loading.progress, time, 0.0);
            }
        }
        let detail_running = self.media.as_ref().is_some_and(|m| !m.detail_pending.is_empty());
        if !work.is_empty() || (detail_running && self.tracks.view == AudioView::Spectrum) {
            ui.ctx().request_repaint_after(LOADING_REFRESH);
        }
        // Rien ne viendra : autant le dire plutôt que laisser une piste vide
        if let Some(&(_, top, bottom)) = rows.iter().find(|r| r.0 == Row::Audio) {
            let idle = self.media.as_ref().is_some_and(|m| m.signals.is_none() && m.signals_progress.is_none());
            if idle {
                tracks.text(
                    row_rect(top, bottom).center(),
                    Align2::CENTER_CENTER,
                    "Analyse audio indisponible pour ce fichier",
                    FontId::proportional(13.0),
                    WEAK,
                );
            }
        }

        // Tête de lecture et repère de survol
        let ruler_top = rows.iter().find(|r| r.0 == Row::Ruler).map_or(body.top(), |r| r.1);
        let head = view.x(body.left(), body.width(), position);
        if head >= body.left() && head <= body.right() {
            painter.line_segment([Pos2::new(head, ruler_top), Pos2::new(head, last)], Stroke::new(1.5, PLAYHEAD));
            painter.add(Shape::convex_polygon(
                vec![
                    Pos2::new(head - 5.0, ruler_top),
                    Pos2::new(head + 5.0, ruler_top),
                    Pos2::new(head, ruler_top + 8.0),
                ],
                PLAYHEAD,
                Stroke::NONE,
            ));
        }
        if let Some(pos) = ui.ctx().pointer_hover_pos().filter(|p| body.contains(*p)) {
            let time = view.time(body.left(), body.width(), pos.x);
            painter.line_segment(
                [Pos2::new(pos.x, ruler_top), Pos2::new(pos.x, last)],
                Stroke::new(1.0, Color32::from_white_alpha(70)),
            );
            let mut text = tc(time, self.fps());
            if let Some(row) = rows
                .iter()
                .find(|r| r.0 == Row::Audio && self.tracks.view == AudioView::Spectrum && pos.y >= r.1 && pos.y < r.2)
            {
                let lanes = self.lanes(self.tracks.audio_channels).len();
                // Aucune piste affichée : pas de fréquence à lire
                if lanes > 0 {
                    let height = (row.2 - row.1) / lanes as f32;
                    let lane = (((pos.y - row.1) / height) as usize).min(lanes - 1) as f32;
                    let band = row_rect(row.1 + lane * height, row.1 + (lane + 1.0) * height);
                    text = format!("{text}  ·  {}", frequency_label(frequency_at(band, pos.y)));
                }
            }
            let on_image = rows.iter().any(|r| r.0 == Row::Strip && pos.y >= r.1 && pos.y < r.2);
            if let Some(chapter) = self.chapter_at(time).filter(|_| on_image) {
                text = format!("{text}  ·  {}", chapter.title);
            }
            let galley = painter.layout_no_wrap(text, FontId::monospace(11.0), Color32::WHITE);
            let at = Pos2::new((pos.x + 8.0).min(body.right() - galley.size().x - 6.0), ruler_top + 4.0);
            painter.rect_filled(Rect::from_min_size(at, galley.size()).expand(2.0), 2.0, Color32::from_black_alpha(190));
            painter.galley(at, galley, Color32::WHITE);
        }

        // Noms et réglages des pistes
        self.headers(ui, full, &rows);
    }

    fn headers(&mut self, ui: &mut egui::Ui, full: Rect, rows: &[(Row, f32, f32)]) {
        let bottom = rows.last().map_or(full.bottom(), |r| r.2);
        let column = Rect::from_min_max(full.min, Pos2::new(full.left() + HEADER_WIDTH, bottom));
        for &(row, top, bottom) in rows {
            let section = Rect::from_min_max(Pos2::new(column.left(), top), Pos2::new(column.right(), bottom));
            match row {
                // La barre de zoom et la règle partagent un seul en-tête
                Row::Overview => continue,
                Row::Ruler => {
                    let top = rows.iter().find(|r| r.0 == Row::Overview).map_or(top, |r| r.1);
                    self.zoom_header(ui, Rect::from_min_max(Pos2::new(column.left(), top), section.max));
                }
                Row::Strip => {
                    section_title(ui, Pos2::new(section.left() + PAD, section.center().y), "Image");
                }
                Row::Sequences => self.sequences_header(ui, section),
                Row::Audio => self.audio_header(ui, section),
            }
            ui.painter().line_segment(
                [Pos2::new(column.left(), bottom), Pos2::new(column.right(), bottom)],
                Stroke::new(2.0, SECTION_LINE),
            );
        }
        ui.painter().line_segment([column.right_top(), column.right_bottom()], Stroke::new(1.0, ROW_LINE));
    }

    fn zoom_header(&mut self, ui: &mut egui::Ui, rect: Rect) {
        let center = rect.center().y;
        let label = Rect::from_min_max(Pos2::new(rect.left() + PAD, center - 9.0), Pos2::new(rect.left() + 48.0, center + 9.0));
        put_left(ui, label, egui::Label::new(egui::RichText::new("Zoom").size(FONT_SIZE).color(NAME)).selectable(false))
            .on_hover_text("Glisser une poignée : zoomer. Glisser la barre : se déplacer. Double-clic : toute la vidéo.");
        let buttons = Rect::from_min_max(
            Pos2::new(rect.left() + 52.0, center - CONTROL_HEIGHT / 2.0),
            Pos2::new(rect.right() - PAD, center + CONTROL_HEIGHT / 2.0),
        );
        let items = [
            ("Auto", self.auto_zoom, "Zoom auto : une séquence choisie remplit la timeline"),
            ("Reset", false, "Toute la vidéo (F)"),
        ];
        match segmented(ui, buttons, &items, 4.0, "zoom") {
            Some(0) => {
                self.auto_zoom = !self.auto_zoom;
                if self.auto_zoom {
                    self.zoom_on_selected();
                }
            }
            Some(1) => {
                self.view = View { start: 0.0, span: self.duration() };
                self.view_changed_at = Instant::now();
            }
            _ => {}
        }
    }

    /// Titre, affichage de la courbe de détection et ses réglages, sensibilité.
    fn sequences_header(&mut self, ui: &mut egui::Ui, rect: Rect) {
        let inner = rect.shrink2(egui::vec2(PAD, 0.0));
        section_title(ui, Pos2::new(inner.left(), rect.top() + 15.0), "Séquences").on_hover_text(
            "Courbe : probabilité de musique. Ligne : seuil de détection. Glisser sur le vide : créer une séquence.",
        );
        let row = Rect::from_min_size(Pos2::new(inner.left(), rect.top() + 27.0), egui::vec2(inner.width(), 20.0));
        let detection = self.media.as_ref().is_some_and(|m| !m.project.detection_off);
        let help = "Détection automatique de la musique. Coupée : ses séquences sont mises de côté (les séquences \
                    retouchées ou ajoutées à la main restent) et un calcul en cours s'arrête.";
        if switch_row(ui, row, "Détection", detection, help).clicked() {
            let ctx = ui.ctx().clone();
            self.set_detection(&ctx, !detection);
        }
        // L'engrenage, juste après le mot : gap, anticipation et post
        let word = ui.painter().layout_no_wrap("Détection".into(), FontId::proportional(FONT_SIZE), NAME).size().x;
        let gear = Rect::from_center_size(Pos2::new(row.left() + word + 11.0, row.center().y), egui::vec2(16.0, 16.0));
        let gear_response = ui
            .interact(gear, ui.scope_id().with("detection-settings"), Sense::click())
            .on_hover_text("Gap, anticipation et post");
        gear_response.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, "Réglages de la détection"));
        paint_gear(ui.painter(), gear.center(), if gear_response.hovered() { Color32::WHITE } else { NAME });
        egui::Popup::from_toggle_button_response(&gear_response)
            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
            .show(|ui| self.detection_controls(ui));

        let slider = Rect::from_min_size(Pos2::new(inner.left(), row.bottom() + 6.0), egui::vec2(inner.width(), 28.0));
        self.sensitivity_box(ui, slider);
    }

    /// La sensibilité, dans le même composant que les réglages de l'engrenage.
    fn sensitivity_box(&mut self, ui: &mut egui::Ui, rect: Rect) {
        let Some(media) = &self.media else { return };
        let enabled = !media.project.scores.is_empty() && !media.project.detection_off;
        let mut settings = media.project.settings.clone();
        let range = detect::MIN_SENSITIVITY..=detect::MAX_SENSITIVITY;
        let response = slider_box(ui, rect, "Sensibilité", &mut settings.sensitivity, range, enabled, crate::SENSITIVITY_HELP);
        let released = response.drag_stopped() || response.clicked();
        self.apply_detection_settings(settings, response.changed(), released);
    }

    /// Le tableau des pistes (afficher, solo, mute), la loudness et ses cibles, la vue.
    fn audio_header(&mut self, ui: &mut egui::Ui, rect: Rect) {
        let inner = rect.shrink2(egui::vec2(PAD, 0.0));
        section_title(ui, Pos2::new(inner.left(), rect.top() + 15.0), "Audio").on_hover_text(
            "Loudness mesurée en BS.1770 (LKFS), sur toute la vidéo. Bleu : intégrée depuis le début, rouge hors \
             de la cible. Blanc : short-term 3 s. Gris : momentary 400 ms. Sa graduation (-60 à 0) lui est propre.",
        );

        // Tout en bas, quelle que soit la hauteur : la vue, et juste au-dessus ce qui ne vaut que pour l'onde
        let selector = Rect::from_min_max(
            Pos2::new(inner.left(), rect.bottom() - PAD - CONTROL_HEIGHT),
            Pos2::new(inner.right(), rect.bottom() - PAD),
        );
        let items = [
            ("Spectre", self.tracks.view == AudioView::Spectrum, "Afficher le spectre (V)"),
            ("Onde", self.tracks.view == AudioView::Waveform, "Afficher l'onde (V)"),
        ];
        match segmented(ui, selector, &items, 0.0, "audio-view") {
            Some(0) => self.tracks.view = AudioView::Spectrum,
            Some(1) => self.tracks.view = AudioView::Waveform,
            _ => {}
        }
        let mut floor = selector.top() - 8.0;
        if self.tracks.view == AudioView::Waveform {
            let row = Rect::from_min_max(Pos2::new(inner.left(), selector.top() - 24.0), Pos2::new(inner.right(), selector.top() - 4.0));
            let help = "Amplifie l'onde pour que la crête du mix remplisse la hauteur. Le même gain s'applique à chaque piste : les écarts entre elles restent justes.";
            if switch_row(ui, row, "Normaliser", self.waveform_normalized, help).clicked() {
                self.waveform_normalized = !self.waveform_normalized;
            }
            floor = row.top() - 8.0;
        }

        let names = self.track_names();
        let tracks: Vec<&str> = std::iter::once("Mix").chain(names.iter().map(String::as_str)).collect();
        let table_top = rect.top() + 30.0;
        let room = floor - table_top - SWITCH_GAP - SWITCH_HEIGHT;
        let row_height = (room / tracks.len() as f32).clamp(15.0, CELL_HEIGHT);
        let table = Rect::from_min_size(Pos2::new(inner.left(), table_top), egui::vec2(inner.width(), row_height * tracks.len() as f32));
        self.track_table(ui, table, &tracks, row_height);

        let loudness = Rect::from_min_size(Pos2::new(inner.left(), table.bottom() + SWITCH_GAP), egui::vec2(inner.width(), SWITCH_HEIGHT));
        let help = "Affiche ou masque les courbes de loudness sur le spectre et les repères du mixeur";
        if switch_row(ui, loudness, "Loudness", self.tracks.loudness, help).clicked() {
            self.tracks.loudness = !self.tracks.loudness;
        }
        let targets_top = loudness.bottom() + 4.0;
        if floor - targets_top >= TARGET_LABEL_HEIGHT + 3.0 * TARGET_ROW_HEIGHT {
            self.target_controls(ui, Rect::from_min_size(Pos2::new(inner.left(), targets_top), egui::vec2(inner.width(), 0.0)));
        }
    }

    /// Une ligne par piste : nom et œil pour l'afficher, puis solo et mute du canal.
    fn track_table(&mut self, ui: &mut egui::Ui, table: Rect, tracks: &[&str], row_height: f32) {
        let painter = ui.painter().clone();
        let eye_left = table.right() - TOGGLE_CELL;
        let mute_left = eye_left - TOGGLE_CELL;
        let solo_left = mute_left - TOGGLE_CELL;
        painter.rect_stroke(table, 3.0, Stroke::new(1.0, CELL_LINE), egui::StrokeKind::Inside);
        for x in [solo_left, mute_left, eye_left] {
            painter.line_segment([Pos2::new(x, table.top()), Pos2::new(x, table.bottom())], Stroke::new(1.0, CELL_LINE));
        }
        for (track, &name) in tracks.iter().enumerate() {
            let top = table.top() + track as f32 * row_height;
            let row = Rect::from_min_max(Pos2::new(table.left(), top), Pos2::new(table.right(), top + row_height));
            if track > 0 {
                painter.line_segment([row.left_top(), row.right_top()], Stroke::new(1.0, CELL_LINE));
            }
            let full_name = track_full_name(name);
            let visible = self.tracks.audio_channels.is_visible(track);
            // Le nom et l'œil affichent ou masquent la piste
            let sense = Sense::click();
            let help = "Afficher ou masquer cette piste";
            let name_cell = Rect::from_min_max(row.min, Pos2::new(solo_left, row.bottom()));
            let eye_cell = Rect::from_min_max(Pos2::new(eye_left, row.top()), row.max);
            let response = ui.interact(name_cell, ui.scope_id().with(("track-visible", track)), sense).on_hover_text(help);
            response.widget_info(|| egui::WidgetInfo::selected(egui::Role::CheckBox, true, visible, full_name));
            let eye_response = ui.interact(eye_cell, ui.scope_id().with(("track-eye", track)), sense).on_hover_text(help);
            let eye_label = format!("Afficher {full_name}");
            eye_response.widget_info(|| egui::WidgetInfo::selected(egui::Role::CheckBox, true, visible, &eye_label));
            for (cell, hovered) in [(name_cell, response.hovered()), (eye_cell, eye_response.hovered())] {
                if hovered {
                    painter.rect_filled(cell.shrink(1.0), 2.0, Color32::from_white_alpha(8));
                }
            }
            painter.text(
                Pos2::new(row.left() + 10.0, row.center().y),
                Align2::LEFT_CENTER,
                full_name,
                FontId::proportional(FONT_SIZE),
                NAME,
            );
            let eye = Pos2::new(eye_left + TOGGLE_CELL / 2.0, row.center().y);
            paint_eye(&painter, eye, if visible { Color32::WHITE } else { Color32::from_gray(105) });
            if response.clicked() || eye_response.clicked() {
                self.tracks.audio_channels.set(track, !visible);
            }

            for (left, letter, color) in [(solo_left, "S", SOLO), (mute_left, "M", MUTE)] {
                let cell = Rect::from_min_max(Pos2::new(left, row.top()), Pos2::new(left + TOGGLE_CELL, row.bottom()));
                // Le mix n'est pas un canal : il n'a ni solo ni mute
                if track == 0 {
                    continue;
                }
                let source = track - 1;
                let on = if letter == "S" { self.monitor.is_solo(source) } else { self.monitor.is_muted(source) };
                let response = ui
                    .interact(cell, ui.scope_id().with((letter, source)), Sense::click())
                    .on_hover_text(format!("{} {name}", if letter == "S" { "Solo" } else { "Mute" }));
                response.widget_info(|| egui::WidgetInfo::selected(egui::Role::Button, true, on, letter));
                if on {
                    painter.rect_filled(cell.shrink(1.5), 2.0, color);
                } else if response.hovered() {
                    painter.rect_filled(cell.shrink(1.0), 2.0, Color32::from_white_alpha(10));
                }
                let text = if on { Color32::BLACK } else { NAME };
                painter.text(cell.center(), Align2::CENTER_CENTER, letter, FontId::proportional(FONT_SIZE), text);
                if response.clicked() {
                    if letter == "S" {
                        self.monitor.toggle_solo(source);
                    } else {
                        self.monitor.toggle_mute(source);
                    }
                }
            }
        }
    }

    /// Les cibles : la courbe et les valeurs passent au rouge en dehors.
    fn target_controls(&mut self, ui: &mut egui::Ui, area: Rect) {
        let enabled = self.tracks.loudness;
        let Some(media) = &mut self.media else { return };
        let mut target = media.project.loudness_target;
        let label = Rect::from_min_size(area.min, egui::vec2(area.width(), TARGET_LABEL_HEIGHT));
        put_left(ui, label, egui::Label::new(egui::RichText::new("Cible").size(FONT_SIZE).color(WEAK)).selectable(false));
        let table = Rect::from_min_size(Pos2::new(area.left(), label.bottom()), egui::vec2(area.width(), 3.0 * TARGET_ROW_HEIGHT));
        let painter = ui.painter().clone();
        painter.rect_stroke(table, 3.0, Stroke::new(1.0, CELL_LINE), egui::StrokeKind::Inside);
        let value_left = table.right() - TARGET_VALUE_WIDTH;
        let rows: [(&str, &str, &mut f32, std::ops::RangeInclusive<f32>); 3] = [
            ("True peak (dB)", "Crête vraie à ne pas dépasser, en dBTP", &mut target.true_peak, -24.0..=0.0),
            ("Intégrée (LKFS)", "Loudness intégrée visée, en LKFS", &mut target.integrated, -50.0..=-5.0),
            ("Tolérance (LU)", "Écart accepté autour de l'intégrée, en LU", &mut target.tolerance, 0.1..=10.0),
        ];
        for (index, (name, help, value, range)) in rows.into_iter().enumerate() {
            let top = table.top() + index as f32 * TARGET_ROW_HEIGHT;
            let row = Rect::from_min_max(Pos2::new(table.left(), top), Pos2::new(table.right(), top + TARGET_ROW_HEIGHT));
            if index > 0 {
                painter.line_segment([row.left_top(), row.right_top()], Stroke::new(1.0, CELL_LINE));
            }
            let cell = Rect::from_min_max(Pos2::new(value_left, row.top()), row.max).shrink(1.0);
            painter.rect_filled(cell, 2.0, CELL_FILL);
            let name_rect = Rect::from_min_max(Pos2::new(row.left() + 10.0, row.top()), Pos2::new(value_left - 4.0, row.bottom()));
            let color = if enabled { NAME } else { WEAK };
            let label = put_left(ui, name_rect, egui::Label::new(egui::RichText::new(name).size(FONT_SIZE).color(color)).selectable(false).truncate());
            let mut cell_ui = ui.new_child(egui::UiBuilder::new().max_rect(cell));
            cell_ui.add_enabled_ui(enabled, |ui| {
                let visuals = &mut ui.visuals_mut().widgets;
                for state in [&mut visuals.inactive, &mut visuals.hovered, &mut visuals.active] {
                    state.weak_bg_fill = CELL_FILL;
                    state.bg_fill = CELL_FILL;
                    state.bg_stroke = Stroke::NONE;
                    state.corner_radius = egui::CornerRadius::same(2);
                }
                ui.style_mut().override_font_id = Some(FontId::proportional(FONT_SIZE));
                ui.spacing_mut().button_padding = egui::vec2(1.0, 0.0);
                ui.add_sized(cell.size(), egui::DragValue::new(value).range(range).speed(0.1).fixed_decimals(1))
                    .labelled_by(label.id)
                    .on_hover_text(help);
            });
        }
        if target != media.project.loudness_target {
            media.project.loudness_target = target;
            media.mark_changed();
        }
    }

    /// Les mesures de la vidéo entier, sur le mix complet : ce sont des données de la fiche.
    pub fn loudness_section(&mut self, ui: &mut egui::Ui) {
        let Some(media) = &self.media else { return };
        egui::CollapsingHeader::new(egui::RichText::new("Loudness de la vidéo").strong()).default_open(true).show(ui, |ui| {
            let Some(film) = media.film_loudness.get(&0) else {
                ui.weak("Disponible après l'analyse audio.");
                return;
            };
            let measure = &film.measure;
            let rows = [
                ("True peak", "dBTP", loudness_text(measure.true_peak), Color32::WHITE),
                ("Intégrée", "LKFS", loudness_text(measure.integrated), Color32::WHITE),
                ("Short-term max", "LKFS", loudness_text(measure.max_short_term), Color32::WHITE),
                ("Momentary max", "LKFS", loudness_text(measure.max_momentary), Color32::WHITE),
                ("LRA", "LU", format!("{:.1}", measure.range), Color32::WHITE),
            ];
            egui::Grid::new("film-loudness").num_columns(3).spacing([8.0, 3.0]).show(ui, |ui| {
                for (name, unit, value, color) in rows {
                    ui.weak(name);
                    ui.label(egui::RichText::new(value).monospace().size(14.0).color(color));
                    ui.weak(unit);
                    ui.end_row();
                }
            });
        });
    }

    /// La barre de zoom : toute la largeur représente la vidéo, la zone claire ce qu'on voit.
    fn paint_overview(&self, painter: &egui::Painter, rect: Rect, duration: f64, position: f64) {
        let (left, width) = zoom_span(rect.left(), rect.width());
        let bar = Rect::from_min_size(Pos2::new(left, rect.top() + 2.0), egui::vec2(width, rect.height() - 4.0));
        painter.rect_filled(bar, 3.0, Color32::from_rgb(30, 33, 38));
        let x = |t: f64| left + (t / duration) as f32 * width;
        let (x0, x1) = zoom_handles(self.view, left, width, duration);
        let window = Rect::from_min_max(Pos2::new(x0, bar.top()), Pos2::new(x1, bar.bottom()));
        painter.rect_filled(window, 3.0, Color32::from_rgb(88, 92, 100));
        // Les séquences, pour se repérer dans la vidéo
        if let Some(media) = &self.media {
            for segment in &media.project.segments {
                let block = Rect::from_min_max(
                    Pos2::new(x(segment.start), bar.bottom() - 4.0),
                    Pos2::new(x(segment.end).max(x(segment.start) + 1.0), bar.bottom() - 1.0),
                );
                painter.rect_filled(block, 0.0, segment_color(segment.origin));
            }
        }
        let head = x(position);
        painter.line_segment([Pos2::new(head, bar.top()), Pos2::new(head, bar.bottom())], Stroke::new(1.0, PLAYHEAD));
        // Une poignée de chaque côté
        for handle in [x0, x1] {
            let shape = Rect::from_center_size(Pos2::new(handle, bar.center().y), egui::vec2(9.0, bar.height()));
            painter.rect_filled(shape, 4.0, ZOOM_BLUE);
        }
    }

    fn paint_strip(&self, painter: &egui::Painter, rect: Rect) {
        let Some(media) = &self.media else { return };
        let (Some(strip), Some(texture)) = (&media.strip, &media.strip_texture) else { return };
        if strip.count == 0 {
            return;
        }
        let slot = rect.height() * 16.0 / 9.0;
        let mut mesh = Mesh::with_texture(texture.id());
        let rows = strip.count.div_ceil(crate::ATLAS_COLUMNS);
        let mut x = rect.left();
        while x < rect.right() {
            let center = self.view.time(rect.left(), rect.width(), x + slot / 2.0);
            let index = ((center.max(0.0) / strip.interval) as usize).min(strip.count - 1);
            let (column, row) = (index % crate::ATLAS_COLUMNS, index / crate::ATLAS_COLUMNS);
            let uv = Rect::from_min_max(
                Pos2::new(column as f32 / crate::ATLAS_COLUMNS as f32, row as f32 / rows as f32),
                Pos2::new((column + 1) as f32 / crate::ATLAS_COLUMNS as f32, (row + 1) as f32 / rows as f32),
            );
            let tile = Rect::from_min_size(Pos2::new(x, rect.top()), egui::vec2(slot, rect.height()));
            mesh.add_rect_with_uv(tile, uv, Color32::WHITE);
            x += slot;
        }
        painter.add(mesh);
    }

    fn mixer_width(&self) -> f32 {
        if self.media.as_ref().is_none_or(|m| m.signals.is_none() && m.layout.is_none()) {
            return 0.0;
        }
        let tracks = self.track_names().len() as f32;
        tracks * MIXER_TRACK_WIDTH + MIXER_SCALE_WIDTH + 2.0 * MIXER_MASTER_BAR + 2.0 + 2.0 * MIXER_PADDING
    }

    /// Niveaux en dB à la tête de lecture : ce qu'on entend de chaque piste, puis le master stéréo.
    pub fn mixer_levels(&self) -> Option<(Vec<f32>, [f32; 2])> {
        let position = self.position();
        self.mixer_values(|channel| {
            channel
                .peaks
                .range((position - METER_WINDOW_S).max(0.0), position + 0.005)
                .map_or(0.0, |(low, high)| low.abs().max(high.abs()))
        })
    }

    /// Loudness en LUFS à la tête de lecture, (short-term, momentary) : chaque piste, puis le mix.
    /// Le mix suit la somme BS.1770 de ce qu'on entend : solos, mutes et faders compris.
    pub fn mixer_loudness(&self) -> Option<(Vec<(f32, f32)>, (f32, f32))> {
        let signals = self.media.as_ref()?.signals.as_ref()?;
        let end = self.position() / BLOCK_S;
        let to_level = |energy: f64| signals::to_lufs(energy).max(METER_FLOOR);
        let windows = [SHORT_TERM_BLOCKS, MOMENTARY_BLOCKS];
        let master = (self.monitor.master_gain() as f64).powi(2);
        let tracks = &signals.channels[1..];
        if tracks.is_empty() {
            let [short, momentary] = windows.map(|width| to_level(sliding_power(&signals.channels[0].energies, end, width) as f64 * master));
            return Some((Vec::new(), (short, momentary)));
        }
        let energy_of = |index: usize, width: f64| -> f64 {
            if !self.monitor.audible(index) {
                return 0.0;
            }
            sliding_power(&tracks[index].energies, end, width) as f64 * (self.monitor.gain(index) as f64).powi(2)
        };
        let shown = (0..tracks.len()).map(|index| (to_level(energy_of(index, windows[0])), to_level(energy_of(index, windows[1])))).collect();
        let [short, momentary] = windows.map(|width| {
            let sum: f64 = tracks.iter().enumerate().map(|(index, track)| signals::loudness_weight(&track.name) * energy_of(index, width)).sum();
            to_level(sum * master)
        });
        Some((shown, (short, momentary)))
    }

    /// Ce qu'on entend de chaque piste, puis le master stéréo, pour une mesure donnée.
    fn mixer_values(&self, level: impl Fn(&signals::Channel) -> f32) -> Option<(Vec<f32>, [f32; 2])> {
        let signals = self.media.as_ref()?.signals.as_ref()?;
        let to_db = |value: f32| if value > 0.0 { (20.0 * value.log10()).max(METER_FLOOR) } else { METER_FLOOR };
        let tracks = &signals.channels[1..];
        if tracks.is_empty() {
            // Source mono : le mix est son unique canal
            let db = to_db(level(&signals.channels[0]) * self.monitor.master_gain());
            return Some((Vec::new(), [db, db]));
        }
        let names: Vec<String> = tracks.iter().map(|c| c.name.clone()).collect();
        // Après le fader : la barre montre ce qui sort de la piste
        let levels: Vec<f32> = tracks.iter().enumerate().map(|(index, channel)| level(channel) * self.monitor.gain(index)).collect();
        let (left, right) = self.monitor.master_levels(&names, &levels);
        let (left, right) = (left * self.monitor.master_gain(), right * self.monitor.master_gain());
        let shown = levels
            .iter()
            .enumerate()
            .map(|(index, value)| if self.monitor.audible(index) { to_db(*value) } else { METER_FLOOR })
            .collect();
        Some((shown, [to_db(left), to_db(right)]))
    }

    /// Le mixeur : une barre et ses boutons S / M par piste, deux barres pour le master.
    fn mixer(&mut self, ui: &mut egui::Ui, rect: Rect) {
        // Avant la fin de l'analyse : le mixeur est là et réglable, ses barres attendent
        let silent = self.track_names().len();
        let (tracks, master) = self.mixer_levels().unwrap_or((vec![METER_FLOOR; silent], [METER_FLOOR; 2]));
        // Montée immédiate, descente progressive : les crêtes restent lisibles
        let targets: Vec<f32> = tracks.iter().copied().chain(master).collect();
        if self.mixer_display.len() != targets.len() {
            self.mixer_display = targets.clone();
        }
        let (loud_tracks, loud_mix) =
            self.mixer_loudness().unwrap_or((vec![(METER_FLOOR, METER_FLOOR); silent], (METER_FLOOR, METER_FLOOR)));
        let loud_targets: Vec<(f32, f32)> = loud_tracks.iter().copied().chain([loud_mix, loud_mix]).collect();
        if self.mixer_loudness_display.len() != loud_targets.len() {
            self.mixer_loudness_display = loud_targets.clone();
        }
        let dt = ui.input(|i| i.stable_dt).min(0.1);
        let mut settling = false;
        for (shown, target) in self.mixer_display.iter_mut().zip(&targets) {
            *shown = target.max(*shown - METER_FALL_DB_PER_S * dt);
            settling |= *shown > *target;
        }
        for (shown, target) in self.mixer_loudness_display.iter_mut().zip(&loud_targets) {
            *shown = (smooth_meter(shown.0, target.0, dt), smooth_meter(shown.1, target.1, dt));
            settling |= (shown.0 - target.0).abs() > 0.05 || (shown.1 - target.1).abs() > 0.05;
        }
        if settling || !self.paused() {
            ui.ctx().request_repaint_after(LOADING_REFRESH);
        }

        let painter = ui.painter().with_clip_rect(rect);
        painter.rect_filled(rect, 0.0, BG);
        painter.line_segment([rect.left_top(), rect.left_bottom()], Stroke::new(1.0, ROW_LINE));
        let names = self.track_names();
        let show_loudness = self.tracks.loudness;
        let top = rect.top() + 20.0;
        let bars_bottom = rect.bottom() - 68.0;
        // Les noms, sous les barres et juste au-dessus des boutons
        let label_y = bars_bottom + 3.0;
        let small = FontId::proportional(10.0);

        // Les pistes
        let mut x = rect.left() + MIXER_PADDING;
        for (index, name) in names.iter().enumerate() {
            let center = x + MIXER_TRACK_WIDTH / 2.0;
            painter.text(Pos2::new(center, label_y), Align2::CENTER_TOP, name, small.clone(), TEXT);
            let bar = Rect::from_min_max(
                Pos2::new(center - MIXER_TRACK_BAR / 2.0, top),
                Pos2::new(center + MIXER_TRACK_BAR / 2.0, bars_bottom),
            );
            let (short, momentary) = if show_loudness { self.mixer_loudness_display[index] } else { (METER_FLOOR, METER_FLOOR) };
            paint_meter(&painter, bar, self.mixer_display[index], short, momentary);
            if show_loudness {
                painter.text(Pos2::new(center, rect.top() + 4.0), Align2::CENTER_TOP, meter_value(short), FontId::proportional(8.5), TEXT);
            }

            // Le fader : une poignée sur la barre, tout en haut à 0 dB
            let fader_y = |db: f32| bars_bottom - meter_position(db) * (bars_bottom - top);
            let area = Rect::from_min_max(Pos2::new(x, top - 6.0), Pos2::new(x + MIXER_TRACK_WIDTH, bars_bottom + 3.0));
            let fader = ui
                .interact(area, ui.scope_id().with(("mixer-fader", index)), Sense::click_and_drag())
                .on_hover_text(format!("{name} : {:.1} dB. Glisser : régler. Double-clic : 0 dB", self.monitor.gain_db(index)))
                .on_hover_cursor(egui::CursorIcon::ResizeVertical);
            fader.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Slider, true, format!("Volume {name} (mixeur)")));
            let before = self.monitor.gain_db(index);
            if fader.double_clicked() {
                self.monitor.set_gain_db(index, 0.0);
            } else if let (true, Some(pos)) = (fader.dragged(), fader.interact_pointer_pos()) {
                let position = ((bars_bottom - pos.y) / (bars_bottom - top)).clamp(0.0, 1.0);
                self.monitor.set_gain_db(index, METER_FLOOR + position * (METER_TOP - METER_FLOOR));
            }
            if self.monitor.gain_db(index) != before {
                if let Some(media) = &mut self.media {
                    media.project.channel_gains = self.monitor.gains().clone();
                    media.mark_changed();
                }
            }
            let handle = Rect::from_center_size(
                Pos2::new(center, fader_y(self.monitor.gain_db(index))),
                egui::vec2(MIXER_TRACK_WIDTH - 4.0, 7.0),
            );
            painter.rect_filled(handle, 2.0, Color32::from_gray(225));
            painter.line_segment(
                [Pos2::new(handle.left() + 2.0, handle.center().y), Pos2::new(handle.right() - 2.0, handle.center().y)],
                Stroke::new(1.0, Color32::from_gray(70)),
            );

            let buttons = Rect::from_min_max(Pos2::new(x + 2.0, bars_bottom + 17.0), Pos2::new(x + MIXER_TRACK_WIDTH - 2.0, rect.bottom()));
            let mut column = ui.new_child(egui::UiBuilder::new().max_rect(buttons).layout(egui::Layout::top_down(egui::Align::Center)));
            column.spacing_mut().item_spacing = egui::vec2(0.0, 2.0);
            column.spacing_mut().button_padding = egui::vec2(2.0, 0.0);
            let solo = column
                .add(channel_button("S", self.monitor.is_solo(index), SOLO).min_size(egui::vec2(20.0, 15.0)))
                .on_hover_text(format!("Solo {name}"));
            solo.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, format!("Solo {name} (mixeur)")));
            if solo.clicked() {
                self.monitor.toggle_solo(index);
            }
            let mute = column
                .add(channel_button("M", self.monitor.is_muted(index), MUTE).min_size(egui::vec2(20.0, 15.0)))
                .on_hover_text(format!("Mute {name}"));
            mute.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, format!("Mute {name} (mixeur)")));
            if mute.clicked() {
                self.monitor.toggle_mute(index);
            }
            x += MIXER_TRACK_WIDTH;
        }

        // La graduation, entre les pistes et le master
        let scale_right = x + MIXER_SCALE_WIDTH - 3.0;
        let meter_y = |db: f32| bars_bottom - meter_position(db) * (bars_bottom - top);
        painter.text(Pos2::new(scale_right, bars_bottom), Align2::RIGHT_CENTER, "-Inf", FontId::proportional(9.0), WEAK);
        for mark in meter_marks(bars_bottom - top) {
            painter.text(
                Pos2::new(scale_right, meter_y(mark)),
                Align2::RIGHT_CENTER,
                if mark > 0.0 { format!("+{mark:.0}") } else { format!("{mark:.0}") },
                FontId::proportional(9.0),
                if mark == 0.0 { Color32::WHITE } else { WEAK },
            );
        }
        x += MIXER_SCALE_WIDTH;

        // Le master : une barre par côté
        let master_width = 2.0 * MIXER_MASTER_BAR + 2.0;
        painter.text(Pos2::new(x + master_width / 2.0, label_y + 13.0), Align2::CENTER_TOP, "Mix", small, Color32::WHITE);
        for (side, name) in ["L", "R"].iter().enumerate() {
            let left = x + side as f32 * (MIXER_MASTER_BAR + 2.0);
            let bar = Rect::from_min_max(Pos2::new(left, top), Pos2::new(left + MIXER_MASTER_BAR, bars_bottom));
            let (short, momentary) = if show_loudness { self.mixer_loudness_display[names.len() + side] } else { (METER_FLOOR, METER_FLOOR) };
            paint_meter(&painter, bar, self.mixer_display[names.len() + side], short, momentary);
            if show_loudness {
                painter.text(Pos2::new(bar.center().x, rect.top() + 4.0), Align2::CENTER_TOP, meter_value(short), FontId::proportional(8.5), TEXT);
            }
            painter.text(Pos2::new(bar.center().x, label_y), Align2::CENTER_TOP, *name, FontId::proportional(9.0), WEAK);
        }
        // Le fader du mix : le volume général, à cheval sur ses deux barres
        let area = Rect::from_min_max(Pos2::new(x, top - 6.0), Pos2::new(x + master_width, bars_bottom + 3.0));
        let fader = ui
            .interact(area, ui.scope_id().with("mixer-master"), Sense::click_and_drag())
            .on_hover_text(format!(
                "Mix : {:.1} dB. Glisser : régler. Double-clic : 0 dB\nRetour stéréo estimé à partir de chaque canal. Barre : crête. Trait blanc : short-term. Trait gris : momentary.",
                self.monitor.master_db()
            ))
            .on_hover_cursor(egui::CursorIcon::ResizeVertical);
        fader.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Slider, true, "Volume Mix (mixeur)"));
        let before = self.monitor.master_db();
        if fader.double_clicked() {
            self.monitor.set_master_db(0.0);
        } else if let (true, Some(pos)) = (fader.dragged(), fader.interact_pointer_pos()) {
            let position = ((bars_bottom - pos.y) / (bars_bottom - top)).clamp(0.0, 1.0);
            self.monitor.set_master_db(METER_FLOOR + position * (METER_TOP - METER_FLOOR));
        }
        if self.monitor.master_db() != before {
            if let Some(media) = &mut self.media {
                media.project.master_gain_db = self.monitor.master_db();
                media.mark_changed();
            }
        }
        let center_y = bars_bottom - meter_position(self.monitor.master_db()) * (bars_bottom - top);
        let handle = Rect::from_center_size(Pos2::new(x + master_width / 2.0, center_y), egui::vec2(master_width + 2.0, 7.0));
        painter.rect_filled(handle, 2.0, Color32::from_gray(225));
        painter.line_segment(
            [Pos2::new(handle.left() + 2.0, handle.center().y), Pos2::new(handle.right() - 2.0, handle.center().y)],
            Stroke::new(1.0, Color32::from_gray(70)),
        );
    }

    pub fn chapter_at(&self, time: f64) -> Option<&signals::Chapter> {
        self.media.as_ref()?.chapters.iter().find(|c| c.start <= time && time < c.end)
    }

    /// Les chapitres de la vidéo : un trait à leur début, leur titre tant qu'il a la place.
    fn paint_chapters(&self, painter: &egui::Painter, rect: Rect) {
        let Some(media) = &self.media else { return };
        let view = self.view;
        for chapter in media.chapters.iter().filter(|c| c.end >= view.start && c.start <= view.end()) {
            let x0 = view.x(rect.left(), rect.width(), chapter.start);
            let x1 = view.x(rect.left(), rect.width(), chapter.end);
            painter.line_segment(
                [Pos2::new(x0, rect.top()), Pos2::new(x0, rect.bottom())],
                Stroke::new(2.0, Color32::WHITE),
            );
            // Le titre reste visible tant qu'on est dans le chapitre, même commencé avant la vue
            let left = x0.max(rect.left()) + 4.0;
            let room = x1.min(rect.right()) - left - 3.0;
            if room < 16.0 {
                continue;
            }
            let mut job = egui::text::LayoutJob::simple_singleline(
                chapter.title.clone(),
                FontId::proportional(10.5),
                Color32::WHITE,
            );
            job.wrap = egui::text::TextWrapping::truncate_at_width(room);
            let galley = painter.layout_job(job);
            let at = Pos2::new(left, rect.top() + 1.0);
            painter.rect_filled(Rect::from_min_size(at, galley.size()).expand2(egui::vec2(2.0, 0.0)), 2.0, Color32::from_black_alpha(185));
            painter.galley(at, galley, Color32::WHITE);
        }
    }

    /// Une seule piste : les blocs des séquences, la courbe de musique par-dessus, puis ce
    /// qui se manipule (poignées, noms).
    fn paint_sequences(&self, painter: &egui::Painter, rect: Rect) {
        let Some(media) = &self.media else { return };
        let view = self.view;
        let block_of = |start: f64, end: f64| {
            Rect::from_min_max(
                Pos2::new(view.x(rect.left(), rect.width(), start), rect.top() + 2.0),
                Pos2::new(view.x(rect.left(), rect.width(), end), rect.bottom() - 2.0),
            )
        };
        let visible = |start: f64, end: f64| end >= view.start && start <= view.end();
        for segment in media.project.segments.iter().filter(|s| visible(s.start, s.end)) {
            let color = segment_color(segment.origin);
            let selected = Some(segment.id) == self.selected;
            painter.rect_filled(block_of(segment.start, segment.end), 2.0, if selected { lighter(color, 0.18) } else { color });
        }

        self.paint_music(painter, rect);

        for (index, segment) in media.project.segments.iter().enumerate() {
            if !visible(segment.start, segment.end) {
                continue;
            }
            let block = block_of(segment.start, segment.end);
            let selected = Some(segment.id) == self.selected;
            if block.width() > 26.0 {
                let label = if segment.name.is_empty() {
                    format!("{}", index + 1)
                } else {
                    format!("{} · {}", index + 1, segment.name)
                };
                let clipped = painter.with_clip_rect(block.intersect(rect));
                clipped.text(
                    Pos2::new(block.left().max(rect.left()) + 8.0, block.top() + 2.0),
                    Align2::LEFT_TOP,
                    label,
                    FontId::proportional(11.0),
                    Color32::WHITE,
                );
            }
            // Poignées de début et de fin
            let handle = if selected { Color32::WHITE } else { Color32::from_white_alpha(130) };
            let width = if selected { 4.0 } else { 2.5 };
            for x in [block.left(), block.right()] {
                painter.rect_filled(Rect::from_center_size(Pos2::new(x, block.center().y), egui::vec2(width, block.height())), 1.0, handle);
            }

            // Ce que l'export sans réencodage prendra réellement avant la poignée
            if selected {
                if let Some(key) = media.strip.as_ref().and_then(|s| s.keyframe_before(segment.start)) {
                    let keyed = view.x(rect.left(), rect.width(), key);
                    if block.left() - keyed >= 1.0 {
                        let pre = Rect::from_min_max(Pos2::new(keyed, block.bottom() - 5.0), Pos2::new(block.left(), block.bottom()));
                        painter.rect_filled(pre, 0.0, Color32::from_rgb(250, 190, 70));
                    }
                }
            }
        }
        paint_legend(painter, rect, &[("Manuelle", MANUAL), ("Retouchée", EDITED), ("Détectée", DETECTED)]);

        // Keyframes, visibles dès qu'on s'approche
        if let Some(strip) = media.strip.as_ref().filter(|_| view.span <= 900.0) {
            let first = strip.keyframes.partition_point(|&k| k < view.start);
            for &key in strip.keyframes[first..].iter().take_while(|&&k| k <= view.end()) {
                let x = view.x(rect.left(), rect.width(), key);
                painter.line_segment(
                    [Pos2::new(x, rect.bottom() - 4.0), Pos2::new(x, rect.bottom())],
                    Stroke::new(1.0, Color32::from_white_alpha(120)),
                );
            }
        }
    }

    fn paint_music(&self, painter: &egui::Painter, rect: Rect) {
        let Some(media) = &self.media else { return };
        if media.project.detection_off {
            return;
        }
        let curve = &media.smoothed;
        let y = |p: f64| rect.bottom() - music_scale(p) * rect.height();
        if !curve.is_empty() {
            let width = rect.width() as usize;
            let levels = smooth_curve(curve, detect::STEP, self.view, width);
            let per_column = self.view.span / width.max(1) as f64;
            // Dans une séquence, sa couleur en plus clair ; ailleurs, en retrait
            let fills: Vec<Color32> = (0..levels.len())
                .map(|column| {
                    let time = self.view.start + (column as f64 + 0.5) * per_column;
                    let inside = media.project.segments.iter().find(|s| s.start <= time && time < s.end);
                    inside.map_or(Color32::from_white_alpha(38), |s| lighter(segment_color(s.origin), 0.42))
                })
                .collect();
            let points: Vec<Pos2> = levels
                .iter()
                .enumerate()
                .map(|(column, &value)| Pos2::new(rect.left() + column as f32 + 0.5, y(value)))
                .collect();
            // Une surface aux bords obliques entre les colonnes, plutôt qu'une barre par colonne
            // (en marches quand la vue est large)…
            let mut mesh = Mesh::default();
            for (column, pair) in points.windows(2).enumerate() {
                let (a, b) = (pair[0], pair[1]);
                if a.y >= rect.bottom() - 0.01 && b.y >= rect.bottom() - 0.01 {
                    continue;
                }
                let fill = fills[column];
                let first = mesh.vertices.len() as u32;
                for corner in [a, b, Pos2::new(b.x, rect.bottom()), Pos2::new(a.x, rect.bottom())] {
                    mesh.colored_vertex(corner, fill);
                }
                mesh.add_triangle(first, first + 1, first + 2);
                mesh.add_triangle(first, first + 2, first + 3);
            }
            painter.add(mesh);
            // …et un trait lissé sur le dessus : un bord net mais sans crénelage
            let mut start = 0;
            while start + 1 < points.len() {
                let fill = fills[start];
                let mut end = start + 1;
                while end + 1 < points.len() && fills[end] == fill {
                    end += 1;
                }
                let line: Vec<Pos2> = points[start..=end].to_vec();
                if line.iter().any(|p| p.y < rect.bottom() - 0.5) {
                    painter.add(Shape::line(line, Stroke::new(1.0, fill.gamma_multiply(0.7))));
                }
                start = end;
            }
            let (_, low) = detect::sensitivity_thresholds(media.project.settings.sensitivity);
            // Aux extrêmes du curseur, la ligne reste visible au bord de la piste
            let threshold = y(low).clamp(rect.top() + 1.0, rect.bottom() - 1.0);
            painter.line_segment(
                [Pos2::new(rect.left(), threshold), Pos2::new(rect.right(), threshold)],
                Stroke::new(1.0, Color32::from_white_alpha(90)),
            );
        }
    }

    /// La piste audio : spectre ou onde, et la loudness toujours par-dessus.
    fn paint_audio(&self, painter: &egui::Painter, rect: Rect) {
        // Le spectre peut arriver avant le reste de l'analyse
        let signals = self.media.as_ref().and_then(|m| m.signals.as_ref());
        let lanes = self.lanes(self.tracks.audio_channels);
        // Toutes les pistes masquées : la zone reste noire
        if lanes.is_empty() {
            painter.rect_filled(rect, 0.0, Color32::BLACK);
            return;
        }
        let lane_height = rect.height() / lanes.len() as f32;
        let gain = signals.map_or(1.0, |s| waveform_gain(s, self.waveform_normalized));
        for (lane, &index) in lanes.iter().enumerate() {
            let channel = signals.and_then(|s| s.channels.get(index));
            let top = rect.top() + lane as f32 * lane_height;
            let band = Rect::from_min_size(Pos2::new(rect.left(), top), egui::vec2(rect.width(), lane_height));
            match self.tracks.view {
                AudioView::Spectrum => {
                    self.paint_spectrum_lane(painter, band, index);
                    self.paint_detail_pending(painter, band, index);
                }
                AudioView::Waveform => {
                    if let Some(channel) = channel {
                        let mut mesh = Mesh::default();
                        paint_peaks(&mut mesh, &channel.peaks, band, self.view, gain, WAVE.gamma_multiply(0.5));
                        painter.add(mesh);
                    }
                }
            }
            // Un canal qu'on n'entend pas s'assombrit
            if index > 0 && !self.monitor.audible(index - 1) {
                painter.rect_filled(band, 0.0, Color32::from_black_alpha(150));
            }
            if let (Some(channel), true) = (channel, self.tracks.loudness) {
                self.paint_loudness_lane(painter, band, index, channel, lane == 0);
            }
            if lane > 0 {
                painter.line_segment(
                    [Pos2::new(band.left(), top), Pos2::new(band.right(), top)],
                    Stroke::new(1.0, Color32::BLACK),
                );
            }
        }
    }

    /// L'image approximative reste visible ; on signale que la version fine arrive.
    fn paint_detail_pending(&self, painter: &egui::Painter, band: Rect, channel: usize) {
        let Some(media) = &self.media else { return };
        if !media.detail_pending.contains(&channel) || media.detail_covers(channel, self.view) {
            return;
        }
        // Le spectre d'ensemble passe d'abord : pas deux indications au même endroit
        if media.spectra.get(&channel).is_some_and(|s| !s.finished) {
            return;
        }
        let time = painter.ctx().input(|i| i.time);
        let lift = if media.signals_progress.is_some() { 24.0 } else { 0.0 };
        paint_track_loading(painter, band, "Spectre fin", None, time, lift);
    }

    fn paint_spectrum_lane(&self, painter: &egui::Painter, band: Rect, channel: usize) {
        let Some(media) = &self.media else { return };
        let full_uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
        let detail = media.details.get(&channel).filter(|_| media.detail_covers(channel, self.view));
        match detail {
            Some((detail, texture)) => {
                let length = (detail.end - detail.start) as f32;
                let uv = Rect::from_min_max(
                    Pos2::new((self.view.start - detail.start) as f32 / length, 0.0),
                    Pos2::new((self.view.end() - detail.start) as f32 / length, 1.0),
                );
                painter.image(texture.id(), band, uv, Color32::WHITE);
            }
            None => {
                if let Some((_, texture)) = media.overviews.get(&channel) {
                    painter.image(texture.id(), band, full_uv, Color32::WHITE);
                }
            }
        }
        // Tant que le spectre arrive, on dit où il en est
        if let Some(state) = media.spectra.get(&channel).filter(|s| !s.finished) {
            let total = (media.project.duration / BLOCK_S).max(1.0);
            let percent = ((state.values.len() / OVERVIEW_BINS) as f64 / total * 100.0).min(99.0) as u32;
            let time = painter.ctx().input(|i| i.time);
            let lift = if media.signals_progress.is_some() { 24.0 } else { 0.0 };
            paint_track_loading(painter, band, "Spectre", known_progress(percent), time, lift);
        }
        // Repères de fréquence : autant que la hauteur de la ligne en laisse lisibles
        for hz in frequency_ticks(band) {
            let y = frequency_y(band, hz);
            painter.text(
                Pos2::new(band.left() + 4.0, y),
                Align2::LEFT_CENTER,
                tick_label(hz),
                FontId::proportional(9.5),
                Color32::from_white_alpha(170),
            );
        }
    }

    fn paint_loudness_lane(&self, painter: &egui::Painter, rect: Rect, index: usize, channel: &signals::Channel, first_lane: bool) {
        let Some(media) = &self.media else { return };
        let target = media.project.loudness_target;
        // La cible d'intégrée vise le programme : un canal seul n'a pas à l'atteindre
        let program = index == 0;
        let film = media.film_loudness.get(&index);
        let measure = film.map(|f| &f.measure);
        let y = |lufs: f32| {
            let t = ((lufs - LOUDNESS_FLOOR) / -LOUDNESS_FLOOR).clamp(0.0, 1.0);
            rect.bottom() - t * rect.height()
        };
        // Le mix est dans la fiche de la vidéo ; un canal a ses propres mesures, sur sa ligne
        let readout = measure.filter(|_| !program).map(|measure| {
            let text = format!(
                "TP {}  I {}  S {}  M {}  LRA {:.1}",
                loudness_text(measure.true_peak),
                loudness_text(measure.integrated),
                loudness_text(measure.max_short_term),
                loudness_text(measure.max_momentary),
                measure.range
            );
            painter.layout_no_wrap(text, FontId::monospace(10.5), Color32::WHITE)
        });
        // La graduation, à droite ; le haut reste à la légende et aux valeurs
        let label_top = rect.top() + 16.0;
        let step = scale_step(rect.height());
        if let Some(step) = step {
            let mut level = -step;
            while level > LOUDNESS_FLOOR {
                let at = y(level);
                painter.line_segment(
                    [Pos2::new(rect.right() - 4.0, at), Pos2::new(rect.right(), at)],
                    Stroke::new(1.0, Color32::from_white_alpha(120)),
                );
                let near_target = program && (at - y(target.integrated)).abs() < 9.0;
                if at >= label_top && !near_target {
                    painter.text(
                        Pos2::new(rect.right() - 6.0, at),
                        Align2::RIGHT_CENTER,
                        format!("{level:.0}"),
                        FontId::proportional(9.5),
                        Color32::from_white_alpha(170),
                    );
                }
                level -= step;
            }
        }
        // La cible et sa tolérance
        if program {
            for level in [target.integrated + target.tolerance, target.integrated - target.tolerance] {
                painter.add(Shape::dashed_line(
                    &[Pos2::new(rect.left(), y(level)), Pos2::new(rect.right(), y(level))],
                    Stroke::new(1.0, Color32::from_white_alpha(90)),
                    5.0,
                    4.0,
                ));
            }
            painter.line_segment(
                [Pos2::new(rect.left(), y(target.integrated)), Pos2::new(rect.right(), y(target.integrated))],
                Stroke::new(1.0, Color32::from_white_alpha(150)),
            );
        }
        if program && y(target.integrated) >= label_top {
            painter.text(
                Pos2::new(rect.right() - 6.0, y(target.integrated)),
                Align2::RIGHT_CENTER,
                format!("{:.0}", target.integrated),
                FontId::proportional(10.0),
                Color32::WHITE,
            );
        }

        let width = rect.width() as usize;
        let curve_points = |values: &[f32], silence_is_a_level: bool| -> Vec<(Pos2, f32)> {
            curve_columns(values, self.view, width, silence_is_a_level)
                .into_iter()
                .map(|(column, value)| (Pos2::new(rect.left() + column as f32 + 0.5, y(value)), value))
                .collect()
        };
        let halo = |points: Vec<Pos2>, thickness: f32| {
            // Un fin détour sombre : la courbe reste lisible sur le spectre comme sur l'onde
            painter.add(Shape::line(points, Stroke::new(thickness + 1.5, Color32::from_black_alpha(130))));
        };
        let trace = |values: &[f32], color: Color32, thickness: f32| {
            let points: Vec<Pos2> = curve_points(values, true).into_iter().map(|p| p.0).collect();
            halo(points.clone(), thickness);
            painter.add(Shape::line(points, Stroke::new(thickness, color)));
        };
        trace(&channel.momentary, LOUD_MOMENTARY, 1.0);
        trace(&channel.short_term, LOUD_SHORT, 1.5);
        if let Some(film) = film {
            // Segment par segment : la couleur dit à chaque instant si la cible est tenue
            let points = curve_points(&film.running, false);
            halo(points.iter().map(|p| p.0).collect(), 2.0);
            for pair in points.windows(2) {
                let color = if program { integrated_color(pair[1].1, target) } else { LOUD_INTEGRATED };
                painter.line_segment([pair[0].0, pair[1].0], Stroke::new(2.0, color));
            }
        }

        let readout_width = readout.as_ref().map_or(0.0, |g| g.size().x + 12.0);
        if first_lane {
            let area = Rect::from_min_max(rect.min, Pos2::new(rect.right() - readout_width, rect.bottom()));
            paint_legend(painter, area, &[("Momentary", LOUD_MOMENTARY), ("Short-term", LOUD_SHORT), ("Intégrée", LOUD_INTEGRATED)]);
        }
        if let Some(galley) = readout {
            let at = Pos2::new(rect.right() - galley.size().x - 6.0, rect.top() + 2.0);
            painter.rect_filled(Rect::from_min_size(at, galley.size()).expand(1.0), 2.0, Color32::from_black_alpha(150));
            painter.galley(at, galley, Color32::WHITE);
        }

        // Au survol, un point suit chaque courbe et donne sa valeur
        let hover = painter.ctx().pointer_hover_pos().filter(|p| rect.contains(*p));
        if let Some(pointer) = hover {
            let column = (pointer.x - rect.left()).max(0.0) as usize;
            let x = rect.left() + column as f32 + 0.5;
            let at_column = |values: &[f32]| loudness_at_column(values, self.view, width, column).filter(|v| *v > NO_LOUDNESS);
            let momentary = at_column(&channel.momentary);
            let short = at_column(&channel.short_term);
            let integrated = film.and_then(|f| at_column(&f.running));
            if let Some(value) = momentary {
                painter.circle(Pos2::new(x, y(value)), 3.0, LOUD_MOMENTARY, Stroke::new(1.0, Color32::BLACK));
            }
            if let Some(value) = short {
                painter.circle(Pos2::new(x, y(value)), 3.5, LOUD_SHORT, Stroke::new(1.0, Color32::BLACK));
            }
            if let Some(value) = integrated {
                let color = if program { integrated_color(value, target) } else { LOUD_INTEGRATED };
                painter.circle(Pos2::new(x, y(value)), 3.5, color, Stroke::new(1.0, Color32::BLACK));
            }
            if let Some(anchor) = integrated.or(short).or(momentary).map(y) {
                let value = |v: Option<f32>| v.map_or("-".to_string(), |v| format!("{v:.1}"));
                let text = format!("M {}  S {}  I {} LKFS", value(momentary), value(short), value(integrated));
                let label = painter.layout_no_wrap(text, FontId::monospace(11.0), Color32::WHITE);
                let size = label.size();
                // À droite du point, ou à gauche près du bord ; toujours dans la ligne
                let left = if x + 10.0 + size.x > rect.right() { x - 10.0 - size.x } else { x + 10.0 };
                let top = (anchor - size.y / 2.0).clamp(rect.top() + 1.0, (rect.bottom() - size.y - 1.0).max(rect.top() + 1.0));
                let at = Pos2::new(left, top);
                painter.rect_filled(Rect::from_min_size(at, size).expand(2.0), 2.0, Color32::from_black_alpha(215));
                painter.galley(at, label, Color32::WHITE);
            }
        }
    }
}

/// Une valeur de loudness, ou un tiret quand il n'y a rien à mesurer.
/// Hauteur relative (0 à 1) d'une probabilité de musique sur la piste des séquences.
/// Échelle logarithmique, rognée aux seuils que le curseur de sensibilité peut atteindre :
/// en bas le seuil à sensibilité maximale, en haut à sensibilité minimale.
fn music_scale(p: f64) -> f32 {
    let bottom = detect::sensitivity_thresholds(detect::MAX_SENSITIVITY).1.log10();
    let top = detect::sensitivity_thresholds(detect::MIN_SENSITIVITY).1.log10();
    ((p.max(1e-9).log10() - bottom) / (top - bottom)).clamp(0.0, 1.0) as f32
}

/// Une courbe par colonne de pixels, sans marches ni dents : interpolée au zoom, puis adoucie
/// sur une largeur qui suit le pas des mesures. L'affichage seul change, pas la détection.
fn smooth_curve(curve: &[f64], step: f64, view: View, width: usize) -> Vec<f64> {
    if width == 0 || curve.is_empty() {
        return vec![0.0; width];
    }
    let per_column = view.span / width as f64;
    let radius = ((step / per_column * 0.75) as usize).clamp(2, 40);
    // Trois moyennes glissantes à la suite : une cloche. Une seule ferait d'une pointe brève,
    // vue de loin, un plateau de plusieurs colonnes.
    const PASSES: usize = 3;
    let pass_radius = radius.div_ceil(2);
    let margin = pass_radius * PASSES;
    let raw = |column: isize| -> f64 {
        let t0 = view.start + column as f64 * per_column;
        let t1 = t0 + per_column;
        if t1 <= 0.0 || t0 >= curve.len() as f64 * step {
            return 0.0;
        }
        if per_column >= step {
            let first = (t0.max(0.0) / step) as usize;
            let last = ((t1 / step).ceil() as usize).max(first + 1).min(curve.len());
            return curve[first.min(last - 1)..last].iter().cloned().fold(0.0, f64::max);
        }
        let position = ((t0 + t1) / 2.0 / step - 0.5).max(0.0);
        let index = (position as usize).min(curve.len() - 1);
        let next = (index + 1).min(curve.len() - 1);
        let fraction = position - index as f64;
        curve[index] * (1.0 - fraction) + curve[next] * fraction
    };
    let mut values: Vec<f64> = (-(margin as isize)..(width + margin) as isize).map(raw).collect();
    let count = (2 * pass_radius + 1) as f64;
    for _ in 0..PASSES {
        let length = values.len() - 2 * pass_radius;
        let mut sum: f64 = values[..2 * pass_radius + 1].iter().sum();
        let mut next = Vec::with_capacity(length);
        for index in 0..length {
            next.push(sum / count);
            if index + 1 < length {
                sum += values[index + 2 * pass_radius + 1] - values[index];
            }
        }
        values = next;
    }
    values.into_iter().map(|value| value.max(0.0)).collect()
}

/// Bouton S ou M du mixeur, plein quand il est actif.
fn channel_button(text: &str, on: bool, color: Color32) -> egui::Button<'static> {
    let label = egui::RichText::new(text.to_string()).size(10.0).color(if on { Color32::BLACK } else { TEXT });
    let button = egui::Button::new(label);
    if on {
        button.fill(color)
    } else {
        button
    }
}

/// Un widget dans `rect`, aligné à gauche et centré verticalement (`ui.put` le centrerait).
fn put_left(ui: &mut egui::Ui, rect: Rect, widget: impl egui::Widget) -> egui::Response {
    let layout = egui::Layout::left_to_right(egui::Align::Center);
    ui.scope_builder(egui::UiBuilder::new().max_rect(rect).layout(layout), |ui| ui.add(widget)).inner
}

/// Case réglable : nom à gauche, valeur à droite, glissière collée au bas de la case.
/// Un clic ou un glissé n'importe où dans la case règle la valeur, au pas de 1.
pub fn slider_box(
    ui: &mut egui::Ui,
    rect: Rect,
    label: &str,
    value: &mut f64,
    range: std::ops::RangeInclusive<f64>,
    enabled: bool,
    help: &str,
) -> egui::Response {
    const RADIUS: u8 = 5;
    const TRACK_HEIGHT: f32 = 3.0;
    let (low, high) = (*range.start(), *range.end());
    let sense = if enabled { Sense::click_and_drag() } else { Sense::hover() };
    let mut response = ui.interact(rect, ui.scope_id().with(("slider-box", label)), sense).on_hover_text(help);
    let inner = rect.shrink(1.0);
    let track = Rect::from_min_max(Pos2::new(inner.left(), inner.bottom() - TRACK_HEIGHT), inner.max);
    // La poignée reste entière dans la case, aux deux bouts
    let (first, last) = (track.left() + 3.0, track.right() - 3.0);
    if response.dragged() || response.clicked() {
        if let Some(pointer) = response.interact_pointer_pos() {
            let share = ((pointer.x - first) / (last - first)).clamp(0.0, 1.0) as f64;
            let next = (low + share * (high - low)).round().clamp(low, high);
            if next != *value {
                *value = next;
                response.mark_changed();
            }
        }
    }
    let shown = *value;
    response.widget_info(|| egui::WidgetInfo::slider(enabled, shown, label));

    let painter = ui.painter();
    let hovered = enabled && (response.hovered() || response.dragged());
    let border = if hovered { Color32::from_rgb(70, 72, 79) } else { CELL_LINE };
    painter.rect_filled(rect, RADIUS, BG);
    // La glissière épouse les coins arrondis du bas
    let bottom_corners = egui::CornerRadius { nw: 0, ne: 0, sw: RADIUS - 1, se: RADIUS - 1 };
    painter.rect_filled(track, bottom_corners, Color32::from_rgb(56, 58, 63));
    painter.rect_stroke(rect, RADIUS, Stroke::new(1.0, border), egui::StrokeKind::Inside);
    let color = if enabled { NAME } else { WEAK };
    let text_y = rect.top() + (rect.height() - TRACK_HEIGHT) / 2.0;
    painter.text(Pos2::new(rect.left() + 10.0, text_y), Align2::LEFT_CENTER, label, FontId::proportional(FONT_SIZE), color);
    painter.text(Pos2::new(rect.right() - 10.0, text_y), Align2::RIGHT_CENTER, format!("{shown:.0}"), FontId::proportional(FONT_SIZE), color);
    let x = first + ((shown - low) / (high - low)).clamp(0.0, 1.0) as f32 * (last - first);
    let handle = Rect::from_center_size(Pos2::new(x, track.center().y - 1.0), egui::vec2(4.0, 9.0));
    painter.rect_filled(handle, 2.0, if enabled { Color32::WHITE } else { WEAK });
    response
}

/// Titre de section en capitales, centré verticalement sur `at`.
fn section_title(ui: &mut egui::Ui, at: Pos2, text: &str) -> egui::Response {
    // Inter Medium, si les polices de l'app sont chargées (pas dans les tests)
    let medium = egui::FontFamily::Name("medium".into());
    let loaded = ui.ctx().fonts(|fonts| fonts.definitions().families.get(&medium).is_some_and(|f| !f.is_empty()));
    let rich = egui::RichText::new(text.to_uppercase()).size(FONT_SIZE).color(TITLE);
    let rich = if loaded { rich.family(medium) } else { rich.strong() };
    let rect = Rect::from_min_max(Pos2::new(at.x, at.y - 9.0), Pos2::new(at.x + 150.0, at.y + 9.0));
    put_left(ui, rect, egui::Label::new(rich).selectable(false))
}

/// Boutons côte à côte. Avec un écart : chacun son cadre ; sans : un cadre commun.
/// Retourne celui qu'on a cliqué.
fn segmented(ui: &mut egui::Ui, rect: Rect, items: &[(&str, bool, &str)], gap: f32, id: &str) -> Option<usize> {
    let painter = ui.painter().clone();
    let count = items.len() as f32;
    let width = (rect.width() - gap * (count - 1.0)) / count;
    if gap == 0.0 {
        painter.rect_stroke(rect, 4.0, Stroke::new(1.0, CELL_LINE), egui::StrokeKind::Inside);
    }
    let mut clicked = None;
    for (index, &(name, selected, help)) in items.iter().enumerate() {
        let cell = Rect::from_min_size(
            Pos2::new(rect.left() + index as f32 * (width + gap), rect.top()),
            egui::vec2(width, rect.height()),
        );
        let response = ui.interact(cell, ui.scope_id().with((id, index)), Sense::click()).on_hover_text(help);
        response.widget_info(|| egui::WidgetInfo::selected(egui::Role::Button, true, selected, name));
        let fill = if gap == 0.0 { cell.shrink(2.0) } else { cell };
        if selected {
            painter.rect_filled(fill, 4.0, SELECTED_FILL);
        } else {
            if gap > 0.0 {
                painter.rect_stroke(cell, 4.0, Stroke::new(1.0, CELL_LINE), egui::StrokeKind::Inside);
            }
            if response.hovered() {
                painter.rect_filled(fill, 4.0, Color32::from_white_alpha(8));
            }
        }
        let color = if selected || response.hovered() { Color32::WHITE } else { TEXT };
        painter.text(cell.center(), Align2::CENTER_CENTER, name, FontId::proportional(FONT_SIZE), color);
        if response.clicked() {
            clicked = Some(index);
        }
    }
    clicked
}

/// Un nom à gauche, un interrupteur à droite ; toute la ligne réagit au clic.
fn switch_row(ui: &mut egui::Ui, rect: Rect, label: &str, on: bool, help: &str) -> egui::Response {
    let response = ui.interact(rect, ui.scope_id().with(("switch", label)), Sense::click()).on_hover_text(help);
    let enabled = ui.is_enabled();
    response.widget_info(|| egui::WidgetInfo::selected(egui::Role::CheckBox, enabled, on, label));
    let painter = ui.painter();
    let color = if enabled { NAME } else { WEAK };
    painter.text(Pos2::new(rect.left(), rect.center().y), Align2::LEFT_CENTER, label, FontId::proportional(FONT_SIZE), color);
    // Un rail deux fois plus large que haut, et un bouton carré de toute sa hauteur, coins arrondis
    const HEIGHT: f32 = 14.0;
    const RADIUS: f32 = 3.0;
    let track = Rect::from_min_size(Pos2::new(rect.right() - 2.0 * HEIGHT, rect.center().y - HEIGHT / 2.0), egui::vec2(2.0 * HEIGHT, HEIGHT));
    let hovered = response.hovered();
    let track_color = if hovered { Color32::from_gray(72) } else { Color32::from_gray(60) };
    painter.rect_filled(track, RADIUS, track_color);
    let knob_left = if on { track.right() - HEIGHT } else { track.left() };
    let knob = Rect::from_min_size(Pos2::new(knob_left, track.top()), egui::vec2(HEIGHT, HEIGHT));
    let knob_color = match (on, enabled) {
        (true, true) => Color32::WHITE,
        (false, true) => Color32::from_gray(150),
        (_, false) => Color32::from_gray(95),
    };
    painter.rect_filled(knob, RADIUS, knob_color);
    response
}

/// Œil : la piste est affichée quand il est blanc.
fn paint_eye(painter: &egui::Painter, center: Pos2, color: Color32) {
    const STEPS: usize = 14;
    let (half_width, half_height) = (6.5, 4.2);
    let point = |t: f32, side: f32| {
        Pos2::new(center.x - half_width + 2.0 * half_width * t, center.y + side * half_height * (std::f32::consts::PI * t).sin())
    };
    let mut outline: Vec<Pos2> = (0..=STEPS).map(|i| point(i as f32 / STEPS as f32, -1.0)).collect();
    outline.extend((1..STEPS).rev().map(|i| point(i as f32 / STEPS as f32, 1.0)));
    painter.add(Shape::convex_polygon(outline, color, Stroke::NONE));
    painter.circle_filled(center, 2.4, BG);
    painter.circle_filled(center, 1.1, color);
}

/// Engrenage des réglages de la détection.
fn paint_gear(painter: &egui::Painter, center: Pos2, color: Color32) {
    for tooth in 0..8 {
        let angle = tooth as f32 * std::f32::consts::TAU / 8.0;
        let direction = egui::vec2(angle.cos(), angle.sin());
        painter.line_segment([center + direction * 3.5, center + direction * 6.2], Stroke::new(2.4, color));
    }
    painter.circle_filled(center, 4.6, color);
    painter.circle_filled(center, 1.9, BG);
}

/// Couleur du dégradé pour un niveau.
fn meter_gradient(db: f32) -> Color32 {
    let db = db.clamp(METER_FLOOR, 0.0);
    let upper = METER_GRADIENT.iter().position(|s| s.0 >= db).unwrap_or(METER_GRADIENT.len() - 1).max(1);
    let (a, b) = (METER_GRADIENT[upper - 1], METER_GRADIENT[upper]);
    let t = (db - a.0) / (b.0 - a.0);
    let mix = |k: usize| (a.1[k] as f32 + (b.1[k] as f32 - a.1[k] as f32) * t) as u8;
    Color32::from_rgb(mix(0), mix(1), mix(2))
}

/// Puissance moyenne sur une fenêtre glissante qui finit à `end` (en blocs). Les blocs du bord
/// comptent au prorata : la valeur évolue en continu, pas par paliers de 100 ms.
fn sliding_power(powers: &[f32], end: f64, width: f64) -> f32 {
    let end = end.min(powers.len() as f64);
    let start = (end - width).max(0.0);
    if end <= start {
        return 0.0;
    }
    let mut sum = 0.0;
    for (index, power) in powers.iter().enumerate().take(end.ceil() as usize).skip(start as usize) {
        let overlap = (end.min(index as f64 + 1.0) - start.max(index as f64)).max(0.0);
        sum += *power as f64 * overlap;
    }
    (sum / (end - start)) as f32
}

/// Un repère rejoint sa cible en douceur.
fn smooth_meter(shown: f32, target: f32, dt: f32) -> f32 {
    let inertia = if target > shown { METER_RISE_S } else { METER_FALL_S };
    shown + (target - shown) * (1.0 - (-dt / inertia).exp())
}

fn meter_value(db: f32) -> String {
    if db <= METER_FLOOR {
        "-Inf".to_string()
    } else {
        format!("{db:.1}")
    }
}

/// Place d'un niveau sur la barre, de 0 (-Inf) à 1 (+6 dB).
fn meter_position(db: f32) -> f32 {
    (db.clamp(METER_FLOOR, METER_TOP) - METER_FLOOR) / (METER_TOP - METER_FLOOR)
}

/// Graduations de la barre : le pas le plus fin qui reste lisible, avec toujours le 0.
fn meter_marks(height: f32) -> Vec<f32> {
    let per_db = height / (METER_TOP - METER_FLOOR);
    let step = [3.0, 6.0, 12.0, 24.0].into_iter().find(|step: &f32| step * per_db >= 14.0).unwrap_or(24.0);
    let mut marks = Vec::new();
    let mut level = step;
    while level <= METER_TOP {
        marks.push(level);
        level += step;
    }
    let mut level = 0.0;
    while level > METER_FLOOR + step / 2.0 {
        marks.push(level);
        level -= step;
    }
    marks
}

/// Une barre de niveau : la crête en dégradé, et deux repères : le short-term en blanc,
/// le momentary en gris, comme sur les courbes de loudness.
fn paint_meter(painter: &egui::Painter, rect: Rect, peak_db: f32, short_db: f32, momentary_db: f32) {
    painter.rect_filled(rect, 1.0, METER_BACK);
    let y = |level: f32| rect.bottom() - meter_position(level) * rect.height();
    if peak_db > METER_FLOOR {
        let mut levels: Vec<f32> = METER_GRADIENT.iter().map(|s| s.0).filter(|level| *level < peak_db).collect();
        levels.push(peak_db.min(METER_TOP));
        let mut mesh = Mesh::default();
        for pair in levels.windows(2) {
            let index = mesh.vertices.len() as u32;
            let (low, high) = (meter_gradient(pair[0]), meter_gradient(pair[1]));
            mesh.colored_vertex(Pos2::new(rect.left(), y(pair[0])), low);
            mesh.colored_vertex(Pos2::new(rect.right(), y(pair[0])), low);
            mesh.colored_vertex(Pos2::new(rect.right(), y(pair[1])), high);
            mesh.colored_vertex(Pos2::new(rect.left(), y(pair[1])), high);
            mesh.add_triangle(index, index + 1, index + 2);
            mesh.add_triangle(index, index + 2, index + 3);
        }
        painter.add(mesh);
    }
    for (level, color) in [(momentary_db, METER_MOMENTARY_LINE), (short_db, Color32::WHITE)] {
        if level > METER_FLOOR {
            let line = Rect::from_min_size(Pos2::new(rect.left(), y(level) - 1.0), egui::vec2(rect.width(), 2.0));
            painter.rect_filled(line, 0.0, color);
        }
    }
}

pub fn loudness_text(value: f32) -> String {
    if value <= NO_LOUDNESS {
        "-".to_string()
    } else {
        format!("{value:.1}")
    }
}

/// Bleu dans la tolérance, rouge à 1 LU au-delà, avec un fondu entre les deux.
fn integrated_color(value: f32, target: LoudnessTarget) -> Color32 {
    let excess = ((value - target.integrated).abs() - target.tolerance).clamp(0.0, 1.0);
    let mix = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * excess) as u8;
    Color32::from_rgb(
        mix(LOUD_INTEGRATED.r(), LOUD_OFF_TARGET.r()),
        mix(LOUD_INTEGRATED.g(), LOUD_OFF_TARGET.g()),
        mix(LOUD_INTEGRATED.b(), LOUD_OFF_TARGET.b()),
    )
}

/// Pas de la graduation, le plus fin qui reste lisible à cette hauteur.
fn scale_step(height: f32) -> Option<f32> {
    let per_lu = height / -LOUDNESS_FLOOR;
    [2.0, 5.0, 10.0, 20.0].into_iter().find(|step| step * per_lu >= SCALE_MIN_SPACING)
}

/// Les points d'une courbe. Le silence absolu est un niveau pour le momentary et le short-term :
/// le retirer ferait relier ses deux bords par une diagonale, qui change à chaque zoom.
/// Pour l'intégrée, `NO_LOUDNESS` veut dire « pas encore de mesure » : rien à tracer.
fn curve_columns(values: &[f32], view: View, width: usize, silence_is_a_level: bool) -> Vec<(usize, f32)> {
    (0..width)
        .filter_map(|column| {
            let value = loudness_at_column(values, view, width, column)?;
            (silence_is_a_level || value > NO_LOUDNESS).then_some((column, value))
        })
        .collect()
}

/// Valeur d'une courbe de loudness sur une colonne de pixels : le maximum de ce qu'elle
/// couvre, comme dans le tracé, pour que le point reste posé sur la courbe.
fn loudness_at_column(values: &[f32], view: View, width: usize, column: usize) -> Option<f32> {
    let t0 = view.start + column as f64 / width as f64 * view.span;
    let t1 = view.start + (column + 1) as f64 / width as f64 * view.span;
    // Au zoom, un bloc couvre plusieurs colonnes : la courbe les relie au lieu de faire des marches
    if t1 - t0 < BLOCK_S {
        return smooth_loudness(values, (t0 + t1) / 2.0);
    }
    let first = (t0 / BLOCK_S) as usize;
    let last = ((t1 / BLOCK_S).ceil() as usize).max(first + 1).min(values.len());
    if first >= last {
        return None;
    }
    Some(values[first..last].iter().cloned().fold(f32::MIN, f32::max))
}

/// Valeur d'une courbe entre deux mesures, posées au centre de leur bloc. Interpolation
/// cubique monotone : la courbe passe par chaque mesure et ne dépasse jamais ses voisines,
/// donc aucun pic inventé. Là où une mesure n'existe pas, on n'interpole pas vers elle.
fn smooth_loudness(values: &[f32], time: f64) -> Option<f32> {
    if time < 0.0 || time / BLOCK_S >= values.len() as f64 {
        return None;
    }
    let position = time / BLOCK_S - 0.5;
    if position <= 0.0 {
        return values.first().copied();
    }
    let index = position as usize;
    if index + 1 >= values.len() {
        return values.last().copied();
    }
    let fraction = (position - index as f64) as f32;
    let (a, b) = (values[index], values[index + 1]);
    if a <= NO_LOUDNESS || b <= NO_LOUDNESS {
        return Some(if fraction < 0.5 { a } else { b });
    }
    let known = |value: f32, fallback: f32| if value <= NO_LOUDNESS { fallback } else { value };
    let before = known(values[index.saturating_sub(1)], a);
    let after = known(values[(index + 2).min(values.len() - 1)], b);
    let slope = |left: f32, right: f32| if left * right <= 0.0 { 0.0 } else { 2.0 * left * right / (left + right) };
    let (start, end) = (slope(a - before, b - a), slope(b - a, after - b));
    let (t, t2, t3) = (fraction, fraction * fraction, fraction * fraction * fraction);
    Some(
        (2.0 * t3 - 3.0 * t2 + 1.0) * a
            + (t3 - 2.0 * t2 + t) * start
            + (-2.0 * t3 + 3.0 * t2) * b
            + (t3 - t2) * end,
    )
}

/// Image du spectrogramme d'un canal pour la portion visible, fréquences hautes en haut.
fn overview_image(values: &[u8], view: View, width: usize) -> egui::ColorImage {
    let columns = values.len() / OVERVIEW_BINS;
    let mut image = egui::ColorImage::filled([width, OVERVIEW_BINS], Color32::BLACK);
    if columns == 0 {
        return image;
    }
    image.pixels.par_chunks_mut(width).enumerate().for_each(|(row, pixels)| {
        let band = OVERVIEW_BINS - 1 - row;
        for (x, pixel) in pixels.iter_mut().enumerate() {
            let t0 = view.start + x as f64 / width as f64 * view.span;
            let t1 = view.start + (x + 1) as f64 / width as f64 * view.span;
            let first = (t0 / BLOCK_S) as usize;
            // Ce qui n'est pas encore calculé reste noir
            if first >= columns {
                continue;
            }
            let last = ((t1 / BLOCK_S).ceil() as usize).clamp(first + 1, columns);
            // Sur une vidéo entière, quelques colonnes suffisent par pixel
            let stride = ((last - first) / 4).max(1);
            let mut value = 0u8;
            let mut column = first;
            while column < last {
                value = value.max(values[column * OVERVIEW_BINS + band]);
                column += stride;
            }
            *pixel = spectrum_color(value);
        }
    });
    image
}

fn paint_ruler(painter: &egui::Painter, rect: Rect, view: View) {
    let step = ruler_step(rect.width() as f64 / view.span);
    let minor = step / 5.0;
    let mut tick = (view.start / minor).floor() as i64;
    loop {
        let time = tick as f64 * minor;
        if time > view.end() {
            break;
        }
        let x = view.x(rect.left(), rect.width(), time);
        let major = tick % 5 == 0;
        let height = if major { 9.0 } else { 4.0 };
        painter.line_segment(
            [Pos2::new(x, rect.bottom() - height), Pos2::new(x, rect.bottom())],
            Stroke::new(1.0, if major { TEXT } else { WEAK }),
        );
        if major {
            painter.text(
                Pos2::new(x + 3.0, rect.top() + 1.0),
                Align2::LEFT_TOP,
                ruler_label(time, step),
                FontId::monospace(10.0),
                TEXT,
            );
        }
        tick += 1;
    }
}

/// Gain d'affichage de l'onde : calé sur la crête du mix, le même pour toutes les pistes.
pub fn waveform_gain(signals: &signals::Signals, normalized: bool) -> f32 {
    let mix = signals.channels.first().map_or(0.0, |c| c.peaks.max_abs());
    if normalized && mix > 1e-4 {
        1.0 / mix
    } else {
        1.0
    }
}

fn paint_peaks(mesh: &mut Mesh, peaks: &Peaks, rect: Rect, view: View, gain: f32, color: Color32) {
    let center = rect.center().y;
    let half = rect.height() / 2.0 - 1.0;
    // Une piste plus forte que le mix s'arrête au bord de sa ligne
    let scale = |value: f32| (value * gain).clamp(-1.0, 1.0);
    let width = rect.width() as usize;
    for column in 0..width {
        let t0 = view.start + column as f64 / width as f64 * view.span;
        let t1 = view.start + (column + 1) as f64 / width as f64 * view.span;
        let Some((low, high)) = peaks.range(t0, t1) else { continue };
        let x = rect.left() + column as f32;
        let top = center - scale(high) * half;
        let bottom = (center - scale(low) * half).max(top + 1.0);
        mesh.add_colored_rect(Rect::from_min_max(Pos2::new(x, top), Pos2::new(x + 1.0, bottom)), color);
    }
}

/// Image de texture du détail du spectrogramme, fréquences hautes en haut.
pub fn detail_image(detail: &signals::Detail) -> egui::ColorImage {
    let mut image = egui::ColorImage::filled([detail.columns, DETAIL_BINS], Color32::BLACK);
    for column in 0..detail.columns {
        for band in 0..DETAIL_BINS {
            let value = detail.values[column * DETAIL_BINS + band];
            image.pixels[(DETAIL_BINS - 1 - band) * detail.columns + column] = spectrum_color(value);
        }
    }
    image
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn music_curve_spans_exactly_the_reachable_thresholds() {
        let threshold = |sensitivity: f64| detect::sensitivity_thresholds(sensitivity).1;
        assert!(music_scale(threshold(detect::MAX_SENSITIVITY)).abs() < 1e-5, "sensibilité maximale : en bas");
        assert!((music_scale(threshold(detect::MIN_SENSITIVITY)) - 1.0).abs() < 1e-5, "minimale : en haut");
        let middle = music_scale(threshold(55.0));
        assert!((middle - 0.5).abs() < 1e-5, "le curseur et la ligne avancent ensemble : {middle}");
        assert_eq!(music_scale(1.0), 1.0, "au-delà : rogné");
        assert_eq!(music_scale(0.0), 0.0);
    }

    #[test]
    fn a_brief_peak_seen_from_afar_has_no_flat_top() {
        // Deux heures sur 1000 colonnes, une seule mesure non nulle
        let mut curve = vec![0.0; (7200.0 / detect::STEP) as usize];
        let middle = curve.len() / 2;
        curve[middle] = 1.0;
        let levels = smooth_curve(&curve, detect::STEP, View { start: 0.0, span: 7200.0 }, 1000);
        let peak = levels.iter().cloned().fold(0.0, f64::max);
        assert_eq!(levels.iter().filter(|&&v| v == peak).count(), 1, "un seul sommet");
        let center = levels.iter().position(|&v| v == peak).unwrap();
        for side in [&levels[center - 3..=center], &levels[center..=center + 3]] {
            assert!(side.windows(2).all(|pair| pair[0] != pair[1]), "pas de palier : {side:?}");
        }
    }

    #[test]
    fn music_curve_has_no_stairs_when_zoomed() {
        let curve = [0.0, 1.0, 0.0, 1.0, 1.0];
        let levels = smooth_curve(&curve, 0.5, View { start: 0.0, span: 2.5 }, 500);
        for pair in levels.windows(2) {
            assert!((pair[1] - pair[0]).abs() < 0.02, "{pair:?}");
        }
        assert!(levels.iter().all(|v| (0.0..=1.0 + 1e-9).contains(v)));
    }

    #[test]
    fn digital_silence_stays_on_the_curve_at_every_zoom() {
        let mut values = vec![-20.0f32; 300];
        values[100..200].fill(NO_LOUDNESS);
        for (span, width) in [(30.0, 1000), (30.0, 137), (12.0, 800), (5.0, 1000)] {
            let view = View { start: 0.0, span };
            let points = curve_columns(&values, view, width, true);
            assert_eq!(points.len(), width, "pas de trou, donc pas de diagonale");
        }
        // L'intégrée n'invente rien tant qu'elle n'existe pas
        let view = View { start: 0.0, span: 30.0 };
        assert!(curve_columns(&vec![NO_LOUDNESS; 300], view, 100, false).is_empty());
    }

    #[test]
    fn frequency_ticks_follow_the_room_available() {
        let band = |height: f32| Rect::from_min_size(Pos2::new(0.0, 0.0), egui::vec2(100.0, height));
        let (tall, medium, short) = (frequency_ticks(band(600.0)), frequency_ticks(band(150.0)), frequency_ticks(band(40.0)));
        assert!(tall.len() >= 20, "{tall:?}");
        assert!(medium.len() < tall.len() && medium.len() >= 6, "{medium:?}");
        assert!(short.len() < medium.len() && short.contains(&1000.0), "{short:?}");
        for ticks in [(&tall, 600.0), (&medium, 150.0), (&short, 40.0)] {
            let mut positions: Vec<f32> = ticks.0.iter().map(|hz| frequency_y(band(ticks.1), *hz)).collect();
            positions.sort_by(f32::total_cmp);
            assert!(positions.windows(2).all(|p| p[1] - p[0] >= FREQUENCY_TICK_SPACING), "{positions:?}");
        }
        assert_eq!((tick_label(1500.0).as_str(), tick_label(20000.0).as_str(), tick_label(300.0).as_str()), ("1,5k", "20k", "300"));
    }

    #[test]
    fn meter_uses_the_spectrogram_colors() {
        for (position, value) in [(1.0, 0u8), (0.5, 128), (0.3, 179), (0.12, 224), (0.0, 255)] {
            let (meter, spectrum) = (meter_gradient(METER_FLOOR * position), spectrum_color(value));
            for (a, b) in [(meter.r(), spectrum.r()), (meter.g(), spectrum.g()), (meter.b(), spectrum.b())] {
                assert!((a as i32 - b as i32).abs() <= 4, "{position} : {meter:?} {spectrum:?}");
            }
        }
    }

    #[test]
    fn loudness_window_moves_continuously_between_blocks() {
        let mut powers = vec![0.01f32; 20];
        powers[10..].fill(1.0);
        let at = |end: f64| sliding_power(&powers, end, 3.0);
        assert!((at(5.0) - 0.01).abs() < 1e-5);
        assert!((at(15.0) - 1.0).abs() < 1e-5);
        // Une image à 60 i/s, soit 1/6 de bloc : aucun palier sur toute la transition
        let mut end = 9.0;
        while end < 14.0 {
            let step = (at(end + 1.0 / 6.0) - at(end)).abs();
            assert!(step < 0.07, "saut de {step} à {end}");
            end += 1.0 / 6.0;
        }
        assert_eq!(sliding_power(&[], 3.0, 3.0), 0.0);
    }

    #[test]
    fn displayed_lines_glide_to_their_target() {
        let next = smooth_meter(-40.0, -20.0, 1.0 / 60.0);
        assert!(next > -40.0 && next < -30.0, "{next}");
        let mut shown = -20.0;
        for _ in 0..120 {
            shown = smooth_meter(shown, -40.0, 1.0 / 60.0);
        }
        assert!((shown + 40.0).abs() < 0.1, "{shown}");
    }

    #[test]
    fn meter_marks_follow_the_height() {
        assert!(meter_marks(1000.0).contains(&-3.0) && meter_marks(1000.0).contains(&3.0));
        assert!(!meter_marks(300.0).contains(&-3.0) && meter_marks(300.0).contains(&-6.0));
        for height in [1000.0, 300.0, 150.0] {
            assert!(meter_marks(height).contains(&0.0), "le 0 est toujours gradué");
        }
        // La poignée à 0 dB laisse de la marge au-dessus d'elle
        assert!(meter_position(0.0) > 0.9 && meter_position(0.0) < 1.0);
        assert_eq!(meter_position(6.0), 1.0);
        assert_eq!(meter_value(METER_FLOOR), "-Inf");
        assert_eq!(meter_value(-12.84), "-12.8");
    }

    #[test]
    fn tinted_colors_keep_their_hue_and_get_lighter() {
        let light = lighter(DETECTED, 0.42);
        assert!(light.r() > DETECTED.r() && light.g() > DETECTED.g() && light.b() > DETECTED.b());
        assert!(light.g() > light.r(), "toujours bleu-vert : {light:?}");
        assert_eq!(lighter(DETECTED, 0.0), DETECTED);
    }

    #[test]
    fn a_partial_spectrum_leaves_the_rest_black() {
        // 10 colonnes calculées sur les 100 de la vue
        let values = vec![255u8; 10 * OVERVIEW_BINS];
        let image = overview_image(&values, View { start: 0.0, span: 10.0 }, 100);
        assert_ne!(image.pixels[5], Color32::BLACK, "le début est visible");
        assert_eq!(image.pixels[50], Color32::BLACK, "la suite attend");
        assert_eq!(image.pixels[99], Color32::BLACK, "la dernière colonne connue ne s'étire pas");
    }

    #[test]
    fn other_spectra_wait_for_the_displayed_one_and_go_one_at_a_time() {
        use crate::SpectrumState;
        use std::collections::HashMap;
        let mut spectra: HashMap<usize, SpectrumState> = HashMap::new();
        assert_eq!(next_background_spectrum(&spectra, 3), Some(0));
        spectra.insert(0, SpectrumState::default());
        assert_eq!(next_background_spectrum(&spectra, 3), None, "la piste affichée passe d'abord");
        spectra.get_mut(&0).unwrap().finished = true;
        assert_eq!(next_background_spectrum(&spectra, 3), Some(1));
        spectra.insert(1, SpectrumState { finished: true, ..Default::default() });
        spectra.insert(2, SpectrumState { finished: true, ..Default::default() });
        assert_eq!(next_background_spectrum(&spectra, 3), None, "tout est prêt");
    }

    #[test]
    fn loading_sits_in_the_bottom_right_corner_of_its_track() {
        let track = Rect::from_min_size(Pos2::new(100.0, 50.0), egui::vec2(800.0, 200.0));
        let frame = loading_corner(track, egui::vec2(200.0, 20.0), 0.0);
        assert!(track.contains_rect(frame));
        assert!(frame.right() > track.right() - 60.0 && frame.bottom() > track.bottom() - 10.0, "{frame:?}");
        assert!(loading_corner(track, egui::vec2(200.0, 20.0), 24.0).bottom() <= frame.top(), "deux chargements s'empilent");
    }

    #[test]
    fn integrated_curve_turns_red_outside_the_tolerance() {
        let target = LoudnessTarget::default();
        assert_eq!(integrated_color(-24.0, target), LOUD_INTEGRATED);
        assert_eq!(integrated_color(-22.0, target), LOUD_INTEGRATED);
        assert_eq!(integrated_color(-26.0, target), LOUD_INTEGRATED);
        assert_eq!(integrated_color(-30.0, target), LOUD_OFF_TARGET);
        assert_eq!(integrated_color(-21.0, target), LOUD_OFF_TARGET);
        let between = integrated_color(-21.5, target);
        assert!(between != LOUD_INTEGRATED && between != LOUD_OFF_TARGET);
    }

    #[test]
    fn loudness_scale_stays_readable_at_every_height() {
        assert_eq!(scale_step(600.0), Some(2.0));
        assert_eq!(scale_step(200.0), Some(5.0));
        assert_eq!(scale_step(90.0), Some(10.0));
        assert_eq!(scale_step(5.0), None);
    }

    /// Temps du centre d'un bloc de mesure.
    fn center(block: usize) -> f64 {
        (block as f64 + 0.5) * BLOCK_S
    }

    #[test]
    fn zoomed_curve_passes_through_every_measure() {
        let values = [-30.0, -20.0, -25.0, -40.0, -22.0, -22.0];
        for (block, value) in values.iter().enumerate() {
            assert!((smooth_loudness(&values, center(block)).unwrap() - value).abs() < 1e-4, "bloc {block}");
        }
    }

    #[test]
    fn zoomed_curve_has_no_stairs() {
        // Une rampe : à l'intérieur d'un même bloc, la valeur doit continuer de monter
        let values: Vec<f32> = (0..50).map(|i| -40.0 + i as f32).collect();
        let view = View { start: 1.0, span: 1.0 }; // 10 blocs sur 200 colonnes
        let curve: Vec<f32> = (0..200).map(|c| loudness_at_column(&values, view, 200, c).unwrap()).collect();
        for pair in curve.windows(2) {
            assert!(pair[1] > pair[0], "{pair:?}");
            assert!(pair[1] - pair[0] < 0.2, "saut de {} : encore une marche", pair[1] - pair[0]);
        }
    }

    #[test]
    fn zoomed_curve_invents_no_peak() {
        let values = [-40.0, -40.0, -10.0, -40.0, -40.0, -39.0, -39.0];
        for step in 0..70 {
            let value = smooth_loudness(&values, step as f64 * 0.01).unwrap();
            assert!((-40.0..=-10.0).contains(&value), "{step} : {value}");
        }
        let flat = smooth_loudness(&values, (center(0) + center(1)) / 2.0).unwrap();
        assert_eq!(flat, -40.0, "un plateau reste plat");
    }

    #[test]
    fn zoomed_curve_does_not_dive_towards_a_missing_measure() {
        let values = [NO_LOUDNESS, NO_LOUDNESS, -25.0, -24.0, -23.0];
        let just_after = smooth_loudness(&values, center(1) + BLOCK_S * 0.75).unwrap();
        assert_eq!(just_after, -25.0);
        assert_eq!(smooth_loudness(&values, center(0) + BLOCK_S * 0.5), Some(NO_LOUDNESS));
        assert_eq!(smooth_loudness(&values, 10.0), None, "pas de valeur au-delà de la mesure");
    }

    #[test]
    fn zoomed_out_curve_keeps_the_loudest_value_of_each_column() {
        let mut values = vec![-30.0; 1000];
        values[505] = -5.0;
        // 100 s sur 100 colonnes : 10 blocs par colonne, le pic ne doit pas disparaître
        let view = View { start: 0.0, span: 100.0 };
        assert_eq!(loudness_at_column(&values, view, 100, 50), Some(-5.0));
    }

    #[test]
    fn hover_point_reads_the_same_value_as_the_curve() {
        // Un bloc de 100 ms par valeur, 10 s de film sur 100 colonnes : une valeur par colonne
        let values: Vec<f32> = (0..100).map(|i| -40.0 + i as f32 * 0.1).collect();
        let view = View { start: 0.0, span: 10.0 };
        let at = |column| loudness_at_column(&values, view, 100, column).unwrap();
        assert!(at(50) >= values[50] && at(50) <= values[51] + 1e-3);
        assert!(at(0) >= -40.0 && at(0) < -39.7);
        // Au-delà de la mesure, pas de valeur inventée
        assert_eq!(loudness_at_column(&values, View { start: 20.0, span: 10.0 }, 100, 5), None);
    }

    #[test]
    fn zoom_keeps_the_point_under_the_cursor() {
        let mut view = View { start: 100.0, span: 1000.0 };
        view.zoom_at(350.0, 0.5, 3600.0);
        assert_eq!(view.span, 500.0);
        assert!((view.time(0.0, 1000.0, 250.0) - 350.0).abs() < 1e-6);
    }

    #[test]
    fn view_never_leaves_the_film() {
        let mut view = View { start: 3500.0, span: 500.0 };
        view.clamp(3600.0);
        assert_eq!(view.start, 3100.0);
        view.zoom_at(3100.0, 100.0, 3600.0);
        assert_eq!(view, View { start: 0.0, span: 3600.0 });
        view.zoom_at(1800.0, 1e-9, 3600.0);
        assert_eq!(view.span, MIN_SPAN);
    }

    #[test]
    fn each_zoom_handle_moves_only_its_own_side() {
        let mut view = View { start: 100.0, span: 100.0 };
        view.set_start(50.0, 600.0);
        assert_eq!((view.start, view.end()), (50.0, 200.0));
        view.set_end(400.0, 600.0);
        assert_eq!((view.start, view.end()), (50.0, 400.0));
        // Une poignée ne traverse pas l'autre, et aucune ne sort de la vidéo
        view.set_start(500.0, 600.0);
        assert_eq!(view.end(), 400.0);
        assert!((view.span - MIN_SPAN).abs() < 1e-9);
        view.set_end(9000.0, 600.0);
        assert_eq!(view.end(), 600.0);
        view.set_start(-50.0, 600.0);
        assert_eq!(view.start, 0.0);
    }

    #[test]
    fn zoom_bar_keeps_a_margin_on_both_sides() {
        let (left, width) = zoom_span(100.0, 1000.0);
        assert_eq!(left, 100.0 + ZOOM_MARGIN);
        assert_eq!(left + width, 1100.0 - ZOOM_MARGIN);
        // Sa poignée la plus à droite reste dans la fenêtre, entière
        let view = View { start: 0.0, span: 3600.0 };
        let (x0, x1) = zoom_handles(view, left, width, 3600.0);
        assert_eq!(x0, left);
        assert!(x1 + 4.5 < 1100.0, "{x1}");
    }

    #[test]
    fn zoom_handles_stay_apart_at_high_zoom() {
        let view = View { start: 3599.5, span: 0.5 };
        let (x0, x1) = zoom_handles(view, 100.0, 1000.0, 3600.0);
        assert!(x1 - x0 >= ZOOM_MIN_WIDTH - 0.01);
        assert!(x1 <= 1100.0, "{x1}");
    }

    #[test]
    fn time_and_x_are_inverse() {
        let view = View { start: 60.0, span: 120.0 };
        let x = view.x(100.0, 800.0, 90.0);
        assert!((view.time(100.0, 800.0, x) - 90.0).abs() < 1e-3);
    }

    #[test]
    fn hidden_tracks_give_their_room_to_the_others() {
        let all = layout_rows(0.0, 600.0, Tracks::default());
        let audio = |rows: &[(Row, f32, f32)]| rows.iter().find(|r| r.0 == Row::Audio).map(|r| r.2 - r.1).unwrap();
        let without_strip = layout_rows(0.0, 600.0, Tracks { strip: false, ..Default::default() });
        assert!(audio(&without_strip) > audio(&all));
        assert_eq!(all.last().unwrap().2, 600.0);
    }

    #[test]
    fn music_and_sequences_share_a_track_and_audio_is_a_single_track() {
        let rows: Vec<Row> = layout_rows(0.0, 600.0, Tracks::default()).iter().map(|r| r.0).collect();
        assert_eq!(rows, [Row::Overview, Row::Ruler, Row::Strip, Row::Sequences, Row::Audio]);
    }

    #[test]
    fn ruler_steps_stay_readable() {
        assert_eq!(ruler_step(1500.0 / 7200.0), 600.0);
        assert_eq!(ruler_step(1500.0 / 10.0), 1.0);
        assert_eq!(ruler_label(3725.0, 60.0), "01:02:05");
    }

    #[test]
    fn frequency_axis_round_trips() {
        let rect = Rect::from_min_size(Pos2::new(0.0, 100.0), egui::vec2(100.0, 200.0));
        let y = frequency_y(rect, 1000.0);
        assert!((frequency_at(rect, y) - 1000.0).abs() < 1.0);
        assert!(frequency_y(rect, 10000.0) < frequency_y(rect, 100.0));
    }
}
