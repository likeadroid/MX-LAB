//! La piste MIDI : les notes de chaque séquence, transcrites en arrière-plan par Basic Pitch,
//! une séquence à la fois, et dessinées en piano roll.

use crate::timeline::{self, Row};
use crate::{signals, App, Job};
use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Stroke};
use mx_lab::transcribe::{self, Note};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

/// Les bornes d'une séquence se retouchent par petits gestes : on attend qu'elles se posent.
const SETTLE: Duration = Duration::from_millis(800);
/// Étendue minimale du piano roll, en demi-tons.
const MIN_RANGE: u8 = 24;
/// Zoom vertical maximal ; au plus près, une octave reste visible.
pub const MAX_ZOOM: f64 = 8.0;
const MIN_VISIBLE: f32 = 12.0;
const HIGHEST_PITCH: u8 = 108;
/// Une note très brève reste cliquable : quelques points de marge de chaque côté.
const NOTE_GRAB: f32 = 3.0;

pub const HELP: &str = "Notes devinées par Basic Pitch (Spotify) sur chaque séquence, à partir de la gauche et de la \
                        droite (sans le centre en 5.1). Une estimation : les dialogues et les sons proches de la \
                        musique peuvent ajouter des notes. Une séquence retouchée est retranscrite.";

/// Notes transcrites d'une séquence, avec les bornes qui ont servi.
pub type Transcription = ((f64, f64), Arc<Vec<Note>>);

fn load_notes(path: &Path) -> Option<Vec<Note>> {
    let bytes = std::fs::read(path).ok()?;
    let json = zstd::decode_all(bytes.as_slice()).ok()?;
    serde_json::from_slice(&json).ok()
}

fn save_notes(path: &Path, notes: &[Note]) {
    if let Ok(json) = serde_json::to_vec(notes) {
        if let Ok(bytes) = zstd::encode_all(json.as_slice(), 3) {
            let _ = std::fs::write(path, bytes);
        }
    }
}

impl App {
    /// Lance la transcription de la prochaine séquence qui n'est pas à jour.
    pub fn update_midi(&mut self, ctx: &egui::Context) {
        if !self.tracks.midi {
            return;
        }
        let generation = self.generation;
        let sender = self.jobs_sender.clone();
        let dragging = !matches!(self.drag, timeline::Drag::None);
        let Some(media) = &mut self.media else { return };
        let segments = &media.project.segments;
        media.midi.retain(|id, _| segments.iter().any(|s| s.id == *id));
        if media.midi_running.is_some() || dragging || media.changed_at.elapsed() < SETTLE || !media.path.is_file() {
            return;
        }
        let Some(layout) = media.layout.clone() else { return };
        let Some(segment) = segments.iter().find(|s| media.midi.get(&s.id).is_none_or(|(bounds, _)| *bounds != (s.start, s.end))) else {
            return;
        };
        let (id, bounds) = (segment.id, (segment.start, segment.end));
        let (path, track) = (media.path.clone(), media.project.audio_track);
        media.midi_running = Some((id, bounds));
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let cache = signals::cache_path(&path, &format!("midi-{track}-{:.3}-{:.3}", bounds.0, bounds.1));
            let cached = cache.as_deref().and_then(load_notes);
            let result = cached.map(Ok).unwrap_or_else(|| {
                let result = crate::guarded(|| transcribe::transcribe(&path, track, &layout, bounds.0, bounds.1));
                if let (Ok(notes), Some(cache)) = (&result, &cache) {
                    save_notes(cache, notes);
                }
                result
            });
            let _ = sender.send((generation, Job::Midi(id, bounds, result)));
            ctx.request_repaint();
        });
    }

    /// Avancement de la transcription, en séquences faites sur le total.
    pub fn midi_progress(&self) -> Option<u32> {
        let media = self.media.as_ref()?;
        media.midi_running?;
        let total = media.project.segments.len().max(1);
        let done = media
            .project
            .segments
            .iter()
            .filter(|s| media.midi.get(&s.id).is_some_and(|(bounds, _)| *bounds == (s.start, s.end)))
            .count();
        Some((done * 100 / total) as u32)
    }

    /// Étendue du piano roll : l'essentiel des notes (2 % à 98 %), pas les quelques notes
    /// isolées qui l'étireraient sur tout le clavier ; élargie pour rester lisible.
    fn midi_range(&self) -> (u8, u8) {
        let mut pitches: Vec<u8> =
            self.media.iter().flat_map(|m| m.midi.values()).flat_map(|(_, notes)| notes.iter().map(|n| n.pitch)).collect();
        pitches.sort_unstable();
        let (low, high) = if pitches.is_empty() {
            (48, 84)
        } else {
            let at = |share: f64| pitches[((pitches.len() - 1) as f64 * share).round() as usize];
            (at(0.02).saturating_sub(2), at(0.98).saturating_add(2))
        };
        if high - low >= MIN_RANGE {
            return (low, high);
        }
        let missing = MIN_RANGE - (high - low);
        let low = low.saturating_sub(missing / 2).max(transcribe::LOWEST_PITCH);
        (low, low + MIN_RANGE)
    }

    /// Ce que montre la piste : l'étendue des notes, resserrée par le zoom vertical autour de
    /// `midi_center` (le milieu des notes tant qu'on n'a pas défilé).
    pub fn midi_view(&self) -> (u8, u8) {
        let (low, high) = self.midi_range();
        let full = (high - low) as f32;
        let span = (full / self.midi_zoom.max(1.0) as f32).max(MIN_VISIBLE).min(full).round();
        let center = self.midi_center.unwrap_or((low as f32 + high as f32) / 2.0);
        let lowest = transcribe::LOWEST_PITCH as f32;
        let first = (center - span / 2.0).round().clamp(lowest, HIGHEST_PITCH as f32 - span);
        (first as u8, (first + span) as u8)
    }

    /// Défile vers le haut ou le bas, en demi-tons.
    pub fn scroll_midi(&mut self, semitones: f32) {
        let (low, high) = self.midi_view();
        let center = (low as f32 + high as f32) / 2.0 + semitones;
        self.midi_center = Some(center.clamp(transcribe::LOWEST_PITCH as f32, HIGHEST_PITCH as f32));
    }

    /// Demi-tons par point, pour faire défiler à la molette.
    pub fn midi_row_height(&self, rect: Rect) -> f32 {
        let (low, high) = self.midi_view();
        rect.height() / (high - low + 1) as f32
    }

    /// La note sous un point de la piste, s'il y en a une.
    pub fn midi_note_at(&self, rect: Rect, pos: Pos2) -> Option<Note> {
        let media = self.media.as_ref()?;
        let pitch = self.midi_pitch_at(rect, pos.y);
        let grab = NOTE_GRAB as f64 * self.view.span / rect.width() as f64;
        let time = self.view.time(rect.left(), rect.width(), pos.x);
        media
            .midi
            .values()
            .flat_map(|(_, notes)| notes.iter())
            .filter(|n| n.pitch == pitch && n.start - grab <= time && time <= n.end + grab)
            .min_by(|a, b| (a.start - time).abs().total_cmp(&(b.start - time).abs()))
            .copied()
    }

    /// Fait entendre une note : une sinusoïde à sa hauteur, de sa durée.
    pub fn play_note(&mut self, note: Note) {
        if self.tone.is_none() {
            self.tone = self.player.as_ref().and_then(|player| player.tone().ok());
        }
        if let Some(tone) = &self.tone {
            tone.play(note.pitch, note.end - note.start);
        }
        self.played_note = Some((note, std::time::Instant::now()));
    }

    /// Hauteur MIDI sous un point de la piste.
    pub fn midi_pitch_at(&self, rect: Rect, y: f32) -> u8 {
        let (low, high) = self.midi_view();
        let rows = (high - low + 1) as f32;
        let row = ((rect.bottom() - y) / rect.height() * rows).floor().clamp(0.0, rows - 1.0);
        low + row as u8
    }

    /// Le piano roll : une ligne par demi-ton, touches noires plus sombres, un repère par do.
    pub fn paint_midi(&self, painter: &egui::Painter, rect: Rect) {
        let Some(media) = &self.media else { return };
        let (low, high) = self.midi_view();
        let rows = (high - low + 1) as f32;
        let row_height = rect.height() / rows;
        let y_of = |pitch: u8| rect.bottom() - (pitch.clamp(low, high) - low + 1) as f32 * row_height;
        // Zoomé, les notes hors de la vue disparaissent ; sinon, les rares notes extrêmes se
        // posent sur la ligne du bord
        let zoomed = self.midi_view() != self.midi_range();
        let played = self.played_note.filter(|(_, at)| at.elapsed() < Duration::from_millis(600)).map(|(note, _)| note);
        for pitch in low..=high {
            let top = y_of(pitch);
            let band = Rect::from_min_max(Pos2::new(rect.left(), top), Pos2::new(rect.right(), top + row_height));
            if matches!(pitch % 12, 1 | 3 | 6 | 8 | 10) {
                painter.rect_filled(band, 0.0, Color32::from_black_alpha(70));
            }
            if pitch % 12 == 0 {
                painter.line_segment([band.left_bottom(), band.right_bottom()], Stroke::new(1.0, Color32::from_white_alpha(16)));
                if row_height * 12.0 >= 20.0 {
                    painter.text(
                        Pos2::new(rect.left() + 4.0, band.center().y),
                        Align2::LEFT_CENTER,
                        transcribe::note_name(pitch),
                        FontId::proportional(9.5),
                        Color32::from_white_alpha(70),
                    );
                }
            }
        }
        let view = self.view;
        for segment in &media.project.segments {
            let Some((_, notes)) = media.midi.get(&segment.id) else { continue };
            if segment.end < view.start || segment.start > view.end() {
                continue;
            }
            let color = timeline::segment_color(segment.origin);
            for note in notes.iter().filter(|n| n.end >= view.start && n.start <= view.end()) {
                if zoomed && !(low..=high).contains(&note.pitch) {
                    continue;
                }
                let x0 = view.x(rect.left(), rect.width(), note.start);
                let x1 = view.x(rect.left(), rect.width(), note.end).max(x0 + 1.0);
                let top = y_of(note.pitch);
                let shape = Rect::from_min_max(Pos2::new(x0, top + 0.5), Pos2::new(x1, top + row_height - 0.5).max(Pos2::new(x1, top + 1.5)));
                // Plus l'activation est forte, plus la note est franche
                let strength = (0.35 + 0.65 * note.velocity.clamp(0.0, 1.0)).clamp(0.0, 1.0);
                let fill = timeline::lighter(color, 0.25).gamma_multiply(strength);
                // La note qu'on vient de faire entendre s'allume un instant
                let fill = if played == Some(*note) { Color32::WHITE } else { fill };
                painter.rect_filled(shape, (row_height * 0.25).min(2.0), fill);
            }
        }
    }

    /// En-tête de la piste : son titre, le nombre de notes et le zoom vertical.
    pub fn midi_header(&mut self, ui: &mut egui::Ui, rect: Rect) {
        let zoom = Rect::from_min_size(
            Pos2::new(rect.left() + timeline::PAD, rect.top() + 48.0),
            egui::vec2(rect.width() - 2.0 * timeline::PAD, 28.0),
        );
        if zoom.bottom() <= rect.bottom() - 4.0 {
            let help = "Zoom vertical : moins de demi-tons, plus grands. Alt + molette sur la piste pour monter ou descendre.";
            let mut value = self.midi_zoom;
            let response = timeline::slider_box(ui, zoom, "Zoom vertical", &mut value, 1.0..=MAX_ZOOM, true, help);
            if response.changed() {
                self.midi_zoom = value;
            }
        }
        timeline::section_title(ui, Pos2::new(rect.left() + timeline::PAD, rect.top() + 15.0), "MIDI").on_hover_text(HELP);
        let Some(media) = &self.media else { return };
        let count: usize = media.midi.values().map(|(_, notes)| notes.len()).sum();
        let text = if media.project.segments.is_empty() {
            "Aucune séquence".to_string()
        } else if media.midi_running.is_some() {
            "Transcription…".to_string()
        } else {
            format!("{count} notes")
        };
        ui.painter().text(
            Pos2::new(rect.left() + timeline::PAD, rect.top() + 34.0),
            Align2::LEFT_CENTER,
            text,
            FontId::proportional(timeline::FONT_SIZE),
            timeline::WEAK,
        );
    }

    /// Ce que la piste MIDI attend encore.
    pub fn midi_loading(&self) -> Option<timeline::Loading> {
        self.midi_progress().map(|progress| timeline::Loading {
            row: Row::Midi,
            label: "Transcription MIDI",
            progress: (progress > 0).then_some(progress),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notes_survive_the_cache() {
        let folder = std::env::temp_dir().join(format!("mx-lab-midi-{}", std::process::id()));
        std::fs::create_dir_all(&folder).unwrap();
        let path = folder.join("notes.zst");
        let notes = vec![Note { start: 1.0, end: 1.5, pitch: 60, velocity: 0.7 }];
        save_notes(&path, &notes);
        assert_eq!(load_notes(&path), Some(notes));
        let _ = std::fs::remove_dir_all(folder);
    }
}
