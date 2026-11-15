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

    /// Hauteur MIDI sous un point de la piste.
    pub fn midi_pitch_at(&self, rect: Rect, y: f32) -> u8 {
        let (low, high) = self.midi_range();
        let rows = (high - low + 1) as f32;
        let row = ((rect.bottom() - y) / rect.height() * rows).floor().clamp(0.0, rows - 1.0);
        low + row as u8
    }

    /// Le piano roll : une ligne par demi-ton, touches noires plus sombres, un repère par do.
    pub fn paint_midi(&self, painter: &egui::Painter, rect: Rect) {
        let Some(media) = &self.media else { return };
        let (low, high) = self.midi_range();
        let rows = (high - low + 1) as f32;
        let row_height = rect.height() / rows;
        // Une note hors de l'étendue se pose sur la ligne du bord
        let y_of = |pitch: u8| rect.bottom() - (pitch.clamp(low, high) - low + 1) as f32 * row_height;
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
                let x0 = view.x(rect.left(), rect.width(), note.start);
                let x1 = view.x(rect.left(), rect.width(), note.end).max(x0 + 1.0);
                let top = y_of(note.pitch);
                let shape = Rect::from_min_max(Pos2::new(x0, top + 0.5), Pos2::new(x1, top + row_height - 0.5).max(Pos2::new(x1, top + 1.5)));
                // Plus l'activation est forte, plus la note est franche
                let strength = (0.35 + 0.65 * note.velocity.clamp(0.0, 1.0)).clamp(0.0, 1.0);
                let fill = timeline::lighter(color, 0.25).gamma_multiply(strength);
                painter.rect_filled(shape, (row_height * 0.25).min(2.0), fill);
            }
        }
    }

    /// En-tête de la piste : son titre et le nombre de notes.
    pub fn midi_header(&mut self, ui: &mut egui::Ui, rect: Rect) {
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
