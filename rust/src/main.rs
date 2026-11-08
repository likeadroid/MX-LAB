#![windows_subsystem = "windows"]

use eframe::egui;
use mse::detect::{self, Settings};
use mse::yamnet::Yamnet;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};

const PANEL_MARGIN: f32 = 8.0;
const VIDEO_EXTENSIONS: &[&str] = &["mp4", "mkv", "avi", "mov", "webm"];
const DROP_ZONE_TEXT: &str = "Glissez vos vidéos ici ou cliquez pour les sélectionner";
const DROP_ZONE_REPLACE: &str = "Glissez ou cliquez pour remplacer";
const READY: &str = "Prêt : lancez l'analyse.";
const NAME_MAX_CHARS: usize = 70;
const CUSTOM_DESTINATION: &str = "Dossier choisi";
const GAP_HELP: &str = "Interruption de la musique tolérée dans un clip. Si la musique reprend dans ce délai, le clip continue ; sinon, un nouveau clip commence.";
const ANTICIPATION_HELP: &str = "Marge ajoutée avant chaque séquence, pour garder quelques secondes avant le début de la musique.";
const POST_HELP: &str = "Marge ajoutée après chaque séquence, pour garder quelques secondes après la fin de la musique.";
const SENSITIVITY_HELP: &str = "Plus elle est haute, plus la détection retient de passages : la musique discrète, mais aussi des passages sans musique. 50 par défaut, 70 pour une vidéo à nappes ou drones discrets.";

#[derive(Default)]
struct Shared {
    status: String,
    running: bool,
}

/// Résultat d'une fenêtre de sélection, ouverte hors du fil de l'interface.
enum Picked {
    Videos(Vec<PathBuf>),
    Folder(PathBuf),
}

struct App {
    videos: Vec<PathBuf>,
    picker: Option<mpsc::Receiver<Option<Picked>>>,
    gap: u32,
    anticipation: u32,
    post: u32,
    sensitivity: u32,
    height: f32,
    custom_destination: bool,
    destination: Option<PathBuf>,
    shared: Arc<Mutex<Shared>>,
}

fn file_name(path: &Path) -> String {
    path.file_name().unwrap_or_default().to_string_lossy().into_owned()
}

impl App {
    fn new() -> Self {
        App {
            videos: Vec::new(),
            picker: None,
            gap: 15,
            anticipation: 6,
            post: 6,
            sensitivity: 50,
            height: 0.0,
            custom_destination: false,
            destination: None,
            shared: Arc::new(Mutex::new(Shared {
                status: "Chargez un ou plusieurs vidéos pour commencer.".into(),
                running: false,
            })),
        }
    }

    /// Ouvre une fenêtre de sélection sans bloquer l'interface.
    fn open_picker(&mut self, ctx: &egui::Context, pick: impl FnOnce() -> Option<Picked> + Send + 'static) {
        let (sender, receiver) = mpsc::channel();
        self.picker = Some(receiver);
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let _ = sender.send(pick());
            ctx.request_repaint();
        });
    }

    fn pick_videos(&mut self, ctx: &egui::Context) {
        self.open_picker(ctx, || {
            rfd::FileDialog::new()
                .set_title("Sélectionner un ou plusieurs vidéos")
                .add_filter("Vidéo", VIDEO_EXTENSIONS)
                .add_filter("Tous les fichiers", &["*"])
                .pick_files()
                .map(Picked::Videos)
        });
    }

    fn pick_destination(&mut self, ctx: &egui::Context) {
        self.open_picker(ctx, || {
            rfd::FileDialog::new().set_title("Dossier de destination").pick_folder().map(Picked::Folder)
        });
    }

    fn poll_picker(&mut self) {
        let Some(receiver) = &self.picker else { return };
        match receiver.try_recv() {
            Err(mpsc::TryRecvError::Empty) => return,
            Ok(Some(Picked::Videos(files))) => self.set_videos(files),
            Ok(Some(Picked::Folder(folder))) => self.destination = Some(folder),
            Ok(None) | Err(mpsc::TryRecvError::Disconnected) => {}
        }
        self.picker = None;
    }

    /// Texte de la zone de dépôt : les vidéos retenues, ou l'invitation à en choisir.
    fn zone_text(&self) -> (String, &'static str) {
        match self.videos.as_slice() {
            [] => (DROP_ZONE_TEXT.to_string(), ""),
            [one] => (shorten(&file_name(one)), DROP_ZONE_REPLACE),
            many => {
                let names: Vec<String> = many.iter().map(|p| file_name(p)).collect();
                (shorten(&format!("{} vidéos : {}", many.len(), names.join(", "))), DROP_ZONE_REPLACE)
            }
        }
    }

    fn drop_videos(&mut self, dropped: Vec<PathBuf>) {
        let videos: Vec<PathBuf> = dropped.into_iter().filter(|p| is_video(p)).collect();
        if videos.is_empty() {
            self.shared.lock().unwrap().status = "Aucune vidéo reconnue dans ce dépôt.".into();
        } else {
            self.set_videos(videos);
        }
    }

    fn set_videos(&mut self, files: Vec<PathBuf>) {
        let status = match files.as_slice() {
            [_] => READY.to_string(),
            many => {
                let names: Vec<String> = many.iter().take(8).map(|p| format!("  • {}", file_name(p))).collect();
                let more = if many.len() > 8 { format!("\n  … (+{})", many.len() - 8) } else { String::new() };
                format!("{} fichiers chargés :\n{}{more}", many.len(), names.join("\n"))
            }
        };
        self.videos = files;
        self.shared.lock().unwrap().status = status;
    }

    /// Dossier de destination : `None` à côté de chaque vidéo, `Err` si le dossier reste à choisir.
    fn export_target(&self) -> Result<Option<PathBuf>, ()> {
        match (self.custom_destination, &self.destination) {
            (false, _) => Ok(None),
            (true, Some(folder)) => Ok(Some(folder.clone())),
            (true, None) => Err(()),
        }
    }

    fn destination_row(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.horizontal(|ui| {
            ui.add_sized([110.0, 20.0], egui::Label::new("Destination"));
            ui.radio_value(&mut self.custom_destination, false, "Même dossier que la vidéo");
            ui.radio_value(&mut self.custom_destination, true, CUSTOM_DESTINATION);
            if self.custom_destination {
                if ui.button("Choisir…").clicked() {
                    self.pick_destination(ctx);
                }
                match &self.destination {
                    Some(folder) => {
                        let path = folder.display().to_string();
                        ui.add(egui::Label::new(&path).truncate()).on_hover_text(&path);
                    }
                    None => {
                        ui.weak("Aucun dossier choisi");
                    }
                }
            }
        });
    }

    fn analyze(&mut self, ctx: &egui::Context) {
        let Ok(destination) = self.export_target() else { return };
        let videos = self.videos.clone();
        let settings = Settings {
            sensitivity: self.sensitivity as f64,
            gap: self.gap as f64,
            anticipation: self.anticipation as f64,
            post: self.post as f64,
        };
        let shared = self.shared.clone();
        shared.lock().unwrap().running = true;
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let total = videos.len();
            let set_status = |text: String| {
                shared.lock().unwrap().status = text;
                ctx.request_repaint();
            };
            let Ok(model) = std::panic::catch_unwind(Yamnet::load) else {
                let mut state = shared.lock().unwrap();
                state.status = "ERREUR : chargement du modèle impossible".to_string();
                state.running = false;
                ctx.request_repaint();
                return;
            };
            let mut results = Vec::new();
            for (index, video) in videos.iter().enumerate() {
                let name = file_name(video);
                let prefix = if total > 1 { format!("[{}/{total}] ", index + 1) } else { String::new() };
                let report = |text: String| set_status(format!("{prefix}{name} : {text}"));
                report("préparation…".into());
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    detect::process_video(&model, video, &settings, destination.as_deref(), &report)
                }));
                let summary = match outcome {
                    Ok(Ok(summary)) => summary,
                    Ok(Err(error)) => format!("ERREUR : {error}"),
                    Err(_) => "ERREUR : échec interne de l'analyse".to_string(),
                };
                results.push((name, summary));
            }
            let text = if let [(name, summary)] = results.as_slice() {
                format!("{name} : {summary}")
            } else {
                let lines: Vec<String> = results.iter().map(|(n, s)| format!("  • {n} : {s}")).collect();
                format!("Terminé ({total} vidéos) :\n{}", lines.join("\n"))
            };
            let mut state = shared.lock().unwrap();
            state.status = text;
            state.running = false;
            ctx.request_repaint();
        });
    }
}

/// Coupe le milieu d'un texte trop long, pour garder la fin d'un nom de fichier.
fn shorten(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= NAME_MAX_CHARS {
        return text.to_string();
    }
    let tail = NAME_MAX_CHARS / 3;
    let head: String = chars[..NAME_MAX_CHARS - tail - 1].iter().collect();
    let end: String = chars[chars.len() - tail..].iter().collect();
    format!("{head}…{end}")
}

fn is_video(path: &Path) -> bool {
    path.extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .is_some_and(|e| VIDEO_EXTENSIONS.contains(&e.as_str()))
}

/// Contour fermé d'un rectangle aux coins arrondis.
fn rounded_outline(rect: egui::Rect, radius: f32) -> Vec<egui::Pos2> {
    let corners = [
        (rect.right_top() + egui::vec2(-radius, radius), -90.0f32),
        (rect.right_bottom() + egui::vec2(-radius, -radius), 0.0),
        (rect.left_bottom() + egui::vec2(radius, -radius), 90.0),
        (rect.left_top() + egui::vec2(radius, radius), 180.0),
    ];
    let mut points = Vec::new();
    for (center, start) in corners {
        for step in 0..=6 {
            let angle = (start + step as f32 * 15.0).to_radians();
            points.push(center + radius * egui::vec2(angle.cos(), angle.sin()));
        }
    }
    points.push(points[0]);
    points
}

/// Zone de dépôt à bordure pointillée, cliquable pour la sélection manuelle.
fn drop_zone(ui: &mut egui::Ui, enabled: bool, dragging: bool, text: &str, hint: &str) -> egui::Response {
    let size = egui::vec2(ui.available_width(), 96.0);
    let response = ui.add_enabled_ui(enabled, |ui| ui.allocate_response(size, egui::Sense::click())).inner;
    response.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, enabled, text));
    let rect = response.rect.shrink(1.0);
    let highlighted = enabled && (dragging || response.hovered());
    let color = if !enabled {
        ui.visuals().weak_text_color()
    } else if dragging {
        ui.visuals().selection.stroke.color
    } else if highlighted {
        ui.visuals().strong_text_color()
    } else {
        ui.visuals().text_color()
    };
    let painter = ui.painter();
    if highlighted {
        painter.rect_filled(rect, 8.0, ui.visuals().widgets.hovered.weak_bg_fill);
    }
    let stroke = egui::Stroke::new(1.5, color);
    painter.extend(egui::Shape::dashed_line(&rounded_outline(rect, 8.0), stroke, 6.0, 5.0));

    // Icône : une flèche qui descend dans un plateau
    let center = rect.center_top() + egui::vec2(0.0, if hint.is_empty() { 34.0 } else { 26.0 });
    let icon = egui::Stroke::new(2.0, color);
    painter.line_segment([center + egui::vec2(0.0, -14.0), center + egui::vec2(0.0, 6.0)], icon);
    painter.line_segment([center + egui::vec2(-7.0, -1.0), center + egui::vec2(0.0, 6.0)], icon);
    painter.line_segment([center + egui::vec2(7.0, -1.0), center + egui::vec2(0.0, 6.0)], icon);
    painter.add(egui::Shape::line(
        vec![
            center + egui::vec2(-14.0, 3.0),
            center + egui::vec2(-14.0, 13.0),
            center + egui::vec2(14.0, 13.0),
            center + egui::vec2(14.0, 3.0),
        ],
        icon,
    ));
    let body = egui::TextStyle::Body.resolve(ui.style());
    let text_y = if hint.is_empty() { 68.0 } else { 56.0 };
    painter.text(rect.center_top() + egui::vec2(0.0, text_y), egui::Align2::CENTER_CENTER, text, body, color);
    if !hint.is_empty() {
        let small = egui::TextStyle::Small.resolve(ui.style());
        let weak = ui.visuals().weak_text_color();
        painter.text(rect.center_top() + egui::vec2(0.0, 76.0), egui::Align2::CENTER_CENTER, hint, small, weak);
    }
    if enabled {
        response.on_hover_cursor(egui::CursorIcon::PointingHand)
    } else {
        response
    }
}

fn slider(ui: &mut egui::Ui, label: &str, help: &str, value: &mut u32, range: std::ops::RangeInclusive<u32>) {
    ui.horizontal(|ui| {
        ui.add_sized([110.0, 20.0], egui::Label::new(label)).on_hover_text(help).on_disabled_hover_text(help);
        ui.spacing_mut().slider_width = (ui.available_width() - 70.0).max(100.0);
        ui.add(egui::Slider::new(value, range)).on_hover_text(help).on_disabled_hover_text(help);
    });
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.draw(ui);
    }
}

impl App {
    fn draw(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.poll_picker();
        let (status, running) = {
            let state = self.shared.lock().unwrap();
            (state.status.clone(), state.running)
        };
        let (dragging, dropped) = ctx.input(|i| {
            let paths: Vec<PathBuf> = i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).collect();
            (!i.raw.hovered_files.is_empty(), paths)
        });
        if !dropped.is_empty() && !running && self.picker.is_none() {
            self.drop_videos(dropped);
        }
        let status = if running { status } else { self.shared.lock().unwrap().status.clone() };
        let busy = running || self.picker.is_some();
        let (zone_text, zone_hint) = self.zone_text();
        let content = egui::CentralPanel::default().show(ui, |ui| {
            let full = egui::vec2(ui.available_width(), 30.0);
            let zone = drop_zone(ui, !busy, dragging, &zone_text, zone_hint);
            let zone = if self.videos.is_empty() {
                zone
            } else {
                let names: Vec<String> = self.videos.iter().map(|p| p.display().to_string()).collect();
                zone.on_hover_text(names.join("\n"))
            };
            if zone.clicked() {
                self.pick_videos(&ctx);
            }
            ui.add_space(8.0);
            ui.add_enabled_ui(!busy, |ui| {
                slider(ui, "Gap (s)", GAP_HELP, &mut self.gap, 1..=60);
                slider(ui, "Anticipation (s)", ANTICIPATION_HELP, &mut self.anticipation, 0..=20);
                slider(ui, "Post (s)", POST_HELP, &mut self.post, 0..=20);
                slider(ui, "Sensibilité", SENSITIVITY_HELP, &mut self.sensitivity, 10..=100);
            });
            ui.add_space(8.0);
            ui.add_enabled_ui(!busy, |ui| self.destination_row(ui, &ctx));
            ui.add_space(8.0);
            let ready = !busy && !self.videos.is_empty() && self.export_target().is_ok();
            let button = egui::Button::new("Analyser et extraire séquences musicales").min_size(full);
            if ui.add_enabled(ready, button).clicked() {
                self.analyze(&ctx);
            }
            ui.add_space(8.0);
            egui::ScrollArea::vertical().max_height(130.0).auto_shrink([false, true]).show(ui, |ui| {
                ui.add(egui::Label::new(status).wrap());
            });
            ui.cursor().min.y
        });
        // La fenêtre suit la hauteur du contenu, qui varie avec le texte d'état
        let height = content.inner + PANEL_MARGIN;
        if (height - self.height).abs() > 0.5 {
            self.height = height;
            let width = ctx.content_rect().width();
            ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(width, height)));
        }
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([600.0, 340.0]).with_min_inner_size([560.0, 200.0]),
        ..Default::default()
    };
    eframe::run_native("Music Sequence Extractor", options, Box::new(|_| Ok(Box::new(App::new()))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::accesskit::Role;
    use egui_kittest::kittest::{NodeT, Queryable};
    use egui_kittest::Harness;

    fn drop_files(harness: &mut Harness<'static, App>, names: &[&str]) {
        for name in names {
            harness.input_mut().dropped_files.push(Arc::new(TestFile(PathBuf::from(name))));
        }
        harness.step();
    }

    #[derive(Debug)]
    struct TestFile(PathBuf);

    impl egui::DroppedFile for TestFile {
        fn path(&self) -> &Path {
            &self.0
        }

        fn bytes(&self) -> Result<Vec<u8>, String> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn manual_destination_requires_a_folder() {
        let mut harness = harness();
        drop_files(&mut harness, &["a.mkv"]);
        harness.step();
        let analyze = "Analyser et extraire séquences musicales";
        assert!(!harness.get_by_label(analyze).accesskit_node().is_disabled());
        harness.get_by_label(CUSTOM_DESTINATION).click();
        harness.step();
        harness.step();
        assert!(harness.state().custom_destination);
        assert!(harness.get_by_label(analyze).accesskit_node().is_disabled());
        harness.state_mut().destination = Some(PathBuf::from("D:\\Clips"));
        harness.step();
        harness.step();
        assert!(!harness.get_by_label(analyze).accesskit_node().is_disabled());
    }

    #[test]
    fn analyze_button_starts_the_analysis() {
        let mut harness = harness();
        drop_files(&mut harness, &["Z:\\absent\\film.mkv"]);
        harness.step();
        harness.get_by_label("Analyser et extraire séquences musicales").click();
        for _ in 0..200 {
            harness.step();
            std::thread::sleep(std::time::Duration::from_millis(50));
            if harness.state().shared.lock().unwrap().status.contains("ERREUR") {
                break;
            }
        }
        let state = harness.state().shared.lock().unwrap();
        assert!(state.status.starts_with("film.mkv : ERREUR"), "{}", state.status);
    }

    #[test]
    fn selected_name_is_shown_in_the_drop_zone() {
        let mut harness = harness();
        drop_files(&mut harness, &["C:\\films\\mon film.mkv"]);
        harness.step();
        harness.get_by_role_and_label(Role::Button, "mon film.mkv");
        harness.get_by_label(READY);
    }

    #[test]
    fn long_names_keep_their_extension() {
        let name = format!("{}.mkv", "a".repeat(200));
        let short = shorten(&name);
        assert_eq!(short.chars().count(), NAME_MAX_CHARS);
        assert!(short.ends_with(".mkv") && short.contains('…'));
    }

    #[test]
    fn drop_zone_is_a_clickable_control() {
        let harness = harness();
        harness.get_by_role_and_label(Role::Button, DROP_ZONE_TEXT);
    }

    #[test]
    fn dropping_keeps_only_videos() {
        let mut harness = harness();
        drop_files(&mut harness, &["a.MKV", "notes.txt", "b.mp4"]);
        assert_eq!(harness.state().videos, [PathBuf::from("a.MKV"), PathBuf::from("b.mp4")]);
        harness.step();
        harness.get_by_label_contains("2 fichiers chargés");
    }

    #[test]
    fn dropping_nothing_usable_keeps_the_selection() {
        let mut harness = harness();
        drop_files(&mut harness, &["a.mkv"]);
        drop_files(&mut harness, &["notes.txt"]);
        assert_eq!(harness.state().videos, [PathBuf::from("a.mkv")]);
        harness.step();
        harness.get_by_label("Aucune vidéo reconnue dans ce dépôt.");
    }

    fn harness() -> Harness<'static, App> {
        Harness::builder()
            .with_size(egui::vec2(600.0, 300.0))
            .build_ui_state(|ui, app: &mut App| app.draw(ui), App::new())
    }

    #[test]
    fn status_is_below_the_analyze_button() {
        let harness = harness();
        let button = harness.get_by_label("Analyser et extraire séquences musicales").rect();
        let status = harness.get_by_label("Chargez un ou plusieurs vidéos pour commencer.").rect();
        assert!(status.top() >= button.bottom(), "{status:?} {button:?}");
    }

    #[test]
    fn every_slider_and_label_describes_itself_on_hover() {
        let rows = [
            ("Gap (s)", GAP_HELP),
            ("Anticipation (s)", ANTICIPATION_HELP),
            ("Post (s)", POST_HELP),
            ("Sensibilité", SENSITIVITY_HELP),
        ];
        for (index, (label, help)) in rows.iter().enumerate() {
            for on_slider in [false, true] {
                let mut harness = harness();
                if on_slider {
                    harness.get_all_by_role(Role::Slider).nth(index).expect("curseur absent").hover();
                } else {
                    harness.get_by_label(label).hover();
                }
                // Laisse passer le délai d'apparition de l'infobulle
                for _ in 0..20 {
                    harness.step();
                }
                assert!(harness.query_by_role_and_label(Role::Label, help).is_some(), "{label}, curseur : {on_slider}");
            }
        }
    }
}
