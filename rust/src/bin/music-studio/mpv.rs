//! Lecteur vidéo : libmpv embarquée.
//!
//! L'interface passe par Direct3D 12 : présenter en OpenGL faisait clignoter toute la fenêtre.
//! mpv garde OpenGL pour son rendu (et le HDR), sur son propre fil et dans un contexte attaché
//! à une fenêtre invisible qui ne présente jamais rien ; chaque image est recopiée vers egui.
//!
//! La DLL est chargée à l'exécution (pas de bibliothèque d'import à fournir) après
//! extraction dans le dossier local de l'utilisateur, comme ffmpeg.

use eframe::egui::{self, Color32};
use glow::HasContext;
use libloading::Library;
use std::ffi::{c_char, c_double, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use windows_sys::Win32::Foundation::{HMODULE, HWND};
use windows_sys::Win32::Graphics::Gdi::{GetDC, ReleaseDC, HDC};
use windows_sys::Win32::Graphics::OpenGL::{
    wglCreateContext, wglDeleteContext, wglGetProcAddress, wglMakeCurrent, ChoosePixelFormat, SetPixelFormat, HGLRC,
    PFD_DOUBLEBUFFER, PFD_DRAW_TO_WINDOW, PFD_SUPPORT_OPENGL, PFD_TYPE_RGBA, PIXELFORMATDESCRIPTOR,
};
use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};
use windows_sys::Win32::UI::WindowsAndMessaging::{CreateWindowExW, DestroyWindow, WS_POPUP};

static COMPRESSED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/libmpv-2.dll.zst"));

const FORMAT_FLAG: c_int = 3;
const FORMAT_DOUBLE: c_int = 5;
const FADERS_LABEL: &str = "faders";
const LIMITER_LABEL: &str = "limiter";
const EVENT_END_FILE: c_int = 7;
const EVENT_FILE_LOADED: c_int = 8;
const EVENT_PLAYBACK_RESTART: c_int = 21;
const EVENT_PROPERTY_CHANGE: c_int = 22;
const SEEK_TIMEOUT: Duration = Duration::from_millis(800);
const RENDER_API_TYPE: c_int = 1;
const RENDER_OPENGL_INIT: c_int = 2;
const RENDER_OPENGL_FBO: c_int = 3;
const RENDER_BLOCK_FOR_TARGET_TIME: c_int = 12;
const RENDER_UPDATE_FRAME: u64 = 1;
const GL_RGBA8: c_int = 0x8058;

#[repr(C)]
struct RenderParam {
    kind: c_int,
    data: *mut c_void,
}

#[repr(C)]
struct OpenGlInit {
    get_proc_address: extern "C" fn(*mut c_void, *const c_char) -> *mut c_void,
    context: *mut c_void,
}

#[repr(C)]
struct OpenGlFbo {
    fbo: c_int,
    width: c_int,
    height: c_int,
    internal_format: c_int,
}

#[repr(C)]
struct Event {
    id: c_int,
    error: c_int,
    reply: u64,
    data: *mut c_void,
}

#[repr(C)]
struct EventProperty {
    name: *const c_char,
    format: c_int,
    data: *mut c_void,
}

type Handle = *mut c_void;
type Callback = extern "C" fn(*mut c_void);

/// Fonctions de libmpv, copiées hors de `Library` qui reste chargée avec elles.
struct Api {
    create: unsafe extern "C" fn() -> Handle,
    initialize: unsafe extern "C" fn(Handle) -> c_int,
    set_option_string: unsafe extern "C" fn(Handle, *const c_char, *const c_char) -> c_int,
    set_property_string: unsafe extern "C" fn(Handle, *const c_char, *const c_char) -> c_int,
    command: unsafe extern "C" fn(Handle, *const *const c_char) -> c_int,
    observe_property: unsafe extern "C" fn(Handle, u64, *const c_char, c_int) -> c_int,
    wait_event: unsafe extern "C" fn(Handle, c_double) -> *const Event,
    set_wakeup_callback: unsafe extern "C" fn(Handle, Callback, *mut c_void),
    terminate_destroy: unsafe extern "C" fn(Handle),
    render_create: unsafe extern "C" fn(*mut Handle, Handle, *mut RenderParam) -> c_int,
    render_set_update_callback: unsafe extern "C" fn(Handle, Callback, *mut c_void),
    render_update: unsafe extern "C" fn(Handle) -> u64,
    render_render: unsafe extern "C" fn(Handle, *mut RenderParam) -> c_int,
    render_free: unsafe extern "C" fn(Handle),
    _library: Library,
}

fn extract_dll() -> Result<PathBuf, String> {
    let base = std::env::var_os("LOCALAPPDATA").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    let folder = base.join("music-sequence-extractor");
    std::fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
    // La taille compressée distingue les versions embarquées
    let target = folder.join(format!("libmpv-{}.dll", COMPRESSED.len()));
    if target.is_file() {
        return Ok(target);
    }
    let partial = folder.join(format!("libmpv-{}.{}.tmp", COMPRESSED.len(), std::process::id()));
    let mut decoder = zstd::Decoder::new(COMPRESSED).map_err(|e| e.to_string())?;
    decoder.window_log_max(27).map_err(|e| e.to_string())?;
    let mut file = std::fs::File::create(&partial).map_err(|e| e.to_string())?;
    std::io::copy(&mut decoder, &mut file).map_err(|e| e.to_string())?;
    drop(file);
    if let Err(error) = std::fs::rename(&partial, &target) {
        let _ = std::fs::remove_file(&partial);
        // Une autre instance a pu l'extraire entre-temps
        if !target.is_file() {
            return Err(error.to_string());
        }
    }
    Ok(target)
}

impl Api {
    fn load() -> Result<Self, String> {
        let path = extract_dll()?;
        unsafe {
            let library = Library::new(&path).map_err(|e| format!("libmpv : {e}"))?;
            macro_rules! symbol {
                ($name:literal) => {
                    *library.get(concat!($name, "\0").as_bytes()).map_err(|e| format!("libmpv : {e}"))?
                };
            }
            Ok(Api {
                create: symbol!("mpv_create"),
                initialize: symbol!("mpv_initialize"),
                set_option_string: symbol!("mpv_set_option_string"),
                set_property_string: symbol!("mpv_set_property_string"),
                command: symbol!("mpv_command"),
                observe_property: symbol!("mpv_observe_property"),
                wait_event: symbol!("mpv_wait_event"),
                set_wakeup_callback: symbol!("mpv_set_wakeup_callback"),
                terminate_destroy: symbol!("mpv_terminate_destroy"),
                render_create: symbol!("mpv_render_context_create"),
                render_set_update_callback: symbol!("mpv_render_context_set_update_callback"),
                render_update: symbol!("mpv_render_context_update"),
                render_render: symbol!("mpv_render_context_render"),
                render_free: symbol!("mpv_render_context_free"),
                _library: library,
            })
        }
    }
}

/// Adresse d'une fonction OpenGL du contexte courant. Les fonctions de base ne sont
/// connues que d'opengl32.dll : wglGetProcAddress rend alors une valeur nulle ou 1, 2, 3, -1.
unsafe fn gl_function(name: *const c_char) -> *mut c_void {
    static OPENGL32: OnceLock<usize> = OnceLock::new();
    if let Some(function) = wglGetProcAddress(name as *const u8) {
        if !matches!(function as usize, 1 | 2 | 3 | usize::MAX) {
            return function as *mut c_void;
        }
    }
    let module = *OPENGL32.get_or_init(|| LoadLibraryA(c"opengl32.dll".as_ptr() as *const u8) as usize) as HMODULE;
    GetProcAddress(module, name as *const u8).map_or(std::ptr::null_mut(), |function| function as *mut c_void)
}

extern "C" fn get_proc_address(_context: *mut c_void, name: *const c_char) -> *mut c_void {
    unsafe { gl_function(name) }
}

/// Contexte OpenGL sans rien à l'écran : la fenêtre n'est jamais montrée, ni le tampon présenté.
struct HiddenGl {
    window: HWND,
    dc: HDC,
    context: HGLRC,
}

impl HiddenGl {
    unsafe fn open() -> Result<Self, String> {
        let class: Vec<u16> = "STATIC\0".encode_utf16().collect();
        let window = CreateWindowExW(
            0,
            class.as_ptr(),
            std::ptr::null(),
            WS_POPUP,
            0,
            0,
            16,
            16,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null(),
        );
        if window.is_null() {
            return Err("création de la fenêtre OpenGL impossible".into());
        }
        let mut hidden = HiddenGl { window, dc: GetDC(window), context: std::ptr::null_mut() };
        let mut format: PIXELFORMATDESCRIPTOR = std::mem::zeroed();
        format.nSize = std::mem::size_of::<PIXELFORMATDESCRIPTOR>() as u16;
        format.nVersion = 1;
        format.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
        format.iPixelType = PFD_TYPE_RGBA as _;
        format.cColorBits = 32;
        let index = ChoosePixelFormat(hidden.dc, &format);
        if index == 0 || SetPixelFormat(hidden.dc, index, &format) == 0 {
            return Err("format de pixels OpenGL refusé".into());
        }
        hidden.context = wglCreateContext(hidden.dc);
        if hidden.context.is_null() || wglMakeCurrent(hidden.dc, hidden.context) == 0 {
            return Err("contexte OpenGL impossible".into());
        }
        Ok(hidden)
    }
}

impl Drop for HiddenGl {
    fn drop(&mut self) {
        unsafe {
            if !self.context.is_null() {
                wglMakeCurrent(std::ptr::null_mut(), std::ptr::null_mut());
                wglDeleteContext(self.context);
            }
            ReleaseDC(self.window, self.dc);
            DestroyWindow(self.window);
        }
    }
}

/// Réveille l'interface quand mpv a du nouveau.
struct Waker {
    ctx: egui::Context,
}

extern "C" fn wake(data: *mut c_void) {
    let waker = unsafe { &*(data as *const Waker) };
    waker.ctx.request_repaint();
}

/// Une image prête à être affichée, de haut en bas.
pub struct VideoFrame {
    pub size: [usize; 2],
    pub pixels: Vec<Color32>,
}

#[derive(Default)]
struct RenderSlot {
    dirty: bool,
    quit: bool,
    size: [i32; 2],
    frame: Option<VideoFrame>,
}

/// Ce que l'interface et le fil de rendu partagent.
struct RenderShared {
    slot: Mutex<RenderSlot>,
    changed: Condvar,
    ctx: egui::Context,
}

extern "C" fn wake_render(data: *mut c_void) {
    let shared = unsafe { &*(data as *const RenderShared) };
    shared.slot.lock().unwrap().dirty = true;
    shared.changed.notify_one();
}

struct SendHandle(Handle);
// mpv autorise l'usage de ses poignées depuis n'importe quel fil
unsafe impl Send for SendHandle {}

fn render_thread(api: Arc<Api>, handle: SendHandle, shared: Arc<RenderShared>, ready: mpsc::Sender<Result<(), String>>) {
    unsafe {
        let hidden = match HiddenGl::open() {
            Ok(hidden) => hidden,
            Err(error) => {
                let _ = ready.send(Err(error));
                return;
            }
        };
        let gl = glow::Context::from_loader_function_cstr(|name| gl_function(name.as_ptr()) as *const c_void);
        let mut init = OpenGlInit { get_proc_address, context: std::ptr::null_mut() };
        let mut params = [
            RenderParam { kind: RENDER_API_TYPE, data: c"opengl".as_ptr() as *mut c_void },
            RenderParam { kind: RENDER_OPENGL_INIT, data: &mut init as *mut _ as *mut c_void },
            RenderParam { kind: 0, data: std::ptr::null_mut() },
        ];
        let mut render: Handle = std::ptr::null_mut();
        if (api.render_create)(&mut render, handle.0, params.as_mut_ptr()) < 0 {
            let _ = ready.send(Err("libmpv : rendu OpenGL impossible".into()));
            return;
        }
        (api.render_set_update_callback)(render, wake_render, Arc::as_ptr(&shared) as *mut c_void);
        let (texture, framebuffer) = match (gl.create_texture(), gl.create_framebuffer()) {
            (Ok(texture), Ok(framebuffer)) => (texture, framebuffer),
            _ => {
                (api.render_free)(render);
                let _ = ready.send(Err("texture de la vidéo impossible".into()));
                return;
            }
        };
        let _ = ready.send(Ok(()));

        let mut size = [0, 0];
        let mut force = false;
        loop {
            let (wanted, dirty) = {
                let mut slot = shared.slot.lock().unwrap();
                while !slot.quit && !slot.dirty && slot.size == size {
                    slot = shared.changed.wait(slot).unwrap();
                }
                if slot.quit {
                    break;
                }
                (slot.size, std::mem::take(&mut slot.dirty))
            };
            let [width, height] = wanted;
            if width <= 0 || height <= 0 {
                size = wanted;
                continue;
            }
            if size != wanted {
                size = wanted;
                gl.bind_texture(glow::TEXTURE_2D, Some(texture));
                for parameter in [glow::TEXTURE_MIN_FILTER, glow::TEXTURE_MAG_FILTER] {
                    gl.tex_parameter_i32(glow::TEXTURE_2D, parameter, glow::LINEAR as i32);
                }
                gl.tex_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    GL_RGBA8,
                    width,
                    height,
                    0,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(None),
                );
                gl.bind_texture(glow::TEXTURE_2D, None);
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
                gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, Some(texture), 0);
                gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                force = true;
            }
            let frame_ready = dirty && (api.render_update)(render) & RENDER_UPDATE_FRAME != 0;
            if !frame_ready && !force {
                continue;
            }
            force = false;

            let mut fbo = OpenGlFbo { fbo: framebuffer.0.get() as c_int, width, height, internal_format: GL_RGBA8 };
            // Sans attendre l'heure d'affichage : mpv appelle quand l'image est due
            let mut block: c_int = 0;
            let mut params = [
                RenderParam { kind: RENDER_OPENGL_FBO, data: &mut fbo as *mut _ as *mut c_void },
                RenderParam { kind: RENDER_BLOCK_FOR_TARGET_TIME, data: &mut block as *mut _ as *mut c_void },
                RenderParam { kind: 0, data: std::ptr::null_mut() },
            ];
            (api.render_render)(render, params.as_mut_ptr());

            let mut pixels = vec![Color32::BLACK; width as usize * height as usize];
            let bytes = std::slice::from_raw_parts_mut(pixels.as_mut_ptr() as *mut u8, pixels.len() * 4);
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
            gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
            gl.read_pixels(0, 0, width, height, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelPackData::Slice(Some(bytes)));
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            for pixel in bytes.chunks_exact_mut(4) {
                pixel[3] = 255;
            }
            shared.slot.lock().unwrap().frame = Some(VideoFrame { size: [width as usize, height as usize], pixels });
            shared.ctx.request_repaint();
        }
        (api.render_free)(render);
        gl.delete_framebuffer(framebuffer);
        gl.delete_texture(texture);
        drop(gl);
        drop(hidden);
    }
}

/// État lu auprès de mpv, sans jamais bloquer l'interface.
#[derive(Clone, Default)]
pub struct State {
    pub position: f64,
    pub duration: f64,
    pub paused: bool,
    pub speed: f64,
    pub fps: f64,
    pub loaded: bool,
    /// Le fichier n'a pas pu s'ouvrir : rien n'est plus en cours de chargement.
    pub failed: bool,
    pub error: Option<String>,
}

const OBSERVED: [(&CStr, c_int); 6] = [
    (c"time-pos", FORMAT_DOUBLE),
    (c"duration", FORMAT_DOUBLE),
    (c"pause", FORMAT_FLAG),
    (c"speed", FORMAT_DOUBLE),
    (c"container-fps", FORMAT_DOUBLE),
    (c"volume", FORMAT_DOUBLE),
];

pub struct Player {
    api: Arc<Api>,
    handle: Handle,
    _waker: Box<Waker>,
    shared: Arc<RenderShared>,
    thread: Option<JoinHandle<()>>,
    seek_started: Option<Instant>,
    /// Gain du mix, et faders et routage en place.
    boost: f64,
    chain: (Option<String>, Option<String>),
    installed_chain: String,
    pub state: State,
}

impl Player {
    pub fn new(ctx: &egui::Context) -> Result<Self, String> {
        let api = Arc::new(Api::load()?);
        unsafe {
            let handle = (api.create)();
            if handle.is_null() {
                return Err("libmpv : création impossible".into());
            }
            let options: [(&CStr, &CStr); 11] = [
                (c"vo", c"libmpv"),
                (c"config", c"no"),
                (c"load-scripts", c"no"),
                (c"osc", c"no"),
                (c"input-default-bindings", c"no"),
                (c"keep-open", c"always"),
                (c"pause", c"yes"),
                (c"hr-seek", c"yes"),
                (c"hwdec", c"auto-copy-safe"),
                // Même piste audio que la détection et l'export, sans sous-titres imposés
                (c"aid", c"1"),
                (c"sid", c"no"),
            ];
            for (name, value) in options {
                (api.set_option_string)(handle, name.as_ptr(), value.as_ptr());
            }
            if (api.initialize)(handle) < 0 {
                (api.terminate_destroy)(handle);
                return Err("libmpv : initialisation impossible".into());
            }
            for (index, (name, format)) in OBSERVED.iter().enumerate() {
                (api.observe_property)(handle, index as u64, name.as_ptr(), *format);
            }

            let shared = Arc::new(RenderShared { slot: Mutex::default(), changed: Condvar::new(), ctx: ctx.clone() });
            let (ready, confirmed) = mpsc::channel();
            let spawned = {
                let (api, shared, handle) = (api.clone(), shared.clone(), SendHandle(handle));
                std::thread::Builder::new()
                    .name("mpv-rendu".into())
                    .spawn(move || render_thread(api, handle, shared, ready))
            };
            let thread = match spawned {
                Ok(thread) => thread,
                Err(error) => {
                    (api.terminate_destroy)(handle);
                    return Err(format!("fil de rendu impossible : {error}"));
                }
            };
            match confirmed.recv() {
                Ok(Ok(())) => {}
                outcome => {
                    let _ = thread.join();
                    (api.terminate_destroy)(handle);
                    return Err(outcome.unwrap_or_else(|_| Err("le fil de rendu s'est arrêté".into())).unwrap_err());
                }
            }

            let waker = Box::new(Waker { ctx: ctx.clone() });
            (api.set_wakeup_callback)(handle, wake, &*waker as *const Waker as *mut c_void);

            Ok(Player {
                api,
                handle,
                _waker: waker,
                shared,
                thread: Some(thread),
                seek_started: None,
                boost: 1.0,
                chain: (None, None),
                installed_chain: String::new(),
                state: State { paused: true, speed: 1.0, ..Default::default() },
            })
        }
    }

    /// Taille voulue pour les images, en pixels de l'écran.
    pub fn set_video_size(&self, size: [i32; 2]) {
        let mut slot = self.shared.slot.lock().unwrap();
        if slot.size != size {
            slot.size = size;
            self.shared.changed.notify_one();
        }
    }

    /// La dernière image rendue, si elle n'a pas encore été prise.
    pub fn take_frame(&self) -> Option<VideoFrame> {
        self.shared.slot.lock().unwrap().frame.take()
    }

    fn command(&self, args: &[&str]) {
        let owned: Vec<CString> = args.iter().filter_map(|a| CString::new(*a).ok()).collect();
        let mut pointers: Vec<*const c_char> = owned.iter().map(|a| a.as_ptr()).collect();
        pointers.push(std::ptr::null());
        unsafe { (self.api.command)(self.handle, pointers.as_ptr()) };
    }

    fn set(&self, name: &CStr, value: &str) {
        if let Ok(value) = CString::new(value) {
            unsafe { (self.api.set_property_string)(self.handle, name.as_ptr(), value.as_ptr()) };
        }
    }

    /// Piste audio à écouter (0 : la première du fichier).
    pub fn set_audio_track(&self, track: usize) {
        self.set(c"aid", &format!("{}", track + 1));
    }

    pub fn load(&mut self, path: &Path) {
        self.state.loaded = false;
        self.state.failed = false;
        self.state.error = None;
        self.command(&["loadfile", &path.to_string_lossy()]);
    }

    /// Va à l'image exacte, ou à la keyframe voisine (plus réactif quand on survole la vidéo).
    pub fn seek(&mut self, position: f64, exact: bool) {
        self.apply_chain();
        let mode = if exact { "absolute+exact" } else { "absolute+keyframes" };
        self.command(&["seek", &format!("{position:.6}"), mode]);
        self.state.position = position;
        self.seek_started = Some(Instant::now());
    }

    /// Faux tant que le dernier déplacement n'a pas abouti : cela évite d'empiler
    /// les demandes quand on fait glisser la tête de lecture.
    pub fn can_seek(&self) -> bool {
        self.state.loaded && self.seek_started.is_none_or(|started| started.elapsed() > SEEK_TIMEOUT)
    }

    pub fn set_paused(&mut self, paused: bool) {
        self.set(c"pause", if paused { "yes" } else { "no" });
        self.state.paused = paused;
    }

    pub fn step_frame(&self, forward: bool) {
        self.command(&[if forward { "frame-step" } else { "frame-back-step" }]);
    }

    pub fn set_speed(&mut self, speed: f64) {
        self.set(c"speed", &format!("{speed}"));
        self.state.speed = speed;
    }

    /// Gain du mix, placé devant le limiteur : rien ne sort au-dessus de 0 dB.
    pub fn set_master_gain(&mut self, gain: f32) {
        self.boost = gain as f64;
        if self.state.loaded {
            self.command(&["af-command", LIMITER_LABEL, "volume", &format!("{:.5}", self.boost), "volume@boost"]);
        } else {
            // Rien ne joue encore : la chaîne partira directement avec le bon gain
            self.apply_chain();
        }
    }

    /// Boucle A-B, ou aucune.
    pub fn set_loop(&self, range: Option<(f64, f64)>) {
        match range {
            Some((a, b)) => {
                self.set(c"ab-loop-a", &format!("{a:.6}"));
                self.set(c"ab-loop-b", &format!("{b:.6}"));
            }
            None => {
                self.set(c"ab-loop-a", "no");
                self.set(c"ab-loop-b", "no");
            }
        }
    }

    /// Chaîne audio : les faders, puis le routage de l'écoute (solo, mute). La remplacer
    /// relance le son et fige brièvement l'image : à ne faire que si sa forme change.
    pub fn set_audio_chain(&mut self, faders: Option<&str>, filter: Option<&str>) {
        self.chain = (faders.map(str::to_string), filter.map(str::to_string));
        self.apply_chain();
    }

    /// Note les réglages déjà envoyés par commandes. Un saut dans la vidéo reconstruit la chaîne
    /// depuis son texte : sans cela, elle reviendrait à son état de départ.
    pub fn remember_routing(&mut self, faders: Option<&str>) {
        self.chain.0 = faders.map(str::to_string);
    }

    /// Pose la chaîne si son texte a changé. Juste avant un saut, le son s'interrompt de toute façon.
    fn apply_chain(&mut self) {
        let text = audio_chain(self.chain.0.as_deref(), self.chain.1.as_deref(), self.boost);
        if text != self.installed_chain {
            self.set(c"af", &text);
            self.installed_chain = text;
        }
    }

    /// Change la part d'un canal vers un côté, en pleine lecture, sans relancer le son.
    pub fn set_route(&self, side: char, channel: usize, gain: f32) {
        self.command(&["af-command", FADERS_LABEL, "volume", &format!("{gain:.5}"), &format!("volume@{side}{channel}")]);
    }

    /// Lit les événements en attente, sans bloquer.
    pub fn poll(&mut self) {
        // Un déplacement refusé par mpv n'envoie aucun signal : ne pas ignorer la position à vie
        if self.seek_started.is_some_and(|started| started.elapsed() > SEEK_TIMEOUT) {
            self.seek_started = None;
        }
        loop {
            let event = unsafe { &*(self.api.wait_event)(self.handle, 0.0) };
            match event.id {
                0 => break,
                EVENT_FILE_LOADED => self.state.loaded = true,
                EVENT_PLAYBACK_RESTART => {
                    self.seek_started = None;
                    // Les sauts internes (boucle, image précédente) passent aussi par ici
                    self.apply_chain();
                }
                EVENT_END_FILE if event.data.is_null() => {}
                EVENT_END_FILE => {
                    // mpv_event_end_file : { reason, error, ... } ; 4 = MPV_END_FILE_REASON_ERROR
                    let values = event.data as *const c_int;
                    if unsafe { *values } == 4 {
                        self.state.failed = true;
                        self.state.error = Some("mpv n'a pas pu ouvrir ce fichier".into());
                    }
                }
                EVENT_PROPERTY_CHANGE if !event.data.is_null() => {
                    let property = unsafe { &*(event.data as *const EventProperty) };
                    if property.data.is_null() {
                        continue;
                    }
                    let name = unsafe { CStr::from_ptr(property.name) };
                    let (flag, number) = unsafe {
                        match property.format {
                            FORMAT_FLAG => (*(property.data as *const c_int) != 0, 0.0),
                            FORMAT_DOUBLE => (false, *(property.data as *const c_double)),
                            _ => continue,
                        }
                    };
                    match name.to_bytes() {
                        // Pendant un déplacement, mpv rapporte encore l'ancienne position
                        b"time-pos" if self.seek_started.is_none() => self.state.position = number,
                        b"duration" => self.state.duration = number,
                        b"pause" => self.state.paused = flag,
                        b"speed" => self.state.speed = number,
                        b"container-fps" => self.state.fps = number,
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }

}

impl Drop for Player {
    fn drop(&mut self) {
        // Le contexte de rendu doit disparaître avant mpv, et sur son propre fil
        self.shared.slot.lock().unwrap().quit = true;
        self.shared.changed.notify_one();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        unsafe { (self.api.terminate_destroy)(self.handle) };
    }
}

/// Texte de l'option `af`. Le graphe des faders contient des crochets : sa longueur annoncée
/// (`%n%`) évite que mpv les prenne pour la fin de la valeur.
fn audio_chain(faders: Option<&str>, filter: Option<&str>, boost: f64) -> String {
    let mut parts = Vec::new();
    if let Some(graph) = faders {
        parts.push(format!("@{FADERS_LABEL}:lavfi=graph=%{}%{graph}", graph.len()));
    }
    if let Some(filter) = filter {
        parts.push(format!("lavfi=[{filter}]"));
    }
    // Toujours en dernier : rien ne sort au-dessus de 0 dB. `level=0` : pas de remise à niveau
    parts.push(format!("@{LIMITER_LABEL}:lavfi=[volume@boost={boost:.5},alimiter=limit=1:level=0]"));
    parts.join(",")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::Monitor;
    use std::ffi::OsStr;

    #[test]
    fn hidden_render_thread_delivers_upright_frames() {
        let mut player = Player::new(&egui::Context::default()).unwrap();
        player.set_video_size([320, 180]);
        // Moitié haute blanche, moitié basse noire : une image à l'envers se verrait
        player.load(Path::new("av://lavfi:color=c=black:s=320x180,drawbox=x=0:y=0:w=320:h=90:color=white:t=fill"));
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            player.poll();
            if let Some(frame) = player.take_frame() {
                assert_eq!(frame.size, [320, 180]);
                assert_eq!(frame.pixels.len(), 320 * 180);
                let [top, _, _, alpha] = frame.pixels[320 * 40 + 160].to_array();
                let [bottom, ..] = frame.pixels[320 * 140 + 160].to_array();
                if top > 200 {
                    assert!(bottom < 50, "haut {top}, bas {bottom}");
                    assert_eq!(alpha, 255);
                    return;
                }
            }
            assert!(Instant::now() < deadline, "aucune image reçue de mpv");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Fait lire un fichier par le vrai mpv, le son étant écrit dans un fichier. Retourne le
    /// niveau RMS des voies gauche et droite.
    fn played_levels(source: &Path, filter: Option<String>, output: &Path) -> (f64, f64) {
        played_chain(source, filter.map(|f| format!("lavfi=[{f}]")), None, output)
    }

    fn route_command(side: char, channel: usize, gain: f32) -> Vec<String> {
        ["af-command", FADERS_LABEL, "volume", &format!("{gain:.5}"), &format!("volume@{side}{channel}")].map(str::to_string).to_vec()
    }

    /// `at_start` : des commandes mpv à envoyer une fois, dès que la lecture démarre.
    fn played_chain(source: &Path, chain: Option<String>, at_start: Option<Vec<Vec<String>>>, output: &Path) -> (f64, f64) {
        let api = Api::load().unwrap();
        let output_text = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let source_text = CString::new(source.to_string_lossy().as_bytes()).unwrap();
        unsafe {
            let handle = (api.create)();
            for (name, value) in [
                (c"config", c"no"),
                (c"load-scripts", c"no"),
                (c"vid", c"no"),
                (c"ao", c"pcm"),
                (c"audio-format", c"s16"),
                (c"audio-channels", c"stereo"),
                (c"untimed", c"yes"),
            ] {
                assert!((api.set_option_string)(handle, name.as_ptr(), value.as_ptr()) >= 0, "{name:?}");
            }
            (api.set_option_string)(handle, c"ao-pcm-file".as_ptr(), output_text.as_ptr());
            // Sans horloge, tout le son serait écrit avant que la commande n'arrive
            if at_start.is_some() {
                (api.set_option_string)(handle, c"pause".as_ptr(), c"yes".as_ptr());
            }
            if let Some(chain) = &chain {
                let af = CString::new(chain.as_str()).unwrap();
                assert!((api.set_option_string)(handle, c"af".as_ptr(), af.as_ptr()) >= 0, "{chain}");
            }
            assert!((api.initialize)(handle) >= 0);
            let args = [c"loadfile".as_ptr(), source_text.as_ptr(), std::ptr::null()];
            assert!((api.command)(handle, args.as_ptr()) >= 0);
            let deadline = Instant::now() + Duration::from_secs(30);
            let mut started = false;
            loop {
                let event = &*(api.wait_event)(handle, 1.0);
                if event.id == EVENT_END_FILE {
                    break;
                }
                if let (EVENT_PLAYBACK_RESTART, Some(commands), false) = (event.id, &at_start, started) {
                    started = true;
                    for command in commands {
                        let texts: Vec<CString> = command.iter().map(|t| CString::new(t.as_str()).unwrap()).collect();
                        let mut pointers: Vec<*const c_char> = texts.iter().map(|t| t.as_ptr()).collect();
                        pointers.push(std::ptr::null());
                        assert!((api.command)(handle, pointers.as_ptr()) >= 0, "commande refusée : {command:?}");
                    }
                    let resume = [c"set".as_ptr(), c"pause".as_ptr(), c"no".as_ptr(), std::ptr::null()];
                    assert!((api.command)(handle, resume.as_ptr()) >= 0);
                }
                assert!(Instant::now() < deadline, "mpv n'a pas fini");
            }
            (api.terminate_destroy)(handle);
        }
        let wav = std::fs::read(output).unwrap();
        let start = wav.windows(4).position(|w| w == b"data").expect("pas de son écrit") + 8;
        // Ce qui était déjà préparé avant les commandes garde l'ancien réglage : on ne le mesure pas
        let skipped = if at_start.is_some() { 48000 * 3 * 4 } else { 0 };
        let samples: Vec<f64> = wav[start + skipped..]
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f64 / 32768.0)
            .collect();
        assert!(samples.len() > 48000, "{} échantillons", samples.len());
        let rms = |side: usize| {
            let values: Vec<f64> = samples.iter().skip(side).step_by(2).copied().collect();
            (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt()
        };
        (rms(0), rms(1))
    }

    /// Solo et mute appliqués en pleine lecture, sans relancer la chaîne, sur le son produit.
    #[test]
    fn solo_and_mute_work_live_through_the_routing_graph() {
        let folder = std::env::temp_dir().join(format!("music-studio-routing-{}", std::process::id()));
        std::fs::create_dir_all(&folder).unwrap();
        // Source 5.1 : le même sinus dans le central et l'arrière gauche
        let source = folder.join("source.mka");
        let (log, ok) = mse::ffmpeg::run(&[
            OsStr::new("-y"),
            OsStr::new("-f"),
            OsStr::new("lavfi"),
            OsStr::new("-i"),
            OsStr::new("sine=frequency=440:duration=20:sample_rate=48000"),
            OsStr::new("-af"),
            OsStr::new("pan=5.1|FC=c0|BL=c0"),
            OsStr::new("-c:a"),
            OsStr::new("flac"),
            source.as_os_str(),
        ])
        .unwrap();
        assert!(ok, "{log}");
        let names: Vec<String> = ["L", "R", "C", "LFE", "Ls", "Rs"].iter().map(|n| n.to_string()).collect();
        let chain = audio_chain(Monitor::default().routing_graph("5.1").as_deref(), None, 1.0);
        let run = |monitor: &Monitor, name: &str| {
            let commands: Vec<Vec<String>> = monitor
                .matrix(&names)
                .iter()
                .enumerate()
                .flat_map(|(c, (l, r))| [route_command('l', c, *l), route_command('r', c, *r)])
                .collect();
            played_chain(&source, Some(chain.clone()), Some(commands), &folder.join(name))
        };
        let mut monitor = Monitor::default();
        monitor.toggle_solo(2); // C : un canal seul sort à plein niveau, des deux côtés
        let (left, right) = run(&monitor, "solo.wav");
        assert!((left - right).abs() < 0.002, "{left} {right}");
        let full = 0.125 * std::f64::consts::FRAC_1_SQRT_2;
        assert!(left > full * 0.9, "{left}");

        monitor.clear();
        monitor.toggle_mute(2); // Il ne reste que l'arrière gauche : rien à droite
        let (left, right) = run(&monitor, "mute.wav");
        std::fs::remove_dir_all(&folder).unwrap();
        assert!(left > 0.01 && right < left * 0.1, "{left} {right}");
    }

    /// Un saut reconstruit la chaîne depuis son texte : une commande seule est alors perdue,
    /// et le texte mis à jour avant le saut garde le réglage.
    #[test]
    fn a_seek_keeps_the_mute_only_if_the_chain_text_is_up_to_date() {
        let folder = std::env::temp_dir().join(format!("music-studio-seek-{}", std::process::id()));
        std::fs::create_dir_all(&folder).unwrap();
        let source = folder.join("source.mka");
        let (log, ok) = mse::ffmpeg::run(&[
            OsStr::new("-y"),
            OsStr::new("-f"),
            OsStr::new("lavfi"),
            OsStr::new("-i"),
            OsStr::new("sine=frequency=440:duration=20:sample_rate=48000"),
            OsStr::new("-af"),
            OsStr::new("pan=5.1|FC=c0|BL=c0"),
            OsStr::new("-c:a"),
            OsStr::new("flac"),
            source.as_os_str(),
        ])
        .unwrap();
        assert!(ok, "{log}");
        let names: Vec<String> = ["L", "R", "C", "LFE", "Ls", "Rs"].iter().map(|n| n.to_string()).collect();
        let start = audio_chain(Monitor::default().routing_graph("5.1").as_deref(), None, 1.0);
        let mut muted = Monitor::default();
        muted.toggle_mute(2);
        let seek: Vec<String> = ["seek", "2", "absolute+exact"].map(str::to_string).to_vec();

        // Commandes seules, puis saut : le centre revient à droite
        let mut lost: Vec<Vec<String>> = muted
            .matrix(&names)
            .iter()
            .enumerate()
            .flat_map(|(c, (l, r))| [route_command('l', c, *l), route_command('r', c, *r)])
            .collect();
        lost.push(seek.clone());
        let (left_lost, right_lost) = played_chain(&source, Some(start.clone()), Some(lost), &folder.join("lost.wav"));

        // Texte à jour avant le saut : le mute tient
        let text = audio_chain(muted.routing_graph("5.1").as_deref(), None, 1.0);
        let kept = vec![["set", "af", text.as_str()].map(str::to_string).to_vec(), seek];
        let (left_kept, right_kept) = played_chain(&source, Some(start), Some(kept), &folder.join("kept.wav"));
        std::fs::remove_dir_all(&folder).unwrap();
        assert!(right_lost > left_lost * 0.2, "le saut devrait perdre la commande : {left_lost} {right_lost}");
        assert!(left_kept > 0.01 && right_kept < left_kept * 0.1, "le mute doit tenir : {left_kept} {right_kept}");
    }

    /// Le limiteur vérifié sur le son produit : pousser le volume ne doit pas écrêter.
    #[test]
    fn boosted_volume_is_limited_and_quiet_sound_is_untouched() {
        let folder = std::env::temp_dir().join(format!("music-studio-limiter-{}", std::process::id()));
        std::fs::create_dir_all(&folder).unwrap();
        let tone = |name: &str, amplitude: f64| {
            let source = folder.join(name);
            let expression = format!("aevalsrc={amplitude}*sin(2*PI*440*t)|{amplitude}*sin(2*PI*440*t):d=5:s=48000");
            let (log, ok) = mse::ffmpeg::run(&[
                OsStr::new("-y"),
                OsStr::new("-f"),
                OsStr::new("lavfi"),
                OsStr::new("-i"),
                OsStr::new(&expression),
                OsStr::new("-c:a"),
                OsStr::new("pcm_f32le"),
                source.as_os_str(),
            ])
            .unwrap();
            assert!(ok, "{log}");
            source
        };
        let loud = tone("loud.wav", 0.9);
        let quiet = tone("quiet.wav", 0.25);
        let (limited, _) = played_chain(&loud, Some(audio_chain(None, None, 2.0)), None, &folder.join("a.wav"));
        // Écrêté, ce sinus aurait un RMS d'environ 0,9 ; limité à 0 dB, au plus 0,71
        assert!(limited > 0.5 && limited < 0.74, "{limited}");
        let (untouched, _) = played_chain(&quiet, Some(audio_chain(None, None, 1.0)), None, &folder.join("b.wav"));
        std::fs::remove_dir_all(&folder).unwrap();
        assert!((untouched - 0.25 * std::f64::consts::FRAC_1_SQRT_2).abs() < 0.005, "{untouched}");
    }

    /// Les faders vérifiés sur le son réellement produit, réglés pendant la lecture.
    #[test]
    fn a_fader_changes_the_sound_without_rebuilding_the_chain() {
        let folder = std::env::temp_dir().join(format!("music-studio-fader-{}", std::process::id()));
        std::fs::create_dir_all(&folder).unwrap();
        let source = folder.join("source.wav");
        let (log, ok) = mse::ffmpeg::run(&[
            OsStr::new("-y"),
            OsStr::new("-f"),
            OsStr::new("lavfi"),
            OsStr::new("-i"),
            OsStr::new("sine=frequency=440:duration=30:sample_rate=48000"),
            OsStr::new("-af"),
            OsStr::new("pan=stereo|c0=c0|c1=c0"),
            source.as_os_str(),
        ])
        .unwrap();
        assert!(ok, "{log}");
        let graph = Monitor::default().routing_graph("stereo").unwrap();
        let chain = audio_chain(Some(&graph), None, 1.0);
        let (left0, right0) = played_chain(&source, Some(chain.clone()), None, &folder.join("plain.wav"));
        assert!(left0 > 0.01 && (left0 - right0).abs() < 0.001, "{left0} {right0}");
        let (left, right) = played_chain(&source, Some(chain), Some(vec![route_command('l', 0, 0.5)]), &folder.join("fader.wav"));
        std::fs::remove_dir_all(&folder).unwrap();
        assert!(left < left0 * 0.6, "le gauche doit baisser : {left} contre {left0}");
        assert!((right - right0).abs() < 0.001, "le droit ne bouge pas : {right} contre {right0}");
    }

    /// Solo et mute vérifiés sur le son réellement produit, pas sur le texte du filtre.
    #[test]
    fn solo_and_mute_change_what_mpv_really_plays() {
        let folder = std::env::temp_dir().join(format!("music-studio-monitor-{}", std::process::id()));
        std::fs::create_dir_all(&folder).unwrap();
        // Source 5.1 : un sinus dans le central et un autre, différent, dans l'arrière gauche
        let source = folder.join("source.mka");
        let (log, ok) = mse::ffmpeg::run(&[
            OsStr::new("-y"),
            OsStr::new("-f"),
            OsStr::new("lavfi"),
            OsStr::new("-i"),
            OsStr::new("sine=frequency=440:duration=2:sample_rate=48000"),
            OsStr::new("-af"),
            OsStr::new("pan=5.1|FC=c0|BL=0.5*c0"),
            OsStr::new("-c:a"),
            OsStr::new("pcm_s16le"),
            source.as_os_str(),
        ])
        .unwrap();
        assert!(ok, "{log}");
        let names: Vec<String> = ["L", "R", "C", "LFE", "Ls", "Rs"].iter().map(|n| n.to_string()).collect();
        let output = folder.join("out.wav");
        const SILENCE: f64 = 1e-4;
        let levels = |set: &dyn Fn(&mut Monitor)| {
            let mut monitor = Monitor::default();
            set(&mut monitor);
            played_levels(&source, monitor.filter(&names), &output)
        };

        // Solo du central : le son sort des deux côtés
        let (left, right) = levels(&|m| m.toggle_solo(2));
        assert!(left > 0.01 && right > 0.01, "{left} {right}");
        assert!((left - right).abs() < 1e-3, "{left} {right}");

        // Solo d'un canal vide : silence
        let (left, right) = levels(&|m| m.toggle_solo(0));
        assert!(left < SILENCE && right < SILENCE, "{left} {right}");

        // Solo du LFE, vide ici : silence, mais surtout un filtre accepté
        let (left, right) = levels(&|m| m.toggle_solo(3));
        assert!(left < SILENCE && right < SILENCE, "{left} {right}");

        // Mute du central : il ne reste que l'arrière gauche, donc à gauche seulement
        let (left, right) = levels(&|m| m.toggle_mute(2));
        assert!(left > 0.01, "{left}");
        assert!(right < SILENCE, "{right}");

        // Mute du central et de l'arrière gauche : plus rien
        let (left, right) = levels(&|m| {
            m.toggle_mute(2);
            m.toggle_mute(4);
        });
        assert!(left < SILENCE && right < SILENCE, "{left} {right}");

        // Sans solo ni mute : la vidéo sort normalement
        let (left, right) = levels(&|_| {});
        assert!(left > 0.01 && right > 0.01, "{left} {right}");

        let _ = std::fs::remove_dir_all(&folder);
    }
}
