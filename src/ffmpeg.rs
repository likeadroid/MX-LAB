//! ffmpeg : sous Windows, embarqué et extrait une fois dans le dossier local de l'utilisateur ;
//! ailleurs, livré à côté de l'exécutable ou installé sur le système.

use std::io::{self, Read};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::OnceLock;

static PATH: OnceLock<Result<PathBuf, String>> = OnceLock::new();

#[cfg(windows)]
static COMPRESSED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ffmpeg.exe.zst"));

#[cfg(windows)]
fn locate() -> Result<PathBuf, String> {
    extract().map_err(|e| format!("extraction de ffmpeg impossible : {e}"))
}

#[cfg(not(windows))]
fn locate() -> Result<PathBuf, String> {
    // Sinon, celui du PATH : un terminal en a souvent un ailleurs
    Ok(crate::paths::find_tool(&["ffmpeg"]).unwrap_or_else(|| PathBuf::from("ffmpeg")))
}

#[cfg(windows)]
fn extract() -> io::Result<PathBuf> {
    use std::fs;
    let folder = crate::paths::data_dir();
    fs::create_dir_all(&folder)?;
    // La taille compressée distingue les versions de ffmpeg embarquées
    let target = folder.join(format!("ffmpeg-{}.exe", COMPRESSED.len()));
    if target.is_file() {
        return Ok(target);
    }
    let partial = folder.join(format!("ffmpeg-{}.{}.tmp", COMPRESSED.len(), std::process::id()));
    let mut decoder = zstd::Decoder::new(COMPRESSED)?;
    decoder.window_log_max(27)?;
    io::copy(&mut decoder, &mut fs::File::create(&partial)?)?;
    if let Err(error) = fs::rename(&partial, &target) {
        let _ = fs::remove_file(&partial);
        // Une autre instance a pu l'extraire entre-temps
        if !target.is_file() {
            return Err(error);
        }
    }
    Ok(target)
}

fn command() -> Result<Command, String> {
    let path = PATH.get_or_init(locate);
    let mut command = Command::new(path.as_ref().map_err(Clone::clone)?);
    command.stdin(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    Ok(command)
}

/// Lance ffmpeg et retourne son journal d'erreurs et son code de sortie.
pub fn run(args: &[&std::ffi::OsStr]) -> Result<(String, bool), String> {
    let output = command()?
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("ffmpeg n'a pas pu démarrer : {e}"))?;
    Ok((String::from_utf8_lossy(&output.stderr).into_owned(), output.status.success()))
}

/// ffmpeg en cours d'exécution, dont on lit la sortie standard.
pub struct Stream {
    child: Child,
    errors: Option<std::thread::JoinHandle<String>>,
}

impl Stream {
    pub fn spawn(args: &[&std::ffi::OsStr]) -> Result<Self, String> {
        let mut child = command()?
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("ffmpeg n'a pas pu démarrer : {e}"))?;
        // Lu à part : un journal plein bloquerait ffmpeg
        let mut stderr = child.stderr.take().unwrap();
        let errors = std::thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = stderr.read_to_end(&mut bytes);
            String::from_utf8_lossy(&bytes).into_owned()
        });
        Ok(Stream { child, errors: Some(errors) })
    }

    pub fn stdout(&mut self) -> &mut impl Read {
        self.child.stdout.as_mut().unwrap()
    }

    pub fn finish(mut self) -> Result<(), String> {
        drop(self.child.stdout.take());
        let status = self.child.wait().map_err(|e| e.to_string())?;
        let message = self.errors.take().unwrap().join().unwrap_or_default();
        if status.success() {
            Ok(())
        } else {
            Err(format!("ffmpeg a échoué : {}", message.trim()))
        }
    }

    /// Comme `finish`, en retournant le journal de ffmpeg.
    pub fn finish_with_log(mut self) -> Result<String, String> {
        drop(self.child.stdout.take());
        let status = self.child.wait().map_err(|e| e.to_string())?;
        let log = self.errors.take().unwrap().join().unwrap_or_default();
        if status.success() {
            Ok(log)
        } else {
            let last = log.lines().rev().find(|line| !line.trim().is_empty()).unwrap_or("");
            Err(format!("ffmpeg a échoué : {}", last.trim()))
        }
    }
}
