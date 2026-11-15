//! Où MX-LAB range ses fichiers et trouve ses outils, selon le système.

use std::path::PathBuf;

/// Dossier local de l'application : réglages, caches, outils extraits.
/// Windows : `%LOCALAPPDATA%\MX-LAB` ; macOS : `~/Library/Application Support/MX-LAB` ;
/// Linux : `$XDG_DATA_HOME/mx-lab` (par défaut `~/.local/share/mx-lab`).
pub fn data_dir() -> PathBuf {
    let home = || std::env::var_os("HOME").map(PathBuf::from);
    let base = if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(|base| PathBuf::from(base).join("MX-LAB"))
    } else if cfg!(target_os = "macos") {
        home().map(|home| home.join("Library/Application Support/MX-LAB"))
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
            .or_else(|| home().map(|home| home.join(".local/share")))
            .map(|base| base.join("mx-lab"))
    };
    base.unwrap_or_else(|| std::env::temp_dir().join("MX-LAB"))
}

/// Outil livré avec MX-LAB ou installé sur le système, dans cet ordre :
/// à côté de l'exécutable, dans `Contents/Frameworks` ou `Contents/MacOS` d'un .app,
/// puis dans les dossiers de Homebrew et du système (une app lancée du Finder n'a pas leur PATH).
#[cfg(not(windows))]
pub fn find_tool(names: &[&str]) -> Option<PathBuf> {
    let mut folders = Vec::new();
    if let Some(exe) = std::env::current_exe().ok().and_then(|exe| exe.parent().map(PathBuf::from)) {
        folders.push(exe.join("../Frameworks"));
        folders.push(exe.join("../lib"));
        folders.push(exe);
    }
    for folder in ["/opt/homebrew/bin", "/opt/homebrew/lib", "/usr/local/bin", "/usr/local/lib", "/usr/bin"] {
        folders.push(PathBuf::from(folder));
    }
    folders.iter().flat_map(|folder| names.iter().map(move |name| folder.join(name))).find(|path| path.is_file())
}

/// Ouvre un dossier dans le gestionnaire de fichiers du système.
pub fn reveal(folder: &std::path::Path) {
    let program = if cfg!(windows) {
        "explorer"
    } else if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    let _ = std::process::Command::new(program).arg(folder).spawn();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_dir_is_absolute_and_named_after_the_app() {
        let folder = data_dir();
        assert!(folder.is_absolute(), "{folder:?}");
        assert!(folder.to_string_lossy().to_lowercase().ends_with("mx-lab"), "{folder:?}");
    }
}
