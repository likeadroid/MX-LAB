//! Projets récents du menu Fichier, gardés d'un lancement à l'autre.

use std::path::{Path, PathBuf};

const MAX_RECENT: usize = 10;
const MAX_LABEL_CHARS: usize = 60;

#[derive(Default)]
pub struct RecentProjects {
    paths: Vec<PathBuf>,
    /// Absent : la liste reste en mémoire (les tests n'écrivent pas chez l'utilisateur).
    file: Option<PathBuf>,
}

fn same_path(a: &Path, b: &Path) -> bool {
    // Windows ne distingue pas les majuscules dans les chemins
    a.to_string_lossy().to_lowercase() == b.to_string_lossy().to_lowercase()
}

/// Le nom montré dans le menu : celui de la vidéo, raccourci s'il est très long.
pub fn label(path: &Path) -> String {
    let stem = path.file_stem().unwrap_or(path.as_os_str()).to_string_lossy();
    if stem.chars().count() <= MAX_LABEL_CHARS {
        return stem.into_owned();
    }
    let kept: String = stem.chars().take(MAX_LABEL_CHARS - 1).collect();
    format!("{kept}…")
}

impl RecentProjects {
    pub fn load() -> Self {
        Self::load_from(mx_lab::paths::data_dir().join("studio-recent.json"))
    }

    fn load_from(file: PathBuf) -> Self {
        let paths = std::fs::read_to_string(&file)
            .ok()
            .and_then(|text| serde_json::from_str::<Vec<PathBuf>>(&text).ok())
            .unwrap_or_default();
        RecentProjects { paths, file: Some(file) }
    }

    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    /// La vidéo qu'on vient d'ouvrir passe en tête.
    pub fn add(&mut self, path: &Path) {
        self.paths.retain(|known| !same_path(known, path));
        self.paths.insert(0, path.to_path_buf());
        self.paths.truncate(MAX_RECENT);
        self.save();
    }

    pub fn remove(&mut self, path: &Path) {
        self.paths.retain(|known| !same_path(known, path));
        self.save();
    }

    pub fn clear(&mut self) {
        self.paths.clear();
        self.save();
    }

    fn save(&self) {
        let Some(file) = &self.file else { return };
        if let Some(folder) = file.parent() {
            let _ = std::fs::create_dir_all(folder);
        }
        if let Ok(text) = serde_json::to_string_pretty(&self.paths) {
            let _ = std::fs::write(file, text);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latest_comes_first_without_duplicates_and_stays_bounded() {
        let mut recent = RecentProjects::default();
        for index in 0..15 {
            recent.add(Path::new(&format!("C:/films/{index}.mkv")));
        }
        assert_eq!(recent.paths().len(), MAX_RECENT);
        assert_eq!(recent.paths()[0], Path::new("C://films//14.mkv"));
        recent.add(Path::new("c://FILMS//10.MKV"));
        assert_eq!(recent.paths().len(), MAX_RECENT);
        assert_eq!(recent.paths()[0], Path::new("c://FILMS//10.MKV"));
    }

    #[test]
    fn the_list_survives_a_restart_and_can_be_cleared() {
        let folder = std::env::temp_dir().join(format!("music-studio-recent-{}", std::process::id()));
        let file = folder.join("recent.json");
        let mut recent = RecentProjects::load_from(file.clone());
        recent.add(Path::new("E://Vidéos//Thème d'été.mp4"));
        let mut reopened = RecentProjects::load_from(file.clone());
        assert_eq!(reopened.paths(), [PathBuf::from("E://Vidéos//Thème d'été.mp4")]);
        reopened.clear();
        assert!(RecentProjects::load_from(file).paths().is_empty());
        let _ = std::fs::remove_dir_all(folder);
    }

    #[test]
    fn long_names_are_shortened_in_the_menu() {
        assert_eq!(label(Path::new("E://Film//Court.mkv")), "Court");
        let long = format!("E://Film//{}.mkv", "a".repeat(90));
        let shown = label(Path::new(&long));
        assert_eq!(shown.chars().count(), MAX_LABEL_CHARS);
        assert!(shown.ends_with('…'));
    }
}
