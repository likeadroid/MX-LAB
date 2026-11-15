//! Moteur de rendu de l'interface, au choix dans Affichage > Rendu.
//!
//! La vidéo n'en dépend pas : mpv rend à part et ses images sont recopiées vers egui.

use std::path::PathBuf;

// Chaque système n'en propose qu'une partie
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderBackend {
    Direct3D12,
    Metal,
    Vulkan,
    OpenGl,
}

impl Default for RenderBackend {
    /// Windows : le seul sans flash constaté, présenter en OpenGL fait clignoter la fenêtre.
    fn default() -> Self {
        Self::ALL[0]
    }
}

impl RenderBackend {
    /// Les moteurs du système, celui par défaut en premier.
    #[cfg(windows)]
    pub const ALL: &[RenderBackend] = &[RenderBackend::Direct3D12, RenderBackend::Vulkan, RenderBackend::OpenGl];
    #[cfg(target_os = "macos")]
    pub const ALL: &[RenderBackend] = &[RenderBackend::Metal, RenderBackend::OpenGl];
    #[cfg(not(any(windows, target_os = "macos")))]
    pub const ALL: &[RenderBackend] = &[RenderBackend::Vulkan, RenderBackend::OpenGl];

    pub fn label(self) -> &'static str {
        match self {
            RenderBackend::Direct3D12 => "Direct3D 12",
            RenderBackend::Metal => "Metal",
            RenderBackend::Vulkan => "Vulkan",
            RenderBackend::OpenGl => "OpenGL",
        }
    }

    fn key(self) -> &'static str {
        match self {
            RenderBackend::Direct3D12 => "dx12",
            RenderBackend::Metal => "metal",
            RenderBackend::Vulkan => "vulkan",
            RenderBackend::OpenGl => "opengl",
        }
    }

    fn parse(text: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|backend| backend.key() == text.trim())
    }

    fn path() -> PathBuf {
        mx_lab::paths::data_dir().join("studio-renderer.txt")
    }

    pub fn load() -> Self {
        std::fs::read_to_string(Self::path()).ok().and_then(|text| Self::parse(&text)).unwrap_or_default()
    }

    pub fn save(self) {
        let path = Self::path();
        if let Some(folder) = path.parent() {
            let _ = std::fs::create_dir_all(folder);
        }
        let _ = std::fs::write(path, self.key());
    }

    /// À appeler au démarrage, avant de créer la fenêtre.
    pub fn apply(self) -> eframe::Renderer {
        match self {
            RenderBackend::OpenGl => eframe::Renderer::Glow,
            backend => {
                std::env::set_var("WGPU_BACKEND", backend.key());
                eframe::Renderer::Wgpu
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_backend_reads_back_from_its_key() {
        for &backend in RenderBackend::ALL {
            assert_eq!(RenderBackend::parse(&format!("{}\n", backend.key())), Some(backend));
        }
        assert_eq!(RenderBackend::parse("inconnu"), None);
    }
}
