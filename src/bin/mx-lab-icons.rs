//! Icônes des paquets, depuis assets/icon.svg :
//!   mx-lab-icons png <taille> <fichier.png>     carré plein (Linux)
//!   mx-lab-icons macos <taille> <fichier.png>   marge et coins arrondis (macOS)

#[path = "../icon_render.rs"]
mod icon_render;

static SVG: &[u8] = include_bytes!("../../assets/icon.svg");

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let [kind, size, path] = arguments.as_slice() else {
        eprintln!("usage : mx-lab-icons png|macos <taille> <fichier.png>");
        std::process::exit(2);
    };
    let size: u32 = size.parse().expect("taille invalide");
    let pixmap = match kind.as_str() {
        "png" => icon_render::render_svg(SVG, size),
        "macos" => icon_render::render_macos(SVG, size),
        other => {
            eprintln!("type d'icône inconnu : {other}");
            std::process::exit(2);
        }
    };
    std::fs::write(path, icon_render::png(size, &icon_render::rgba(&pixmap))).expect("écriture de l'icône");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn icons_render_the_svg_and_encode_valid_png() {
        let full = icon_render::rgba(&icon_render::render_svg(SVG, 64));
        assert_eq!(full[3], 255, "le carré plein est opaque dans le coin");
        let mac = icon_render::rgba(&icon_render::render_macos(SVG, 64));
        assert_eq!(mac[3], 0, "la version macOS a une marge transparente");
        let center = (32 * 64 + 32) * 4;
        assert_eq!(mac[center + 3], 255);
        let file = icon_render::png(64, &full);
        assert_eq!(&file[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(&file[file.len() - 8..], b"IEND\xae\x42\x60\x82");
    }
}
