// L'icône de MX-LAB, rendue depuis assets/icon.svg. Inclus par build.rs (icône de l'exe Windows
// et de la fenêtre) et par mx-lab-icons (icônes des paquets macOS et Linux).

use resvg::tiny_skia::{FillRule, FilterQuality, Mask, PathBuilder, Pixmap, PixmapPaint, Transform};

/// Taille du rendu de référence, ensuite réduit : resvg plante sur certains filtres en petite
/// taille, et un flou rendu en grand puis réduit reste fidèle au dessin.
const MASTER_SIZE: u32 = 1024;

/// Le SVG rendu en carré de `size` pixels, sur toute la surface.
pub fn render_svg(svg: &[u8], size: u32) -> Pixmap {
    let tree = resvg::usvg::Tree::from_data(svg, &resvg::usvg::Options::default()).expect("icône SVG illisible");
    let master = MASTER_SIZE.max(size);
    let mut pixmap = Pixmap::new(master, master).expect("taille d'icône nulle");
    let scale = master as f32 / tree.size().width().max(tree.size().height());
    resvg::render(&tree, Transform::from_scale(scale, scale), &mut pixmap.as_mut());
    // Réduire de moitié en moitié, puis d'un coup en bicubique : sans crénelage
    while pixmap.width() / 2 >= size && pixmap.width() > size {
        pixmap = resize(&pixmap, pixmap.width() / 2, FilterQuality::Bilinear);
    }
    if pixmap.width() != size {
        pixmap = resize(&pixmap, size, FilterQuality::Bicubic);
    }
    pixmap
}

fn resize(source: &Pixmap, size: u32, quality: FilterQuality) -> Pixmap {
    let mut target = Pixmap::new(size, size).expect("taille d'icône nulle");
    let scale = size as f32 / source.width() as f32;
    let paint = PixmapPaint { quality, ..PixmapPaint::default() };
    target.draw_pixmap(0, 0, source.as_ref(), &paint, Transform::from_scale(scale, scale), None);
    target
}

/// Version macOS : dessin de 824 sur 1024 au centre, coins arrondis, marge transparente,
/// comme les icônes du système.
pub fn render_macos(svg: &[u8], size: u32) -> Pixmap {
    let content = ((size as f32 * 824.0 / 1024.0).round() as u32).max(1);
    let offset = (size - content) / 2;
    let inner = render_svg(svg, content);
    let radius = content as f32 * 0.225;
    let (x, y, side) = (offset as f32, offset as f32, content as f32);
    // Rectangle aux coins arrondis, arcs approchés par des courbes de Bézier
    let k = radius * (1.0 - 0.552_284_8);
    let mut builder = PathBuilder::new();
    builder.move_to(x + radius, y);
    builder.line_to(x + side - radius, y);
    builder.cubic_to(x + side - k, y, x + side, y + k, x + side, y + radius);
    builder.line_to(x + side, y + side - radius);
    builder.cubic_to(x + side, y + side - k, x + side - k, y + side, x + side - radius, y + side);
    builder.line_to(x + radius, y + side);
    builder.cubic_to(x + k, y + side, x, y + side - k, x, y + side - radius);
    builder.line_to(x, y + radius);
    builder.cubic_to(x, y + k, x + k, y, x + radius, y);
    builder.close();
    let path = builder.finish().expect("contour de l'icône");
    let mut mask = Mask::new(size, size).expect("masque de l'icône");
    mask.fill_path(&path, FillRule::Winding, true, Transform::identity());
    let mut pixmap = Pixmap::new(size, size).expect("taille d'icône nulle");
    pixmap.draw_pixmap(
        offset as i32,
        offset as i32,
        inner.as_ref(),
        &PixmapPaint::default(),
        Transform::identity(),
        Some(&mask),
    );
    pixmap
}

/// Pixels RGBA, non prémultipliés.
pub fn rgba(pixmap: &Pixmap) -> Vec<u8> {
    pixmap
        .pixels()
        .iter()
        .flat_map(|pixel| {
            let color = pixel.demultiply();
            [color.red(), color.green(), color.blue(), color.alpha()]
        })
        .collect()
}

/// PNG RGBA sans compression (blocs « stored ») : assez pour une icône, sans dépendance.
pub fn png(size: u32, rgba: &[u8]) -> Vec<u8> {
    let mut raw = Vec::with_capacity(rgba.len() + size as usize);
    for row in rgba.chunks_exact(size as usize * 4) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    let mut zlib = vec![0x78, 0x01];
    let mut blocks = raw.chunks(65535).peekable();
    while let Some(block) = blocks.next() {
        zlib.push(blocks.peek().is_none() as u8);
        let length = block.len() as u16;
        zlib.extend_from_slice(&length.to_le_bytes());
        zlib.extend_from_slice(&(!length).to_le_bytes());
        zlib.extend_from_slice(block);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in &raw {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    zlib.extend_from_slice(&((b << 16) | a).to_be_bytes());

    let mut header = Vec::new();
    header.extend_from_slice(&size.to_be_bytes());
    header.extend_from_slice(&size.to_be_bytes());
    header.extend_from_slice(&[8, 6, 0, 0, 0]);
    let mut file = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    for (kind, data) in [(b"IHDR", header.as_slice()), (b"IDAT", zlib.as_slice()), (b"IEND", &[][..])] {
        file.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let start = file.len();
        file.extend_from_slice(kind);
        file.extend_from_slice(data);
        let crc = crc32(&file[start..]);
        file.extend_from_slice(&crc.to_be_bytes());
    }
    file
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xedb8_8320 } else { crc >> 1 };
        }
    }
    !crc
}
