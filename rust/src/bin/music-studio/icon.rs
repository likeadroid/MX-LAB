// L'icône de Music Studio, dessinée : barres de niveau et tête de lecture rouge.
// Incluse aussi par build.rs, pour l'icône de l'exe.

/// Pixels RGBA d'une icône carrée de `size` pixels.
pub fn icon_rgba(size: u32) -> Vec<u8> {
    const SAMPLES: u32 = 4;
    const BARS: [(f32, f32); 5] = [(0.17, 0.36), (0.31, 0.62), (0.45, 0.44), (0.59, 0.72), (0.73, 0.50)];
    const BAR_WIDTH: f32 = 0.10;
    const RADIUS: f32 = 0.2;
    let mut pixels = Vec::with_capacity((size * size * 4) as usize);
    for py in 0..size {
        for px in 0..size {
            let mut sum = [0.0f32; 4];
            for sy in 0..SAMPLES {
                for sx in 0..SAMPLES {
                    let x = (px as f32 + (sx as f32 + 0.5) / SAMPLES as f32) / size as f32;
                    let y = (py as f32 + (sy as f32 + 0.5) / SAMPLES as f32) / size as f32;
                    // Carré aux coins arrondis
                    let dx = (x - 0.5).abs() - (0.5 - RADIUS);
                    let dy = (y - 0.5).abs() - (0.5 - RADIUS);
                    let outside = dx.max(0.0).hypot(dy.max(0.0));
                    if outside > RADIUS {
                        continue;
                    }
                    let mut color = [28.0, 31.0, 38.0];
                    for (left, height) in BARS {
                        if x >= left && x < left + BAR_WIDTH && y <= 0.84 && y >= 0.84 - height {
                            color = [62.0, 190.0, 165.0];
                        }
                    }
                    // Tête de lecture
                    let head = 0.52;
                    if (x - head).abs() < 0.022 && y > 0.12 && y < 0.9 {
                        color = [255.0, 72.0, 72.0];
                    }
                    if y >= 0.08 && y < 0.2 && (x - head).abs() < (0.2 - y) * 0.75 {
                        color = [255.0, 72.0, 72.0];
                    }
                    for channel in 0..3 {
                        sum[channel] += color[channel];
                    }
                    sum[3] += 255.0;
                }
            }
            let count = (SAMPLES * SAMPLES) as f32;
            let alpha = sum[3] / count;
            for channel in 0..3 {
                let value = if sum[3] > 0.0 { sum[channel] / (sum[3] / 255.0) } else { 0.0 };
                pixels.push(value as u8);
            }
            pixels.push(alpha as u8);
        }
    }
    pixels
}