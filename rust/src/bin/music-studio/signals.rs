//! Analyse audio affichée sur la timeline : forme d'onde, spectrogramme, loudness
//! BS.1770 pour le mix et pour chaque canal. Vignettes et keyframes de la vidéo.
//! Tout est mis en cache hors du dossier de la vidéo : c'est recalculable.

use mse::ffmpeg;
use rayon::prelude::*;
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::ffi::OsStr;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const SAMPLE_RATE: usize = 48000; // coefficients BS.1770 donnés pour 48 kHz
pub const PEAK_HOP: usize = 240; // 200 mesures de crête par seconde
pub const BLOCK: usize = 4800; // 100 ms : pas de la loudness et du spectrogramme
pub const BLOCK_S: f64 = BLOCK as f64 / SAMPLE_RATE as f64;
pub const OVERVIEW_BINS: usize = 256;
pub const DETAIL_BINS: usize = 512;
pub const F_MIN: f64 = 30.0;
pub const F_MAX: f64 = 20000.0;
/// Analyse multi-résolution : une fenêtre longue sépare les notes graves, une fenêtre
/// courte garde les attaques nettes dans les aigus. (taille de FFT, fréquence maximale)
///
/// Une FFT a des raies régulièrement espacées alors que l'axe est logarithmique : les graves,
/// très étirés à l'écran, manquent de raies. D'où des fenêtres d'autant plus longues qu'on descend.
const LONGEST_FFT: usize = 32768;
const OVERVIEW_RESOLUTIONS: [(usize, f64); 3] = [(LONGEST_FFT, 150.0), (8192, 800.0), (4096, F_MAX)];
const DETAIL_RESOLUTIONS: [(usize, f64); 5] =
    [(LONGEST_FFT, 150.0), (16384, 400.0), (8192, 1000.0), (4096, 3000.0), (2048, F_MAX)];
/// Zoom serré : on voit le temps de près, une fenêtre de 0,7 s y étalerait trop les attaques.
const CLOSE_DETAIL_RESOLUTIONS: [(usize, f64); 4] = [(16384, 150.0), (8192, 500.0), (4096, 1500.0), (2048, F_MAX)];
const CLOSE_DETAIL_SPAN: f64 = 30.0;
/// Largeur du fondu entre deux résolutions, pour ne pas voir de cassure dans l'image.
const CROSSFADE_RATIO: f64 = 1.25;
const FLOOR_DB: f32 = -100.0;
pub const NO_LOUDNESS: f32 = -120.0;
const CACHE_VERSION: u32 = 1;
const SIGNALS_CACHE_VERSION: u32 = 7;
/// Suréchantillonnage du true peak (BS.1770) : 4 fois, sur 12 échantillons.
const TRUE_PEAK_TAPS: usize = 12;
const TRUE_PEAK_PHASES: usize = 3;
/// Histogramme du gating : 0,1 LU par case, de -70 à +30 LUFS.
const GATE_BINS: usize = 1000;

/// Noms des canaux dans l'ordre ffmpeg. `None` : disposition inconnue, réduite en stéréo.
pub fn channel_names(layout: &str) -> Option<Vec<&'static str>> {
    Some(match layout {
        "mono" => vec![],
        "stereo" => vec!["L", "R"],
        "2.1" => vec!["L", "R", "LFE"],
        "3.0" => vec!["L", "R", "C"],
        "4.0" => vec!["L", "R", "C", "Cs"],
        "quad" | "quad(side)" => vec!["L", "R", "Ls", "Rs"],
        "5.0" | "5.0(side)" => vec!["L", "R", "C", "Ls", "Rs"],
        "5.1" | "5.1(side)" => vec!["L", "R", "C", "LFE", "Ls", "Rs"],
        "6.1" => vec!["L", "R", "C", "LFE", "Cs", "Ls", "Rs"],
        "7.1" | "7.1(wide)" => vec!["L", "R", "C", "LFE", "Lb", "Rb", "Ls", "Rs"],
        _ => return None,
    })
}

/// Pondération d'un canal dans la loudness du programme (BS.1770).
pub fn loudness_weight(name: &str) -> f64 {
    match name {
        "LFE" => 0.0,
        "Ls" | "Rs" | "Lb" | "Rb" | "Cs" => 1.41,
        _ => 1.0,
    }
}

/// Part d'un canal dans le mix affiché.
fn mix_weight(name: &str) -> f32 {
    match name {
        "LFE" => 0.0,
        "L" | "R" => 1.0,
        _ => std::f32::consts::FRAC_1_SQRT_2,
    }
}

pub fn to_lufs(energy: f64) -> f32 {
    if energy <= 0.0 {
        NO_LOUDNESS
    } else {
        ((-0.691 + 10.0 * energy.log10()) as f32).max(NO_LOUDNESS)
    }
}

/// Filtre de pondération K (BS.1770), en deux étages.
#[derive(Clone, Default)]
struct KWeighting {
    shelf: [f64; 2],
    high_pass: [f64; 2],
}

impl KWeighting {
    #[inline]
    fn process(&mut self, x: f64) -> f64 {
        const B1: [f64; 3] = [1.53512485958697, -2.69169618940638, 1.19839281085285];
        const A1: [f64; 2] = [-1.69065929318241, 0.73248077421585];
        const B2: [f64; 3] = [1.0, -2.0, 1.0];
        const A2: [f64; 2] = [-1.99004745483398, 0.99007225036621];
        let y1 = B1[0] * x + self.shelf[0];
        self.shelf[0] = B1[1] * x - A1[0] * y1 + self.shelf[1];
        self.shelf[1] = B1[2] * x - A1[1] * y1;
        let y2 = B2[0] * y1 + self.high_pass[0];
        self.high_pass[0] = B2[1] * y1 - A2[0] * y2 + self.high_pass[1];
        self.high_pass[1] = B2[2] * y1 - A2[1] * y2;
        y2
    }
}

/// Mesure de crête vraie : retrouve les crêtes qui tombent entre deux échantillons.
struct TruePeak {
    phases: [[f32; TRUE_PEAK_TAPS]; TRUE_PEAK_PHASES],
    /// Écrit deux fois, pour lire les derniers échantillons d'un seul tenant.
    history: [f32; TRUE_PEAK_TAPS * 2],
    at: usize,
}

/// Fonction de Bessel modifiée d'ordre 0, pour la fenêtre de Kaiser.
fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term) = (1.0, 1.0);
    for k in 1..30 {
        term *= (x / 2.0 / k as f64).powi(2);
        sum += term;
    }
    sum
}

impl TruePeak {
    fn new() -> Self {
        const KAISER_BETA: f64 = 5.0;
        let half = TRUE_PEAK_TAPS as f64 / 2.0;
        let mut phases = [[0.0; TRUE_PEAK_TAPS]; TRUE_PEAK_PHASES];
        for (phase, taps) in phases.iter_mut().enumerate() {
            let back = half - (phase + 1) as f64 / 4.0;
            let raw: Vec<f64> = (0..TRUE_PEAK_TAPS)
                .map(|k| {
                    let d = k as f64 - back;
                    let sinc = if d == 0.0 { 1.0 } else { (std::f64::consts::PI * d).sin() / (std::f64::consts::PI * d) };
                    let edge = (1.0 - (d / half).powi(2)).max(0.0).sqrt();
                    sinc * bessel_i0(KAISER_BETA * edge) / bessel_i0(KAISER_BETA)
                })
                .collect();
            let total: f64 = raw.iter().sum();
            for (tap, value) in taps.iter_mut().zip(&raw) {
                *tap = (value / total) as f32;
            }
        }
        TruePeak { phases, history: [0.0; TRUE_PEAK_TAPS * 2], at: 0 }
    }

    /// Crête de l'échantillon et des 3 points interpolés qui le précèdent.
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        self.at = (self.at + TRUE_PEAK_TAPS - 1) % TRUE_PEAK_TAPS;
        self.history[self.at] = x;
        self.history[self.at + TRUE_PEAK_TAPS] = x;
        let recent = &self.history[self.at..self.at + TRUE_PEAK_TAPS];
        let mut sums = [0.0f32; TRUE_PEAK_PHASES];
        for (k, value) in recent.iter().enumerate() {
            for phase in 0..TRUE_PEAK_PHASES {
                sums[phase] += self.phases[phase][k] * value;
            }
        }
        sums.iter().fold(x.abs(), |peak, sum| peak.max(sum.abs()))
    }
}

/// Une taille de fenêtre de l'analyse.
struct Resolution {
    size: usize,
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    buffer: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    power: Vec<f32>,
}

/// D'où vient une bande de l'image : positions fractionnaires dans une résolution, et poids.
#[derive(Clone, Copy)]
struct BandSource {
    resolution: usize,
    low: f64,
    high: f64,
    weight: f32,
}

/// Calcule des colonnes de spectrogramme en bandes logarithmiques (dB vers 0-255).
pub struct Spectrum {
    resolutions: Vec<Resolution>,
    bands: Vec<Vec<BandSource>>,
}

impl Spectrum {
    pub fn overview() -> Self {
        Self::new(OVERVIEW_BINS, &OVERVIEW_RESOLUTIONS)
    }

    /// `span` : durée analysée, en secondes.
    pub fn detail(span: f64) -> Self {
        if span <= CLOSE_DETAIL_SPAN {
            Self::new(DETAIL_BINS, &CLOSE_DETAIL_RESOLUTIONS)
        } else {
            Self::new(DETAIL_BINS, &DETAIL_RESOLUTIONS)
        }
    }

    fn new(bins: usize, sizes: &[(usize, f64)]) -> Self {
        let mut planner = FftPlanner::new();
        let resolutions: Vec<Resolution> = sizes
            .iter()
            .map(|&(size, _)| {
                let fft = planner.plan_fft_forward(size);
                let scratch = vec![Complex::default(); fft.get_inplace_scratch_len()];
                let window = (0..size)
                    .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / size as f32).cos())
                    .collect();
                let buffer = vec![Complex::default(); size];
                Resolution { size, fft, window, buffer, scratch, power: vec![0.0; size / 2 + 1] }
            })
            .collect();
        let ratio = F_MAX / F_MIN;
        let bands = (0..bins)
            .map(|band| {
                let low_hz = F_MIN * ratio.powf(band as f64 / bins as f64);
                let high_hz = F_MIN * ratio.powf((band + 1) as f64 / bins as f64);
                let center = (low_hz * high_hz).sqrt();
                let source = |resolution: usize, weight: f32| {
                    let hz_per_bin = SAMPLE_RATE as f64 / sizes[resolution].0 as f64;
                    BandSource { resolution, low: low_hz / hz_per_bin, high: high_hz / hz_per_bin, weight }
                };
                // La résolution de la zone, avec un fondu autour de chaque changement
                let main = sizes.iter().position(|s| center <= s.1).unwrap_or(sizes.len() - 1);
                if main + 1 < sizes.len() {
                    let position = (center / sizes[main].1).ln() / CROSSFADE_RATIO.ln();
                    if position > -1.0 {
                        let next = ((position + 1.0) / 2.0) as f32;
                        return vec![source(main, 1.0 - next), source(main + 1, next)];
                    }
                }
                if main > 0 {
                    let position = (center / sizes[main - 1].1).ln() / CROSSFADE_RATIO.ln();
                    if position < 1.0 {
                        let next = ((position + 1.0) / 2.0) as f32;
                        return vec![source(main - 1, 1.0 - next), source(main, next)];
                    }
                }
                vec![source(main, 1.0)]
            })
            .collect();
        Spectrum { resolutions, bands }
    }

    /// `samples` : exactement LONGEST_FFT échantillons ; toutes les fenêtres sont centrées dessus.
    pub fn column(&mut self, samples: &[f32], out: &mut [u8]) {
        for resolution in &mut self.resolutions {
            let offset = (samples.len() - resolution.size) / 2;
            for (i, x) in samples[offset..offset + resolution.size].iter().enumerate() {
                resolution.buffer[i] = Complex::new(x * resolution.window[i], 0.0);
            }
            resolution.fft.process_with_scratch(&mut resolution.buffer, &mut resolution.scratch);
            // Une sinusoïde à pleine échelle donne |X| = N / 4 avec une fenêtre de Hann
            let scale = 4.0 / resolution.size as f32;
            for (power, value) in resolution.power.iter_mut().zip(&resolution.buffer) {
                *power = value.norm_sqr() * scale * scale;
            }
        }
        for (value, sources) in out.iter_mut().zip(&self.bands) {
            let db: f32 = sources
                .iter()
                .map(|source| source.weight * band_db(&self.resolutions[source.resolution].power, source.low, source.high))
                .sum();
            *value = ((db - FLOOR_DB) / -FLOOR_DB * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
}

/// Niveau d'une bande : le maximum des raies qu'elle couvre. Plus étroite qu'une raie, elle est
/// interpolée entre ses deux voisines, ce qui évite les marches d'escalier dans les graves.
fn band_db(power: &[f32], low: f64, high: f64) -> f32 {
    let last = power.len() - 1;
    let value = if high - low < 1.0 {
        let center = (low + high) / 2.0;
        let index = (center as usize).min(last - 1);
        let fraction = (center - index as f64) as f32;
        // En décibels : une moyenne des puissances comblerait les creux entre deux notes voisines
        let (a, b) = (10.0 * power[index].max(1e-12).log10(), 10.0 * power[index + 1].max(1e-12).log10());
        return a * (1.0 - fraction) + b * fraction;
    } else {
        let first = (low.ceil() as usize).min(last);
        let end = (high.floor() as usize).clamp(first, last);
        power[first..=end].iter().copied().fold(0.0f32, f32::max)
    };
    10.0 * value.max(1e-12).log10()
}

/// Crêtes [min, max] sur plusieurs niveaux de détail, pour dessiner sans tout parcourir.
#[derive(Default, Clone)]
pub struct Peaks {
    pub levels: Vec<Vec<[i16; 2]>>,
}

const PEAK_REDUCTION: usize = 8;

impl Peaks {
    fn build(base: Vec<[i16; 2]>) -> Self {
        let mut levels = vec![base];
        while levels.last().unwrap().len() > 2048 {
            let next = levels
                .last()
                .unwrap()
                .chunks(PEAK_REDUCTION)
                .map(|c| {
                    let min = c.iter().map(|p| p[0]).min().unwrap_or(0);
                    let max = c.iter().map(|p| p[1]).max().unwrap_or(0);
                    [min, max]
                })
                .collect();
            levels.push(next);
        }
        Peaks { levels }
    }

    /// Plus haute crête du canal entier, en -1..1.
    pub fn max_abs(&self) -> f32 {
        let coarsest = self.levels.last().map_or(&[][..], |level| level.as_slice());
        let top = coarsest.iter().map(|p| (p[0] as i32).abs().max((p[1] as i32).abs())).max().unwrap_or(0);
        top as f32 / 32767.0
    }

    /// Crêtes (min, max, en -1..1) d'un intervalle de temps en secondes.
    pub fn range(&self, from: f64, to: f64) -> Option<(f32, f32)> {
        let base_rate = (SAMPLE_RATE / PEAK_HOP) as f64;
        let wanted = (to - from) * base_rate;
        let mut level = 0;
        let mut factor = 1.0;
        while level + 1 < self.levels.len() && factor * PEAK_REDUCTION as f64 * 8.0 <= wanted {
            level += 1;
            factor *= PEAK_REDUCTION as f64;
        }
        let data = &self.levels[level];
        let rate = base_rate / factor;
        let first = (from.max(0.0) * rate) as usize;
        let last = ((to * rate).ceil() as usize).max(first + 1).min(data.len());
        if first >= last {
            return None;
        }
        let min = data[first..last].iter().map(|p| p[0]).min()?;
        let max = data[first..last].iter().map(|p| p[1]).max()?;
        Some((min as f32 / 32767.0, max as f32 / 32767.0))
    }
}

pub struct Channel {
    pub name: String,
    pub peaks: Peaks,
    pub momentary: Vec<f32>,
    pub short_term: Vec<f32>,
    /// Énergie pondérée K par bloc : de quoi tout mesurer sur n'importe quelle portion.
    pub energies: Vec<f32>,
    /// Crête vraie (linéaire) par bloc.
    pub true_peaks: Vec<f32>,
}

/// Les mesures d'une portion de la vidéo.
#[derive(Clone, Debug)]
pub struct Measure {
    pub true_peak: f32,
    pub integrated: f32,
    pub range: f32,
    pub max_short_term: f32,
    pub max_momentary: f32,
}

impl Channel {
    pub fn measure(&self, start: f64, end: f64) -> Measure {
        let count = self.energies.len();
        let first = ((start.max(0.0) / BLOCK_S) as usize).min(count);
        let last = ((end / BLOCK_S).ceil() as usize).clamp(first, count);
        let blocks: Vec<f64> = self.energies[first..last].iter().map(|&e| e as f64).collect();
        let peak = self.true_peaks.get(first..last).unwrap_or(&[]).iter().copied().fold(0.0, f32::max);
        Measure {
            true_peak: if peak > 0.0 { 20.0 * peak.log10() } else { NO_LOUDNESS },
            integrated: running_integrated(&blocks).last().copied().unwrap_or(NO_LOUDNESS),
            range: loudness_range(&blocks),
            max_short_term: loudest_window(&blocks, 30),
            max_momentary: loudest_window(&blocks, 4),
        }
    }

    /// L'intégrée de la vidéo depuis son début, un point par bloc ; `NO_LOUDNESS` tant qu'elle n'existe pas.
    pub fn running_integrated(&self) -> Vec<f32> {
        let blocks: Vec<f64> = self.energies.iter().map(|&e| e as f64).collect();
        running_integrated(&blocks)
    }
}

/// Canal 0 : le mix complet ; ensuite les canaux de la source.
pub struct Signals {
    pub channels: Vec<Channel>,
}

/// Analyse progressive d'un canal.
struct ChannelAnalyzer {
    name: String,
    peaks: Vec<[i16; 2]>,
    peak_low: f32,
    peak_high: f32,
    peak_count: usize,
    filter: KWeighting,
    energy: f64,
    block_count: usize,
    blocks: Vec<f64>,
    true_peak: Option<TruePeak>,
    block_peak: f32,
    true_peaks: Vec<f32>,
}

/// Ce qu'un canal livre à la fin de l'analyse.
struct Finished {
    name: String,
    peaks: Vec<[i16; 2]>,
    blocks: Vec<f64>,
    true_peaks: Vec<f32>,
}

impl ChannelAnalyzer {
    /// `true_peak` : à éviter sur le mix d'une source à plusieurs canaux, qui n'en a pas besoin.
    fn new(name: &str, true_peak: bool) -> Self {
        ChannelAnalyzer {
            name: name.to_string(),
            peaks: Vec::new(),
            peak_low: f32::MAX,
            peak_high: f32::MIN,
            peak_count: 0,
            filter: KWeighting::default(),
            energy: 0.0,
            block_count: 0,
            blocks: Vec::new(),
            true_peak: true_peak.then(TruePeak::new),
            block_peak: 0.0,
            true_peaks: Vec::new(),
        }
    }

    fn flush_peak(&mut self) {
        let clamp = |x: f32| (x.clamp(-1.0, 1.0) * 32767.0) as i16;
        self.peaks.push([clamp(self.peak_low), clamp(self.peak_high)]);
        self.peak_low = f32::MAX;
        self.peak_high = f32::MIN;
        self.peak_count = 0;
    }

    fn flush_block(&mut self) {
        self.blocks.push(self.energy / self.block_count as f64);
        self.true_peaks.push(self.block_peak);
        self.block_peak = 0.0;
        self.energy = 0.0;
        self.block_count = 0;
    }

    fn feed(&mut self, samples: &[f32]) {
        for &x in samples {
            self.peak_low = self.peak_low.min(x);
            self.peak_high = self.peak_high.max(x);
            if let Some(true_peak) = &mut self.true_peak {
                self.block_peak = self.block_peak.max(true_peak.process(x));
            }
            self.peak_count += 1;
            if self.peak_count == PEAK_HOP {
                self.flush_peak();
            }
            let weighted = self.filter.process(x as f64);
            self.energy += weighted * weighted;
            self.block_count += 1;
            if self.block_count == BLOCK {
                self.flush_block();
            }
        }
    }

    fn finish(mut self) -> Finished {
        if self.peak_count > 0 {
            self.flush_peak();
        }
        if self.block_count > 0 {
            self.flush_block();
        }
        Finished {
            name: self.name,
            peaks: self.peaks,
            blocks: self.blocks,
            true_peaks: self.true_peaks,
        }
    }
}

/// Moyenne glissante des énergies de bloc, en LUFS.
fn sliding_lufs(blocks: &[f64], width: usize) -> Vec<f32> {
    let mut sum = 0.0;
    (0..blocks.len())
        .map(|i| {
            sum += blocks[i];
            if i >= width {
                sum -= blocks[i - width];
            }
            // Le début, plus court que la fenêtre, reste une mesure valable : pas de trou
            to_lufs((sum / (i.min(width - 1) + 1) as f64).max(0.0))
        })
        .collect()
}

/// Énergies moyennes sur `width` blocs, uniquement pour les fenêtres complètes.
fn full_windows(blocks: &[f64], width: usize) -> Vec<f64> {
    blocks.windows(width).map(|w| w.iter().sum::<f64>() / width as f64).collect()
}

/// La fenêtre la plus forte. Une portion plus courte que la fenêtre compte pour une fenêtre.
fn loudest_window(blocks: &[f64], width: usize) -> f32 {
    let width = width.min(blocks.len());
    if width == 0 {
        return NO_LOUDNESS;
    }
    to_lufs(full_windows(blocks, width).into_iter().fold(0.0, f64::max))
}

/// Loudness intégrée avec double gating (BS.1770-4), telle qu'elle est à chaque bloc :
/// le dernier point est la mesure de toute la portion. Un histogramme évite de tout
/// reparcourir à chaque bloc, comme dans ebur128.
fn running_integrated(blocks: &[f64]) -> Vec<f32> {
    let bin = |lufs: f32| (((lufs + 70.0) * 10.0) as usize).min(GATE_BINS - 1);
    let mut sums = vec![0.0f64; GATE_BINS];
    let mut counts = vec![0u32; GATE_BINS];
    let (mut audible_sum, mut audible_count) = (0.0, 0u32);
    let mut running = vec![NO_LOUDNESS; blocks.len().min(3)];
    for z in full_windows(blocks, 4) {
        let lufs = to_lufs(z);
        if lufs > -70.0 {
            let index = bin(lufs);
            sums[index] += z;
            counts[index] += 1;
            audible_sum += z;
            audible_count += 1;
        }
        running.push(if audible_count == 0 {
            NO_LOUDNESS
        } else {
            let first = bin((to_lufs(audible_sum / audible_count as f64) - 10.0).max(-70.0));
            let kept: u32 = counts[first..].iter().sum();
            to_lufs(sums[first..].iter().sum::<f64>() / kept.max(1) as f64)
        });
    }
    running
}

/// Plage de loudness (LRA, EBU Tech 3342).
fn loudness_range(blocks: &[f64]) -> f32 {
    let windows = full_windows(blocks, 30);
    let audible: Vec<f64> = windows.iter().copied().filter(|&z| to_lufs(z) > -70.0).collect();
    if audible.is_empty() {
        return 0.0;
    }
    let gate = to_lufs(audible.iter().sum::<f64>() / audible.len() as f64) - 20.0;
    let mut kept: Vec<f32> = audible.iter().map(|&z| to_lufs(z)).filter(|&l| l > gate).collect();
    if kept.len() < 2 {
        return 0.0;
    }
    kept.sort_by(f32::total_cmp);
    let at = |p: f64| kept[((kept.len() - 1) as f64 * p).round() as usize];
    at(0.95) - at(0.10)
}

fn loudness_channel(name: String, peaks: Vec<[i16; 2]>, blocks: &[f64], true_peaks: Vec<f32>) -> Channel {
    Channel {
        name,
        peaks: Peaks::build(peaks),
        momentary: sliding_lufs(blocks, 4),
        short_term: sliding_lufs(blocks, 30),
        energies: blocks.iter().map(|&e| e as f32).collect(),
        true_peaks,
    }
}

/// Analyse complète d'un flux entrelacé : le découpage en morceaux est libre.
pub struct Analysis {
    names: Vec<&'static str>,
    mix_weights: Vec<f32>,
    mix: ChannelAnalyzer,
    channels: Vec<ChannelAnalyzer>,
    planar: Vec<Vec<f32>>,
    mix_samples: Vec<f32>,
}

impl Analysis {
    pub fn new(names: Vec<&'static str>) -> Self {
        let count = names.len().max(1);
        let mix_weights: Vec<f32> = if names.is_empty() {
            vec![1.0]
        } else {
            let weights: Vec<f32> = names.iter().map(|n| mix_weight(n)).collect();
            let total: f32 = weights.iter().sum();
            weights.iter().map(|w| w / total).collect()
        };
        Analysis {
            channels: names.iter().map(|n| ChannelAnalyzer::new(n, true)).collect(),
            mix: ChannelAnalyzer::new("Mix", names.is_empty()),
            mix_weights,
            names,
            planar: vec![Vec::new(); count],
            mix_samples: Vec::new(),
        }
    }

    pub fn feed(&mut self, interleaved: &[f32]) {
        let count = self.planar.len();
        let frames = interleaved.len() / count;
        self.mix_samples.clear();
        self.mix_samples.resize(frames, 0.0);
        for plane in &mut self.planar {
            plane.clear();
        }
        for (frame, mixed) in interleaved.chunks_exact(count).zip(&mut self.mix_samples) {
            for (channel, &x) in frame.iter().enumerate() {
                self.planar[channel].push(x);
                *mixed += x * self.mix_weights[channel];
            }
        }
        let planar = &self.planar;
        let mix_samples = &self.mix_samples;
        let mix = &mut self.mix;
        let channels = &mut self.channels;
        rayon::join(
            || mix.feed(mix_samples),
            || channels.par_iter_mut().zip(planar.par_iter()).for_each(|(a, s)| a.feed(s)),
        );
    }

    pub fn finish(self) -> Signals {
        let names = self.names;
        let mix = self.mix.finish();
        let parts: Vec<Finished> = self.channels.into_iter().map(ChannelAnalyzer::finish).collect();
        // Le programme somme les énergies des canaux et garde la plus haute de leurs crêtes ;
        // une source mono est son propre programme
        let (program, program_peaks): (Vec<f64>, Vec<f32>) = if parts.is_empty() {
            (mix.blocks, mix.true_peaks)
        } else {
            let length = parts.iter().map(|p| p.blocks.len()).min().unwrap_or(0);
            (0..length)
                .map(|i| {
                    let energy: f64 = names.iter().zip(&parts).map(|(n, p)| loudness_weight(n) * p.blocks[i]).sum();
                    (energy, parts.iter().map(|p| p.true_peaks[i]).fold(0.0, f32::max))
                })
                .unzip()
        };
        let mut channels = vec![loudness_channel(mix.name, mix.peaks, &program, program_peaks)];
        for part in parts {
            channels.push(loudness_channel(part.name, part.peaks, &part.blocks, part.true_peaks));
        }
        Signals { channels }
    }
}

fn read_all_into(reader: &mut impl Read, buffer: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0;
    while filled < buffer.len() {
        let read = reader.read(&mut buffer[filled..])?;
        if read == 0 {
            break;
        }
        filled += read;
    }
    Ok(filled)
}

pub fn analyze(video: &Path, track: usize, layout: &str, duration: f64, progress: &(dyn Fn(u32) + Sync)) -> Result<Signals, String> {
    let known = channel_names(layout);
    let names = known.clone().unwrap_or_else(|| vec!["L", "R"]);
    let rate = SAMPLE_RATE.to_string();
    let map = format!("0:a:{track}");
    let mut args: Vec<&OsStr> = vec![
        OsStr::new("-v"),
        OsStr::new("error"),
        OsStr::new("-i"),
        video.as_os_str(),
        OsStr::new("-map"),
        OsStr::new(&map),
        OsStr::new("-vn"),
        OsStr::new("-ar"),
        OsStr::new(&rate),
    ];
    if known.is_none() {
        args.extend([OsStr::new("-ac"), OsStr::new("2")]);
    }
    args.extend([OsStr::new("-f"), OsStr::new("f32le"), OsStr::new("-")]);
    let mut stream = ffmpeg::Stream::spawn(&args)?;

    let count = names.len().max(1);
    let mut analysis = Analysis::new(names);
    let mut raw = vec![0u8; SAMPLE_RATE * 5 * count * 4];
    let mut samples: Vec<f32> = Vec::with_capacity(SAMPLE_RATE * 5 * count);
    let mut total = 0usize;
    loop {
        let read = read_all_into(stream.stdout(), &mut raw).map_err(|e| e.to_string())?;
        let whole = read / (4 * count) * (4 * count);
        if whole == 0 {
            break;
        }
        samples.clear();
        samples.extend(raw[..whole].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())));
        analysis.feed(&samples);
        total += samples.len() / count;
        if duration > 0.0 {
            progress(((total as f64 / SAMPLE_RATE as f64 / duration * 100.0) as u32).min(99));
        }
        if read < raw.len() {
            break;
        }
    }
    stream.finish()?;
    Ok(analysis.finish())
}

thread_local! {
    /// Préparer une analyse coûte cher : chaque fil de calcul garde la sienne.
    static OVERVIEW_SPECTRUM: std::cell::RefCell<Option<Spectrum>> = const { std::cell::RefCell::new(None) };
}

/// Calcule des colonnes de la vue d'ensemble, centrées sur leur bloc, sur tous les cœurs.
/// `samples` commence à l'échantillon `base` ; ce qui manque autour compte pour du silence.
fn fill_columns(samples: &[f32], base: usize, first: usize, count: usize) -> Vec<u8> {
    let mut values = vec![0u8; count * OVERVIEW_BINS];
    values.par_chunks_mut(OVERVIEW_BINS).enumerate().for_each_init(
        || vec![0.0f32; LONGEST_FFT],
        |window, (offset, out)| {
            let center = (first + offset) * BLOCK + BLOCK / 2;
            let start = center as i64 - (LONGEST_FFT / 2) as i64 - base as i64;
            for (i, value) in window.iter_mut().enumerate() {
                let index = start + i as i64;
                *value = if index >= 0 && (index as usize) < samples.len() { samples[index as usize] } else { 0.0 };
            }
            OVERVIEW_SPECTRUM.with(|spectrum| spectrum.borrow_mut().get_or_insert_with(Spectrum::overview).column(window, out));
        },
    );
    values
}

/// Le filtre qui isole une piste affichée : un canal, ou le mix.
fn lane_filter(layout: &str, channel: usize) -> String {
    match channel_names(layout) {
        Some(names) if channel > 0 && !names.is_empty() => format!("pan=mono|c0=c{}", channel - 1),
        Some(names) if !names.is_empty() => {
            let total: f32 = names.iter().map(|n| mix_weight(n)).sum();
            let terms: Vec<String> =
                names.iter().enumerate().map(|(c, n)| format!("{:.5}*c{c}", mix_weight(n) / total)).collect();
            format!("pan=mono|c0={}", terms.join("+"))
        }
        _ => "aformat=channel_layouts=mono".to_string(),
    }
}

/// Spectre de toute la vidéo pour une piste, livré au fur et à mesure : (première colonne, valeurs).
pub fn overview_spectrum(
    video: &Path,
    layout: &str,
    channel: usize,
    track: usize,
    on_part: &mut dyn FnMut(usize, Vec<u8>),
) -> Result<(), String> {
    let cache = cache_path(video, &track_cache_kind(&format!("spectrum{channel}"), track));
    if let Some(values) = cache.as_deref().and_then(read_cache) {
        on_part(0, values);
        return Ok(());
    }
    let (filter, map, rate) = (lane_filter(layout, channel), format!("0:a:{track}"), SAMPLE_RATE.to_string());
    let args: [&OsStr; 13] = [
        OsStr::new("-v"),
        OsStr::new("error"),
        OsStr::new("-i"),
        video.as_os_str(),
        OsStr::new("-map"),
        OsStr::new(&map),
        OsStr::new("-af"),
        OsStr::new(&filter),
        OsStr::new("-ar"),
        OsStr::new(&rate),
        OsStr::new("-f"),
        OsStr::new("f32le"),
        OsStr::new("-"),
    ];
    let mut stream = ffmpeg::Stream::spawn(&args)?;
    let mut raw = vec![0u8; SAMPLE_RATE * 30 * 4];
    let (mut samples, mut base, mut next): (Vec<f32>, usize, usize) = (Vec::new(), 0, 0);
    let mut all = Vec::new();
    loop {
        let read = read_all_into(stream.stdout(), &mut raw).map_err(|e| e.to_string())?;
        let ended = read < raw.len();
        samples.extend(raw[..read / 4 * 4].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())));
        let total = base + samples.len();
        let margin = BLOCK / 2 + LONGEST_FFT / 2;
        let ready = if ended {
            total.div_ceil(BLOCK)
        } else if total >= margin {
            (total - margin) / BLOCK + 1
        } else {
            0
        };
        if ready > next {
            let values = fill_columns(&samples, base, next, ready - next);
            all.extend_from_slice(&values);
            on_part(next, values);
            next = ready;
            let keep = (next * BLOCK + BLOCK / 2).saturating_sub(LONGEST_FFT / 2).max(base);
            samples.drain(..(keep - base).min(samples.len()));
            base = keep;
        }
        if ended {
            break;
        }
    }
    stream.finish()?;
    if let Some(cache) = &cache {
        let _ = write_cache(cache, &all);
    }
    Ok(())
}

/// Spectrogramme fin d'un intervalle, calculé à la demande quand on zoome.
pub struct Detail {
    pub channel: usize,
    pub start: f64,
    pub end: f64,
    pub columns: usize,
    pub values: Vec<u8>, // DETAIL_BINS valeurs par colonne
}

pub fn detail_spectrogram(
    video: &Path,
    layout: &str,
    channel: usize,
    start: f64,
    end: f64,
    columns: usize,
    track: usize,
) -> Result<Detail, String> {
    let before = LONGEST_FFT as f64 / 2.0 / SAMPLE_RATE as f64;
    let from = (start - before).max(0.0);
    let (from_text, length) = (format!("{from:.6}"), format!("{:.6}", end - from + before));
    let rate = SAMPLE_RATE.to_string();
    let map = format!("0:a:{track}");
    let downmix = if channel > 0 && channel_names(layout).is_some() {
        format!("pan=mono|c0=c{}", channel - 1)
    } else {
        "aformat=channel_layouts=mono".to_string()
    };
    let args: [&OsStr; 17] = [
        OsStr::new("-v"),
        OsStr::new("error"),
        OsStr::new("-ss"),
        OsStr::new(&from_text),
        OsStr::new("-t"),
        OsStr::new(&length),
        OsStr::new("-i"),
        video.as_os_str(),
        OsStr::new("-map"),
        OsStr::new(&map),
        OsStr::new("-af"),
        OsStr::new(&downmix),
        OsStr::new("-ar"),
        OsStr::new(&rate),
        OsStr::new("-f"),
        OsStr::new("f32le"),
        OsStr::new("-"),
    ];
    let mut stream = ffmpeg::Stream::spawn(&args)?;
    let mut raw = Vec::new();
    stream.stdout().read_to_end(&mut raw).map_err(|e| e.to_string())?;
    stream.finish()?;
    let samples: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();

    let offset = (start - from) * SAMPLE_RATE as f64;
    let per_column = (end - start) * SAMPLE_RATE as f64 / columns as f64;
    let mut values = vec![0u8; columns * DETAIL_BINS];
    values.par_chunks_mut(DETAIL_BINS).enumerate().for_each_init(
        || (Spectrum::detail(end - start), vec![0.0f32; LONGEST_FFT]),
        |(spectrum, padded), (column, out)| {
            let center = offset + (column as f64 + 0.5) * per_column;
            let first = center as i64 - LONGEST_FFT as i64 / 2;
            if first >= 0 && first as usize + LONGEST_FFT <= samples.len() {
                spectrum.column(&samples[first as usize..first as usize + LONGEST_FFT], out);
            } else {
                // Bords de la vidéo : on complète par du silence
                for (i, value) in padded.iter_mut().enumerate() {
                    let index = first + i as i64;
                    *value = if index >= 0 && (index as usize) < samples.len() { samples[index as usize] } else { 0.0 };
                }
                spectrum.column(padded, out);
            }
        },
    );
    Ok(Detail { channel, start, end, columns, values })
}

/// Vignettes et positions des keyframes.
pub struct Strip {
    pub width: usize,
    pub height: usize,
    pub interval: f64,
    pub count: usize,
    pub pixels: Vec<u8>, // RGB, vignette après vignette
    pub keyframes: Vec<f64>,
}

const THUMB_WIDTH: usize = 96;
const THUMB_HEIGHT: usize = 54;
const MAX_THUMBS: f64 = 2000.0;

fn parse_keyframes(log: &str) -> Vec<f64> {
    let mut times: Vec<f64> = log
        .lines()
        .filter(|line| line.contains("Parsed_showinfo"))
        .filter_map(|line| line.split_once("pts_time:")?.1.split_whitespace().next()?.parse().ok())
        .collect();
    times.sort_by(f64::total_cmp);
    times.dedup();
    times
}

/// Ne décode que les keyframes : rapide, et donne d'un coup vignettes et positions de coupe.
pub fn build_strip(video: &Path, duration: f64, progress: &(dyn Fn(u32) + Sync)) -> Result<Strip, String> {
    let interval = (duration / MAX_THUMBS).max(10.0);
    let filter = format!(
        "showinfo,fps=1/{interval},scale={THUMB_WIDTH}:{THUMB_HEIGHT}:force_original_aspect_ratio=decrease,\
         pad={THUMB_WIDTH}:{THUMB_HEIGHT}:(ow-iw)/2:(oh-ih)/2"
    );
    let args: [&OsStr; 20] = [
        OsStr::new("-hide_banner"),
        OsStr::new("-v"),
        OsStr::new("info"),
        // Décodage par la carte graphique s'il y en a une : 5 fois plus vite sur une vidéo 4K.
        // Sans carte adaptée, ffmpeg revient seul au processeur.
        OsStr::new("-hwaccel"),
        OsStr::new("auto"),
        OsStr::new("-skip_frame"),
        OsStr::new("nokey"),
        OsStr::new("-i"),
        video.as_os_str(),
        OsStr::new("-map"),
        OsStr::new("0:v:0"),
        OsStr::new("-an"),
        OsStr::new("-sn"),
        OsStr::new("-vf"),
        OsStr::new(&filter),
        OsStr::new("-f"),
        OsStr::new("rawvideo"),
        OsStr::new("-pix_fmt"),
        OsStr::new("rgb24"),
        OsStr::new("-"),
    ];
    let mut stream = ffmpeg::Stream::spawn(&args)?;
    let frame_size = THUMB_WIDTH * THUMB_HEIGHT * 3;
    let mut pixels = Vec::new();
    let mut frame = vec![0u8; frame_size];
    loop {
        let read = read_all_into(stream.stdout(), &mut frame).map_err(|e| e.to_string())?;
        if read < frame_size {
            break;
        }
        pixels.extend_from_slice(&frame);
        if duration > 0.0 {
            let done = (pixels.len() / frame_size) as f64 * interval / duration;
            progress(((done * 100.0) as u32).min(99));
        }
    }
    let log = stream.finish_with_log()?;
    Ok(Strip {
        width: THUMB_WIDTH,
        height: THUMB_HEIGHT,
        interval,
        count: pixels.len() / frame_size,
        pixels,
        keyframes: parse_keyframes(&log),
    })
}

/// Un chapitre inscrit dans le fichier de la vidéo.
#[derive(Clone, PartialEq, Debug)]
pub struct Chapter {
    pub start: f64,
    pub end: f64,
    pub title: String,
}

/// Lit les chapitres dans le journal de `ffmpeg -i`.
fn parse_chapters(log: &str) -> Vec<Chapter> {
    let mut chapters: Vec<Chapter> = Vec::new();
    let mut titled = true;
    for line in log.lines().map(str::trim) {
        // Les titres des pistes, plus loin, ne sont pas ceux des chapitres
        if line.starts_with("Stream #") {
            break;
        }
        if let Some((_, times)) = line.strip_prefix("Chapter #").and_then(|rest| rest.split_once(": start ")) {
            let Some((start, end)) = times.split_once(", end ") else { continue };
            if let (Ok(start), Ok(end)) = (start.trim().parse(), end.trim().parse()) {
                chapters.push(Chapter { start, end, title: format!("Chapitre {}", chapters.len() + 1) });
                titled = false;
            }
        } else if let (false, Some((name, title))) = (titled, line.split_once(":")) {
            if name.trim() == "title" && !title.trim().is_empty() {
                if let Some(chapter) = chapters.last_mut() {
                    chapter.title = title.trim().to_string();
                }
                titled = true;
            }
        }
    }
    chapters
}

/// Une piste audio du fichier.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct AudioTrack {
    pub label: String,
}

/// Ce que les métadonnées du fichier disent de lui.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct MediaInfo {
    pub resolution: String,
    pub video: String,
    pub audio: Vec<AudioTrack>,
}

fn codec_name(codec: &str) -> String {
    match codec {
        "hevc" => "H.265",
        "h264" => "H.264",
        "av1" => "AV1",
        "vp9" => "VP9",
        "mpeg4" => "MPEG-4",
        "mpeg2video" => "MPEG-2",
        "aac" => "AAC",
        "ac3" => "Dolby Digital",
        "eac3" => "Dolby Digital+",
        "truehd" => "Dolby TrueHD",
        "dts" => "DTS",
        "flac" => "FLAC",
        "opus" => "Opus",
        "mp3" => "MP3",
        other => return other.to_uppercase(),
    }
    .to_string()
}

fn parse_media_info(log: &str) -> MediaInfo {
    let mut info = MediaInfo::default();
    let mut current_audio = None;
    for line in log.lines().map(str::trim) {
        if line.starts_with("Stream #") {
            current_audio = None;
            let language = line.split_once('(').and_then(|(_, rest)| rest.split_once("):")).map(|(l, _)| l).filter(|l| *l != "und");
            if let Some((_, rest)) = line.split_once("Video: ") {
                if info.video.is_empty() && !line.contains("attached pic") {
                    let codec = rest.split([' ', ',']).next().unwrap_or("");
                    let deep = rest.contains("p10") || rest.contains("10le");
                    info.video = format!("{}{}", codec_name(codec), if deep { " · 10 bits" } else { "" });
                    info.resolution = rest
                        .split(',')
                        .flat_map(|part| part.split_whitespace())
                        .find(|word| word.split_once('x').is_some_and(|(w, h)| w.parse::<u32>().is_ok() && h.parse::<u32>().is_ok()))
                        .map_or(String::new(), |size| size.replace('x', " × "));
                }
            } else if let Some((_, rest)) = line.split_once("Audio: ") {
                let codec = rest.split([' ', ',']).next().unwrap_or("");
                let layout = rest.split_once(" Hz, ").and_then(|(_, after)| after.split(',').next()).unwrap_or("").trim();
                let mut label = format!("{} {layout}", codec_name(codec)).trim().to_string();
                if let Some(language) = language {
                    label = format!("{} · {label}", language.to_uppercase());
                }
                info.audio.push(AudioTrack { label });
                current_audio = Some(info.audio.len() - 1);
            }
        } else if let (Some(index), Some((name, title))) = (current_audio, line.split_once(':')) {
            if name.trim() == "title" && !title.trim().is_empty() {
                let track = &mut info.audio[index];
                track.label = format!("{} · {}", track.label, title.trim());
            }
        }
    }
    info
}

pub fn read_media_info(video: &Path) -> MediaInfo {
    let args = [OsStr::new("-hide_banner"), OsStr::new("-i"), video.as_os_str()];
    ffmpeg::run(&args).map(|(log, _)| parse_media_info(&log)).unwrap_or_default()
}

/// Nom de cache d'un calcul qui dépend de la piste audio.
pub fn track_cache_kind(kind: &str, track: usize) -> String {
    if track == 0 {
        kind.to_string()
    } else {
        format!("{kind}-a{track}")
    }
}

/// Un fichier sans chapitres, ou illisible, n'en a simplement aucun.
pub fn read_chapters(video: &Path) -> Vec<Chapter> {
    let args = [OsStr::new("-hide_banner"), OsStr::new("-i"), video.as_os_str()];
    ffmpeg::run(&args).map(|(log, _)| parse_chapters(&log)).unwrap_or_default()
}

// --- Cache --------------------------------------------------------------------

/// Empreinte de la vidéo : chemin, taille et date. Une vidéo remplacé invalide le cache.
pub fn cache_path(video: &Path, kind: &str) -> Option<PathBuf> {
    let metadata = std::fs::metadata(video).ok()?;
    let modified = metadata.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs();
    let identity = format!("{}|{}|{}", video.display(), metadata.len(), modified);
    let mut hash: u64 = 0xcbf29ce484222325; // FNV-1a
    for byte in identity.bytes() {
        hash = (hash ^ byte as u64).wrapping_mul(0x100000001b3);
    }
    let base = std::env::var_os("LOCALAPPDATA").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    let folder = base.join("music-sequence-extractor").join("studio-cache");
    std::fs::create_dir_all(&folder).ok()?;
    Some(folder.join(format!("{hash:016x}.{kind}.zst")))
}

#[derive(Default)]
struct Writer(Vec<u8>);

impl Writer {
    fn count(&mut self, value: usize) {
        self.0.extend_from_slice(&(value as u64).to_le_bytes());
    }
    fn f64(&mut self, value: f64) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }
    fn bytes(&mut self, value: &[u8]) {
        self.count(value.len());
        self.0.extend_from_slice(value);
    }
    fn f32s(&mut self, value: &[f32]) {
        self.count(value.len());
        self.0.extend(value.iter().flat_map(|v| v.to_le_bytes()));
    }
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Option<&'a [u8]> {
        if self.0.len() < count {
            return None;
        }
        let (head, tail) = self.0.split_at(count);
        self.0 = tail;
        Some(head)
    }
    fn count(&mut self) -> Option<usize> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?) as usize)
    }
    fn f64(&mut self) -> Option<f64> {
        Some(f64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn bytes(&mut self) -> Option<&'a [u8]> {
        let count = self.count()?;
        self.take(count)
    }
    fn f32s(&mut self) -> Option<Vec<f32>> {
        let count = self.count()?;
        Some(self.take(count.checked_mul(4)?)?.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
    }
}

fn write_cache(path: &Path, data: &[u8]) -> Result<(), String> {
    let partial = path.with_extension(format!("{}.tmp", std::process::id()));
    let file = std::fs::File::create(&partial).map_err(|e| e.to_string())?;
    let mut encoder = zstd::Encoder::new(file, 3).map_err(|e| e.to_string())?;
    std::io::copy(&mut &*data, &mut encoder).map_err(|e| e.to_string())?;
    encoder.finish().map_err(|e| e.to_string())?;
    std::fs::rename(&partial, path).map_err(|e| e.to_string())
}

fn read_cache(path: &Path) -> Option<Vec<u8>> {
    let file = std::fs::File::open(path).ok()?;
    let mut data = Vec::new();
    zstd::Decoder::new(file).ok()?.read_to_end(&mut data).ok()?;
    Some(data)
}

impl Signals {
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let mut out = Writer::default();
        out.count(SIGNALS_CACHE_VERSION as usize);
        out.count(self.channels.len());
        for channel in &self.channels {
            out.bytes(channel.name.as_bytes());
            let base = &channel.peaks.levels[0];
            out.count(base.len());
            out.0.extend(base.iter().flat_map(|p| [p[0].to_le_bytes(), p[1].to_le_bytes()]).flatten());
            out.f32s(&channel.momentary);
            out.f32s(&channel.short_term);
            out.f32s(&channel.energies);
            out.f32s(&channel.true_peaks);
        }
        write_cache(path, &out.0)
    }

    pub fn load(path: &Path) -> Option<Self> {
        let data = read_cache(path)?;
        let mut input = Reader(&data);
        if input.count()? != SIGNALS_CACHE_VERSION as usize {
            return None;
        }
        let count = input.count()?;
        let mut channels = Vec::new();
        for _ in 0..count {
            let name = String::from_utf8(input.bytes()?.to_vec()).ok()?;
            let peak_count = input.count()?;
            let raw = input.take(peak_count.checked_mul(4)?)?;
            let base = raw
                .chunks_exact(4)
                .map(|b| [i16::from_le_bytes([b[0], b[1]]), i16::from_le_bytes([b[2], b[3]])])
                .collect();
            channels.push(Channel {
                name,
                peaks: Peaks::build(base),
                momentary: input.f32s()?,
                short_term: input.f32s()?,
                energies: input.f32s()?,
                true_peaks: input.f32s()?,
            });
        }
        Some(Signals { channels })
    }
}

impl Strip {
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let mut out = Writer::default();
        out.count(CACHE_VERSION as usize);
        out.count(self.width);
        out.count(self.height);
        out.f64(self.interval);
        out.count(self.count);
        out.bytes(&self.pixels);
        out.count(self.keyframes.len());
        for time in &self.keyframes {
            out.f64(*time);
        }
        write_cache(path, &out.0)
    }

    pub fn load(path: &Path) -> Option<Self> {
        let data = read_cache(path)?;
        let mut input = Reader(&data);
        if input.count()? != CACHE_VERSION as usize {
            return None;
        }
        let width = input.count()?;
        let height = input.count()?;
        let interval = input.f64()?;
        let count = input.count()?;
        let pixels = input.bytes()?.to_vec();
        let key_count = input.count()?;
        let mut keyframes = Vec::with_capacity(key_count);
        for _ in 0..key_count {
            keyframes.push(input.f64()?);
        }
        Some(Strip { width, height, interval, count, pixels, keyframes })
    }

    /// Keyframe sur laquelle l'export sans réencodage calera un début à `time`.
    pub fn keyframe_before(&self, time: f64) -> Option<f64> {
        let index = self.keyframes.partition_point(|&k| k <= time + 1e-6);
        index.checked_sub(1).map(|i| self.keyframes[i])
    }

    pub fn nearest_keyframe(&self, time: f64) -> Option<f64> {
        let index = self.keyframes.partition_point(|&k| k < time);
        let after = self.keyframes.get(index).copied();
        let before = index.checked_sub(1).map(|i| self.keyframes[i]);
        match (before, after) {
            (Some(b), Some(a)) => Some(if time - b <= a - time { b } else { a }),
            (one, other) => one.or(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(frequency: f64, level_db: f64, seconds: f64, channels: usize) -> Vec<f32> {
        let amplitude = 10f64.powf(level_db / 20.0);
        let frames = (seconds * SAMPLE_RATE as f64) as usize;
        let mut samples = Vec::with_capacity(frames * channels);
        for i in 0..frames {
            let x = amplitude * (std::f64::consts::TAU * frequency * i as f64 / SAMPLE_RATE as f64).sin();
            samples.extend(std::iter::repeat_n(x as f32, channels));
        }
        samples
    }

    fn analyze_samples(names: Vec<&'static str>, samples: &[f32]) -> Signals {
        let mut analysis = Analysis::new(names);
        for chunk in samples.chunks(48000 * 2) {
            analysis.feed(chunk);
        }
        analysis.finish()
    }

    /// Cas de test EBU Tech 3341 n° 1 : sinus 1 kHz stéréo à -23 dBFS = -23 LUFS.
    #[test]
    fn stereo_reference_tone_measures_minus_23_lufs() {
        let signals = analyze_samples(vec!["L", "R"], &sine(1000.0, -23.0, 20.0, 2));
        let program = &signals.channels[0];
        let measure = program.measure(0.0, f64::INFINITY);
        assert!((measure.integrated + 23.0).abs() < 0.1, "{}", measure.integrated);
        let last = *program.short_term.last().unwrap();
        assert!((last + 23.0).abs() < 0.1, "{last}");
        assert!(measure.range.abs() < 0.2, "{}", measure.range);
        assert!((measure.true_peak + 23.0).abs() < 0.1, "{}", measure.true_peak);
        assert!((measure.max_short_term + 23.0).abs() < 0.1, "{}", measure.max_short_term);
        assert!((measure.max_momentary + 23.0).abs() < 0.1, "{}", measure.max_momentary);
    }

    /// Cas EBU Tech 3341 n° 3 et 4 : le gating écarte les passages faibles. Durées raccourcies.
    #[test]
    fn gating_ignores_the_quiet_passages() {
        let mut samples = sine(1000.0, -36.0, 10.0, 2);
        samples.extend(sine(1000.0, -23.0, 30.0, 2));
        samples.extend(sine(1000.0, -36.0, 10.0, 2));
        let measure = analyze_samples(vec!["L", "R"], &samples).channels[0].measure(0.0, f64::INFINITY);
        assert!((measure.integrated + 23.0).abs() < 0.1, "{}", measure.integrated);
    }

    /// Le calcul progressif doit donner la même chose que le calcul direct, à chaque instant.
    #[test]
    fn running_integrated_matches_the_direct_measure() {
        let mut samples = sine(1000.0, -40.0, 8.0, 1);
        samples.extend(sine(400.0, -20.0, 8.0, 1));
        samples.extend(sine(3000.0, -30.0, 8.0, 1));
        let signals = analyze_samples(vec![], &samples);
        let running = signals.channels[0].running_integrated();
        assert_eq!(running.len(), signals.channels[0].energies.len());
        assert_eq!(running[2], NO_LOUDNESS, "pas encore de fenêtre complète");
        for i in (10..running.len()).step_by(25) {
            let blocks: Vec<f64> = signals.channels[0].energies[..=i].iter().map(|&e| e as f64).collect();
            let windows = full_windows(&blocks, 4);
            let audible: Vec<f64> = windows.into_iter().filter(|&z| to_lufs(z) > -70.0).collect();
            let gate = to_lufs(audible.iter().sum::<f64>() / audible.len() as f64) - 10.0;
            let kept: Vec<f64> = audible.into_iter().filter(|&z| to_lufs(z) > gate).collect();
            let direct = to_lufs(kept.iter().sum::<f64>() / kept.len() as f64);
            assert!((running[i] - direct).abs() < 0.05, "bloc {i} : {} contre {direct}", running[i]);
        }
    }

    #[test]
    fn a_measure_only_looks_at_its_portion() {
        let mut samples = sine(1000.0, -40.0, 10.0, 1);
        samples.extend(sine(1000.0, -20.0, 10.0, 1));
        samples.extend(sine(1000.0, -30.0, 10.0, 1));
        let signals = analyze_samples(vec![], &samples);
        // Un seul canal à -20 dBFS : 3 dB sous la référence stéréo, soit -23 LUFS
        let middle = signals.channels[0].measure(11.0, 19.0);
        assert!((middle.integrated + 23.0).abs() < 0.1, "{}", middle.integrated);
        assert!((middle.true_peak + 20.0).abs() < 0.1, "{}", middle.true_peak);
        assert!((middle.max_short_term + 23.0).abs() < 0.1, "{}", middle.max_short_term);
        assert!(middle.range.abs() < 0.2, "{}", middle.range);
    }

    /// Cas EBU Tech 3341 n° 15 à 19 : sinus dont les crêtes tombent entre les échantillons.
    /// Tolérance de la norme : +0,2 / -0,4 dB.
    #[test]
    fn true_peak_finds_the_peaks_between_samples() {
        for (ratio, phase) in [(0.25, 45.0f64), (0.25, 22.5), (0.125, 22.5), (0.1, 0.0), (0.4, 45.0)] {
            let frames = 48000;
            let amplitude = 10f64.powf(-6.0 / 20.0);
            let samples: Vec<f32> = (0..frames)
                .map(|i| (amplitude * (std::f64::consts::TAU * ratio * i as f64 + phase.to_radians()).sin()) as f32)
                .collect();
            let measure = analyze_samples(vec![], &samples).channels[0].measure(0.0, f64::INFINITY);
            assert!((-6.4..=-5.8).contains(&measure.true_peak), "{ratio} fs, {phase}° : {}", measure.true_peak);
        }
    }

    /// À fs/4 et 45°, chaque échantillon vaut 0,707 de la crête : 3 dB que le pic par échantillon manque.
    #[test]
    fn true_peak_is_above_the_sample_peak() {
        let amplitude = 10f64.powf(-6.0 / 20.0);
        let samples: Vec<f32> = (0..48000)
            .map(|i| (amplitude * (std::f64::consts::TAU * 0.25 * i as f64 + std::f64::consts::FRAC_PI_4).sin()) as f32)
            .collect();
        let sample_peak = 20.0 * samples.iter().fold(0.0f32, |p, x| p.max(x.abs())).log10();
        let true_peak = analyze_samples(vec![], &samples).channels[0].measure(0.0, f64::INFINITY).true_peak;
        assert!(true_peak - sample_peak > 2.8, "{sample_peak} {true_peak}");
    }

    #[test]
    fn program_true_peak_is_the_highest_of_its_channels() {
        let frames = 48000 * 5;
        let tone = sine(1000.0, -3.0, 5.0, 1);
        let mut samples = vec![0.0f32; frames * 6];
        for i in 0..frames {
            samples[i * 6 + 3] = tone[i]; // LFE : hors de la loudness, mais pas des crêtes
        }
        let signals = analyze_samples(vec!["L", "R", "C", "LFE", "Ls", "Rs"], &samples);
        let measure = signals.channels[0].measure(0.0, f64::INFINITY);
        assert!((measure.true_peak + 3.0).abs() < 0.1, "{}", measure.true_peak);
    }

    #[test]
    fn lfe_does_not_count_in_program_loudness() {
        let frames = 48000 * 10;
        let tone = sine(1000.0, -23.0, 10.0, 1);
        let mut samples = vec![0.0f32; frames * 6];
        for i in 0..frames {
            samples[i * 6] = tone[i]; // L
            samples[i * 6 + 3] = 0.9; // LFE : ne doit rien changer
        }
        let with_lfe = analyze_samples(vec!["L", "R", "C", "LFE", "Ls", "Rs"], &samples);
        // Un seul canal à -23 dBFS = 3 dB sous la référence stéréo
        assert!((with_lfe.channels[0].measure(0.0, f64::INFINITY).integrated + 26.0).abs() < 0.2);
    }

    #[test]
    fn surround_channels_get_the_bs1770_gain() {
        let frames = 48000 * 10;
        let tone = sine(1000.0, -23.0, 10.0, 1);
        let mut samples = vec![0.0f32; frames * 6];
        for i in 0..frames {
            samples[i * 6 + 4] = tone[i]; // Ls
        }
        let signals = analyze_samples(vec!["L", "R", "C", "LFE", "Ls", "Rs"], &samples);
        let expected = -26.0 + 10.0 * 1.41f32.log10();
        let integrated = signals.channels[0].measure(0.0, f64::INFINITY).integrated;
        assert!((integrated - expected).abs() < 0.2, "{integrated}");
    }

    #[test]
    fn spectrogram_shows_the_tone_at_its_frequency() {
        let samples = sine(1000.0, -6.0, 3.0, 1);
        let spectrogram = fill_columns(&samples, 0, 0, samples.len().div_ceil(BLOCK));
        let columns = spectrogram.len() / OVERVIEW_BINS;
        assert_eq!(columns, 30);
        let column = &spectrogram[10 * OVERVIEW_BINS..11 * OVERVIEW_BINS];
        let loudest = column.iter().enumerate().max_by_key(|(_, v)| **v).unwrap().0;
        let band_hz = F_MIN * (F_MAX / F_MIN).powf(loudest as f64 / OVERVIEW_BINS as f64);
        assert!((800.0..1250.0).contains(&band_hz), "{band_hz}");
        // -6 dBFS, à une marge près due à la fenêtre
        let db = column[loudest] as f32 / 255.0 * 100.0 - 100.0;
        assert!((db + 6.0).abs() < 3.0, "{db}");
    }

    /// Colonne de détail d'un signal centré sur la fenêtre, en dB par bande.
    fn detail_column(samples: &[f32]) -> Vec<f32> {
        let mut out = vec![0u8; DETAIL_BINS];
        Spectrum::detail(60.0).column(samples, &mut out);
        out.iter().map(|v| *v as f32 / 255.0 * -FLOOR_DB + FLOOR_DB).collect()
    }

    fn band_of(hz: f64) -> usize {
        ((hz / F_MIN).ln() / (F_MAX / F_MIN).ln() * DETAIL_BINS as f64) as usize
    }

    fn tones(frequencies: &[f64]) -> Vec<f32> {
        (0..LONGEST_FFT)
            .map(|i| {
                let t = i as f64 / SAMPLE_RATE as f64;
                frequencies.iter().map(|f| (std::f64::consts::TAU * f * t).sin()).sum::<f64>() as f32 * 0.1
            })
            .collect()
    }

    fn peak_near(column: &[f32], hz: f64) -> f32 {
        let band = band_of(hz);
        column[band.saturating_sub(6)..(band + 6).min(column.len())].iter().copied().fold(f32::MIN, f32::max)
    }

    #[test]
    fn close_notes_stay_separate_in_the_bass() {
        // Deux notes graves à 30 Hz d'écart : il faut un creux bien net entre les deux
        let column = detail_column(&tones(&[60.0, 90.0]));
        let valley = column[band_of(72.0)..band_of(78.0)].iter().copied().fold(f32::MAX, f32::min);
        assert!(peak_near(&column, 60.0) - valley > 12.0, "{} {valley}", peak_near(&column, 60.0));
        assert!(peak_near(&column, 90.0) - valley > 12.0, "{} {valley}", peak_near(&column, 90.0));
    }

    /// Écart en dB entre deux notes et le creux qui les sépare.
    fn separation(low: f64, high: f64) -> f32 {
        let column = detail_column(&tones(&[low, high]));
        let (a, b) = (band_of(low), band_of(high));
        let peak = column[a.saturating_sub(4)..b + 4].iter().copied().fold(f32::MIN, f32::max);
        let middle = (low * high).sqrt();
        let m = band_of(middle);
        let valley = column[m - 1..=m + 1].iter().copied().fold(f32::MAX, f32::min);
        peak - valley
    }

    /// Le demi-ton, la plus petite distance entre deux notes : sa largeur en Hz diminue
    /// en descendant, c'est le vrai test de la définition dans les graves.
    #[test]
    fn neighbouring_semitones_stay_separate_low_in_the_bass() {
        for (name, low, high) in [
            ("do 2 / do# 2", 65.41, 69.30),
            ("la 2 / sib 2", 110.0, 116.54),
            ("mi 3 / fa 3", 164.81, 174.61),
            ("la 3 / sib 3", 220.0, 233.08),
            ("la 4 / sib 4", 440.0, 466.16),
        ] {
            let gap = separation(low, high);
            assert!(gap > 6.0, "{name} : seulement {gap:.1} dB entre les notes et le creux");
        }
    }

    #[test]
    fn overview_stays_aligned_with_the_picture() {
        // Un clic à 5,05 s : il doit tomber dans la colonne 50, pas plusieurs colonnes à côté
        let frames = 48000 * 10;
        let mut samples = vec![0.0f32; frames];
        samples[(5.05 * 48000.0) as usize] = 1.0;
        let spectrum = &fill_columns(&samples, 0, 0, samples.len().div_ceil(BLOCK));
        assert_eq!(spectrum.len() / OVERVIEW_BINS, 100, "autant de colonnes que de blocs");
        let band = ((8000.0f64 / F_MIN).ln() / (F_MAX / F_MIN).ln() * OVERVIEW_BINS as f64) as usize;
        let loudest = (0..100).max_by_key(|c| spectrum[c * OVERVIEW_BINS + band]).unwrap();
        assert_eq!(loudest, 50);

        // Une note grave de 5 à 6 s : son centre reste à 5,5 s malgré la fenêtre longue
        let mut samples = vec![0.0f32; frames];
        for i in 48000 * 5..48000 * 6 {
            samples[i] = (std::f64::consts::TAU * 80.0 * i as f64 / 48000.0).sin() as f32 * 0.5;
        }
        let spectrum = &fill_columns(&samples, 0, 0, samples.len().div_ceil(BLOCK));
        let band = ((80.0f64 / F_MIN).ln() / (F_MAX / F_MIN).ln() * OVERVIEW_BINS as f64) as usize;
        let top = (0..100).map(|c| spectrum[c * OVERVIEW_BINS + band]).max().unwrap();
        let strong: Vec<usize> = (0..100).filter(|c| spectrum[c * OVERVIEW_BINS + band] + 10 >= top).collect();
        let center = (strong[0] + strong[strong.len() - 1]) as f64 / 2.0;
        assert!((center - 55.0).abs() <= 1.5, "{strong:?}");
    }

    #[test]
    fn close_zoom_keeps_bass_attacks_sharper() {
        // Un clic à 0,25 s du centre : la fenêtre du zoom serré (± 0,17 s) ne le voit plus à 100 Hz
        let mut samples = vec![0.0f32; LONGEST_FFT];
        samples[LONGEST_FFT / 2 + 12000] = 1.0;
        let mut close = vec![0u8; DETAIL_BINS];
        Spectrum::detail(10.0).column(&samples, &mut close);
        let mut wide = vec![0u8; DETAIL_BINS];
        Spectrum::detail(120.0).column(&samples, &mut wide);
        let band = band_of(100.0);
        // Un seul échantillon pèse peu dans une longue fenêtre : on vérifie qu'il est vu ou non
        assert!(wide[band] >= 10 && close[band] == 0, "{} {}", wide[band], close[band]);
    }

    #[test]
    fn close_harmonics_stay_separate_in_the_mids() {
        let column = detail_column(&tones(&[1000.0, 1200.0]));
        let valley = column[band_of(1080.0)..band_of(1120.0)].iter().copied().fold(f32::MAX, f32::min);
        assert!(peak_near(&column, 1000.0) - valley > 15.0, "{valley}");
    }

    #[test]
    fn treble_attacks_stay_sharp_and_bass_notes_stay_long() {
        // Un clic à 30 ms du centre : hors de la fenêtre courte des aigus (± 21 ms)
        let mut samples = vec![0.0f32; LONGEST_FFT];
        samples[LONGEST_FFT / 2 + 1440] = 1.0;
        let far = detail_column(&samples);
        // Le même clic au centre
        let mut samples = vec![0.0f32; LONGEST_FFT];
        samples[LONGEST_FFT / 2] = 1.0;
        let near = detail_column(&samples);
        assert!(near[band_of(8000.0)] - far[band_of(8000.0)] > 30.0, "{} {}", near[band_of(8000.0)], far[band_of(8000.0)]);
        // Les graves, eux, l'étalent encore : la fenêtre longue le voit toujours
        assert!(far[band_of(100.0)] > FLOOR_DB + 10.0, "{}", far[band_of(100.0)]);
    }

    #[test]
    fn a_tone_keeps_its_level_across_resolutions() {
        for hz in [100.0, 300.0, 700.0, 1500.0, 5000.0] {
            let column = detail_column(&tones(&[hz]));
            // Une fenêtre de Hann perd au plus 1,42 dB entre deux raies, plus 0,39 dB d'arrondi sur 8 bits
            assert!((peak_near(&column, hz) + 20.0).abs() < 1.85, "{hz} Hz : {}", peak_near(&column, hz));
        }
    }

    #[test]
    fn bands_blend_their_resolutions_without_a_gap() {
        for spectrum in [Spectrum::overview(), Spectrum::detail(60.0), Spectrum::detail(5.0)] {
            for sources in &spectrum.bands {
                let total: f32 = sources.iter().map(|s| s.weight).sum();
                assert!((total - 1.0).abs() < 1e-5, "{total}");
                assert!(sources.iter().all(|s| s.weight >= 0.0));
            }
        }
    }

    #[test]
    fn peaks_find_the_extremes_at_every_zoom() {
        let mut samples = vec![0.0f32; 48000 * 60];
        samples[48000 * 30] = 0.5;
        samples[48000 * 40] = -0.25;
        let signals = analyze_samples(vec![], &samples);
        let peaks = &signals.channels[0].peaks;
        assert!(peaks.levels.len() > 1);
        let (min, max) = peaks.range(0.0, 60.0).unwrap();
        assert!((max - 0.5).abs() < 1e-3 && (min + 0.25).abs() < 1e-3, "{min} {max}");
        let (_, max) = peaks.range(29.99, 30.01).unwrap();
        assert!((max - 0.5).abs() < 1e-3, "{max}");
        let (min, max) = peaks.range(10.0, 20.0).unwrap();
        assert_eq!((min, max), (0.0, 0.0));
    }

    #[test]
    fn signals_survive_the_cache() {
        let folder = std::env::temp_dir().join(format!("music-studio-test-{}", std::process::id()));
        std::fs::create_dir_all(&folder).unwrap();
        let path = folder.join("signals.zst");
        let signals = analyze_samples(vec!["L", "R"], &sine(440.0, -12.0, 2.0, 2));
        signals.save(&path).unwrap();
        let loaded = Signals::load(&path).unwrap();
        std::fs::remove_dir_all(&folder).unwrap();
        assert_eq!(loaded.channels.len(), 3);
        let (a, b) = (&signals.channels[1], &loaded.channels[1]);
        assert_eq!(a.name, b.name);
        assert_eq!(a.peaks.levels[0], b.peaks.levels[0]);
        assert_eq!(a.momentary, b.momentary);
        assert_eq!(a.energies, b.energies);
        assert_eq!(a.true_peaks, b.true_peaks);
    }

    #[test]
    fn chapters_are_read_from_the_ffmpeg_log() {
        let log = "Input #0, matroska,webm, from 'film.mkv':
  Metadata:
    title           : Paolo S01E03 The Heart of Roisson
  Duration: 00:50:47.68, start: 0.000000, bitrate: 3696 kb/s
  Chapters:
    Chapter #0:0: start 0.000000, end 93.000000
      Metadata:
        title           : Chapter 01
    Chapter #0:1: start 93.000000, end 129.000000
      Metadata:
        title           : Intro : la suite
    Chapter #0:2: start 129.000000, end 200.500000
    Chapter #0:3: start 200.500000, end 300.000000
      Metadata:
        title           : Credits
  Stream #0:0: Video: h264
      Metadata:
        title           : Forced
At least one output file must be specified
";
        let chapters = parse_chapters(log);
        assert_eq!(chapters.len(), 4);
        assert_eq!(chapters[0], Chapter { start: 0.0, end: 93.0, title: "Chapter 01".into() });
        assert_eq!(chapters[1].title, "Intro : la suite");
        assert_eq!(chapters[2].title, "Chapitre 3", "sans titre : un numéro");
        assert_eq!((chapters[3].end, chapters[3].title.as_str()), (300.0, "Credits"), "le titre d'une piste ne compte pas");
    }

    #[test]
    fn media_info_comes_from_the_streams() {
        let log = "  Stream #0:0: Video: hevc (Main 10), yuv420p10le(tv, progressive), 3840x2076 [SAR 1:1 DAR 320:173], 23.98 fps (default)
  Stream #0:1(eng): Audio: eac3, 48000 Hz, 5.1(side), fltp, 640 kb/s (default)
      Metadata:
        title           : Original
  Stream #0:2(fre): Audio: aac (LC), 48000 Hz, stereo, fltp
  Stream #0:3(eng): Subtitle: subrip (srt)
      Metadata:
        title           : Forced
";
        let info = parse_media_info(log);
        assert_eq!(info.resolution, "3840 × 2076");
        assert_eq!(info.video, "H.265 · 10 bits");
        assert_eq!(info.audio.len(), 2);
        assert_eq!(info.audio[0].label, "ENG · Dolby Digital+ 5.1(side) · Original");
        assert_eq!(info.audio[1].label, "FRE · AAC stereo");
    }

    /// De vrais échantillons décodés par ffmpeg : livrés par morceaux, puis relus du cache.
    #[test]
    fn spectrum_arrives_in_parts_and_comes_back_from_the_cache() {
        let folder = std::env::temp_dir().join(format!("music-studio-spectrum-{}", std::process::id()));
        std::fs::create_dir_all(&folder).unwrap();
        let source = folder.join("source.wav");
        let (log, ok) = ffmpeg::run(&[
            OsStr::new("-y"),
            OsStr::new("-f"),
            OsStr::new("lavfi"),
            OsStr::new("-i"),
            OsStr::new("sine=frequency=1000:duration=70:sample_rate=48000"),
            OsStr::new("-af"),
            OsStr::new("pan=stereo|c0=c0|c1=c0"),
            source.as_os_str(),
        ])
        .unwrap();
        assert!(ok, "{log}");
        let collect = |parts: &mut Vec<(usize, usize)>| {
            let mut values = Vec::new();
            overview_spectrum(&source, "stereo", 1, 0, &mut |first, part| {
                assert_eq!(first * OVERVIEW_BINS, values.len(), "les morceaux se suivent");
                parts.push((first, part.len() / OVERVIEW_BINS));
                values.extend_from_slice(&part);
            })
            .unwrap();
            values
        };
        let mut parts = Vec::new();
        let first = collect(&mut parts);
        assert!(parts.len() >= 2, "{parts:?}");
        assert_eq!(first.len() / OVERVIEW_BINS, 700);
        let column = &first[300 * OVERVIEW_BINS..301 * OVERVIEW_BINS];
        let loudest = column.iter().enumerate().max_by_key(|(_, v)| **v).unwrap().0;
        let hz = F_MIN * (F_MAX / F_MIN).powf(loudest as f64 / OVERVIEW_BINS as f64);
        assert!((800.0..1250.0).contains(&hz), "{hz}");
        let mut cached = Vec::new();
        let again = collect(&mut cached);
        if let Some(path) = cache_path(&source, "spectrum1") {
            let _ = std::fs::remove_file(path);
        }
        std::fs::remove_dir_all(&folder).unwrap();
        assert_eq!(cached.len(), 1, "un seul morceau depuis le cache");
        assert_eq!(again, first);
    }

    #[test]
    fn lane_filter_isolates_a_channel_or_builds_the_mix() {
        assert_eq!(lane_filter("5.1", 3), "pan=mono|c0=c2");
        let mix = lane_filter("stereo", 0);
        assert_eq!(mix, "pan=mono|c0=0.50000*c0+0.50000*c1");
        assert_eq!(lane_filter("mono", 0), "aformat=channel_layouts=mono");
    }

    #[test]
    fn a_film_without_chapters_has_none() {
        let log = "Input #0, mov,mp4, from 'a.mp4':\n  Metadata:\n    title : Melancholia\n  Duration: 02:15:00\n  Stream #0:0: Video: h264\n";
        assert!(parse_chapters(log).is_empty());
    }

    #[test]
    fn keyframes_are_read_from_showinfo() {
        let log = "[Parsed_showinfo_0 @ 0x1] n:   0 pts:    83 pts_time:0.083   duration:42\n\
                   [Parsed_showinfo_0 @ 0x1] n:   1 pts: 10000 pts_time:10      duration:42\n\
                   frame=   10 fps=0.0\n";
        assert_eq!(parse_keyframes(log), [0.083, 10.0]);
    }

    #[test]
    fn export_start_lands_on_the_previous_keyframe() {
        let strip = Strip {
            width: 0,
            height: 0,
            interval: 10.0,
            count: 0,
            pixels: Vec::new(),
            keyframes: vec![0.0, 5.0, 10.0],
        };
        assert_eq!(strip.keyframe_before(7.0), Some(5.0));
        assert_eq!(strip.keyframe_before(5.0), Some(5.0));
        assert_eq!(strip.nearest_keyframe(7.0), Some(5.0));
        assert_eq!(strip.nearest_keyframe(8.0), Some(10.0));
    }
}
