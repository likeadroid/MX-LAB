//! YAMNet : spectrogramme log-mel flottant puis MobileNet quantifié en int8.
//! Les arrondis suivent ceux de XNNPACK (via LiteRT), le moteur qui a servi aux réglages.

use crate::tflite::{self, Model};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

pub const WINDOW: usize = 15600; // entrée fixe de YAMNet (0,975 s)
pub const HOP: usize = 7680; // 0,48 s
pub const MUSIC_CLASS: usize = 132; // classe « Music » d'AudioSet

const FRAMES: usize = 96;
const FRAME_HOP: usize = 160;
const FRAME_LENGTH: usize = 400;
const FFT_LENGTH: usize = 512;
const BINS: usize = 257;
const MELS: usize = 64;

static MODEL: &[u8] = include_bytes!("../../models/yamnet.tflite");

/// Noms des 521 classes, rangés dans le fichier du modèle.
pub fn class_labels() -> Vec<&'static str> {
    const MARKER: &[u8] = b"yamnet_label_list.txtSpeech";
    let start = MODEL.windows(MARKER.len()).position(|w| w == MARKER).expect("noms des classes absents")
        + b"yamnet_label_list.txt".len();
    let text = String::from_utf8_lossy(&MODEL[start..]);
    let names: Vec<usize> = text.lines().take(521).map(str::len).collect();
    let mut labels = Vec::with_capacity(521);
    let mut at = start;
    for length in names {
        labels.push(std::str::from_utf8(&MODEL[at..at + length]).expect("nom de classe illisible"));
        at += length + 1;
    }
    labels
}

fn activation_range(activation: i8, scale: f32, zero_point: i32) -> (i32, i32) {
    let quantize = |v: f32| zero_point + (v / scale).round() as i32;
    match activation {
        0 => (-128, 127),
        1 => (quantize(0.0).max(-128), 127),
        2 => (quantize(-1.0).max(-128), quantize(1.0).min(127)),
        3 => (quantize(0.0).max(-128), quantize(6.0).min(127)),
        other => panic!("activation {other} non gérée"),
    }
}

fn same_padding(input: usize, kernel: usize, stride: usize) -> usize {
    let output = input.div_ceil(stride);
    ((output - 1) * stride + kernel).saturating_sub(input) / 2
}

struct Requant {
    scale: f32,
    bias: i32,
}

struct Conv {
    depthwise: bool,
    filter: Vec<i16>,
    kernel_h: usize,
    kernel_w: usize,
    out_channels: usize,
    stride_h: usize,
    stride_w: usize,
    same: bool,
    input_offset: i16,
    output_zero: i32,
    min: i32,
    max: i32,
    requant: Vec<Requant>,
}

struct Mean {
    scale: f32,
    input_zero: i32,
    output_zero: i32,
    min: i32,
    max: i32,
}

struct Classifier {
    /// 1024 poids par classe.
    weights: Vec<i16>,
    biases: Vec<i32>,
    input_offset: i16,
    scale: f32,
    /// Échelle du logit réel, avant l'arrondi sur 8 bits de la sortie.
    logit_scale: f32,
    output_zero: i32,
    min: i32,
    max: i32,
    /// Sigmoïde quantifiée, indexée par la sortie int8 + 128.
    logistic: [f32; 256],
    logistic_q: [i8; 256],
}

enum Layer {
    Conv(Conv),
    Mean(Mean),
}

struct Traced {
    layer: Layer,
    output: i32,
}

pub struct Yamnet {
    window: Vec<f32>,
    mel: Vec<f32>,
    mel_bias: Vec<f32>,
    input_scale: f32,
    input_zero: i32,
    layers: Vec<Traced>,
    classifier: Classifier,
    fft: Arc<dyn Fft<f64>>,
}

/// Observe les tenseurs intermédiaires, pour les comparer à TensorFlow Lite.
pub type Trace<'a> = Option<&'a mut dyn FnMut(i32, &[u8])>;

impl Yamnet {
    pub fn load() -> Self {
        let model = Model::parse(MODEL);
        let tensor = |i: i32| &model.tensors[i as usize];

        // Frontal : MUL = fenêtre de Hann, FULLY_CONNECTED flottant = banc de filtres mel
        let mul = model.operators.iter().find(|o| o.code == 18).expect("MUL absent");
        let window = tensor(mul.inputs[1]).as_f32();
        let mel_op = model.operators.iter().find(|o| o.code == tflite::FULLY_CONNECTED).unwrap();
        let mel = tensor(mel_op.inputs[1]).as_f32();
        let mel_bias = tensor(mel_op.inputs[2]).as_f32();
        assert_eq!((window.len(), mel.len()), (FRAME_LENGTH, MELS * BINS));
        let quantize = model.operators.iter().find(|o| o.code == 114).expect("QUANTIZE absent");
        let quantized = tensor(quantize.outputs[0]);

        let mut layers = Vec::new();
        let mut classifier = None;
        let mut channels = 1;
        for op in &model.operators {
            match op.code {
                tflite::CONV_2D | tflite::DEPTHWISE_CONV_2D => {
                    let (input, filter, bias, output) = (
                        tensor(op.inputs[0]),
                        tensor(op.inputs[1]),
                        tensor(op.inputs[2]),
                        tensor(op.outputs[0]),
                    );
                    let depthwise = op.code == tflite::DEPTHWISE_CONV_2D;
                    let (same, stride_h, stride_w, activation) = if depthwise {
                        model.depthwise_options(op)
                    } else {
                        model.conv_options(op)
                    };
                    let out_channels = if depthwise { filter.shape[3] } else { filter.shape[0] };
                    let output_zero = output.zero_point[0] as i32;
                    let (min, max) = activation_range(activation, output.scale[0], output_zero);
                    let bias = bias.as_i32();
                    let requant = (0..out_channels)
                        .map(|c| {
                            // Calcul flottant de XNNPACK, le moteur qui a servi aux réglages
                            Requant { scale: input.scale[0] * filter.scale[c] / output.scale[0], bias: bias[c] }
                        })
                        .collect();
                    assert!(filter.zero_point.iter().all(|&z| z == 0));
                    layers.push(Traced {
                        layer: Layer::Conv(Conv {
                            depthwise,
                            filter: filter.as_i16_from_i8(),
                            kernel_h: filter.shape[1],
                            kernel_w: filter.shape[2],
                            out_channels,
                            stride_h,
                            stride_w,
                            same,
                            input_offset: -(input.zero_point[0] as i16),
                            output_zero,
                            min,
                            max,
                            requant,
                        }),
                        output: op.outputs[0],
                    });
                    channels = out_channels;
                }
                tflite::MEAN => {
                    let (input, output) = (tensor(op.inputs[0]), tensor(op.outputs[0]));
                    layers.push(Traced {
                        layer: Layer::Mean(Mean {
                            scale: input.scale[0] / output.scale[0],
                            input_zero: input.zero_point[0] as i32,
                            output_zero: output.zero_point[0] as i32,
                            min: -128,
                            max: 127,
                        }),
                        output: op.outputs[0],
                    });
                }
                tflite::FULLY_CONNECTED if tensor(op.inputs[0]).scale.first().is_some_and(|&s| s != 0.0) => {
                    let (input, weights, bias, output) = (
                        tensor(op.inputs[0]),
                        tensor(op.inputs[1]),
                        tensor(op.inputs[2]),
                        tensor(op.outputs[0]),
                    );
                    let scale = input.scale[0] * weights.scale[0] / output.scale[0];
                    let output_zero = output.zero_point[0] as i32;
                    let (min, max) =
                        activation_range(model.fully_connected_activation(op), output.scale[0], output_zero);
                    assert_eq!((weights.shape[0], weights.shape[1], channels), (521, 1024, 1024));
                    assert_eq!(weights.zero_point[0], 0);
                    let all = weights.as_i16_from_i8();
                    let sigmoid_op = model.operators.iter().find(|o| o.code == tflite::LOGISTIC).unwrap();
                    let sigmoid = tensor(sigmoid_op.outputs[0]);
                    let mut logistic = [0f32; 256];
                    let mut logistic_q = [0i8; 256];
                    for v in 0..256usize {
                        let x = output.scale[0] * (v as i32 - 128 - output_zero) as f32;
                        let y = 1.0 / (1.0 + (-x).exp());
                        let q = ((y / sigmoid.scale[0]).round() as i32 + sigmoid.zero_point[0] as i32).clamp(-128, 127);
                        logistic_q[v] = q as i8;
                        logistic[v] = sigmoid.scale[0] * (q - sigmoid.zero_point[0] as i32) as f32;
                    }
                    classifier = Some(Classifier {
                        weights: all,
                        biases: bias.as_i32(),
                        input_offset: -(input.zero_point[0] as i16),
                        scale,
                        logit_scale: input.scale[0] * weights.scale[0],
                        output_zero,
                        min,
                        max,
                        logistic,
                        logistic_q,
                    });
                }
                _ => {}
            }
        }
        Yamnet {
            window,
            mel,
            mel_bias,
            input_scale: quantized.scale[0],
            input_zero: quantized.zero_point[0] as i32,
            layers,
            classifier: classifier.expect("classifieur absent"),
            fft: FftPlanner::<f64>::new().plan_fft_forward(FFT_LENGTH),
        }
    }

    /// P(musique) d'une fenêtre de 15 600 échantillons à 16 kHz.
    pub fn music_probability(&self, waveform: &[f32]) -> f32 {
        self.run(waveform, None)
    }

    pub fn run(&self, waveform: &[f32], mut trace: Trace) -> f32 {
        let x = self.embedding(waveform, &mut trace);
        let (q, p) = self.classifier.run(&x, MUSIC_CLASS);
        if let Some(t) = trace.as_mut() {
            t(119, bytemuck_i8(&[q]));
        }
        p
    }

    /// Logits des classes demandées, pour une même fenêtre. Pris avant l'arrondi sur 8 bits :
    /// la sortie du modèle avance par pas de 0,4 %, ce qui efface les classes discrètes.
    pub fn class_logits(&self, waveform: &[f32], classes: &[usize]) -> Vec<f32> {
        let x = self.embedding(waveform, &mut None);
        classes.iter().map(|&class| self.classifier.logit(&x, class)).collect()
    }

    /// Les 1024 valeurs que le classifieur reçoit.
    fn embedding(&self, waveform: &[f32], trace: &mut Trace) -> Vec<i8> {
        assert_eq!(waveform.len(), WINDOW);
        let spectrogram = self.log_mel(waveform);
        if let Some(t) = trace.as_mut() {
            let bytes: Vec<u8> = spectrogram.iter().flat_map(|v| v.to_le_bytes()).collect();
            t(25, &bytes);
        }
        let mut x: Vec<i8> = spectrogram
            .iter()
            .map(|&v| ((v / self.input_scale).round() as i32 + self.input_zero).clamp(-128, 127) as i8)
            .collect();
        if let Some(t) = trace.as_mut() {
            t(28, bytemuck_i8(&x));
        }
        let (mut h, mut w, mut c) = (FRAMES, MELS, 1);
        for traced in &self.layers {
            match &traced.layer {
                Layer::Conv(conv) => {
                    let (out, oh, ow) = run_conv(conv, &x, h, w, c);
                    x = out;
                    h = oh;
                    w = ow;
                    c = conv.out_channels;
                }
                Layer::Mean(mean) => {
                    x = mean.run(&x, h, w, c);
                    h = 1;
                    w = 1;
                }
            }
            if let Some(t) = trace.as_mut() {
                t(traced.output, bytemuck_i8(&x));
            }
        }
        x
    }

    /// Spectrogramme log-mel de 96 trames × 64 bandes.
    fn log_mel(&self, waveform: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(FRAMES * MELS);
        let mut buffer = vec![Complex::new(0f64, 0f64); FFT_LENGTH];
        let mut scratch = vec![Complex::new(0f64, 0f64); self.fft.get_inplace_scratch_len()];
        let mut magnitude = [0f32; BINS];
        for frame in 0..FRAMES {
            let samples = &waveform[frame * FRAME_HOP..frame * FRAME_HOP + FRAME_LENGTH];
            for (i, b) in buffer.iter_mut().enumerate() {
                *b = if i < FRAME_LENGTH {
                    Complex::new((samples[i] * self.window[i]) as f64, 0.0)
                } else {
                    Complex::new(0.0, 0.0)
                };
            }
            self.fft.process_with_scratch(&mut buffer, &mut scratch);
            for (m, b) in magnitude.iter_mut().zip(&buffer) {
                let (re, im) = (b.re as f32, b.im as f32);
                *m = (re * re + im * im).sqrt();
            }
            for band in 0..MELS {
                let row = &self.mel[band * BINS..(band + 1) * BINS];
                let sum: f32 = row.iter().zip(&magnitude).map(|(a, b)| a * b).sum();
                out.push((sum + self.mel_bias[band]).ln());
            }
        }
        out
    }
}

fn bytemuck_i8(x: &[i8]) -> &[u8] {
    // i8 et u8 ont la même représentation
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len()) }
}

impl Mean {
    fn run(&self, x: &[i8], h: usize, w: usize, c: usize) -> Vec<i8> {
        let count = (h * w) as i32;
        (0..c)
            .map(|channel| {
                let mut sum = 0i32;
                for pixel in 0..h * w {
                    sum += x[pixel * c + channel] as i32;
                }
                let centered = (sum - self.input_zero * count) as f32;
                let rounded = (centered * (self.scale / count as f32)).round_ties_even() as i32;
                (rounded + self.output_zero).clamp(self.min, self.max) as i8
            })
            .collect()
    }
}

impl Classifier {
    fn sum(&self, x: &[i8], class: usize) -> i32 {
        let mut sum = self.biases[class];
        for (&v, &w) in x.iter().zip(&self.weights[class * 1024..(class + 1) * 1024]) {
            sum += (v as i32 + self.input_offset as i32) * w as i32;
        }
        sum
    }

    fn logit(&self, x: &[i8], class: usize) -> f32 {
        self.sum(x, class) as f32 * self.logit_scale
    }

    fn run(&self, x: &[i8], class: usize) -> (i8, f32) {
        let sum = self.sum(x, class);
        let q = ((sum as f32 * self.scale).round_ties_even() as i32 + self.output_zero).clamp(self.min, self.max);
        let index = (q + 128) as usize;
        (self.logistic_q[index], self.logistic[index])
    }
}

/// Produit scalaire i16 : boucle simple, vectorisée par le compilateur.
#[inline(always)]
fn dot(a: &[i16], b: &[i16]) -> i32 {
    a.iter().zip(b).map(|(&x, &y)| x as i32 * y as i32).sum()
}

#[inline(always)]
fn convolve(conv: &Conv, input: &[i16], h: usize, w: usize, c: usize, out: &mut [i8], oh: usize, ow: usize) {
    let (pad_h, pad_w) = if conv.same {
        (same_padding(h, conv.kernel_h, conv.stride_h), same_padding(w, conv.kernel_w, conv.stride_w))
    } else {
        (0, 0)
    };
    let (kh, kw) = (conv.kernel_h, conv.kernel_w);
    let mut acc = vec![0i32; conv.out_channels];
    for oy in 0..oh {
        for ox in 0..ow {
            let origin_y = (oy * conv.stride_h) as isize - pad_h as isize;
            let origin_x = (ox * conv.stride_w) as isize - pad_w as isize;
            acc.iter_mut().for_each(|a| *a = 0);
            for ky in 0..kh {
                let iy = origin_y + ky as isize;
                if iy < 0 || iy >= h as isize {
                    continue;
                }
                for kx in 0..kw {
                    let ix = origin_x + kx as isize;
                    if ix < 0 || ix >= w as isize {
                        continue;
                    }
                    let pixel = &input[(iy as usize * w + ix as usize) * c..][..c];
                    if conv.depthwise {
                        let filter = &conv.filter[(ky * kw + kx) * c..][..c];
                        for ((a, &p), &f) in acc.iter_mut().zip(pixel).zip(filter) {
                            *a += p as i32 * f as i32;
                        }
                    } else {
                        let tap = (ky * kw + kx) * c;
                        for (oc, a) in acc.iter_mut().enumerate() {
                            let filter = &conv.filter[oc * kh * kw * c + tap..][..c];
                            *a += dot(pixel, filter);
                        }
                    }
                }
            }
            let target = &mut out[(oy * ow + ox) * conv.out_channels..][..conv.out_channels];
            for ((t, &a), r) in target.iter_mut().zip(&acc).zip(&conv.requant) {
                let v = ((a + r.bias) as f32 * r.scale).round_ties_even() as i32 + conv.output_zero;
                *t = v.clamp(conv.min, conv.max) as i8;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn convolve_avx2(conv: &Conv, input: &[i16], h: usize, w: usize, c: usize, out: &mut [i8], oh: usize, ow: usize) {
    convolve(conv, input, h, w, c, out, oh, ow)
}

fn run_conv(conv: &Conv, x: &[i8], h: usize, w: usize, c: usize) -> (Vec<i8>, usize, usize) {
    let (oh, ow) = if conv.same {
        (h.div_ceil(conv.stride_h), w.div_ceil(conv.stride_w))
    } else {
        ((h - conv.kernel_h) / conv.stride_h + 1, (w - conv.kernel_w) / conv.stride_w + 1)
    };
    let input: Vec<i16> = x.iter().map(|&v| v as i16 + conv.input_offset).collect();
    let mut out = vec![0i8; oh * ow * conv.out_channels];
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY : AVX2 vient d'être détecté sur ce processeur.
        unsafe { convolve_avx2(conv, &input, h, w, c, &mut out, oh, ow) };
        return (out, oh, ow);
    }
    convolve(conv, &input, h, w, c, &mut out, oh, ow);
    (out, oh, ow)
}
