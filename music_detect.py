import os
import re
import subprocess
import sys
import tempfile

import ffmpeg
import numpy as np

SAMPLE_RATE = 16000
WINDOW = 15600  # entrée fixe de YAMNet (0,975 s)
HOP = 7680      # 0,48 s
STEP = HOP / SAMPLE_RATE
MUSIC_CLASS = 132  # classe « Music » d'AudioSet

SMOOTHING_S = 3
MIN_DURATION_S = 5
HIGH_THRESHOLD = 0.05  # un passage net déclenche la détection...
LOW_THRESHOLD = 0.05   # ...qui se prolonge tant que la musique continue
DEFAULT_SENSITIVITY = 50
MAX_SIDE_ATTENUATION_DB = 40  # au-delà, gauche - droite est considéré comme vide

# Canaux à garder : le central porte généralement les dialogues, qui masquent
# la musique pour le modèle. Ordre ffmpeg : 5.1 = FL FR FC LFE BL BR.
CHANNELS_WITHOUT_CENTER = {
    '5.1': (0, 1, 4, 5),
    '5.1(side)': (0, 1, 4, 5),
    '5.0': (0, 1, 3, 4),
    '5.0(side)': (0, 1, 3, 4),
    '7.1': (0, 1, 4, 5, 6, 7),
    '7.1(wide)': (0, 1, 4, 5, 6, 7),
}

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
_interpreter = None


def _model_path():
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, 'models', 'yamnet.tflite')


def _get_interpreter():
    global _interpreter
    if _interpreter is None:
        from ai_edge_litert.interpreter import Interpreter
        _interpreter = Interpreter(model_path=_model_path())
        _interpreter.allocate_tensors()
    return _interpreter


def probe_audio(video_path):
    """Retourne (disposition des canaux, durée en s) de la 1re piste audio.

    ffprobe n'étant pas embarqué, on lit le résumé affiché par `ffmpeg -i`.
    """
    info = subprocess.run(['ffmpeg', '-hide_banner', '-i', video_path],
                          stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE, encoding='utf-8', errors='replace',
                          creationflags=_NO_WINDOW).stderr
    audio = re.search(r'Audio:.*?\d+ Hz, ([^,]+),', info)
    if audio is None:
        raise ValueError('aucune piste audio trouvée')
    duration = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', info)
    seconds = 0.0
    if duration:
        seconds = int(duration[1]) * 3600 + int(duration[2]) * 60 + float(duration[3])
    return audio[1].strip(), seconds


def _mean_volume(video_path, audio_filter):
    """Niveau moyen en dB d'un mix de la 1re piste audio."""
    info = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', video_path, '-map', '0:a:0', '-vn',
                           '-af', f'{audio_filter},volumedetect', '-f', 'null', '-'],
                          stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE, encoding='utf-8', errors='replace',
                          creationflags=_NO_WINDOW).stderr
    level = re.search(r'mean_volume: (-?[\d.]+|-?inf) dB', info)
    return float(level[1]) if level else float('-inf')


def choose_mixes(video_path, layout):
    """Retourne (mixes à analyser, dialogues atténués ?).

    Le mix atténué fait ressortir la musique sous les voix : sans canal central en
    5.1, gauche moins droite en stéréo (muet sur un faux stéréo). Il perd la musique
    placée au centre, d'où l'analyse complémentaire du mix mono.
    """
    channels = CHANNELS_WITHOUT_CENTER.get(layout)
    if channels is not None:
        weight = 1 / len(channels)
        mix = '+'.join(f'{weight}*c{c}' for c in channels)
        return [['-af', f'pan=mono|c0={mix}'], ['-ac', '1']], True
    if layout == 'stereo':
        side, mono = 'pan=mono|c0=0.5*c0-0.5*c1', 'pan=mono|c0=0.5*c0+0.5*c1'
        if _mean_volume(video_path, side) >= _mean_volume(video_path, mono) - MAX_SIDE_ATTENUATION_DB:
            return [['-af', side], ['-af', mono]], True
        return [['-af', mono]], False
    return [['-ac', '1']], False


def music_probabilities(video_path, mix_args, duration=0.0, progress=None):
    """Calcule P(musique) toutes les 0,48 s, sans charger tout l'audio en mémoire."""
    interpreter = _get_interpreter()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    command = ['ffmpeg', '-v', 'error', '-i', video_path, '-map', '0:a:0', '-vn',
               *mix_args, '-ar', str(SAMPLE_RATE), '-f', 'f32le', '-']
    scores = []
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=errors, creationflags=_NO_WINDOW)
        try:
            buffer = np.frombuffer(process.stdout.read(WINDOW * 4), dtype=np.float32)
            while len(buffer) == WINDOW:
                interpreter.set_tensor(input_detail['index'],
                                       buffer.reshape(input_detail['shape']).copy())
                interpreter.invoke()
                scores.append(float(interpreter.get_tensor(output_detail['index']).reshape(-1)[MUSIC_CLASS]))
                if progress and duration and len(scores) % 500 == 0:
                    progress(min(99, int(len(scores) * STEP / duration * 100)))
                new = np.frombuffer(process.stdout.read(HOP * 4), dtype=np.float32)
                if len(new) < HOP:
                    break
                buffer = np.concatenate((buffer[HOP:], new))
        finally:
            process.stdout.close()
            code = process.wait()
        if code != 0:
            errors.seek(0)
            message = errors.read().decode('utf-8', 'replace').strip()
            raise RuntimeError(f'ffmpeg a échoué : {message}')
    return np.array(scores, dtype=np.float32)


def sensitivity_thresholds(sensitivity):
    """Curseur 10-100 : haut = plus sensible = seuils plus bas (défaut 50)."""
    factor = 2 ** ((DEFAULT_SENSITIVITY - sensitivity) / 25)
    return min(HIGH_THRESHOLD * factor, 0.95), LOW_THRESHOLD * factor


def scores_to_segments(scores, sensitivity=DEFAULT_SENSITIVITY, gap=15.0, anticipation=6.0, post=6.0, duration=None):
    """Transforme P(musique) en segments (début, fin) en secondes.

    Le gap mesure le vrai temps sans musique : anticipation et post ne
    s'appliquent qu'après cette fusion, pour ne pas ponter davantage que le gap.
    """
    if len(scores) == 0:
        return []
    high, low = sensitivity_thresholds(sensitivity)
    width = max(1, int(SMOOTHING_S / STEP))
    smooth = np.convolve(scores, np.ones(width) / width, mode='same')

    above = np.concatenate(([0], (smooth > low).astype(np.int8), [0]))
    changes = np.flatnonzero(np.diff(above))
    runs = [(a, b) for a, b in zip(changes[::2], changes[1::2]) if smooth[a:b].max() > high]

    merged = []
    for start, end in runs:
        if merged and (start - merged[-1][1]) * STEP <= gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    limit = duration if duration else len(scores) * STEP
    segments = []
    for start, end in merged:
        if (end - start) * STEP < MIN_DURATION_S:
            continue
        begin = float(max(0.0, start * STEP - anticipation))
        finish = float(min(limit, end * STEP + post))
        if segments and begin <= segments[-1][1]:
            segments[-1] = (segments[-1][0], finish)
        else:
            segments.append((begin, finish))
    return segments


def detect_music_probabilities(video_path, duration, mixes, progress=None):
    """P(musique) maximale entre les mixes analysés."""
    scores = None
    for index, mix_args in enumerate(mixes):
        report = None
        if progress:
            report = lambda p, i=index: progress((i * 100 + p) // len(mixes))
        current = music_probabilities(video_path, mix_args, duration, report)
        if scores is None:
            scores = current
        else:
            size = min(len(scores), len(current))
            scores = np.maximum(scores[:size], current[:size])
    return scores


def detect_music_segments(video_path, sensitivity=DEFAULT_SENSITIVITY, gap=15.0, anticipation=6.0, post=6.0, progress=None):
    """Retourne (segments, dialogues atténués ?) pour une vidéo."""
    layout, duration = probe_audio(video_path)
    mixes, dialogue_reduced = choose_mixes(video_path, layout)
    print(f"[DEBUG] Audio : {layout}, {duration:.0f}s, mixes : {mixes}")
    scores = detect_music_probabilities(video_path, duration, mixes, progress)
    segments = scores_to_segments(scores, sensitivity, gap, anticipation, post, duration or None)
    return segments, dialogue_reduced


def extract_segments(video_path, segments, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Container de sortie identique à la source : garantit la compatibilité
    # du flux copié (-c copy). Copier vers .mp4 depuis un .mkv/.avi peut échouer
    # selon les codecs (audio DTS, sous-titres...). On retombe sur .mp4 si besoin.
    ext = os.path.splitext(video_path)[1].lower()
    if ext not in ('.mp4', '.mkv', '.avi', '.mov', '.webm'):
        ext = '.mp4'
    out_files = []
    for idx, (start, end) in enumerate(segments):
        # Formatage des timestamps hh-mm-ss
        def format_ts(ts):
            h = int(ts // 3600)
            m = int((ts % 3600) // 60)
            s = int(ts % 60)
            return f"{h:02d}-{m:02d}-{s:02d}"
        start_str = format_ts(start)
        end_str = format_ts(end)
        out_file = os.path.join(output_dir, f'segment_{idx+1:03d}_{start_str}_{end_str}{ext}')
        # -ss en option d'ENTREE : ffmpeg cale automatiquement le début sur la
        # keyframe précédente (« keyframe cut » facon LosslessCut) → image dès la
        # 1re frame et A/V synchro. Ne PAS ajouter de -ss en sortie (tronque la
        # vidéo et provoque le démarrage « audio sans image »).
        (
            ffmpeg
            .input(video_path, ss=start)
            .output(out_file, t=end-start, c='copy', avoid_negative_ts='make_zero')
            .run(overwrite_output=True)
        )
        out_files.append(out_file)
    return out_files
