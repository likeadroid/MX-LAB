import sys
import os

def _setup_bundled_ffmpeg():
    """Rend ffmpeg accessible via le PATH.

    - En exe PyInstaller (one-file), le binaire embarqué est extrait dans
      sys._MEIPASS.
    - En script, on ajoute le dossier ffmpeg-min (ou ffmpeg/bin) du projet.
    """
    candidates = []
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        candidates.append(meipass)
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(here, 'ffmpeg-min'))
        candidates.append(os.path.join(here, 'ffmpeg', 'bin'))
    for d in candidates:
        if d and os.path.isdir(d):
            os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')

_setup_bundled_ffmpeg()

import tkinter as tk
from tkinter import ttk, filedialog


class MusicSequenceExtractor:
    def __init__(self, root):
        self.root = root
        self.video_paths = []
        self.init_ui()

    def init_ui(self):
        self.root.title('Music Sequence Extractor')
        self.root.minsize(560, 360)
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill='both', expand=True)

        self.btn_load = ttk.Button(frame, text='Sélectionner des vidéos', command=self.load_video)
        self.btn_load.pack(fill='x')

        self.label = ttk.Label(frame, text='Chargez un ou plusieurs films pour commencer.', wraplength=520, justify='left')
        self.label.pack(fill='x', pady=(8, 12))

        # Curseurs de paramètres : (attribut, libellé, min, max, défaut)
        self.sliders = {}
        for attr, text, mn, mx, default in (
            ('gap', 'Gap (s)', 1, 60, 15),
            ('anticipation', 'Anticipation (s)', 0, 20, 6),
            ('post', 'Post (s)', 0, 20, 6),
            ('sens', 'Sensibilité', 10, 100, 50),
        ):
            row = ttk.Frame(frame)
            row.pack(fill='x', pady=2)
            ttk.Label(row, text=text, width=16).pack(side='left')
            var = tk.IntVar(value=default)
            scale = tk.Scale(row, from_=mn, to=mx, orient='horizontal',
                             variable=var, resolution=1, showvalue=True)
            scale.pack(side='left', fill='x', expand=True)
            self.sliders[attr] = var

        self.btn_analyze = ttk.Button(frame, text='Analyser et extraire séquences musicales',
                                      command=self.analyze_video, state='disabled')
        self.btn_analyze.pack(fill='x', pady=(8, 0))

    def _set_status(self, text):
        self.label.config(text=text)
        self.root.update()  # rafraîchit l'UI pendant le traitement (thread principal)

    def _truncate(self, path, max_chars=120):
        if len(path) <= max_chars:
            return path
        step = max_chars // 3
        parts = [path[i:i + step] for i in range(0, len(path), step)]
        return '\n'.join(parts[:3])

    def load_video(self):
        file_paths = filedialog.askopenfilenames(
            title='Sélectionner un ou plusieurs films',
            filetypes=[('Vidéo', '*.mp4 *.mkv *.avi'), ('Tous les fichiers', '*.*')])
        if file_paths:
            self.video_paths = list(file_paths)
            n = len(self.video_paths)
            if n == 1:
                self.label.config(text=f'Fichier chargé : {self._truncate(self.video_paths[0])}')
            else:
                names = '\n'.join('  • ' + os.path.basename(p) for p in self.video_paths[:8])
                if n > 8:
                    names += f'\n  … (+{n - 8})'
                self.label.config(text=f'{n} fichiers chargés :\n{names}')
            self.btn_analyze.config(state='normal')

    def analyze_video(self):
        if not self.video_paths:
            return
        self.btn_analyze.config(state='disabled')
        self.btn_load.config(state='disabled')
        total = len(self.video_paths)
        results = []  # (nom fichier, résumé)
        try:
            for idx, video_path in enumerate(self.video_paths, 1):
                name = os.path.basename(video_path)
                try:
                    summary = self._process_video(video_path, idx, total)
                except Exception as e:
                    summary = f'ERREUR : {e}'
                    print(f"[DEBUG] Exception sur {name} : {e}")
                results.append((name, summary))
        finally:
            self.btn_analyze.config(state='normal')
            self.btn_load.config(state='normal')
        # Récapitulatif final
        if total == 1:
            self._set_status(f'{results[0][0]} → {results[0][1]}')
        else:
            lines = '\n'.join(f'  • {n} → {s}' for n, s in results)
            self._set_status(f'Terminé ({total} vidéos) :\n{lines}')

    def _process_video(self, video_path, idx, total):
        """Traite une vidéo : détection puis découpe.

        Retourne un court résumé (ex. '12 clips → music-sequences').
        """
        from music_detect import detect_music_segments, extract_segments

        prefix = f'[{idx}/{total}] ' if total > 1 else ''
        short = os.path.basename(video_path)
        self._set_status(f'{prefix}{short} : détection des séquences musicales…')
        segments, dialogue_reduced = detect_music_segments(
            video_path,
            sensitivity=self.sliders['sens'].get(),
            gap=self.sliders['gap'].get(),
            anticipation=self.sliders['anticipation'].get(),
            post=self.sliders['post'].get(),
            progress=lambda p: self._set_status(f'{prefix}{short} : détection des séquences musicales… {p} %'),
        )
        print(f"[DEBUG] Détection musicale terminée, segments trouvés : {segments}")
        note = '' if dialogue_reduced else ' (voix non séparables : musique sous dialogues parfois manquée)'
        if not segments:
            return f'aucune séquence détectée{note}'

        self._set_status(f'{prefix}{short} : {len(segments)} séquences, découpage…')
        # Création automatique du dossier d'export à côté de la vidéo
        video_dir = os.path.dirname(video_path)
        base_export_dir = os.path.join(video_dir, 'music-sequences')
        export_dir = base_export_dir
        count = 1
        while os.path.exists(export_dir):
            export_dir = f"{base_export_dir} {count}"
            count += 1
        os.makedirs(export_dir)
        print(f"[DEBUG] Dossier d'export créé : {export_dir}")
        files = extract_segments(video_path, segments, export_dir)
        print(f"[DEBUG] Découpage vidéo terminé, fichiers créés : {files}")
        return f'{len(files)} clips → {os.path.basename(export_dir)}{note}'


if __name__ == '__main__':
    root = tk.Tk()
    app = MusicSequenceExtractor(root)
    root.mainloop()
