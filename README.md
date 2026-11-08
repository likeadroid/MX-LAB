# Music Sequence Extractor

Outil Python avec interface graphique (tkinter) pour extraire automatiquement les séquences musicales d'une vidéo et les découper sans perte.

## Fonctionnalités
- Détection de la musique avec le modèle YAMNet (`models/yamnet.tflite`), sur le mix mono et sur un mix qui atténue les dialogues : sans le canal central en 5.1, gauche moins droite en stéréo
- Découpe vidéo sans réencodage, calée sur les keyframes (FFmpeg)
- Traitement de plusieurs vidéos à la suite

Sur une source mono ou un faux stéréo, les voix ne peuvent pas être séparées et de la musique sous dialogues peut être manquée.

## Réglages
- **Gap** : interruption de la musique tolérée dans un clip. Si la musique reprend dans ce délai, le clip continue ; sinon, un nouveau clip commence
- **Anticipation / Post** : marge ajoutée avant et après chaque séquence, après la fusion
- **Sensibilité** : plus elle est haute, plus la détection retient de passages

## Installation
Python 3.10 et FFmpeg dans `ffmpeg-min/` ou sur le PATH :

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt pyinstaller
```

## Utilisation
```bash
.venv/Scripts/python.exe main.py
```

## Générer l'exe
À lancer depuis `.venv`, qui ne contient pas les anciennes dépendances :

```bash
.venv/Scripts/python.exe -m PyInstaller main.spec --noconfirm
```

## Version Rust
`rust/` contient un portage complet, sans Python ni LiteRT : YAMNet y tourne sur un petit moteur écrit pour ce modèle. Le modèle et FFmpeg sont intégrés à l'exe ; FFmpeg est extrait au premier lancement dans `%LOCALAPPDATA%\music-sequence-extractor`.

Rust et les outils C++ de Visual Studio sont nécessaires pour compiler :

```bash
cd rust
cargo build --release
```

Cette version accepte le glisser-déposer des vidéos. Les clips vont à côté de chaque vidéo, ou dans un dossier choisi, où chaque vidéo a son sous-dossier `music-sequences - <nom>`.

L'exe est créé dans `rust/target/release/`. `mse-cli` y sert à valider la détection sans interface.

## Inspirations
- [LosslessCut](https://github.com/mifi/lossless-cut)
