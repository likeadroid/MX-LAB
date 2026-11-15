# MX-LAB

Logiciel pour étudier la musique d'un film à l'image : timeline de la vidéo, détection de la musique, onde, spectre, loudness, lecteur précis, puis correction et export des séquences musicales sans réencodage.

## Fonctionnalités
- **Timeline** : image (vignettes et chapitres), musique et séquences, audio (spectre ou onde, loudness superposée), avec zoom par barre et lecture à l'image près (moteur libmpv)
- **Détection de la musique** avec le modèle YAMNet (`models/yamnet.tflite`), exécuté par un petit moteur écrit pour ce modèle, sur le mix et sur un mix qui atténue les dialogues
- **Loudness** BS.1770 par canal (TP, I, S, M, LRA), avec cibles réglables
- **Mixeur** : solo, mute et fader par canal
- **Séquences** : tableau modifiable (bornes, mesures, mood, ensemble, instruments), notes, export sans perte calé sur les keyframes (FFmpeg)
- **Projet** `<vidéo>.mx-lab.json` enregistré à côté de la vidéo

## Compiler
Rust et, sous Windows, les outils C++ de Visual Studio sont nécessaires. FFmpeg (`ffmpeg-min/ffmpeg.exe`) et libmpv (`vendor/libmpv/libmpv-2.dll`) ne sont pas dans le dépôt : ils sont à placer à ces endroits avant de compiler, puis sont intégrés à l'exe et extraits au premier lancement dans `%LOCALAPPDATA%\MX-LAB`.

```bash
cargo build --release
```

L'exe est créé dans `target/release/mx-lab.exe`. Pour itérer plus vite : `cargo build --profile studio-dev`.

## macOS et Linux
MX-LAB compile et passe ses tests sous macOS et Linux (voir la CI). FFmpeg et libmpv n'y sont pas intégrés à l'exécutable : ils sont cherchés à côté de lui, puis dans Homebrew et le système.

```bash
# macOS
brew install ffmpeg mpv
# Ubuntu / Debian
sudo apt install ffmpeg libmpv2
```

```bash
cargo build --release
```

La vidéo y passe par le rendu logiciel de mpv. Données locales : `~/Library/Application Support/MX-LAB` (macOS), `~/.local/share/mx-lab` (Linux). Pas encore de paquet `.app` ni d'AppImage.

## Licence
GPL-3.0, voir `LICENSE`.

## Crédits
- [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) (Apache 2.0)
- Police [Inter](https://rsms.me/inter/) (SIL Open Font License, `assets/fonts/OFL-Inter.txt`)
- [mpv](https://mpv.io/) et [FFmpeg](https://ffmpeg.org/)
- Inspiration : [LosslessCut](https://github.com/mifi/lossless-cut)
