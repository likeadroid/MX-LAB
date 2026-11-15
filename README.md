# MX-LAB

Logiciel pour étudier la musique d'un film à l'image : timeline de la vidéo, détection de la musique, onde, spectre, loudness, lecteur précis, puis correction et export des séquences musicales sans réencodage.

## Fonctionnalités
- **Timeline** : image (vignettes et chapitres), musique et séquences, audio (spectre ou onde, loudness superposée), avec zoom par barre et lecture à l'image près (moteur libmpv)
- **Détection de la musique** avec le modèle YAMNet (`models/yamnet.tflite`), exécuté par un petit moteur écrit pour ce modèle, sur le mix et sur un mix qui atténue les dialogues
- **Loudness** BS.1770 par canal (TP, I, S, M, LRA), avec cibles réglables
- **Mixeur** : solo, mute et fader par canal
- **Séquences** : tableau modifiable (bornes, mesures, mood, ensemble, instruments), notes, export sans perte calé sur les keyframes (FFmpeg)
- **MIDI (expérimental)** : à activer dans Affichage > Pistes. Les notes de chaque séquence, transcrites par le modèle Basic Pitch (à partir de L et R en 5.1), affichées en piano roll ; un clic sur une note la fait entendre
- **Projet** `<vidéo>.mx-lab.json` enregistré à côté de la vidéo

## Installer
Prêt à l'emploi, sans rien installer d'autre : ffmpeg et libmpv sont livrés avec l'app.

- **macOS** (Apple Silicon, macOS 12 ou plus) : `MX-LAB-macOS.zip`, à décompresser, puis glisser `MX-LAB.app` dans Applications. L'app n'est pas signée par un compte développeur Apple : au premier lancement, clic droit > Ouvrir (ou Réglages Système > Confidentialité et sécurité > Ouvrir quand même).
- **Linux** (x86_64 ; Ubuntu 24.04, Fedora 40 ou plus récent) : `MX-LAB-x86_64.AppImage`, à rendre exécutable puis lancer.

Les paquets sont publiés dans les releases du dépôt, et fabriqués par la CI à chaque push (onglet Actions). Données locales : `%LOCALAPPDATA%\MX-LAB` (Windows), `~/Library/Application Support/MX-LAB` (macOS), `~/.local/share/mx-lab` (Linux).

## Compiler
Avec Rust. `cargo build --profile studio-dev` pour itérer plus vite.

**Windows** (outils C++ de Visual Studio requis) : FFmpeg (`ffmpeg-min/ffmpeg.exe`) et libmpv (`vendor/libmpv/libmpv-2.dll`) ne sont pas dans le dépôt. Placés à ces endroits avant de compiler, ils sont intégrés à `target/release/mx-lab.exe` et extraits au premier lancement.

**macOS et Linux** : ffmpeg et libmpv viennent du système pour développer, et sont copiés dans le paquet par `packaging/macos.sh` (dylibbundler) ou `packaging/linux.sh` (linuxdeploy). La vidéo y passe par le rendu logiciel de mpv.

```bash
# macOS
brew install ffmpeg mpv dylibbundler
cargo build --release && bash packaging/macos.sh
# Ubuntu
sudo apt install ffmpeg libmpv2
cargo build --release && bash packaging/linux.sh
```

`mx-lab --self-check` vérifie que ffmpeg et libmpv livrés fonctionnent.

## Licence
GPL-3.0, voir `LICENSE`.

## Crédits
- [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) (Apache 2.0)
- [Basic Pitch](https://github.com/spotify/basic-pitch) de Spotify, modèle `models/basic-pitch-nmp.onnx` (Apache 2.0), exécuté par [tract](https://github.com/sonos/tract)
- Police [Inter](https://rsms.me/inter/) (SIL Open Font License, `assets/fonts/OFL-Inter.txt`)
- [mpv](https://mpv.io/) et [FFmpeg](https://ffmpeg.org/)
- Inspiration : [LosslessCut](https://github.com/mifi/lossless-cut)
