#!/usr/bin/env bash
# Fabrique MX-LAB-x86_64.AppImage, autonome : ffmpeg et libmpv, avec leurs bibliothèques,
# sont copiés dans l'AppImage (les bibliothèques de base du bureau, comme OpenGL, restent celles
# du système, selon la liste d'exclusion de linuxdeploy).
# Prérequis (Ubuntu) : apt install ffmpeg libmpv2 ; cargo build --release
set -euo pipefail
cd "$(dirname "$0")/.."

appdir="target/package/AppDir"
rm -rf target/package
mkdir -p "$appdir" target/package

linuxdeploy="target/package/linuxdeploy-x86_64.AppImage"
curl -fsSL -o "$linuxdeploy" \
  https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage
chmod +x "$linuxdeploy"

libmpv="$(ldconfig -p | sed -n 's/.*libmpv\.so\.2 .*=> \(.*\)$/\1/p' | head -1)"
test -n "$libmpv"

target/release/mx-lab --icon-png 256 target/package/mx-lab.png

export APPIMAGE_EXTRACT_AND_RUN=1
export OUTPUT=MX-LAB-x86_64.AppImage
# libmpv est chargée à l'exécution : la nommer, sinon linuxdeploy ne la verrait pas
"$linuxdeploy" --appdir "$appdir" \
  --executable target/release/mx-lab \
  --executable "$(command -v ffmpeg)" \
  --library "$libmpv" \
  --desktop-file packaging/mx-lab.desktop \
  --icon-file target/package/mx-lab.png \
  --output appimage
mv MX-LAB-x86_64.AppImage target/package/
du -sh target/package/MX-LAB-x86_64.AppImage
