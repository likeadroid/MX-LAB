#!/usr/bin/env bash
# Fabrique MX-LAB.app, autonome : ffmpeg et libmpv (avec toutes leurs bibliothèques Homebrew)
# sont copiés dans l'app, qui ne dépend plus de Homebrew. Puis MX-LAB-macOS.zip.
# Prérequis : brew install ffmpeg mpv dylibbundler ; cargo build --release
set -euo pipefail
cd "$(dirname "$0")/.."

brew_prefix="$(brew --prefix)"
version="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)"
app="target/package/MX-LAB.app"
contents="$app/Contents"
rm -rf target/package
mkdir -p "$contents/MacOS" "$contents/Frameworks" "$contents/Resources"

cp target/release/mx-lab "$contents/MacOS/mx-lab"
cp "$brew_prefix/bin/ffmpeg" "$contents/MacOS/ffmpeg"
cp "$brew_prefix/lib/libmpv.2.dylib" "$contents/Frameworks/libmpv.2.dylib"
chmod u+w "$contents/MacOS/ffmpeg" "$contents/Frameworks/libmpv.2.dylib"

# Copie les bibliothèques dont dépendent ffmpeg et libmpv, et réécrit leurs chemins
dylibbundler -of -b \
  -x "$contents/MacOS/ffmpeg" \
  -x "$contents/Frameworks/libmpv.2.dylib" \
  -d "$contents/Frameworks" \
  -p @executable_path/../Frameworks/ \
  -s "$brew_prefix/lib" < /dev/null
install_name_tool -id @executable_path/../Frameworks/libmpv.2.dylib "$contents/Frameworks/libmpv.2.dylib"

# Plus aucune référence à Homebrew : l'app doit tourner sans lui
if otool -L "$contents/MacOS/ffmpeg" "$contents"/Frameworks/*.dylib | grep -E "$brew_prefix|/usr/local/(opt|Cellar)"; then
  echo "Des bibliothèques pointent encore vers Homebrew" >&2
  exit 1
fi

# Icône
iconset="target/package/mx-lab.iconset"
mkdir -p "$iconset"
for size in 16 32 128 256 512; do
  "$contents/MacOS/mx-lab" --icon-png "$size" "$iconset/icon_${size}x${size}.png"
  "$contents/MacOS/mx-lab" --icon-png "$((size * 2))" "$iconset/icon_${size}x${size}@2x.png"
done
iconutil -c icns "$iconset" -o "$contents/Resources/mx-lab.icns"

cat > "$contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>MX-LAB</string>
  <key>CFBundleDisplayName</key><string>MX-LAB</string>
  <key>CFBundleIdentifier</key><string>io.github.likeadroid.mx-lab</string>
  <key>CFBundleExecutable</key><string>mx-lab</string>
  <key>CFBundleIconFile</key><string>mx-lab</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$version</string>
  <key>CFBundleVersion</key><string>$version</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

# Signature ad hoc : indispensable sur Apple Silicon, après la réécriture des bibliothèques
codesign --force --sign - "$contents"/Frameworks/*.dylib "$contents/MacOS/ffmpeg"
codesign --force --sign - "$app"

ditto -c -k --keepParent "$app" target/package/MX-LAB-macOS.zip
du -sh "$app" target/package/MX-LAB-macOS.zip
