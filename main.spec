# -*- mode: python ; coding: utf-8 -*-
import os

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[
        ('ffmpeg-min/ffmpeg.exe', '.'),
        *collect_dynamic_libs('ai_edge_litert'),
    ],
    datas=[
        ('models/yamnet.tflite', 'models'),
    ],
    hiddenimports=collect_submodules('ai_edge_litert'),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'pandas', 'IPython', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'librosa', 'scipy', 'numba', 'llvmlite', 'sklearn'],
    noarchive=False,
    optimize=0,
)
# LiteRT plante (violation d'accès) avec un runtime C++ plus ancien que celui
# avec lequel il est compilé : on embarque celui de Windows, à jour.
VC_RUNTIME = {'msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll'}
system32 = os.path.join(os.environ['SystemRoot'], 'System32')
a.binaries = [
    (name, os.path.join(system32, name), kind) if name.lower() in VC_RUNTIME else (name, path, kind)
    for name, path, kind in a.binaries
]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='music-sequence-extractor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
