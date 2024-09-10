# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['app_win.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('C:/Users/Vince/OneDrive/Desktop/code/Percept_Data_Analysis_App/.venv/Lib/site-packages/kaleido', 'kaleido'), 
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Percept App',
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
    icon='Icon.ico'
)
