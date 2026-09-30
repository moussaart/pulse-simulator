# -*- mode: python ; coding: utf-8 -*-

import dis

original_get_const_info = dis._get_const_info
def patched_get_const_info(arg, constants):
    try:
        return original_get_const_info(arg, constants)
    except IndexError:
        return (None, None)
dis._get_const_info = patched_get_const_info

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('assets', 'assets'),
        ('data', 'data'),
        ('src\\UWB data', 'src\\UWB data'),
        ('src\\user_algorithms', 'src\\user_algorithms'),
        ('src\\core\\uwb\\uwb_hardware_profiles.json', 'src\\core\\uwb')
    ],
    hiddenimports=[
        # Imported lazily inside main.py's worker branch; PyInstaller cannot see
        # through a function-local import that is only reached under a flag.
        'src.api.ai_training_worker',
        'src.api.ai_training_engine',
        'src.api.training_metrics',
    ],
    hookspath=[],
    hooksconfig={},
    # multiprocessing support, so a frozen build can spawn the training worker.
    runtime_hooks=['src\\api\\pulse_freeze_hook.py'],
    excludes=['tensorflow', 'keras', 'tensorboard', 'ml_dtypes', 'h5py', 'ipython', 'tkinter', 'torch'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PULSE_Simulator_App',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets\\logo.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PULSE_Simulator_App',

)
