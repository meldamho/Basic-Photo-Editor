# Build Windows .exe using PyInstaller
# Usage: from repo root, run:  powershell -ExecutionPolicy Bypass -File build_exe.ps1
$ErrorActionPreference = "Stop"

$python = "./.venv/Scripts/python.exe"

if (-not (Test-Path $python)) {
    Write-Error "Python venv not found at $python. Activate/create it first."
}

Write-Host "Installing PyInstaller if missing..."
& $python -m pip install -q --upgrade pip
& $python -m pip install -q pyinstaller

Write-Host "Resolving matplotlib data path..."
$mplData = & $python -c "import matplotlib as mpl, pathlib; print(pathlib.Path(mpl.get_data_path()).resolve())"
if (-not $mplData) { Write-Error "Failed to locate matplotlib data path" }

$extraData = "${mplData};matplotlib/mpl-data"

Write-Host "Building exe..."
& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --name BasicPhotoEditor `
    --add-data "$extraData" `
    --collect-submodules PySide6 `
    --collect-data PySide6 `
    --collect-submodules matplotlib `
    --collect-data matplotlib `
    main.py

Write-Host "Build complete. See dist/BasicPhotoEditor/"
