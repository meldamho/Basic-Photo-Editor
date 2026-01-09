<<<<<<< HEAD
# Basic-Photo-Editor
=======
# Basic Photo Editor

Local desktop photo editor for an Image Processing class. Built with Python 3.10+, PySide6, OpenCV, NumPy, and Matplotlib. Phased development.

## Setup
1. Create/activate a Python 3.10+ environment (venv recommended).
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the app: `python main.py`.

## Windows .exe build (PyInstaller)
- Ensure the venv is active and PySide6/Matplotlib are installed (`pip install -r requirements.txt`).
- From repo root run: `powershell -ExecutionPolicy Bypass -File build_exe.ps1`
	- The script installs PyInstaller if missing, collects PySide6 plugins and Matplotlib data/backends, and writes the exe to `dist/BasicPhotoEditor/`.
- Alternatively, use the provided spec: `pyinstaller basic_photo_editor.spec --noconfirm --clean`
	- If running manually, ensure Matplotlib data is added (the spec already collects it).

## Phases
We will build this project phase-by-phase, keeping each step minimal and testable before moving on.
>>>>>>> 426af35 (Initial commit)
