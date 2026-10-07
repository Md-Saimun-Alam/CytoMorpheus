@echo off
REM Build CytoMorpheus Analyzer as a Windows application (one folder, exe inside).
REM Run from this folder, in the environment that runs cytomorpheus_analyzer.py.
pip install pyinstaller
pyinstaller --noconfirm --clean --onedir --windowed --name CytoMorpheus ^
  --icon app_icon.ico ^
  --add-data "app_icon.ico;." ^
  --collect-all cellpose ^
  --collect-submodules torchvision ^
  --collect-submodules scipy ^
  --hidden-import PIL._tkinter_finder ^
  cytomorpheus_analyzer.py
REM Trained weights are not included. Copy them to dist\CytoMorpheus\models\<architecture>\
echo.
echo Done. Run dist\CytoMorpheus\CytoMorpheus.exe
pause
