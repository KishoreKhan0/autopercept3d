@echo off
echo Setting up AutoPercept3D Step 1...

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Using Python launcher: py
    py -m venv .venv
) else (
    echo Python launcher 'py' not found. Trying 'python'...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

python -m pip install --upgrade pip
pip install -e .

echo.
echo Setup complete.
echo To activate later, run:
echo .venv\Scripts\activate.bat
