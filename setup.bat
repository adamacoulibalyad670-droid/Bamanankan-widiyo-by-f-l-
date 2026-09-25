@echo off
REM Bamanankan Video Dubber Pro — setup.bat
REM سكريبت التشغيل السريع للـ Windows

echo.
echo Bamanankan Video Dubber Pro - Setup
echo.

python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python not found. Please install Python 3.8+
    exit /b 1
)

ffmpeg --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: FFmpeg not found.
    echo Visit: https://ffmpeg.org/download.html
    exit /b 1
)

echo Creating virtual environment...
python -m venv .venv

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Installing requirements...
pip install --upgrade pip
pip install -r requirements.txt

echo Installing Whosper...
pip install git+https://github.com/sudoping01/whosper.git

echo Installing MALIBA-AI...
pip install maliba-ai

echo.
echo Setup completed successfully!
echo.
echo To run the app:
echo   .venv\Scripts\activate.bat
echo   streamlit run app.py
echo.
