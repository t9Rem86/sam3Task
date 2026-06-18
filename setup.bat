@echo off
REM ==========================================================================
REM Construction Detector - Windows setup script
REM Installs SAM 3 via HuggingFace Transformers + dependencies.
REM
REM Requires Python 3.10+ (3.12 recommended). Check with:  py --version
REM An NVIDIA GPU is strongly recommended; the model runs at 1008px.
REM ==========================================================================

echo === Construction Detector setup (Windows) ===

REM --- Pick a Python launcher (prefer 3.12) ---
set PY=py -3.12
%PY% --version >nul 2>&1
if errorlevel 1 set PY=py -3
%PY% --version >nul 2>&1
if errorlevel 1 set PY=python

echo [1/4] Creating virtual environment (venv)...
%PY% -m venv venv
if errorlevel 1 goto :error
call venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo [2/4] Installing PyTorch...
REM Detect NVIDIA GPU via nvidia-smi.
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo   No NVIDIA GPU detected -^> installing CPU torch.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
) else (
    echo   NVIDIA GPU detected -^> installing CUDA 12.1 torch.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
)
if errorlevel 1 goto :error

echo [3/4] Installing Transformers (with SAM 3) and dependencies...
pip install "transformers>=5.12.0" || pip install git+https://github.com/huggingface/transformers.git
pip install -r requirements.txt
if errorlevel 1 goto :error

echo [4/4] HuggingFace authentication...
echo   SAM 3 weights are gated at 'facebook/sam3'.
echo   1) Request access at: https://huggingface.co/facebook/sam3
echo   2) Run: hf auth login
echo.
echo === Done! ===
echo.
echo Activate the environment:  venv\Scripts\activate
echo.
echo Quick start:
echo   python main.py "https://www.youtube.com/watch?v=XXXX" -o out.mp4
echo   python main.py --demo -o demo_out.mp4 --device cpu --max-frames 30 --frame-stride 15
goto :eof

:error
echo.
echo *** Setup failed. See the error above. ***
exit /b 1
