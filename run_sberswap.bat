@echo off
setlocal enabledelayedexpansion
echo Starting SberSwap Face Swap...

:: Set working directory to script location
cd /d "%~dp0"

:: Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python and add it to your system PATH.
    pause
    exit /b 1
)

:: Check if virtual environment exists
if exist "venv\" (
    echo Virtual environment found. Activating...
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Creating...
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    call venv\Scripts\activate.bat
    
    echo Upgrading pip...
    python -m pip install --upgrade pip
    
    :: Run setup.py to install requirements and download models
    if exist "setup.py" (
        echo Running setup.py to install all requirements and download models...
        python setup.py
        if %ERRORLEVEL% neq 0 (
            echo ERROR: setup.py failed. Some dependencies may not be installed.
            echo Please check the error messages above.
            pause
        )
    ) else (
        echo ERROR: setup.py not found!
        echo Cannot install requirements automatically.
        echo Please make sure setup.py is in the same directory.
        pause
        exit /b 1
    )
)

:: Create temporary check script for dependency verification
(
    echo import sys
    echo try:
    echo     import ailia
    echo     print^("AILIA_OK"^)
    echo except:
    echo     print^("AILIA_MISSING"^)
    echo try:
    echo     import tkinterdnd2
    echo     print^("TKINTERDND2_OK"^)
    echo except:
    echo     print^("TKINTERDND2_MISSING"^)
) > check_deps.py

:: Check critical dependencies
echo.
echo Checking dependencies...

:: Run the check directly in the activated environment
python check_deps.py > dep_check_result.txt
for /f "tokens=*" %%i in (dep_check_result.txt) do (
    if "%%i"=="AILIA_MISSING" (
        echo WARNING: ailia module not found.
        echo Please run setup.py to install missing dependencies.
    ) else if "%%i"=="AILIA_OK" (
        echo [OK] ailia is installed
    )
    
    if "%%i"=="TKINTERDND2_MISSING" (
        echo WARNING: tkinterdnd2 module not found.
        echo Please run setup.py to install missing dependencies.
    ) else if "%%i"=="TKINTERDND2_OK" (
        echo [OK] tkinterdnd2 is installed
    )
)

:: Clean up temporary files
del check_deps.py >nul 2>&1
del dep_check_result.txt >nul 2>&1

:: Check and create necessary directories
if not exist "output\" (
    echo Creating output directory...
    mkdir output
)

:: Check if models directory exists and has required files
set "missing_models=0"
if not exist "models\" (
    echo WARNING: models directory not found!
    set "missing_models=1"
) else (
    :: Check for essential model files
    if not exist "models\G_unet_2blocks.onnx" set "missing_models=1"
    if not exist "models\scrfd_10g_bnkps.onnx" set "missing_models=1"
    if not exist "models\arcface_backbone.onnx" set "missing_models=1"
    if not exist "models\face_landmarks.onnx" set "missing_models=1"
)

if "%missing_models%"=="1" (
    echo.
    echo WARNING: Required model files are missing!
    
    if exist "setup.py" (
        set /p "want_download=Would you like to run setup.py to download the model files now? [Y/N] "
        
        if /i "!want_download!"=="Y" (
            echo.
            echo Running setup.py to download models...
            python setup.py
            if !ERRORLEVEL! neq 0 (
                echo ERROR: Failed to download models.
                echo Please check your internet connection or download models manually from:
                echo https://storage.googleapis.com/ailia-models/sber-swap/
                pause
            ) else (
                echo Model download complete!
            )
        ) else (
            echo You can run setup.py later to download the models.
            echo The application may not work without model files.
            pause
        )
    ) else (
        echo setup.py not found. Please download model files manually from:
        echo https://storage.googleapis.com/ailia-models/sber-swap/
        echo and place them in the "models" directory.
        pause
    )
)

:: Show final status of model files
echo.
echo Checking model files status...
set "all_models_found=1"
if not exist "models\G_unet_2blocks.onnx" (
    echo [-] G_unet_2blocks.onnx - Missing
    set "all_models_found=0"
) else (
    echo [+] G_unet_2blocks.onnx - Found
)
if not exist "models\scrfd_10g_bnkps.onnx" (
    echo [-] scrfd_10g_bnkps.onnx - Missing
    set "all_models_found=0"
) else (
    echo [+] scrfd_10g_bnkps.onnx - Found
)
if not exist "models\arcface_backbone.onnx" (
    echo [-] arcface_backbone.onnx - Missing
    set "all_models_found=0"
) else (
    echo [+] arcface_backbone.onnx - Found
)
if not exist "models\face_landmarks.onnx" (
    echo [-] face_landmarks.onnx - Missing
    set "all_models_found=0"
) else (
    echo [+] face_landmarks.onnx - Found
)

:: Finally, run the application
echo.
echo Launching SberSwap GUI...
echo ========================================
python sber-swap-gui.py

:: Check exit status
if %ERRORLEVEL% neq 0 (
    echo.
    echo ========================================
    echo Application exited with an error.
    echo.
    echo If you see import errors or missing modules:
    echo 1. Run setup.py to install all dependencies
    echo 2. Make sure all model files are properly downloaded
    echo 3. Verify your virtual environment is properly activated
    echo.
    set /p "want_verbose=Would you like to see detailed error information? [Y/N] "
    if /i "!want_verbose!"=="Y" (
        echo.
        echo Running with verbose output...
        python -c "exec(open('sber-swap-gui.py').read())"
    )
    pause
) else (
    echo.
    echo Application closed successfully.
)
endlocal