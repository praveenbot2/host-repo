@echo off
setlocal

set "ROOT=%~dp0"
set "VENV=%ROOT%.venv"
set "PYTHON=py -3.13"

echo [1/4] Checking virtual environment...
if not exist "%VENV%\Scripts\python.exe" (
    echo Creating virtual environment at %VENV% ...
    %PYTHON% -m venv "%VENV%"
    if errorlevel 1 goto :error
) else (
    echo Reusing existing virtual environment.
)

echo [2/4] Upgrading pip...
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error

echo [3/4] Installing dependencies...
"%VENV%\Scripts\python.exe" -m pip install -r "%ROOT%requirements.txt"
if errorlevel 1 goto :error

echo [4/4] Starting application...
"%VENV%\Scripts\python.exe" "%ROOT%run.py"
goto :eof

:error
echo.
echo Setup failed. Review the messages above.
exit /b 1