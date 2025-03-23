@echo off
REM Run Red Teaming Framework helper script
REM Usage: run_red_teamer.bat [command] [arguments]

REM Set Python path - change if needed
set PYTHON=python

REM Check if Python is installed
%PYTHON% --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in the PATH.
    echo Please install Python 3.8 or later.
    exit /b 1
)

REM Check if redteamer package is installed
%PYTHON% -c "import redteamer" >nul 2>&1
if errorlevel 1 (
    echo Installing Red Teaming Framework...
    %PYTHON% -m pip install -e .
    if errorlevel 1 (
        echo Error: Failed to install Red Teaming Framework.
        exit /b 1
    )
    echo Red Teaming Framework installed successfully.
)

REM Check environment variables for API keys
if not defined OPENAI_API_KEY (
    echo Warning: OPENAI_API_KEY environment variable is not set.
    echo Set it with: set OPENAI_API_KEY=your_api_key
)

if not defined ANTHROPIC_API_KEY (
    echo Warning: ANTHROPIC_API_KEY environment variable is not set.
    echo Set it with: set ANTHROPIC_API_KEY=your_api_key
)

if not defined GOOGLE_API_KEY (
    echo Warning: GOOGLE_API_KEY environment variable is not set.
    echo Set it with: set GOOGLE_API_KEY=your_api_key
)

REM Create necessary directories
mkdir results 2>nul
mkdir datasets 2>nul
mkdir reports 2>nul

REM Run the command
if "%1"=="" (
    echo Red Teaming Framework - Interactive CLI
    echo --------------------------------------
    echo.
    echo First time using the framework? Start with:
    echo   run_red_teamer.bat quickstart
    echo.
    echo This will guide you through:
    echo - Setting up your API keys
    echo - Running a simple red team evaluation
    echo - Generating a report
    echo.
    echo Other example commands:
    echo   run_red_teamer.bat test       - Test a model interactively
    echo   run_red_teamer.bat run        - Run a red team evaluation with interactive configuration
    echo   run_red_teamer.bat dataset create      - Create a dataset interactively
    echo   run_red_teamer.bat dataset add-vector  - Add vectors to a dataset interactively
    echo   run_red_teamer.bat info                - Show framework information
    %PYTHON% -m redteamer.cli info
) else (
    %PYTHON% -m redteamer.cli %*
) 