@echo off
REM Batch file to set up the environment, install dependencies, and run Fooocus

REM Create a Python virtual environment if it does not exist
if not exist "fooocus_env" (
    python -m venv fooocus_env
)

REM Activate the environment
call fooocus_env\Scripts\activate.bat

REM Upgrade pip and install required packages
python -m pip install --upgrade pip
pip install -r requirements_versions.txt

REM Launch Fooocus
python entry_with_update.py %*

pause
