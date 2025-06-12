@echo off
REM Update the repository
git pull
REM Batch file to set up the environment, install dependencies, and run Fooocus
git pull
REM Create a Python virtual environment if it does not exist
if not exist "fooocus_env" (
    python -m venv fooocus_env
)

REM Activate the environment
call fooocus_env\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install PyTorch with CUDA support following the official guide
pip install torch==2.1.0 torchvision==0.16.0 --extra-index-url https://download.pytorch.org/whl/cu121

REM Install the remaining required packages
pip install -r requirements_versions.txt

REM Launch Fooocus
python entry_with_update.py %*

pause
