import os
import sys
import shutil
import subprocess
from pathlib import Path
import urllib.request
import zipfile
import tempfile

# Model files to download
MODEL_FILES = [
    'G_unet_2blocks.onnx',
    'G_unet_2blocks.onnx.prototxt',
    'scrfd_10g_bnkps.onnx',
    'scrfd_10g_bnkps.onnx.prototxt',
    'arcface_backbone.onnx',
    'arcface_backbone.onnx.prototxt',
    'face_landmarks.onnx',
    'face_landmarks.onnx.prototxt',
    '10_net_G.onnx',
    '10_net_G.onnx.prototxt'
]

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/sber-swap/'

# Required Python packages
REQUIRED_PACKAGES = [
    'numpy',
    'scikit-image',
    'opencv-python',
    'Pillow',
    'ailia',
    'tkinterdnd2'
]

def print_step(message):
    """Print a formatted step message"""
    print("\n" + "=" * 80)
    print(f"  {message}")
    print("=" * 80)

def install_requirements():
    """Install required Python packages"""
    print_step("Installing required Python packages")
    
    # Check if pip is available
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', '--version'])
    except subprocess.CalledProcessError:
        print("Error: pip is not available. Please install pip first.")
        return False
    
    # Install each package
    for package in REQUIRED_PACKAGES:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', package
            ])
        except subprocess.CalledProcessError:
            print(f"Error installing {package}. Please install it manually.")
            return False
    
    print("All required packages installed successfully!")
    return True

def download_models():
    """Download model files"""
    print_step("Downloading model files")
    
    # Create models directory
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Download each model file
    for model_file in MODEL_FILES:
        target_path = os.path.join(models_dir, model_file)
        
        # Skip if file already exists
        if os.path.exists(target_path):
            print(f"File already exists: {model_file}")
            continue
        
        url = REMOTE_PATH + model_file
        print(f"Downloading {model_file}...")
        
        try:
            urllib.request.urlretrieve(url, target_path)
            print(f"Downloaded: {model_file}")
        except Exception as e:
            print(f"Error downloading {model_file}: {str(e)}")
            return False
    
    print("All model files downloaded successfully!")
    return True

def copy_additional_files():
    """Copy additional required files from the original repository"""
    print_step("Setting up additional files")
    
    # In a real application, we would download or extract these files
    # For this example, we assume they're already available in the same directory
    
    source_dir = os.path.dirname(os.path.abspath(__file__))
    required_files = [
        'face_detect_crop.py',
        'face_align.py',
        'image_infer.py',
        'masks.py'
    ]
    
    # Check if the required files exist
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(source_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print("The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease download these files from the original repository.")
        return False
    
    print("All required files are present!")
    return True

def create_shortcut():
    """Create desktop shortcut for Windows"""
    print_step("Creating desktop shortcut")
    
    # Only on Windows
    if sys.platform != 'win32':
        print("Skipping shortcut creation (not on Windows)")
        return True
    
    try:
        # Path to the main script
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sber-swap-gui.py')
        
        # Path to the desktop
        desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
        
        # Create shortcut
        import winshell
        from win32com.client import Dispatch
        
        shortcut_path = os.path.join(desktop_path, 'SberSwap Face Swap.lnk')
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(shortcut_path)
        shortcut.Targetpath = sys.executable
        shortcut.Arguments = f'"{script_path}"'
        shortcut.WorkingDirectory = os.path.dirname(script_path)
        shortcut.save()
        
        print(f"Shortcut created at: {shortcut_path}")
        return True
    except Exception as e:
        print(f"Error creating shortcut: {str(e)}")
        print("You can manually create a shortcut to sber-swap-gui.py")
        return False

def create_batch_file():
    """Create a batch file to run the application on Windows"""
    print_step("Creating batch file")
    
    # Only on Windows
    if sys.platform != 'win32':
        print("Skipping batch file creation (not on Windows)")
        return True
    
    try:
        # Path to the main script
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sber-swap-gui.py')
        
        # Create batch file
        batch_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_sberswap.bat')
        
        with open(batch_path, 'w') as f:
            f.write(f'@echo off\n')
            f.write(f'echo Starting SberSwap Face Swap...\n')
            f.write(f'"{sys.executable}" "{script_path}"\n')
            f.write(f'if %ERRORLEVEL% neq 0 pause\n')
        
        print(f"Batch file created at: {batch_path}")
        return True
    except Exception as e:
        print(f"Error creating batch file: {str(e)}")
        return False

def main():
    """Main setup function"""
    print_step("SberSwap GUI Setup")
    print("This script will set up the SberSwap GUI application.")
    
    # Create output and models directories
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements. Please install them manually.")
    
    # Download models
    if not download_models():
        print("Failed to download model files. Please download them manually.")
    
    # Copy additional files
    if not copy_additional_files():
        print("Missing required files. Please download them from the original repository.")
    
    # Create shortcuts and batch files
    if sys.platform == 'win32':
        try:
            import winshell
            import win32com.client
            create_shortcut()
        except ImportError:
            print("winshell and pywin32 are required for shortcut creation.")
            print("You can install them with: pip install winshell pywin32")
        
        # create_batch_file()
    
    print_step("Setup Complete")
    print("You can now run the SberSwap GUI application by executing sber-swap-gui.py")
    print("On Windows, you can also use the created shortcut or run_sberswap.bat")

if __name__ == "__main__":
    main()