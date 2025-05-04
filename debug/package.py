import os
import sys
import shutil
import subprocess
from pathlib import Path

def print_step(message):
    """Print a formatted step message"""
    print("\n" + "=" * 80)
    print(f"  {message}")
    print("=" * 80)

def check_pyinstaller():
    """Check if PyInstaller is installed"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'show', 'pyinstaller'], stdout=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False

def install_pyinstaller():
    """Install PyInstaller"""
    print("Installing PyInstaller...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])
        return True
    except subprocess.CalledProcessError:
        print("Failed to install PyInstaller. Please install it manually.")
        return False

def create_executable():
    """Create standalone executable using PyInstaller"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sber-swap-gui.py')
    
    # Define additional data files
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'webapp/models')
    
    # Create PyInstaller command
    cmd = [
        sys.executable, '-m', 'pyinstaller',
        '--name=SberSwapGUI',
        '--onedir',
        '--windowed',
        '--icon=sberswap_icon.ico' if os.path.exists('sberswap_icon.ico') else '',
        f'--add-data={models_dir}{os.pathsep}models',
        script_path
    ]
    
    # Add required modules that might need explicit inclusion
    required_modules = [
        'face_detect_crop', 'face_align', 'image_infer', 'masks'
    ]
    
    for module in required_modules:
        module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{module}.py')
        if os.path.exists(module_path):
            cmd.append(f'--add-data={module_path}{os.pathsep}.')
    
    # Remove empty arguments
    cmd = [arg for arg in cmd if arg]
    
    print("Running PyInstaller with command:")
    print(" ".join(cmd))
    
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"PyInstaller failed: {e}")
        return False

def copy_additional_files():
    """Copy additional files to the dist directory"""
    dist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dist', 'SberSwapGUI')
    
    # Create output directory in dist
    output_dir = os.path.join(dist_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a readme file in the dist directory
    readme_path = os.path.join(dist_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write("SberSwap GUI Application\n")
        f.write("=======================\n\n")
        f.write("A face swapping application with drag and drop support.\n\n")
        f.write("Usage:\n")
        f.write("1. Drag and drop a source face image (left panel)\n")
        f.write("2. Drag and drop a target image (middle panel)\n")
        f.write("3. Click 'Process' to swap faces\n")
        f.write("4. Results are saved in the 'output' folder\n\n")
        f.write("Note: For first-time use, it may take a moment to load all the models.\n")

    print(f"Created README.txt in {dist_dir}")
    return True

def main():
    """Main packaging function"""
    print_step("SberSwap GUI Packaging")
    print("This script will create a standalone executable for SberSwap GUI.")
    
    # Check if PyInstaller is installed
    if not check_pyinstaller():
        print("PyInstaller is not installed.")
        if not install_pyinstaller():
            print("Exiting.")
            return
    
    # Create executable
    if not create_executable():
        print("Failed to create executable.")
        return
    
    # Copy additional files
    if not copy_additional_files():
        print("Failed to copy additional files.")
        return
    
    print_step("Packaging Complete")
    print("The standalone executable has been created in the 'dist/SberSwapGUI' directory.")
    print("You can distribute this entire folder to run the application without Python installed.")

if __name__ == "__main__":
    main()