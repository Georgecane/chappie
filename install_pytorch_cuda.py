#!/usr/bin/env python
"""
Script to help install the correct version of PyTorch with CUDA support.
This script checks your system and provides the appropriate pip command to install PyTorch.
"""

import os
import sys
import platform
import subprocess
import logging
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def check_python_version() -> str:
    """Get the Python version."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"

def check_cuda_version() -> Optional[str]:
    """Check the installed CUDA version."""
    # Try to get CUDA version from nvidia-smi
    try:
        if platform.system() == "Windows":
            # Find nvidia-smi
            nvidia_smi_path = None
            try:
                result = subprocess.run(["where", "nvidia-smi"], capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    nvidia_smi_path = result.stdout.strip().split("\n")[0]
                else:
                    # Try common locations
                    common_paths = [
                        os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe"),
                        os.path.join(os.environ.get("ProgramW6432", "C:\\Program Files"), "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe"),
                        "C:\\Windows\\System32\\nvidia-smi.exe"
                    ]
                    for path in common_paths:
                        if os.path.exists(path):
                            nvidia_smi_path = path
                            break
            except Exception:
                pass
            
            if nvidia_smi_path:
                # Run nvidia-smi to get CUDA version
                result = subprocess.run([nvidia_smi_path], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    # Parse output to find CUDA version
                    for line in result.stdout.split("\n"):
                        if "CUDA Version:" in line:
                            return line.split("CUDA Version:")[1].strip()
        else:
            # Linux/Mac
            try:
                result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    # Parse output to find CUDA version
                    for line in result.stdout.split("\n"):
                        if "CUDA Version:" in line:
                            return line.split("CUDA Version:")[1].strip()
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"Error checking CUDA version: {e}")
    
    # Try to get CUDA version from environment variable
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        # Extract version from path (e.g., C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8)
        try:
            version_part = os.path.basename(cuda_path)
            if version_part.startswith("v"):
                return version_part[1:]
        except Exception:
            pass
    
    return None

def get_pytorch_install_command(cuda_version: Optional[str], python_version: str) -> str:
    """Get the appropriate pip command to install PyTorch with CUDA support."""
    # Default to CPU if no CUDA version is found
    if not cuda_version:
        return "pip install torch torchvision torchaudio"
    
    # Convert CUDA version to the format used by PyTorch
    cuda_major_minor = ".".join(cuda_version.split(".")[:2])
    
    # Map CUDA version to PyTorch CUDA version
    pytorch_cuda_version = None
    if float(cuda_major_minor) >= 12.0:
        pytorch_cuda_version = "12.1"
    elif float(cuda_major_minor) >= 11.8:
        pytorch_cuda_version = "11.8"
    elif float(cuda_major_minor) >= 11.7:
        pytorch_cuda_version = "11.7"
    elif float(cuda_major_minor) >= 11.6:
        pytorch_cuda_version = "11.6"
    elif float(cuda_major_minor) >= 11.3:
        pytorch_cuda_version = "11.3"
    elif float(cuda_major_minor) >= 10.2:
        pytorch_cuda_version = "10.2"
    else:
        logger.warning(f"CUDA version {cuda_version} is too old for recent PyTorch versions")
        logger.warning("Defaulting to CPU version")
        return "pip install torch torchvision torchaudio"
    
    # Get the appropriate pip command
    if platform.system() == "Windows":
        return f"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu{pytorch_cuda_version.replace('.', '')}"
    else:
        # Linux/Mac
        return f"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu{pytorch_cuda_version.replace('.', '')}"

def main():
    """Main function."""
    print("\n" + "=" * 80)
    print(" PyTorch with CUDA Installation Helper ".center(80, "="))
    print("=" * 80 + "\n")
    
    # Check Python version
    python_version = check_python_version()
    print(f"Python version: {python_version}")
    
    # Check CUDA version
    cuda_version = check_cuda_version()
    if cuda_version:
        print(f"CUDA version: {cuda_version}")
    else:
        print("CUDA version: Not found")
    
    # Get PyTorch install command
    install_command = get_pytorch_install_command(cuda_version, python_version)
    
    print("\nTo install PyTorch with CUDA support, run the following command:")
    print("\n" + "=" * 80)
    print(install_command)
    print("=" * 80 + "\n")
    
    # Additional information
    print("After installation, verify that PyTorch can access your GPU with:")
    print("python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\"")
    
    # Offer to run the command
    if platform.system() == "Windows":
        print("\nWould you like to run this command now? (y/n)")
        choice = input().strip().lower()
        if choice == 'y':
            print("\nRunning installation command...")
            subprocess.run(install_command, shell=True)
            
            print("\nVerifying installation...")
            subprocess.run("python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\"", shell=True)
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
