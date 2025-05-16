#!/usr/bin/env python
"""
Dedicated script for checking NVIDIA GPU status and compatibility with PyTorch.
This script provides detailed diagnostics about NVIDIA GPUs and CUDA availability.
"""

import os
import sys
import platform
import subprocess
import logging
from typing import List, Optional, Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gpu_diagnostics.log')
    ]
)
logger = logging.getLogger(__name__)

def check_pytorch_installation() -> Dict[str, Any]:
    """Check PyTorch installation and CUDA support."""
    result = {
        "pytorch_installed": False,
        "version": None,
        "cuda_built": False,
        "cuda_version": None,
        "cuda_available": False,
        "device_count": 0,
        "devices": []
    }

    try:
        import torch
        result["pytorch_installed"] = True
        result["version"] = torch.__version__

        # Check if PyTorch was built with CUDA
        result["cuda_built"] = hasattr(torch, 'cuda')

        if result["cuda_built"]:
            # Get CUDA version
            if hasattr(torch.version, 'cuda'):
                result["cuda_version"] = torch.version.cuda

            # Check if CUDA is available
            result["cuda_available"] = torch.cuda.is_available()

            if result["cuda_available"]:
                # Get device count and properties
                result["device_count"] = torch.cuda.device_count()

                for i in range(result["device_count"]):
                    props = torch.cuda.get_device_properties(i)
                    device_info = {
                        "index": i,
                        "name": props.name,
                        "total_memory_gb": props.total_memory / (1024**3),
                        "cuda_capability": f"{props.major}.{props.minor}",
                        "multi_processor_count": props.multi_processor_count
                    }
                    result["devices"].append(device_info)

    except ImportError:
        logger.error("PyTorch is not installed")
    except Exception as e:
        logger.error(f"Error checking PyTorch installation: {e}")

    return result

def find_nvidia_smi() -> Optional[str]:
    """Find the nvidia-smi executable."""
    if platform.system() == "Windows":
        # Try using 'where' command
        try:
            result = subprocess.run(["where", "nvidia-smi"], capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                return result.stdout.strip().split("\n")[0]
        except Exception:
            pass

        # Try common locations
        common_paths = [
            os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe"),
            os.path.join(os.environ.get("ProgramW6432", "C:\\Program Files"), "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe"),
            "C:\\Windows\\System32\\nvidia-smi.exe"
        ]

        for path in common_paths:
            if os.path.exists(path):
                return path
    else:
        # Linux/Mac - try common locations
        for path in ["/usr/bin/nvidia-smi", "/usr/local/bin/nvidia-smi"]:
            if os.path.exists(path):
                return path

        # Try using 'which' command
        try:
            result = subprocess.run(["which", "nvidia-smi"], capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

    return None

def check_nvidia_driver() -> Dict[str, Any]:
    """Check NVIDIA driver installation and GPU information."""
    result = {
        "driver_found": False,
        "driver_version": None,
        "gpus": [],
        "has_gtx": False,
        "nvidia_smi_path": None,
        "nvidia_smi_output": None
    }

    # Find nvidia-smi
    nvidia_smi_path = find_nvidia_smi()
    result["nvidia_smi_path"] = nvidia_smi_path

    if not nvidia_smi_path:
        logger.warning("nvidia-smi not found. NVIDIA drivers may not be installed.")
        return result

    # Check driver version
    try:
        cmd_version = [nvidia_smi_path, "--query-gpu=driver_version", "--format=csv,noheader"]
        proc = subprocess.run(cmd_version, capture_output=True, text=True, timeout=5)

        if proc.returncode == 0:
            result["driver_found"] = True
            result["driver_version"] = proc.stdout.strip()
    except Exception as e:
        logger.error(f"Error checking driver version: {e}")
        return result

    # Get GPU information
    try:
        cmd_gpus = [nvidia_smi_path, "--query-gpu=gpu_name,memory.total,memory.free,memory.used", "--format=csv"]
        proc = subprocess.run(cmd_gpus, capture_output=True, text=True, timeout=5)

        if proc.returncode == 0:
            # Skip header line
            lines = proc.stdout.strip().split("\n")[1:]

            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpu_info = {
                        "name": parts[0],
                        "memory_total": parts[1],
                        "memory_free": parts[2],
                        "memory_used": parts[3]
                    }
                    result["gpus"].append(gpu_info)

                    # Check if it's a GeForce GTX
                    if "GeForce GTX" in parts[0]:
                        result["has_gtx"] = True
    except Exception as e:
        logger.error(f"Error getting GPU information: {e}")

    # Get full nvidia-smi output
    try:
        proc = subprocess.run([nvidia_smi_path], capture_output=True, text=True, timeout=5)
        if proc.returncode == 0:
            result["nvidia_smi_output"] = proc.stdout
    except Exception:
        pass

    return result

def check_environment_variables() -> Dict[str, str]:
    """Check relevant environment variables for CUDA."""
    relevant_vars = [
        "CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES",
        "PATH", "LD_LIBRARY_PATH", "PYTHONPATH"
    ]

    return {var: os.environ.get(var, "Not set") for var in relevant_vars}

def print_diagnostics(pytorch_info: Dict[str, Any], driver_info: Dict[str, Any], env_vars: Dict[str, str]):
    """Print diagnostic information in a readable format."""
    print("\n" + "=" * 80)
    print(" NVIDIA GPU AND PYTORCH DIAGNOSTICS ".center(80, "="))
    print("=" * 80 + "\n")

    # System information
    print("SYSTEM INFORMATION:")
    print(f"  OS: {platform.system()} {platform.release()} {platform.version()}")
    print(f"  Python: {platform.python_version()}")
    print(f"  Architecture: {platform.machine()}")
    print()

    # PyTorch information
    print("PYTORCH INFORMATION:")
    if pytorch_info["pytorch_installed"]:
        print(f"  PyTorch version: {pytorch_info['version']}")
        print(f"  Built with CUDA: {pytorch_info['cuda_built']}")
        if pytorch_info["cuda_built"]:
            print(f"  CUDA version: {pytorch_info['cuda_version']}")
            print(f"  CUDA available: {pytorch_info['cuda_available']}")
            print(f"  GPU count: {pytorch_info['device_count']}")

            for device in pytorch_info["devices"]:
                print(f"\n  Device {device['index']}: {device['name']}")
                print(f"    Memory: {device['total_memory_gb']:.2f} GB")
                print(f"    CUDA capability: {device['cuda_capability']}")
                print(f"    Multiprocessors: {device['multi_processor_count']}")
    else:
        print("  PyTorch is not installed")
    print()

    # NVIDIA driver information
    print("NVIDIA DRIVER INFORMATION:")
    if driver_info["driver_found"]:
        print(f"  Driver version: {driver_info['driver_version']}")
        print(f"  nvidia-smi path: {driver_info['nvidia_smi_path']}")
        print(f"  GeForce GTX detected: {driver_info['has_gtx']}")

        if driver_info["gpus"]:
            print("\n  GPUs detected by nvidia-smi:")
            for i, gpu in enumerate(driver_info["gpus"]):
                print(f"    GPU {i}: {gpu['name']}")
                print(f"      Memory total: {gpu['memory_total']}")
                print(f"      Memory free: {gpu['memory_free']}")
                print(f"      Memory used: {gpu['memory_used']}")
    else:
        print("  NVIDIA drivers not found or not functioning properly")
        if driver_info["nvidia_smi_path"]:
            print(f"  nvidia-smi found at: {driver_info['nvidia_smi_path']} but failed to run")
        else:
            print("  nvidia-smi not found in PATH or common locations")
    print()

    # Environment variables
    print("RELEVANT ENVIRONMENT VARIABLES:")
    for var, value in env_vars.items():
        if var in ["PATH", "LD_LIBRARY_PATH", "PYTHONPATH"]:
            print(f"  {var}: {'Set (too long to display)' if value != 'Not set' else 'Not set'}")
        else:
            print(f"  {var}: {value}")
    print()

    # Full nvidia-smi output
    if driver_info["nvidia_smi_output"]:
        print("NVIDIA-SMI OUTPUT:")
        print("-" * 80)
        print(driver_info["nvidia_smi_output"])
        print("-" * 80)

    # Summary and recommendations
    print("\nDIAGNOSIS SUMMARY:")
    if pytorch_info["pytorch_installed"] and pytorch_info["cuda_available"] and driver_info["driver_found"]:
        print("[SUCCESS] PyTorch with CUDA support is properly installed and can access your NVIDIA GPU.")
    elif not pytorch_info["pytorch_installed"]:
        print("[ERROR] PyTorch is not installed. Please install PyTorch with CUDA support.")
    elif not pytorch_info["cuda_built"]:
        print("[ERROR] PyTorch is installed but was not built with CUDA support.")
        print("        Please reinstall PyTorch with CUDA support from https://pytorch.org/get-started/locally/")
    elif not pytorch_info["cuda_available"]:
        if driver_info["driver_found"]:
            print("[ERROR] PyTorch cannot access your NVIDIA GPU despite drivers being installed.")
            print("        This could be due to:")
            print("        - PyTorch CUDA version mismatch with driver")
            print("        - Environment variables not properly set")
            print("        - Hardware issues")
        else:
            print("[ERROR] NVIDIA drivers are not installed or not functioning properly.")
            print("        Please install the latest NVIDIA drivers for your GPU.")

    print("\n" + "=" * 80 + "\n")

def main():
    """Main function to run diagnostics."""
    logger.info("Starting NVIDIA GPU diagnostics")

    # Check PyTorch installation
    pytorch_info = check_pytorch_installation()

    # Check NVIDIA driver
    driver_info = check_nvidia_driver()

    # Check environment variables
    env_vars = check_environment_variables()

    # Print diagnostics
    print_diagnostics(pytorch_info, driver_info, env_vars)

    # Save diagnostics to file
    with open("gpu_diagnostics_report.txt", "w") as f:
        # Redirect stdout to file
        old_stdout = sys.stdout
        sys.stdout = f
        print_diagnostics(pytorch_info, driver_info, env_vars)
        sys.stdout = old_stdout

    logger.info("Diagnostics complete. Report saved to gpu_diagnostics_report.txt")

    # Return success status
    return pytorch_info["cuda_available"] and driver_info["driver_found"]

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
