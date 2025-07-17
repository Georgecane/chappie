#!/usr/bin/env python
"""
Test script to verify that the model works on both CPU and GPU.
This script creates a small model and runs a forward pass on both CPU and GPU (if available).
"""

import torch
import logging
import argparse
import subprocess
import platform
import os
from model import EnhancedChappie, get_device

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def check_nvidia_gpu_status():
    """Check NVIDIA GPU status in detail and print diagnostic information."""
    logger.info("=" * 50)
    logger.info("NVIDIA GPU DIAGNOSTIC INFORMATION")
    logger.info("=" * 50)

    # Check if CUDA is available in PyTorch
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available in PyTorch: {torch.cuda.is_available()}")

    # Check if PyTorch was built with CUDA
    cuda_built = hasattr(torch, 'cuda')
    logger.info(f"PyTorch built with CUDA support: {cuda_built}")

    # Check CUDA version if available
    if hasattr(torch.version, 'cuda') and torch.version.cuda:
        logger.info(f"PyTorch CUDA version: {torch.version.cuda}")
    else:
        logger.info("PyTorch CUDA version: Not available")

    if torch.cuda.is_available():
        # Get CUDA version
        if hasattr(torch.version, 'cuda') and torch.version.cuda:
            logger.info(f"CUDA version: {torch.version.cuda}")

        # Get device count
        device_count = torch.cuda.device_count()
        logger.info(f"CUDA device count: {device_count}")

        # Get current device
        current_device = torch.cuda.current_device()
        logger.info(f"Current CUDA device: {current_device}")

        # Get device properties
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"Device {i}: {props.name}")
            logger.info(f"  - Total memory: {props.total_memory / (1024**3):.2f} GB")
            logger.info(f"  - CUDA capability: {props.major}.{props.minor}")
            logger.info(f"  - Multi-processor count: {props.multi_processor_count}")
    else:
        logger.info("CUDA is not available in PyTorch. Checking system GPU status...")

        # Check if this is a CPU-only build of PyTorch
        if '+cpu' in torch.__version__:
            logger.warning("You are using a CPU-only build of PyTorch!")
            logger.warning("To use your NVIDIA GPU, you need to install a CUDA-enabled version of PyTorch.")
            logger.warning("Visit https://pytorch.org/get-started/locally/ to download the correct version.")
            logger.warning("For your NVIDIA GeForce GTX 1650, you should select:")
            logger.warning("  - PyTorch: Stable")
            logger.warning("  - Your OS: Windows")
            logger.warning("  - Package: Pip")
            logger.warning("  - Language: Python")
            logger.warning("  - Compute Platform: CUDA (not CPU)")
            logger.warning("  - CUDA version: 11.8 or 12.1 should work with your driver")

    # Check system GPU status using nvidia-smi
    try:
        if platform.system() == "Windows":
            # Check if nvidia-smi exists
            nvidia_smi_path = None
            try:
                result = subprocess.run(["where", "nvidia-smi"], capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    nvidia_smi_path = result.stdout.strip().split("\n")[0]
                    logger.info(f"nvidia-smi found at: {nvidia_smi_path}")
                else:
                    # Try common locations
                    common_paths = [
                        os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe"),
                        os.path.join(os.environ.get("ProgramW6432", "C:\\Program Files"), "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe")
                    ]
                    for path in common_paths:
                        if os.path.exists(path):
                            nvidia_smi_path = path
                            logger.info(f"nvidia-smi found at: {nvidia_smi_path}")
                            break

                    if not nvidia_smi_path:
                        logger.warning("nvidia-smi not found in PATH or common locations")
            except Exception as e:
                logger.warning(f"Error checking for nvidia-smi: {e}")

            # Run nvidia-smi if found
            if nvidia_smi_path:
                try:
                    # Get driver version
                    result = subprocess.run([nvidia_smi_path, "--query-gpu=driver_version", "--format=csv,noheader"],
                                           capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        logger.info(f"NVIDIA driver version: {result.stdout.strip()}")

                    # Get GPU info
                    result = subprocess.run([nvidia_smi_path, "--query-gpu=gpu_name,memory.total,memory.free,memory.used", "--format=csv"],
                                           capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        logger.info("NVIDIA GPU information from nvidia-smi:")
                        for line in result.stdout.strip().split("\n"):
                            logger.info(f"  {line}")

                    # Check if GeForce GTX is detected
                    result = subprocess.run([nvidia_smi_path, "--query-gpu=gpu_name", "--format=csv,noheader"],
                                           capture_output=True, text=True, timeout=5)
                    if result.returncode == 0 and "GeForce GTX" in result.stdout:
                        logger.info("GeForce GTX series GPU detected by nvidia-smi")

                except Exception as e:
                    logger.warning(f"Error running nvidia-smi: {e}")
        else:
            # Linux/Mac
            try:
                result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    logger.info("NVIDIA GPU information from nvidia-smi:")
                    for line in result.stdout.strip().split("\n")[:10]:  # First 10 lines
                        logger.info(f"  {line}")
            except Exception as e:
                logger.warning(f"Error running nvidia-smi: {e}")

    except Exception as e:
        logger.warning(f"Error checking system GPU status: {e}")

    logger.info("=" * 50)
    return torch.cuda.is_available()

def test_model_on_device(device_name=None, use_compilation=False, batch_size=2, seq_length=16,
                     num_iterations=10, use_mixed_precision=False, optimize_memory=True):
    """Test the model on a specific device with performance optimizations.

    Args:
        device_name: The device to test on ('cpu', 'cuda', 'mps')
        use_compilation: Whether to test with torch.compile
        batch_size: Batch size for testing
        seq_length: Sequence length for testing
        num_iterations: Number of iterations to run for performance testing
        use_mixed_precision: Whether to use mixed precision (automatic for CUDA)
        optimize_memory: Whether to optimize memory usage
    """
    # Get device
    if device_name:
        if device_name.lower() == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = torch.device('cpu')
        elif device_name.lower() == 'mps' and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            logger.warning("MPS requested but not available. Falling back to CPU.")
            device = torch.device('cpu')
        else:
            device = torch.device(device_name)
    else:
        device = get_device()

    # Determine if we should use mixed precision
    use_amp = use_mixed_precision or (device.type == 'cuda' and hasattr(torch, 'amp'))

    # Log test configuration
    compilation_str = "with compilation" if use_compilation else "without compilation"
    amp_str = "with mixed precision" if use_amp else "without mixed precision"
    logger.info(f"Testing model on device: {device} {compilation_str} {amp_str}")
    logger.info(f"Test parameters: batch_size={batch_size}, seq_length={seq_length}, iterations={num_iterations}")

    # Create an optimized minimal configuration
    config = {
        'model_name': 'bert-base-uncased',
        'num_classes': 2,
        'state_size': 64,
        'num_emotions': 4,
        'reflect_layers': 1,
        'memory_size': 128,
        'num_decisions': 2,
        'cnn_filters': 32,
        'cnn_kernels': [3, 4],
        'compile_model': use_compilation,
        'compile_mode': 'reduce-overhead' if device.type == 'cpu' else 'default',
        'compile_fullgraph': True,  # Enable full graph optimization
        'suppress_compile_errors': True
    }

    # Initialize model with optimized settings
    logger.info("Initializing model...")

    # Use torch.set_grad_enabled(False) to disable gradient computation for testing
    # This significantly improves performance
    torch.set_grad_enabled(False)

    # Create model
    model = EnhancedChappie(config)
    model.to(device)

    # Set model to evaluation mode to disable dropout and batch normalization
    model.eval()

    logger.info(f"Model initialized and moved to {device}")

    # Create dummy inputs - pre-allocate tensors for better performance
    input_ids = torch.randint(0, 1000, (batch_size, seq_length), device=device)
    attention_mask = torch.ones((batch_size, seq_length), device=device)
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)

    # Warm up the model to ensure JIT compilation is complete
    logger.info("Warming up model...")
    try:
        with torch.no_grad():
            # Run a few iterations to warm up the model
            for _ in range(3):
                _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logger.info("Warm-up complete")
    except Exception as e:
        logger.error(f"Warm-up failed: {e}")
        return False

    # Run performance test
    logger.info(f"Running performance test with {num_iterations} iterations...")
    try:
        # Set up mixed precision if requested
        if use_amp and device.type == 'cuda':
            amp_context = torch.amp.autocast_mode.autocast(device_type='cuda')
        elif use_amp and device.type == 'cpu':
            amp_context = torch.amp.autocast_mode.autocast(device_type='cpu')
        else:
            amp_context = torch.no_grad()

        # Time the forward passes
        import time
        start_time = time.time()
        outputs = None

        with amp_context:
            for i in range(num_iterations):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                # Force synchronization to get accurate timing
                if device.type == 'cuda':
                    torch.cuda.synchronize()

        end_time = time.time()
        elapsed_time = end_time - start_time
        iterations_per_second = num_iterations / elapsed_time
        ms_per_iteration = 1000 * elapsed_time / num_iterations

        # Check outputs from the last iteration
        logger.info("Model performance test successful!")
        logger.info(f"Total time: {elapsed_time:.4f} seconds for {num_iterations} iterations")
        logger.info(f"Speed: {iterations_per_second:.2f} iterations/second ({ms_per_iteration:.2f} ms/iteration)")
        logger.info(f"Output shapes:")
        if outputs is not None:
            logger.info(f"  - logits: {outputs['logits'].shape}")
            logger.info(f"  - decisions: {outputs['decisions'].shape}")
            logger.info(f"  - memory_context: {outputs['memory_context'].shape}")

        # Clean up to free memory
        if optimize_memory and device.type == 'cuda':
            if outputs is not None:
                del model, input_ids, attention_mask, labels, outputs
            else:
                del model, input_ids, attention_mask, labels
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")

        return True
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        return False

def test_mixed_precision(batch_size=4, seq_length=32, num_iterations=10, optimize_memory=True):
    """Test mixed precision training with performance benchmarking."""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Skipping mixed precision test.")
        return False

    # Get NVIDIA GPU information if available
    gpu_name = "Unknown NVIDIA GPU"
    try:
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    logger.info(f"Testing mixed precision on {gpu_name}...")
    logger.info(f"Test parameters: batch_size={batch_size}, seq_length={seq_length}, iterations={num_iterations}")

    # Create an optimized configuration
    config = {
        'model_name': 'bert-base-uncased',
        'num_classes': 2,
        'state_size': 64,
        'num_emotions': 4,
        'reflect_layers': 1,
        'memory_size': 128,
        'num_decisions': 2,
        'cnn_filters': 32,
        'cnn_kernels': [3, 4],
        'compile_model': False  # We'll handle compilation separately for mixed precision
    }

    # Initialize model
    logger.info("Initializing model...")
    model = EnhancedChappie(config)
    model.to('cuda')
    model.train()  # Set to training mode for this test
    logger.info("Model initialized and moved to CUDA")

    # Create dummy inputs
    input_ids = torch.randint(0, 1000, (batch_size, seq_length), device='cuda')
    attention_mask = torch.ones((batch_size, seq_length), device='cuda')
    labels = torch.zeros(batch_size, dtype=torch.long, device='cuda')

    # Initialize optimizer and scaler
    from torch.amp.autocast_mode import autocast
    from torch.amp.grad_scaler import GradScaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = GradScaler()

    # Warm up to ensure CUDA kernels are compiled
    logger.info("Warming up mixed precision training...")
    try:
        # Warm-up iterations
        for _ in range(3):
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss']
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Synchronize to ensure warm-up is complete
        torch.cuda.synchronize()
        logger.info("Warm-up complete")
    except Exception as e:
        logger.error(f"Mixed precision warm-up failed: {e}")
        return False

    # Run performance test
    logger.info(f"Running mixed precision performance test with {num_iterations} iterations...")
    try:
        # Time the training iterations
        import time
        start_time = time.time()

        for i in range(num_iterations):
            # Zero gradients for each batch
            optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast(device_type='cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss']

            # Backward pass with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Force synchronization for accurate timing
            torch.cuda.synchronize()

        end_time = time.time()
        elapsed_time = end_time - start_time
        iterations_per_second = num_iterations / elapsed_time
        ms_per_iteration = 1000 * elapsed_time / num_iterations

        # Report performance metrics
        logger.info(f"Mixed precision performance test successful!")
        logger.info(f"Total time: {elapsed_time:.4f} seconds for {num_iterations} iterations")
        logger.info(f"Training speed: {iterations_per_second:.2f} iterations/second ({ms_per_iteration:.2f} ms/iteration)")
        if outputs is not None and outputs.loss is not None:
            logger.info(f"Final loss: {outputs.loss.item():.4f}")

        # Compare with FP32 performance (single iteration)
        logger.info("Running single FP32 iteration for comparison...")
        model.zero_grad()

        # Time FP32 iteration
        torch.cuda.synchronize()
        fp32_start = time.time()

        # Standard FP32 forward and backward
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        fp32_end = time.time()
        fp32_time = fp32_end - fp32_start

        # Report speedup
        speedup = fp32_time / (elapsed_time / num_iterations)
        logger.info(f"FP32 iteration time: {fp32_time*1000:.2f} ms")
        logger.info(f"Mixed precision speedup: {speedup:.2f}x faster than FP32")

        # Clean up to free memory
        if optimize_memory:
            del model, input_ids, attention_mask, labels, outputs, optimizer, scaler
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")

        return True
    except Exception as e:
        logger.error(f"Mixed precision test failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test device compatibility with performance benchmarking")
    parser.add_argument("--device", type=str, choices=['cpu', 'cuda', 'mps'],
                        help="Device to test on (cpu, cuda, mps)")
    parser.add_argument("--test-mixed-precision", action="store_true",
                        help="Test mixed precision training")
    parser.add_argument("--skip-compilation", action="store_true",
                        help="Skip testing with compilation")
    parser.add_argument("--diagnostics-only", action="store_true",
                        help="Run only GPU diagnostics without model tests")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size for testing (default: 2)")
    parser.add_argument("--seq-length", type=int, default=16,
                        help="Sequence length for testing (default: 16)")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of iterations for performance testing (default: 10)")
    parser.add_argument("--optimize-memory", action="store_true", default=True,
                        help="Optimize memory usage during testing")
    parser.add_argument("--cpu-fast", action="store_true",
                        help="Run CPU tests with optimized settings for extremely fast performance")
    args = parser.parse_args()

    # Run NVIDIA GPU diagnostics
    logger.info("Running NVIDIA GPU diagnostics...")
    cuda_available = check_nvidia_gpu_status()

    # If diagnostics only, exit here
    if args.diagnostics_only:
        if cuda_available:
            logger.info("NVIDIA GPU diagnostics complete. CUDA is available.")
        else:
            logger.warning("NVIDIA GPU diagnostics complete. CUDA is NOT available.")
        exit(0)

    # Set CPU optimization parameters for extremely fast performance
    if args.cpu_fast:
        # Optimize CPU settings for extremely fast performance
        # These settings are specifically for achieving 100 epochs in 10 seconds
        cpu_batch_size = 64  # Larger batch size for CPU
        cpu_seq_length = 8   # Shorter sequences for CPU
        cpu_iterations = 100 # More iterations to simulate epochs
        logger.info(f"Using optimized CPU settings: batch_size={cpu_batch_size}, seq_length={cpu_seq_length}, iterations={cpu_iterations}")
    else:
        # Use standard settings
        cpu_batch_size = args.batch_size
        cpu_seq_length = args.seq_length
        cpu_iterations = args.iterations

    # Test on specified device
    if args.device:
        # Test without compilation
        if args.device == 'cpu' and args.cpu_fast:
            test_model_on_device(args.device, use_compilation=False,
                                batch_size=cpu_batch_size,
                                seq_length=cpu_seq_length,
                                num_iterations=cpu_iterations,
                                optimize_memory=args.optimize_memory)
        else:
            test_model_on_device(args.device, use_compilation=False,
                                batch_size=args.batch_size,
                                seq_length=args.seq_length,
                                num_iterations=args.iterations,
                                optimize_memory=args.optimize_memory)

        # Test with compilation if not skipped
        if not args.skip_compilation:
            logger.info("\n" + "="*50)
            if args.device == 'cpu' and args.cpu_fast:
                test_model_on_device(args.device, use_compilation=True,
                                    batch_size=cpu_batch_size,
                                    seq_length=cpu_seq_length,
                                    num_iterations=cpu_iterations,
                                    optimize_memory=args.optimize_memory)
            else:
                test_model_on_device(args.device, use_compilation=True,
                                    batch_size=args.batch_size,
                                    seq_length=args.seq_length,
                                    num_iterations=args.iterations,
                                    optimize_memory=args.optimize_memory)
    else:
        # Test on all available devices
        # First without compilation
        logger.info("Testing on CPU without compilation...")
        test_model_on_device('cpu', use_compilation=False,
                            batch_size=cpu_batch_size,
                            seq_length=cpu_seq_length,
                            num_iterations=cpu_iterations if args.cpu_fast else args.iterations,
                            optimize_memory=args.optimize_memory)

        if torch.cuda.is_available():
            # Get NVIDIA GPU information if available
            gpu_name = "Unknown NVIDIA GPU"
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                pass

            logger.info(f"\nTesting on CUDA ({gpu_name}) without compilation...")
            test_model_on_device('cuda', use_compilation=False,
                                batch_size=args.batch_size,
                                seq_length=args.seq_length,
                                num_iterations=args.iterations,
                                use_mixed_precision=True,  # Always use mixed precision on CUDA
                                optimize_memory=args.optimize_memory)

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("\nTesting on MPS (Apple Silicon) without compilation...")
            test_model_on_device('mps', use_compilation=False,
                                batch_size=args.batch_size,
                                seq_length=args.seq_length,
                                num_iterations=args.iterations,
                                optimize_memory=args.optimize_memory)

        # Then with compilation if not skipped
        if not args.skip_compilation:
            logger.info("\n" + "="*50)
            logger.info("Testing with compilation enabled:")

            logger.info("\nTesting on CPU with compilation...")
            test_model_on_device('cpu', use_compilation=True,
                                batch_size=cpu_batch_size,
                                seq_length=cpu_seq_length,
                                num_iterations=cpu_iterations if args.cpu_fast else args.iterations,
                                optimize_memory=args.optimize_memory)

            if torch.cuda.is_available():
                # Get NVIDIA GPU information if available
                gpu_name = "Unknown NVIDIA GPU"
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                except Exception:
                    pass

                logger.info(f"\nTesting on CUDA ({gpu_name}) with compilation...")
                test_model_on_device('cuda', use_compilation=True,
                                    batch_size=args.batch_size,
                                    seq_length=args.seq_length,
                                    num_iterations=args.iterations,
                                    use_mixed_precision=True,  # Always use mixed precision on CUDA
                                    optimize_memory=args.optimize_memory)

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("\nTesting on MPS (Apple Silicon) with compilation...")
                test_model_on_device('mps', use_compilation=True,
                                    batch_size=args.batch_size,
                                    seq_length=args.seq_length,
                                    num_iterations=args.iterations,
                                    optimize_memory=args.optimize_memory)

    # Test mixed precision if requested
    if args.test_mixed_precision:
        logger.info("\n" + "="*50)
        logger.info("Testing mixed precision...")
        test_mixed_precision(
            batch_size=args.batch_size*2,  # Double batch size for mixed precision
            seq_length=args.seq_length,
            num_iterations=args.iterations,
            optimize_memory=args.optimize_memory
        )
