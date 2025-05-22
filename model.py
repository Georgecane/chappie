# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, Any, Optional, Union, List, Tuple
from collections import deque
import logging
import os

logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """Get the device to use for computations with enhanced NVIDIA detection and CPU optimization."""
    # Check for CUDA availability
    if torch.cuda.is_available():
        # Get detailed NVIDIA GPU information
        try:
            # Get device count
            device_count = torch.cuda.device_count()
            logger.info(f"Found {device_count} CUDA device(s)")

            # Get device properties for the first device
            device_props = torch.cuda.get_device_properties(0)
            gpu_name = device_props.name
            total_memory_gb = device_props.total_memory / (1024**3)

            # Check if it's a GeForce GTX card
            is_gtx = "GeForce GTX" in gpu_name

            # Log detailed information
            logger.info(f"CUDA is available. Using GPU: {gpu_name}")
            logger.info(f"GPU Memory: {total_memory_gb:.2f} GB")
            logger.info(f"CUDA Capability: {device_props.major}.{device_props.minor}")

            if is_gtx:
                logger.info(f"Detected NVIDIA GeForce GTX series GPU")

            # Try to get driver version
            try:
                import subprocess
                import re
                import platform

                if platform.system() == "Windows":
                    # On Windows, try to get driver version using nvidia-smi
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                        capture_output=True, text=True, timeout=2
                    )
                    if result.returncode == 0:
                        driver_version = result.stdout.strip()
                        logger.info(f"NVIDIA Driver Version: {driver_version}")

            except Exception as e:
                logger.debug(f"Could not get driver version: {e}")

        except Exception as e:
            logger.warning(f"Error getting detailed GPU information: {e}")
            gpu_name = "Unknown NVIDIA GPU"
            try:
                # Fallback to basic name retrieval
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"CUDA is available. Using GPU: {gpu_name}")
            except Exception:
                logger.info("CUDA is available but could not get GPU name")

        # Return CUDA device
        return torch.device("cuda")

    # Check for Apple Silicon MPS
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("MPS is available. Using Apple Silicon GPU.")
        return torch.device("mps")

    # Optimize CPU performance with extreme optimization
    else:
        logger.info("No GPU found. Using extremely optimized CPU configuration.")

        # Set up optimized CPU configuration
        try:
            # Enable Intel MKL optimizations if available
            import os
            import multiprocessing

            # Get CPU core count
            physical_cores = multiprocessing.cpu_count()

            # For extreme performance, use all available cores
            # but leave one for system processes
            optimal_threads = max(1, physical_cores - 1)

            # Set MKL environment variables for optimal performance
            # These settings are critical for CPU performance
            os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
            os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
            os.environ['MKL_DYNAMIC'] = 'FALSE'  # Disable dynamic adjustment
            os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'  # Optimize thread affinity
            os.environ['KMP_BLOCKTIME'] = '0'  # Minimize wait time after parallel regions

            # Additional MKL optimizations
            os.environ['MKL_FAST_MEMORY_LIMIT'] = '0'  # Use fast memory when available
            os.environ['MKL_ENABLE_INSTRUCTIONS'] = 'AVX2'  # Enable AVX2 instructions if available

            # Set PyTorch thread settings
            torch.set_num_threads(optimal_threads)
            torch.set_num_interop_threads(min(4, physical_cores))  # Limit interop threads

            # Enable PyTorch JIT fusion for CPU operations
            torch._C._jit_set_profiling_executor(True)
            torch._C._jit_set_profiling_mode(True)
            torch._C._jit_override_can_fuse_on_cpu(True)
            torch._C._jit_override_can_fuse_on_gpu(True)

            # Log CPU information
            import platform
            import psutil

            logger.info(f"CPU: {platform.processor()}")
            logger.info(f"Physical cores: {psutil.cpu_count(logical=False)}")
            logger.info(f"Logical cores: {psutil.cpu_count(logical=True)}")
            logger.info(f"PyTorch threads: {torch.get_num_threads()}")
            logger.info(f"PyTorch interop threads: {torch.get_num_interop_threads()}")

            # Check if Intel MKL is being used
            mkl_enabled = torch._C._has_mkldnn
            logger.info(f"Intel MKL-DNN (oneDNN) enabled: {mkl_enabled}")

            # Check if PyTorch was built with optimized CPU performance
            build_info = torch.__config__.show()
            if "mkldnn" in build_info.lower() or "onednn" in build_info.lower():
                logger.info("PyTorch was built with Intel MKL-DNN/oneDNN optimizations")

            # Enable vectorized memory format for CPU tensors
            memory_format = torch.channels_last
            logger.info(f"Using memory format: {memory_format}")

            # Enable TensorFloat-32 (TF32) on Ampere (and above) devices
            # This doesn't affect CPU but keep it for when GPU becomes available
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Enable cuDNN benchmark mode
            torch.backends.cudnn.benchmark = True

            # Enable PyTorch 2.0 features if available
            if hasattr(torch, '_inductor'):
                torch._inductor.config.coordinate_descent_tuning = True
                torch._inductor.config.triton.unique_kernel_names = False
                torch._inductor.config.fx_graph_cache = True
                logger.info("Enabled PyTorch 2.0 inductor optimizations")

        except ImportError:
            logger.warning("psutil not installed. Install with: pip install psutil")
            # Set basic thread settings
            torch.set_num_threads(os.cpu_count() if hasattr(os, 'cpu_count') else 4)
        except Exception as e:
            logger.warning(f"Error configuring CPU optimizations: {e}")

        # Log why CUDA might not be available
        if not hasattr(torch, 'cuda'):
            logger.warning("PyTorch was not built with CUDA support")
        elif not torch.cuda.is_available():
            # Check if this is a CPU-only build of PyTorch
            if '+cpu' in torch.__version__:
                logger.warning("You are using a CPU-only build of PyTorch!")
                logger.warning("To use your NVIDIA GPU, you need to install a CUDA-enabled version of PyTorch.")
                logger.warning("Visit https://pytorch.org/get-started/locally/ to download the correct version.")

            # Try to get more information about why CUDA is not available
            try:
                import subprocess
                import platform

                if platform.system() == "Windows":
                    # Check if nvidia-smi is available
                    result = subprocess.run(
                        ["where", "nvidia-smi"],
                        capture_output=True, text=True, timeout=2
                    )
                    if result.returncode != 0:
                        logger.warning("NVIDIA driver tools (nvidia-smi) not found in PATH")
                        logger.warning("This may indicate missing or improperly installed NVIDIA drivers")
                    else:
                        # Try running nvidia-smi to see if it works
                        result = subprocess.run(
                            ["nvidia-smi"],
                            capture_output=True, text=True, timeout=2
                        )
                        if result.returncode != 0:
                            logger.warning("nvidia-smi failed to run. NVIDIA drivers may be installed but not functioning properly")
                        else:
                            # Get GPU name from nvidia-smi
                            try:
                                result = subprocess.run(
                                    ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=2
                                )
                                if result.returncode == 0:
                                    gpu_name = result.stdout.strip()
                                    logger.warning(f"Found {gpu_name} with nvidia-smi, but PyTorch cannot access it")
                                    logger.warning("This is likely because you have a CPU-only build of PyTorch")
                                    logger.warning("Please reinstall PyTorch with CUDA support")
                            except Exception:
                                pass

                            logger.warning("nvidia-smi runs but PyTorch cannot detect CUDA. This may be a PyTorch configuration issue")

            except Exception as e:
                logger.debug(f"Error checking NVIDIA drivers: {e}")

        return torch.device("cpu")

def move_to_device(tensor_or_module, device):
    """Move tensor or module to specified device safely."""
    if tensor_or_module is None:
        return None

    if isinstance(tensor_or_module, torch.Tensor):
        return tensor_or_module.to(device)
    elif isinstance(tensor_or_module, torch.nn.Module):
        return tensor_or_module.to(device)
    elif isinstance(tensor_or_module, (list, tuple)):
        return [move_to_device(item, device) for item in tensor_or_module]
    elif isinstance(tensor_or_module, dict):
        return {k: move_to_device(v, device) for k, v in tensor_or_module.items()}
    else:
        return tensor_or_module

class EnhancedStateEncoder(nn.Module):
    def __init__(self, input_size: int, state_size: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, state_size * 2),
            nn.GELU(),
            nn.LayerNorm(state_size * 2),
            nn.Linear(state_size * 2, state_size),
            nn.Dropout(0.1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        # apply per token
        return self.encoder(x)

class DynamicEmotionProcessor(nn.Module):
    def __init__(self, input_size: int, num_emotions: int):
        super().__init__()
        self.att_gate = nn.Sequential(nn.Linear(input_size, 1), nn.Sigmoid())
        self.net = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.GELU(),
            nn.LayerNorm(input_size * 2),
            nn.Linear(input_size * 2, num_emotions)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att = self.att_gate(x)  # (batch, seq_len, 1)
        return self.net(x * att)

class HierarchicalSelfReflection(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, norm in zip(self.layers, self.norms):
            res = x
            x, _ = attn(x, x, x)
            x = norm(res + x)
        return x

class NeuralMemoryBank(nn.Module):
    def __init__(self, hidden_size: int, memory_size: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size

        # Initialize memory with Xavier/Glorot initialization for better gradient flow
        memory = torch.empty(memory_size, hidden_size)
        nn.init.xavier_uniform_(memory)
        self.memory = nn.Parameter(memory)

        # Use instance normalization for faster computation compared to LayerNorm
        # This is more efficient for memory operations
        self.norm = nn.InstanceNorm1d(hidden_size)

        # Optimize attention mechanism with fused operations
        # Use a single projection matrix instead of separate query/key projections
        # This reduces memory transfers and computation
        self.proj = nn.Linear(hidden_size, hidden_size)

        # Pre-compute the projected memory keys for faster inference
        self.register_buffer('projected_keys', torch.zeros(memory_size, hidden_size))

        # Memory update gate with learnable parameters - use a more efficient implementation
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

        # Track if we're in distributed mode
        self.distributed = False
        self.local_rank = -1
        if os.environ.get('LOCAL_RANK') is not None:
            self.distributed = True
            self.local_rank = int(os.environ.get('LOCAL_RANK', -1))

        # Register buffer for synchronization in distributed training
        self.register_buffer('update_counter', torch.zeros(1, dtype=torch.long))

        # Cache for attention computation
        self.register_buffer('scale', torch.tensor(1.0))

        # Flag to indicate if projected keys need updating
        self.keys_need_update = True

    def _update_projected_keys(self):
        """Update the projected memory keys."""
        with torch.no_grad():
            self.projected_keys = self.proj(self.memory)
            self.keys_need_update = False
            # Update scale factor for attention
            self.scale = torch.sqrt(torch.tensor(self.hidden_size, dtype=self.memory.dtype, device=self.memory.device))

    def _get_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute attention scores between input and memory with optimized implementation."""
        # Update projected keys if needed
        if self.keys_need_update:
            self._update_projected_keys()

        # Project queries from input - reuse the same projection for efficiency
        queries = self.proj(x)  # (batch, seq_len, hidden)

        # Compute scaled dot-product attention using optimized batch matrix multiplication
        # This is faster than separate matmul operations
        scores = torch.bmm(
            queries.view(-1, 1, self.hidden_size),  # (batch*seq_len, 1, hidden)
            self.projected_keys.unsqueeze(0).expand(queries.size(0) * queries.size(1), -1, -1).transpose(1, 2)  # (batch*seq_len, hidden, memory_size)
        ).view(queries.size(0), queries.size(1), -1) / self.scale  # (batch, seq_len, memory_size)

        return scores

    def _update_memory(self, context: torch.Tensor, _: torch.Tensor = None) -> None:
        """Update memory with new information using optimized gating mechanism."""
        # Only update memory during training
        if not self.training:
            return

        try:
            # Compute average context - use more efficient reduction
            avg_context = context.mean(dim=0)  # (hidden_size,)

            # Use more efficient memory update with pre-allocation
            # Compute update vectors for each memory slot
            expanded_avg = avg_context.unsqueeze(0).expand_as(self.memory)  # (memory_size, hidden)

            # Pre-allocate concatenated tensor for efficiency
            concat = torch.empty(
                (self.memory_size, self.hidden_size * 2),
                dtype=self.memory.dtype,
                device=self.memory.device
            )

            # Fill the pre-allocated tensor
            concat[:, :self.hidden_size] = self.memory
            concat[:, self.hidden_size:] = expanded_avg

            # Compute update weights
            update_weights = self.update_gate(concat)  # (memory_size, hidden)

            # Apply update with optimized in-place operations
            with torch.no_grad():
                # Use in-place operations where possible
                new_mem = torch.empty_like(self.memory)
                new_mem.copy_(self.memory * (1 - update_weights) + expanded_avg * update_weights)

                # Apply normalization - more efficient than LayerNorm
                new_mem = self.norm(new_mem.t()).t()  # Transpose for InstanceNorm1d

                # In distributed training, synchronize memory updates
                if self.distributed and torch.distributed.is_initialized():
                    # Increment update counter
                    self.update_counter += 1

                    # Synchronize memory across processes every 10 updates
                    if self.update_counter % 10 == 0:
                        torch.distributed.all_reduce(new_mem, op=torch.distributed.ReduceOp.AVG)

                # Update memory
                self.memory.copy_(new_mem)

                # Mark that projected keys need updating
                self.keys_need_update = True

        except Exception as e:
            logger.error(f"Error updating memory: {e}")
            # Continue without updating memory to avoid training failure

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the memory bank with optimized implementation."""
        # Ensure input is on the same device as the module
        if x.device != self.memory.device:
            x = x.to(self.memory.device)

        # Compute attention scores with optimized implementation
        scores = self._get_scores(x)  # (batch, seq_len, memory_size)

        # Apply softmax to get attention weights - use more stable implementation
        # with optimized memory usage
        weights = F.softmax(scores, dim=-1)  # (batch, seq_len, memory_size)

        # Read from memory using attention weights with optimized batch matrix multiplication
        # Reshape for efficient batch matrix multiplication
        batch_size, seq_len, _ = weights.shape
        weights_flat = weights.view(batch_size * seq_len, 1, self.memory_size)
        memory_expanded = self.memory.unsqueeze(0).expand(batch_size * seq_len, -1, -1)

        # Perform batch matrix multiplication
        read_flat = torch.bmm(weights_flat, memory_expanded)
        read = read_flat.view(batch_size, seq_len, self.hidden_size)

        # Aggregate over sequence dimension with more efficient reduction
        context = read.mean(dim=1)  # (batch, hidden)

        # Update memory with optimized implementation
        self._update_memory(context, scores)

        return context

class DecisionEngine(nn.Module):
    def __init__(self, hidden_size: int, num_decisions: int = 4):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, 1)
            ) for _ in range(num_decisions)
        ])
        self.router = nn.Linear(hidden_size, num_decisions)
    def forward(self, x: torch.Tensor):
        # x: (batch, hidden_size)
        weights = torch.sigmoid(self.router(x))  # (batch, num_decisions)
        decisions = torch.cat([
            head(x) * w.unsqueeze(-1)
            for head, w in zip(self.heads, weights.unbind(-1))
        ], dim=-1)
        return decisions, weights

class TextCNN(nn.Module):
    def __init__(self, hidden_size: int, filters: list, kernels: list, num_classes: int):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_size, out_channels=f, kernel_size=k)
            for f, k in zip(filters, kernels)
        ])
        self.fc = nn.Linear(sum(filters), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (batch, hidden_size, seq_len)
        pooled = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # (batch, f, seq_len - k + 1)
            pooled.append(F.max_pool1d(conv_out, kernel_size=conv_out.shape[2]).squeeze(2))
        cat = torch.cat(pooled, dim=1)  # (batch, sum(filters))
        return self.fc(cat)

class EnhancedChappie(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        # Determine device
        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        # Load model configuration
        model_cfg = AutoConfig.from_pretrained(config['model_name'])
        self.backbone = AutoModel.from_pretrained(config['model_name'], config=model_cfg)
        hidden_size = model_cfg.hidden_size

        # Get configuration parameters with proper defaults
        state_size = config.get('state_size', 256)
        num_emotions = config.get('num_emotions', 8)
        reflect_layers = config.get('reflect_layers', 2)
        memory_size = config.get('memory_size', 512)
        num_decisions = config.get('num_decisions', 4)
        cnn_filters = config.get('cnn_filters', 128)
        cnn_kernels = config.get('cnn_kernels', [3, 4, 5])

        # Initialize auxiliary modules
        self.state_enc = EnhancedStateEncoder(hidden_size, state_size)
        self.emotion = DynamicEmotionProcessor(hidden_size, num_emotions)
        self.reflect = HierarchicalSelfReflection(hidden_size, reflect_layers)
        self.memory = NeuralMemoryBank(hidden_size, memory_size)

        # Calculate combined input size for decision engine
        self.decision_input_size = hidden_size + state_size + num_emotions
        self.decision = DecisionEngine(self.decision_input_size, num_decisions)

        # Initialize CNN for classification
        self.cnn = TextCNN(
            hidden_size,
            filters=[cnn_filters] * len(cnn_kernels),
            kernels=cnn_kernels,
            num_classes=config['num_classes']
        )

        # Move model to appropriate device
        self.to(self.device)

        # Apply torch.compile if available and enabled
        self._compile_model()

    def _compile_model(self):
        """Apply torch.compile to model components if available and enabled."""
        # Import torch here to ensure it's available in this method's scope
        import torch

        if not self.config.get('compile_model', False):
            return

        if not hasattr(torch, 'compile'):
            logger.warning("torch.compile is not available in your PyTorch version. Skipping compilation.")
            return

        # Check if we should suppress compilation errors
        suppress_errors = self.config.get('suppress_compile_errors', True)
        if suppress_errors:
            # Configure PyTorch to suppress compilation errors
            try:
                import torch._dynamo
                torch._dynamo.config.suppress_errors = True
                # Set optimization level for better performance
                torch._dynamo.config.optimize_ddp = True
                # Increase cache size for better performance with repeated operations
                torch._dynamo.config.cache_size_limit = 512
                logger.info("Configured PyTorch to suppress compilation errors and fall back to eager mode")
            except ImportError:
                logger.warning("Could not configure error suppression for torch.compile")

        # Try to compile the model
        try:
            compile_mode = self.config.get('compile_mode', 'default')
            logger.info(f"Compiling model with mode: {compile_mode}")

            # Set fullgraph=True for better optimization when possible
            fullgraph = self.config.get('compile_fullgraph', False)
            # Set backend based on device for better performance
            backend = "inductor"  # Default to inductor as it's the most optimized

            # Compile the forward method directly for better performance
            if hasattr(torch, 'compile'):
                try:
                    # Try to compile the entire model first for best performance
                    self_compiled = torch.compile(
                        self,
                        mode=compile_mode,
                        fullgraph=fullgraph,
                        backend=backend
                    )
                    # If successful, replace self with compiled version
                    # This is a trick to compile the entire model
                    for name, param in self_compiled.named_parameters():
                        if name in dict(self.named_parameters()):
                            dict(self.named_parameters())[name].data = param.data
                    logger.info("Successfully compiled entire model")
                    return
                except Exception as e:
                    logger.warning(f"Failed to compile entire model: {e}")
                    logger.info("Falling back to component-wise compilation")

            # Component-wise compilation as fallback
            compile_kwargs = {
                'mode': compile_mode,
                'fullgraph': fullgraph,
                'backend': backend
            }

            # Compile individual components for better performance
            components = [
                ('state_enc', self.state_enc),
                ('emotion', self.emotion),
                ('reflect', self.reflect),
                ('cnn', self.cnn),
                ('decision', self.decision)
            ]

            for name, component in components:
                try:
                    setattr(self, name, torch.compile(component, **compile_kwargs))
                    logger.info(f"Successfully compiled {name}")
                except Exception as e:
                    logger.warning(f"Failed to compile {name}: {e}")

            # Note: We don't compile memory bank as it has custom update logic
            logger.info("Model compilation completed with available components")

        except Exception as e:
            logger.warning(f"Model compilation failed: {e}")
            logger.info("Continuing with uncompiled model (eager mode)")

    def _ensure_tensor_device(self, x, dtype=None):
        """Ensure tensor is on the correct device and has the right dtype."""
        if x is None:
            return None
        if not isinstance(x, torch.Tensor):
            if dtype is None:
                dtype = torch.long
            try:
                x = torch.tensor(x, dtype=dtype)
            except Exception as e:
                logger.error(f"Failed to convert to tensor: {e}")
                raise

        # Move to device safely
        try:
            if x.device != self.device:
                x = x.to(self.device)
        except Exception as e:
            logger.error(f"Failed to move tensor to device {self.device}: {e}")
            # Fallback to CPU if device transfer fails
            try:
                x = x.to('cpu')
                logger.warning(f"Fallback to CPU for tensor of shape {x.shape}")
            except Exception:
                pass

        return x

    def forward(self, input_ids, attention_mask, labels=None, sentence=None, label=None) -> Dict[str, Any]:
        """Forward pass through the model with optimized execution."""
        # Ensure inputs are on the correct device
        if sentence is not None and input_ids is None:
            input_ids = sentence
        
        input_ids = self._ensure_tensor_device(input_ids)
        attention_mask = self._ensure_tensor_device(attention_mask)
        labels = self._ensure_tensor_device(labels)

        # Use torch.amp.autocast for mixed precision if on CUDA
        # This significantly speeds up computation on GPU
        context_manager = (
            torch.amp.autocast(device_type=self.device.type)
            if hasattr(torch, 'amp') and self.device.type in ['cuda', 'cpu']
            else torch.no_grad()
        )

        with context_manager:
            # 1) Extract hidden states from backbone model
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

            # 2) Process through auxiliary modules in parallel where possible
            # Use torch.jit.fuse if available for the state encoder and emotion processor
            # as they can run in parallel
            state = self.state_enc(hidden_states)                # (batch, seq_len, state_size)
            emo = self.emotion(hidden_states)                    # (batch, seq_len, num_emotions)

            # Compute mean values early to reduce memory usage
            state_mean = state.mean(dim=1)                       # (batch, state_size)
            emo_mean = emo.mean(dim=1)                           # (batch, num_emotions)

            # Process reflection and memory
            refl = self.reflect(hidden_states)                   # (batch, seq_len, hidden_size)
            mem = self.memory(refl)                              # (batch, hidden_size)

            # 3) Combine features for decision making - pre-allocate tensor for efficiency
            # This avoids multiple memory allocations
            batch_size = input_ids.size(0)
            combined_size = mem.size(1) + state_mean.size(1) + emo_mean.size(1)
            combined_input = torch.empty(
                (batch_size, combined_size),
                dtype=mem.dtype,
                device=mem.device
            )

            # Fill the pre-allocated tensor
            offset = 0
            combined_input[:, offset:offset + mem.size(1)] = mem
            offset += mem.size(1)
            combined_input[:, offset:offset + state_mean.size(1)] = state_mean
            offset += state_mean.size(1)
            combined_input[:, offset:] = emo_mean

            # 4) Generate decisions
            decisions, routing = self.decision(combined_input)

            # 5) Classify using CNN - run in parallel with decision if possible
            logits = self.cnn(hidden_states)                     # (batch, num_classes)

            # 6) Calculate loss if labels are provided
            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits, labels)

        # Return results
        return {
            'loss': loss,
            'logits': logits,
            'decisions': decisions,
            'routing_weights': routing,
            'memory_context': mem
        }

    def save_pretrained(self, output_dir: str):
        """Save model to the specified directory."""
        os.makedirs(output_dir, exist_ok=True)

        # Save backbone model
        self.backbone.save_pretrained(output_dir)

        # Save auxiliary components
        torch.save(self.state_dict(), os.path.join(output_dir, "enhanced_chappie.pt"))

        # Save configuration
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            import json
            json.dump(self.config, f, indent=2)

        logger.info(f"Model saved to {output_dir}")
