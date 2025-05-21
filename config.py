"""
Configuration module for EnhancedChappie model and training.
"""
import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
import torch

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the EnhancedChappie model."""
    model_name: str = "bert-base-uncased"
    num_classes: int = 2
    state_size: int = 256
    num_emotions: int = 8
    reflect_layers: int = 2
    memory_size: int = 512
    num_decisions: int = 4
    cnn_filters: int = 128
    cnn_kernels: List[int] = field(default_factory=lambda: [3, 4, 5])
    compile_model: bool = False
    compile_mode: str = "default"  # Options: default, reduce-overhead, max-autotune
    suppress_compile_errors: bool = True  # Whether to suppress compilation errors and fall back to eager mode

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}

    def check_compilation_available(self) -> bool:
        """Check if compilation is available on this system with enhanced NVIDIA detection."""
        # First check if torch.compile is available
        if not hasattr(torch, 'compile'):
            logger.warning("torch.compile is not available in your PyTorch version")
            return False

        # Check CUDA availability for NVIDIA GPUs
        if torch.cuda.is_available():
            try:
                # Get CUDA version
                cuda_version = torch.version.cuda
                logger.info(f"CUDA version: {cuda_version}")

                # Get GPU information
                device_count = torch.cuda.device_count()
                logger.info(f"Found {device_count} CUDA device(s)")

                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    gpu_name = props.name
                    logger.info(f"GPU {i}: {gpu_name}")

                    # Check if it's a GeForce GTX
                    if "GeForce GTX" in gpu_name:
                        logger.info(f"Detected NVIDIA GeForce GTX series GPU")

                        # Check CUDA compute capability
                        compute_capability = f"{props.major}.{props.minor}"
                        logger.info(f"CUDA Compute Capability: {compute_capability}")

                        # PyTorch 2.0+ compilation works best with compute capability 7.0+
                        if float(compute_capability) < 7.0:
                            logger.warning(f"Your GPU has compute capability {compute_capability}")
                            logger.warning("torch.compile works best with compute capability 7.0 or higher")
                            logger.warning("Compilation may still work but with reduced performance")
            except Exception as e:
                logger.warning(f"Error checking CUDA device properties: {e}")

        # Check if we're on Windows and have the cl compiler
        import platform
        import subprocess
        import shutil

        if platform.system() == "Windows":
            # Check if cl.exe is in PATH
            cl_path = shutil.which("cl")
            if cl_path:
                logger.info(f"Found Microsoft C/C++ compiler at: {cl_path}")
            else:
                # Try to find Visual Studio compiler
                try:
                    # Try running a simple command to check if cl is available through VS environment
                    result = subprocess.run(
                        ["where", "cl"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        cl_path = result.stdout.strip().split("\n")[0]
                        logger.info(f"Found Microsoft C/C++ compiler at: {cl_path}")
                    else:
                        # Try to check for Visual Studio installation
                        vs_paths = [
                            os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "Microsoft Visual Studio"),
                            os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "Microsoft Visual Studio")
                        ]

                        vs_found = False
                        for vs_path in vs_paths:
                            if os.path.exists(vs_path):
                                logger.info(f"Found Visual Studio installation at: {vs_path}")
                                logger.info("But cl.exe is not in PATH. You may need to run from a Developer Command Prompt")
                                vs_found = True
                                break

                        if not vs_found:
                            logger.warning("Microsoft Visual Studio not found")

                        logger.warning("Microsoft C/C++ compiler (cl.exe) not found in PATH")
                        logger.warning("torch.compile may not work correctly on Windows without it")
                        return False
                except (subprocess.SubprocessError, FileNotFoundError):
                    logger.warning("Could not check for Microsoft C/C++ compiler")
                    logger.warning("torch.compile may not work correctly on Windows")
                    return False

        # On Linux/Mac, check for gcc/clang
        elif platform.system() in ["Linux", "Darwin"]:
            compiler_found = False
            for compiler in ["gcc", "clang"]:
                compiler_path = shutil.which(compiler)
                if compiler_path:
                    compiler_found = True
                    logger.info(f"Found compiler: {compiler} at {compiler_path}")
                    break

            if not compiler_found:
                logger.warning(f"No suitable compiler (gcc/clang) found for {platform.system()}")
                logger.warning("torch.compile may not work correctly without it")
                return False

        # Check for NVIDIA driver
        try:
            import subprocess
            import platform

            if platform.system() == "Windows":
                # Try to run nvidia-smi
                nvidia_smi_path = shutil.which("nvidia-smi")
                if nvidia_smi_path:
                    result = subprocess.run(
                        [nvidia_smi_path, "--query-gpu=driver_version", "--format=csv,noheader"],
                        capture_output=True, text=True, timeout=2
                    )
                    if result.returncode == 0:
                        driver_version = result.stdout.strip()
                        logger.info(f"NVIDIA Driver Version: {driver_version}")
        except Exception as e:
            logger.debug(f"Error checking NVIDIA driver: {e}")

        return True

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.model_name:
            raise ValueError("model_name must be specified")
        if self.num_classes < 1:
            raise ValueError("num_classes must be at least 1")
        if self.state_size < 1:
            raise ValueError("state_size must be at least 1")
        if self.num_emotions < 1:
            raise ValueError("num_emotions must be at least 1")
        if self.reflect_layers < 1:
            raise ValueError("reflect_layers must be at least 1")
        if self.memory_size < 1:
            raise ValueError("memory_size must be at least 1")
        if self.num_decisions < 1:
            raise ValueError("num_decisions must be at least 1")
        if self.cnn_filters < 1:
            raise ValueError("cnn_filters must be at least 1")
        if not self.cnn_kernels:
            raise ValueError("cnn_kernels must not be empty")
        if self.compile_mode not in ["default", "reduce-overhead", "max-autotune"]:
            raise ValueError("compile_mode must be one of: default, reduce-overhead, max-autotune")

        # Check compilation availability if enabled
        if self.compile_model:
            if not hasattr(torch, 'compile'):
                logger.warning("torch.compile is not available in your PyTorch version")
                logger.warning("Setting compile_model to False")
                self.compile_model = False
            elif not self.check_compilation_available():
                if not self.suppress_compile_errors:
                    raise ValueError("Compilation is enabled but required compiler is not available")
                logger.warning("Required compiler for torch.compile not found")
                logger.warning("Compilation will be attempted but may fall back to eager mode")


@dataclass
class TrainingConfig:
    """Configuration for training."""
    output_dir: str = "./out"
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    num_train_epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    logging_steps: int = 50
    save_total_limit: int = 1
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "matthews_correlation"
    greater_is_better: bool = True

    # Mixed precision settings
    fp16: bool = True
    fp16_opt_level: str = "O1"

    # Gradient accumulation
    gradient_accumulation_steps: int = 2

    # Learning rate scheduler
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.1

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0

    # Distributed training
    local_rank: int = -1

    # Checkpointing
    resume_from_checkpoint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        # Parameters that are not valid for TrainingArguments
        excluded_params = [
            'resume_from_checkpoint',
            'early_stopping_patience',
            'early_stopping_threshold'
        ]

        return {k: v for k, v in self.__dict__.items()
                if k not in excluded_params}

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.per_device_train_batch_size < 1:
            raise ValueError("per_device_train_batch_size must be at least 1")
        if self.per_device_eval_batch_size < 1:
            raise ValueError("per_device_eval_batch_size must be at least 1")
        if self.num_train_epochs < 1:
            raise ValueError("num_train_epochs must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")


@dataclass
class ChappieConfig:
    """Combined configuration for EnhancedChappie model and training."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ChappieConfig':
        """Create config from dictionary."""
        # Process model config with type conversion
        model_params = {}
        for k, v in config_dict.items():
            if k in ModelConfig.__annotations__:
                expected_type = ModelConfig.__annotations__[k]
                # Handle special cases like Union types
                if hasattr(expected_type, "__origin__") and expected_type.__origin__ is Union:
                    # For Optional types (Union[X, None])
                    if type(None) in expected_type.__args__:
                        if v is not None:
                            # Get the non-None type
                            non_none_type = next(t for t in expected_type.__args__ if t is not type(None))
                            try:
                                model_params[k] = non_none_type(v)
                            except (ValueError, TypeError):
                                model_params[k] = v
                        else:
                            model_params[k] = None
                # Handle List type
                elif hasattr(expected_type, "__origin__") and expected_type.__origin__ is list:
                    if not isinstance(v, list):
                        # Convert to list if it's not already
                        model_params[k] = [v]
                    else:
                        model_params[k] = v
                # Handle primitive types
                elif expected_type in (int, float, bool, str):
                    try:
                        model_params[k] = expected_type(v)
                    except (ValueError, TypeError):
                        model_params[k] = v
                else:
                    model_params[k] = v

        # Process training config with type conversion
        training_params = {}
        for k, v in config_dict.items():
            if k in TrainingConfig.__annotations__:
                expected_type = TrainingConfig.__annotations__[k]
                # Handle special cases like Union types
                if hasattr(expected_type, "__origin__") and expected_type.__origin__ is Union:
                    # For Optional types (Union[X, None])
                    if type(None) in expected_type.__args__:
                        if v is not None:
                            # Get the non-None type
                            non_none_type = next(t for t in expected_type.__args__ if t is not type(None))
                            try:
                                training_params[k] = non_none_type(v)
                            except (ValueError, TypeError):
                                training_params[k] = v
                        else:
                            training_params[k] = None
                # Handle List type
                elif hasattr(expected_type, "__origin__") and expected_type.__origin__ is list:
                    if not isinstance(v, list):
                        # Convert to list if it's not already
                        training_params[k] = [v]
                    else:
                        training_params[k] = v
                # Handle primitive types
                elif expected_type in (int, float, bool, str):
                    try:
                        training_params[k] = expected_type(v)
                    except (ValueError, TypeError):
                        training_params[k] = v
                else:
                    training_params[k] = v

        model_config = ModelConfig(**model_params)
        training_config = TrainingConfig(**training_params)
        return cls(model=model_config, training=training_config)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ChappieConfig':
        """Load config from YAML file."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Log the loaded configuration for debugging
        logger.debug(f"Loaded configuration from {yaml_path}: {config_dict}")

        # Check for potential type issues
        for key, value in config_dict.items():
            if key == 'learning_rate' and isinstance(value, str):
                logger.warning(f"learning_rate is a string '{value}', will attempt to convert to float")
            elif key in ['num_train_epochs', 'per_device_train_batch_size', 'per_device_eval_batch_size'] and isinstance(value, str):
                logger.warning(f"{key} is a string '{value}', will attempt to convert to int")

        return cls.from_dict(config_dict)

    def save_to_yaml(self, yaml_path: str) -> None:
        """Save config to YAML file."""
        config_dict = {
            **self.model.to_dict(),
            **self.training.to_dict()
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def validate(self) -> None:
        """Validate all configuration parameters."""
        self.model.validate()
        self.training.validate()


def get_default_config() -> ChappieConfig:
    """Get default configuration."""
    return ChappieConfig()

