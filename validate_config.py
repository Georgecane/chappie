#!/usr/bin/env python
"""
Utility script to validate a configuration file.
This helps catch configuration errors before starting a training run.
"""

import argparse
import logging
import sys
from config import ChappieConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def validate_config(config_path):
    """Validate a configuration file."""
    try:
        logger.info(f"Loading configuration from {config_path}")
        config = ChappieConfig.from_yaml(config_path)

        # Print configuration details
        logger.info("Model configuration:")
        for key, value in config.model.__dict__.items():
            logger.info(f"  {key}: {value}")

        logger.info("Training configuration:")
        for key, value in config.training.__dict__.items():
            logger.info(f"  {key}: {value}")

        # Check compilation settings
        if config.model.compile_model:
            logger.info("Checking compilation availability...")
            import torch

            if not hasattr(torch, 'compile'):
                logger.warning("⚠️ torch.compile is not available in your PyTorch version")
                logger.warning("  - PyTorch 2.0 or later is required for compilation")
                logger.warning("  - Training will proceed without compilation")
            else:
                # Check if compilation is available on this system
                compilation_available = config.model.check_compilation_available()
                if not compilation_available:
                    if config.model.suppress_compile_errors:
                        logger.warning("⚠️ Compilation is enabled but required compiler not found")
                        logger.warning("  - Training will attempt compilation but may fall back to eager mode")
                        logger.warning("  - Set suppress_compile_errors: false to fail instead of falling back")
                    else:
                        logger.error("❌ Compilation is enabled but required compiler not found")
                        logger.error("  - Training will fail unless suppress_compile_errors is set to true")
                else:
                    logger.info("✓ Compilation is available and properly configured")

        # Validate configuration
        logger.info("Validating configuration...")
        config.validate()

        logger.info("Configuration is valid!")
        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a configuration file")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    if validate_config(args.config):
        sys.exit(0)
    else:
        sys.exit(1)
