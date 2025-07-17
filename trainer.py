# trainer.py
import os
import torch
import numpy as np
import argparse
import logging
from pathlib import Path
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from datasets import load_dataset
from evaluate import load as load_metric
from model import EnhancedChappie, get_device
from config import ChappieConfig, ModelConfig, TrainingConfig
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chappie_training.log')
    ]
)
logger = logging.getLogger(__name__)

def compute_metrics(p):
    """Compute evaluation metrics."""
    preds = np.argmax(p.predictions, axis=1)
    return {
        'matthews_correlation': load_metric('matthews_correlation').compute(
            predictions=preds,
            references=p.label_ids
        )['matthews_correlation']
    }

def filter_batch_for_model(batch):
    """Filter batch to only include arguments expected by the model."""
    expected_keys = {'input_ids', 'attention_mask', 'labels', 'sentence', 'label'}
    return {k: v for k, v in batch.items() if k in expected_keys}

def clear_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def get_memory_usage():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3     # GB
        return allocated, cached
    return 0.0, 0.0

def get_scheduler(optimizer, scheduler_type, num_warmup_steps, num_training_steps):
    """Get the appropriate learning rate scheduler."""
    if scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

def train_with_amp(model, train_dataloader, eval_dataloader, config, device):
    """Custom training loop with Automatic Mixed Precision."""
    logger.info(f"Starting custom AMP training loop on device: {device}")

    # Clear memory before starting
    clear_memory()

    # Monitor initial memory usage
    if torch.cuda.is_available():
        allocated, cached = get_memory_usage()
        logger.info(f"Initial GPU memory: {allocated:.2f} GB allocated, {cached:.2f} GB cached")

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model.backbone, 'gradient_checkpointing_enable'):
        model.backbone.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing for memory efficiency")

    # Ensure model is on the correct device
    model = model.to(device)

    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )

    # Setup gradient scaler for mixed precision training
    use_amp = config.training.fp16 and torch.cuda.is_available()
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = GradScaler(device_type) if use_amp else None

    # Calculate total training steps
    num_epochs = config.training.num_train_epochs
    num_update_steps_per_epoch = len(train_dataloader) // config.training.gradient_accumulation_steps
    num_training_steps = num_update_steps_per_epoch * num_epochs

    # Setup learning rate scheduler
    num_warmup_steps = int(num_training_steps * config.training.warmup_ratio)
    scheduler = get_scheduler(
        optimizer,
        config.training.lr_scheduler_type,
        num_warmup_steps,
        num_training_steps
    )

    # Training loop
    best_metric = float('-inf') if config.training.greater_is_better else float('inf')
    patience_counter = 0

    # Create output directory
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # Progress bar for training
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        # Training step
        for step, batch in enumerate(progress_bar):
            try:
                # Debug logging for first batch
                if step == 0:
                    logger.info("Debug - Original batch keys and types:")
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            logger.info(f"  {k}: {type(v)} {v.dtype} {v.shape}")
                        else:
                            logger.info(f"  {k}: {type(v)}")

                # Filter batch to only include expected model arguments
                batch = filter_batch_for_model(batch)

                # Debug logging for filtered batch
                if step == 0:
                    logger.info("Debug - Filtered batch keys and types:")
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            logger.info(f"  {k}: {type(v)} {v.dtype} {v.shape}")
                        else:
                            logger.info(f"  {k}: {type(v)}")

                # Convert to proper tensor types and move to device safely
                processed_batch = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        # Ensure proper dtype for different keys
                        if k in ['input_ids', 'attention_mask']:
                            processed_batch[k] = v.long().to(device)
                        elif k == 'labels':
                            processed_batch[k] = v.long().to(device)
                        else:
                            processed_batch[k] = v.to(device)
                    elif isinstance(v, list) and k == 'labels':
                        # Convert list of labels to tensor
                        processed_batch[k] = torch.tensor(v, dtype=torch.long, device=device)
                    elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
                        # Handle nested lists (like input_ids, attention_mask)
                        processed_batch[k] = torch.tensor(v, dtype=torch.long, device=device)
                    else:
                        processed_batch[k] = v

                batch = processed_batch

                # Memory monitoring before forward pass
                if step % 50 == 0 and torch.cuda.is_available():
                    allocated, cached = get_memory_usage()
                    logger.debug(f"Step {step} GPU memory: {allocated:.2f} GB allocated, {cached:.2f} GB cached")

                # Forward pass with mixed precision if available
                if use_amp:
                    with autocast(device_type=device_type):
                        outputs = model(**batch)
                        loss = outputs['loss']
                        loss = loss / config.training.gradient_accumulation_steps

                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()

                    # Update weights with gradient accumulation
                    if (step + 1) % config.training.gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                else:
                    # Standard forward and backward pass without AMP
                    outputs = model(**batch)
                    loss = outputs['loss']
                    loss = loss / config.training.gradient_accumulation_steps

                    # Backward pass
                    loss.backward()

                    # Update weights with gradient accumulation
                    if (step + 1) % config.training.gradient_accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                # Update progress bar
                total_loss += loss.item() * config.training.gradient_accumulation_steps
                progress_bar.set_postfix({"loss": total_loss / (step + 1)})

                # Clear memory periodically to prevent accumulation
                if step % 100 == 0:
                    clear_memory()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"CUDA out of memory at step {step}: {e}")
                    logger.info("Clearing GPU cache and skipping batch...")
                    clear_memory()
                    # Skip this batch and continue
                    continue
                else:
                    logger.error(f"Runtime error during training step: {e}")
                    continue
            except Exception as e:
                logger.error(f"Error during training step: {e}")
                # Skip this batch and continue
                continue

        # Clear memory before evaluation
        clear_memory()

        # Evaluation step
        model.eval()
        eval_loss = 0
        eval_preds = []
        eval_labels = []

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                try:
                    # Filter batch to only include expected model arguments
                    batch = filter_batch_for_model(batch)

                    # Move batch to device safely
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                    # Forward pass with mixed precision if available
                    if use_amp:
                        with autocast(device_type=device_type):
                            outputs = model(**batch)
                    else:
                        outputs = model(**batch)

                    # Collect evaluation results
                    if outputs['loss'] is not None:
                        eval_loss += outputs['loss'].item()
                    eval_preds.append(outputs['logits'].cpu().numpy())
                    eval_labels.append(batch['labels'].cpu().numpy())

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.error(f"CUDA out of memory during evaluation: {e}")
                        logger.info("Clearing GPU cache and skipping evaluation batch...")
                        clear_memory()
                        continue
                    else:
                        logger.error(f"Runtime error during evaluation step: {e}")
                        continue
                except Exception as e:
                    logger.error(f"Error during evaluation step: {e}")
                    # Skip this batch and continue
                    continue

        # Compute metrics
        if eval_preds and eval_labels:  # Check if we have any valid predictions
            try:
                eval_preds = np.concatenate(eval_preds, axis=0)
                eval_labels = np.concatenate(eval_labels, axis=0)

                metrics = compute_metrics(type('obj', (), {
                    'predictions': eval_preds,
                    'label_ids': eval_labels
                }))

                # Log metrics
                if len(eval_dataloader) > 0:
                    eval_loss /= len(eval_dataloader)
                metrics['eval_loss'] = eval_loss

                logger.info(f"Epoch {epoch+1}/{num_epochs}: {metrics}")

                # Check for improvement
                metric_value = metrics[config.training.metric_for_best_model]
                improved = (config.training.greater_is_better and metric_value > best_metric) or \
                          (not config.training.greater_is_better and metric_value < best_metric)

                if improved:
                    best_metric = metric_value
                    patience_counter = 0

                    # Save best model
                    logger.info(f"New best model with {config.training.metric_for_best_model} = {best_metric}")
                    model.save_pretrained(output_dir / "best_model")

                    # Save tokenizer
                    if hasattr(model, 'tokenizer'):
                        model.tokenizer.save_pretrained(output_dir / "best_model")
                else:
                    patience_counter += 1
                    logger.info(f"No improvement. Patience: {patience_counter}/{config.training.early_stopping_patience}")

                # Early stopping
                if patience_counter >= config.training.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            except Exception as e:
                logger.error(f"Error computing metrics: {e}")
        else:
            logger.warning("No valid predictions for evaluation. Skipping metrics computation.")

        # Save checkpoint
        try:
            checkpoint_dir = output_dir / f"checkpoint-{epoch+1}"
            checkpoint_dir.mkdir(exist_ok=True, parents=True)
            model.save_pretrained(checkpoint_dir)

            # Save optimizer and scheduler state
            checkpoint_dict = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
            }

            if use_amp:
                checkpoint_dict['scaler'] = scaler.state_dict()

            torch.save(checkpoint_dict, checkpoint_dir / "optimizer.pt")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

    # Load best model
    try:
        logger.info(f"Loading best model with {config.training.metric_for_best_model} = {best_metric}")
        best_model_path = output_dir / "best_model"
        if best_model_path.exists():
            # Load best model weights
            best_state_dict = torch.load(best_model_path / "enhanced_chappie.pt",
                                        map_location=device)
            model.load_state_dict(best_state_dict)
    except Exception as e:
        logger.error(f"Error loading best model: {e}")

    # Final memory cleanup
    clear_memory()

    # Report final memory usage
    if torch.cuda.is_available():
        allocated, cached = get_memory_usage()
        logger.info(f"Final GPU memory: {allocated:.2f} GB allocated, {cached:.2f} GB cached")

    return model, best_metric

def main(config_path=None):
    """Main training function."""
    # Load configuration
    if config_path and os.path.exists(config_path):
        logger.info(f"Loading configuration from {config_path}")
        config = ChappieConfig.from_yaml(config_path)
    else:
        logger.info("Using default configuration")
        config = ChappieConfig()

    # Validate configuration
    config.validate()

    # Set environment variables
    os.environ['HF_DATASETS_OFFLINE'] = '1'

    # Determine device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info("Loading dataset")
    dataset = load_dataset('glue', 'cola', cache_dir='./cache')

    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.model.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

    # Preprocess dataset
    logger.info("Preprocessing dataset")
    def preprocess(examples):
        # Tokenize the sentences
        encodings = tokenizer(
            examples['sentence'],
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors=None  # Return as lists, not tensors
        )

        # Convert to proper types
        return {
            'input_ids': encodings['input_ids'],  # Already lists of integers
            'attention_mask': encodings['attention_mask'],  # Already lists of integers
            'labels': examples['label']  # Already list of integers
        }

    processed_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset['train'].column_names  # Remove original columns to avoid conflicts
    )

    # Initialize model
    logger.info("Initializing model")
    model = EnhancedChappie(config.model.to_dict())

    # Choose training method
    if config.training.fp16 and torch.cuda.is_available():
        logger.info("Using custom AMP training loop")

        # Create data loaders with custom collate function
        def collate_fn(batch):
            """Custom collate function to ensure proper tensor types."""
            # Extract keys
            keys = batch[0].keys()

            # Create batch dict
            batch_dict = {}
            for key in keys:
                if key in ['input_ids', 'attention_mask', 'labels']:
                    # Convert to tensors with proper dtype
                    batch_dict[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)
                else:
                    # Keep other data as is
                    batch_dict[key] = [item[key] for item in batch]

            return batch_dict

        train_dataloader = DataLoader(
            processed_dataset['train'],
            batch_size=config.training.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        eval_dataloader = DataLoader(
            processed_dataset['validation'],
            batch_size=config.training.per_device_eval_batch_size,
            collate_fn=collate_fn
        )

        # Train with custom AMP loop
        model, _ = train_with_amp(
            model,
            train_dataloader,
            eval_dataloader,
            config,
            device
        )
        # Note: best_metric is returned by train_with_amp but we don't need it here
    else:
        logger.info("Using Hugging Face Trainer")

        # Convert config to TrainingArguments
        training_args = TrainingArguments(
            **config.training.to_dict()
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset['train'],
            eval_dataset=processed_dataset['validation'],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=config.training.early_stopping_patience,
                    early_stopping_threshold=config.training.early_stopping_threshold
                )
            ]
        )

        # Train model
        if config.training.resume_from_checkpoint and os.path.exists(config.training.resume_from_checkpoint):
            logger.info(f"Resuming training from {config.training.resume_from_checkpoint}")
            trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)
        else:
            logger.info("Starting training from scratch")
            trainer.train()

        # Evaluate model
        metrics = trainer.evaluate()
        logger.info(f"Final evaluation metrics: {metrics}")

        # Save model
        output_dir = Path(config.training.output_dir) / "final_model"
        output_dir.mkdir(exist_ok=True, parents=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        logger.info(f"Model saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EnhancedChappie model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    main(args.config)
