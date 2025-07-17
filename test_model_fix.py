#!/usr/bin/env python
"""
Simple test script to verify that the model fixes work correctly.
This script tests basic model functionality without running full training.
"""

import torch
import logging
from transformers import AutoTokenizer
from model import EnhancedChappie, get_device
from config import ChappieConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def test_model_forward_pass():
    """Test basic model forward pass functionality."""
    logger.info("Testing model forward pass...")

    # Load configuration
    config = ChappieConfig()

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Create model
    logger.info("Creating model...")
    model = EnhancedChappie(config.model.to_dict())
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

    # Create test sentences
    test_sentences = [
        "This is a grammatically correct sentence.",
        "This sentence is not correct grammatical."
    ]

    # Test preprocessing similar to training
    logger.info("Testing data preprocessing...")
    for i, sentence in enumerate(test_sentences):
        logger.info(f"Testing sentence {i+1}: '{sentence}'")

        # Tokenize
        encoding = tokenizer(
            sentence,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )

        # Create a batch dict similar to what the trainer would create
        batch = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor([1])  # Dummy label
        }

        # Test forward pass
        try:
            with torch.no_grad():
                outputs = model(**batch)

            logger.info(f"✓ Forward pass successful for sentence {i+1}")
            logger.info(f"  - Output shapes: logits={outputs['logits'].shape}, decisions={outputs['decisions'].shape}")
            logger.info(f"  - Loss: {outputs['loss'].item() if outputs['loss'] is not None else 'None'}")

        except Exception as e:
            logger.error(f"✗ Forward pass failed for sentence {i+1}: {e}")
            return False

    return True

def test_batch_processing():
    """Test batch processing with multiple sentences."""
    logger.info("Testing batch processing...")

    # Load configuration
    config = ChappieConfig()

    # Create model
    model = EnhancedChappie(config.model.to_dict())
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

    # Create test batch
    test_sentences = [
        "This is a grammatically correct sentence.",
        "This sentence is not correct grammatical.",
        "Another example of a proper sentence.",
        "Bad grammar this is having."
    ]

    labels = [1, 0, 1, 0]  # Dummy labels

    # Tokenize batch
    encoding = tokenizer(
        test_sentences,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )

    # Create batch dict
    batch = {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'labels': torch.tensor(labels)
    }

    logger.info(f"Batch size: {batch['input_ids'].shape[0]}")
    logger.info(f"Sequence length: {batch['input_ids'].shape[1]}")

    # Test forward pass
    try:
        with torch.no_grad():
            outputs = model(**batch)

        logger.info("✓ Batch processing successful")
        logger.info(f"  - Logits shape: {outputs['logits'].shape}")
        logger.info(f"  - Decisions shape: {outputs['decisions'].shape}")
        logger.info(f"  - Memory context shape: {outputs['memory_context'].shape}")
        logger.info(f"  - Loss: {outputs['loss'].item() if outputs['loss'] is not None else 'None'}")

        return True

    except Exception as e:
        logger.error(f"✗ Batch processing failed: {e}")
        return False

def test_dataset_compatibility():
    """Test compatibility with dataset fields (including 'idx')."""
    logger.info("Testing dataset compatibility with extra fields...")

    # Load configuration
    config = ChappieConfig()

    # Create model
    model = EnhancedChappie(config.model.to_dict())
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

    # Create a batch that mimics what comes from the CoLA dataset
    sentence = "This is a test sentence."
    encoding = tokenizer(
        sentence,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )

    # Simulate dataset batch with extra fields (like 'idx')
    batch_with_extra_fields = {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'labels': torch.tensor([1]),
        'idx': torch.tensor([0]),  # This field was causing the error
        'sentence': ['This is a test sentence.'],  # Original sentence
        'extra_field': torch.tensor([42])  # Another extra field
    }

    logger.info("Testing with extra dataset fields...")
    logger.info(f"Batch keys: {list(batch_with_extra_fields.keys())}")

    # Test the filter function from trainer
    from trainer import filter_batch_for_model
    filtered_batch = filter_batch_for_model(batch_with_extra_fields)

    logger.info(f"Filtered batch keys: {list(filtered_batch.keys())}")

    # Test forward pass with filtered batch
    try:
        with torch.no_grad():
            outputs = model(**filtered_batch)

        logger.info("✓ Dataset compatibility test successful")
        logger.info("  - Extra fields properly filtered out")
        logger.info(f"  - Forward pass completed successfully")

        return True

    except Exception as e:
        logger.error(f"✗ Dataset compatibility test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Running Model Fix Verification Tests")
    logger.info("=" * 60)

    tests = [
        ("Basic Forward Pass", test_model_forward_pass),
        ("Batch Processing", test_batch_processing),
        ("Dataset Compatibility", test_dataset_compatibility)
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
            results[test_name] = False

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! The model fixes are working correctly.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
