import torch
import torch.nn as nn
import os
from pathlib import Path
import logging
import json
import datetime
from transformers import (
    AutoTokenizer, 
    AutoModel,
    GPT2Model,
    T5EncoderModel,
    set_seed
)
from torch.optim import Adam
import numpy as np
from typing import Dict, List, Tuple
import torch.nn.functional as F

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chappie.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        self.hidden_size = 768
        self.num_emotions = 8
        self.max_seq_length = 512
        self.batch_size = 16
        self.learning_rate = 1e-4
        self.num_epochs = 10
        self.warmup_steps = 100
        self.gradient_accumulation_steps = 1
        self.seed = 42
        self.cache_dir = "./chappie_cache"
        self.checkpoint_dir = "./checkpoints"
        self.model_version = "1.0.0"

class CacheManager:
    def __init__(self, config: Config):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.model_cache_dir = self.cache_dir / "models"
        self.tokenizer_cache_dir = self.cache_dir / "tokenizers"
        self.setup_cache()

    def setup_cache(self):
        """Setup cache directories and environment"""
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ['TRANSFORMERS_CACHE'] = str(self.model_cache_dir)

    def get_cache_path(self, model_name: str) -> str:
        return str(self.model_cache_dir / model_name)

class EmotionalPredictionSynthesizer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.emotion_embedding = nn.Linear(config.hidden_size, config.hidden_size)
        self.emotion_classifier = nn.Linear(config.hidden_size, config.num_emotions)
        self.emotion_attention = nn.MultiheadAttention(config.hidden_size, 8)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, hidden_states):
        emotion_embeds = self.dropout(self.emotion_embedding(hidden_states))
        attention_output, attention_weights = self.emotion_attention(
            emotion_embeds, 
            emotion_embeds, 
            emotion_embeds
        )
        attention_output = self.dropout(attention_output)
        emotion_scores = self.emotion_classifier(attention_output)
        return emotion_scores, attention_output, attention_weights

class SelfTuningNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.adaptation_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.performance_tracker = []
        self.learning_rate = config.learning_rate
        
    def forward(self, x):
        x = self.dropout(x)
        adapted_features = self.adaptation_layer(x)
        adapted_features = self.layer_norm(adapted_features)
        return adapted_features
    
    def adjust_parameters(self, performance_metric):
        self.performance_tracker.append(performance_metric)
        if len(self.performance_tracker) > 10:
            if performance_metric < np.mean(self.performance_tracker[-10:]):
                self.learning_rate *= 0.95

class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attention = nn.MultiheadAttention(config.hidden_size, 8)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Self-attention
        attention_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + self.dropout(attention_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        return x

class ChappieIntegratedModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        set_seed(config.seed)
        
        # Initialize cache manager
        self.cache_manager = CacheManager(config)
        
        # Initialize base models with consistent cache directories
        self.bert = AutoModel.from_pretrained(
            "bert-base-uncased",
            cache_dir=self.cache_manager.get_cache_path("bert")
        )
        self.gpt2 = GPT2Model.from_pretrained(
            "gpt2",
            cache_dir=self.cache_manager.get_cache_path("gpt2")
        )
        self.t5 = T5EncoderModel.from_pretrained(
            "google/flan-t5-small",
            cache_dir=self.cache_manager.get_cache_path("t5")
        )
        
        # Use consistent hidden size for all components
        self.hidden_size = self.bert.config.hidden_size  # Use BERT's hidden size for consistency
        self.emotion_embedding = nn.Linear(self.hidden_size, self.hidden_size)
        self.emotion_classifier = nn.Linear(self.hidden_size, self.config.num_emotions)
        self.emotion_attention = nn.MultiheadAttention(self.hidden_size, 8)

        # Initialize tokenizers with consistent padding
        self.tokenizers = {}
        self.tokenizers['bert'] = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            cache_dir=self.cache_manager.get_cache_path("bert_tokenizer")
        )
        
        # Setup GPT-2 tokenizer with padding
        gpt2_tokenizer = AutoTokenizer.from_pretrained(
            "gpt2",
            cache_dir=self.cache_manager.get_cache_path("gpt2_tokenizer")
        )
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        self.tokenizers['gpt2'] = gpt2_tokenizer
        
        # Setup T5 tokenizer
        self.tokenizers['t5'] = AutoTokenizer.from_pretrained(
            "google/flan-t5-small",
            cache_dir=self.cache_manager.get_cache_path("t5_tokenizer")
        )
        
        # Initialize components
        self.eps = EmotionalPredictionSynthesizer(config)
        self.self_tuning = SelfTuningNetwork(config)
        
        # Transformer blocks for feature processing
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(3)
        ])
        
        # Integration layers
        self.integration_layer = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.final_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, text: str, inputs) -> Dict[str, torch.Tensor]:
        # Process input
        text = inputs['input_text']  # Extract text from the input dictionary
        encodings = self.process_input(text)
        
        bert_output = self.bert(**encodings['bert']).last_hidden_state
        gpt2_output = self.gpt2(**encodings['gpt2']).last_hidden_state
        t5_output = self.t5(**encodings['t5']).last_hidden_state
        
        # Ensure all outputs have the same dimension
        min_dim = min(self.bert.config.hidden_size, self.gpt2.config.n_embd, self.t5.config.hidden_size)
        bert_output = bert_output[:, :min_dim]
        gpt2_output = gpt2_output[:, :min_dim]
        t5_output = t5_output[:, :min_dim]

        # Reshape T5 output to match the hidden size of BERT and GPT-2
        if t5_output.size(-1) != self.hidden_size:
            t5_output = F.pad(t5_output, (0, self.hidden_size - t5_output.size(-1)), "constant", 0)
            
        # Process through transformer blocks
        for transformer in self.transformer_blocks:
            bert_output = transformer(bert_output)
            gpt2_output = transformer(gpt2_output)
            t5_output = transformer(t5_output)
        
        # Global average pooling
        bert_output = torch.mean(bert_output, dim=1)
        gpt2_output = torch.mean(gpt2_output, dim=1)
        t5_output = torch.mean(t5_output, dim=1)
        
        # Combine features
        combined_features = torch.cat([bert_output, gpt2_output, t5_output], dim=-1)
        integrated_features = self.integration_layer(combined_features)
        integrated_features = self.layer_norm(integrated_features)
        integrated_features = self.dropout(integrated_features)
        
        # Self-tuning and emotional prediction
        tuned_features = self.self_tuning(integrated_features)
        emotion_scores, emotion_context, attention_weights = self.eps(tuned_features)
        
        # Final processing
        output_features = self.final_layer(emotion_context)
        output_features = self.layer_norm(output_features)
        
        return {
            'emotion_scores': emotion_scores,
            'features': output_features,
            'attention_weights': attention_weights,
            'bert_output': bert_output,
            'gpt2_output': gpt2_output,
            't5_output': t5_output
        }

    def process_input(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Process the input text using the appropriate tokenizers and models.
        
        Args:
            text (str): Input text to be processed.
        
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the encoded inputs for each model.
        """
        # Tokenize the input text
        try:
            bert_inputs = self.tokenizers['bert'].encode_plus(
                text,
                max_length=self.config.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            gpt2_inputs = self.tokenizers['gpt2'].encode_plus(
                text,
                max_length=self.config.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            t5_inputs = self.tokenizers['t5'].encode_plus(
                text,
                max_length=self.config.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        except Exception as e:
            logger.error(f"Error during tokenization: {str(e)}")
            raise

        return {
            'bert': bert_inputs,
            'gpt2': gpt2_inputs,
            't5': t5_inputs
        }

    def save_checkpoint(self, epoch: int, optimizer, loss: float):
        """Saves the model checkpoint."""
        # Ensure the checkpoint directory exists
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        logger.info(f"Checkpoint saved at {checkpoint_path}")
        
    def export_model(self, export_path: str):
        """Exports the model to the specified path."""
        torch.save(self.state_dict(), export_path)
        logger.info(f"Model exported to {export_path}")

class ChappieTrainer:
    def __init__(self, model: ChappieIntegratedModel, config: Config):
        self.model = model
        self.config = config
        self.optimizer = Adam(model.parameters(), lr=config.learning_rate)
        self.best_loss = float('inf')
        
    def train(self, train_dataloader):
        for epoch in range(self.config.num_epochs):
            total_loss = 0
            for batch in train_dataloader:
                loss = self.train_step(batch)
                total_loss += loss
            
            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}, Loss: {avg_loss:.4f}")
            
            # Save checkpoint every 100 epochs
            if (epoch + 1) % 100 == 0:
                self.model.save_checkpoint(epoch + 1, self.optimizer, avg_loss)
                
    def train_step(self, batch: Dict[str, torch.Tensor]):
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(batch['input_text'])
        loss = self.calculate_loss(outputs, batch)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def calculate_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]):
        emotion_loss = F.cross_entropy(outputs['emotion_scores'], batch['emotion_labels'])
        feature_loss = F.mse_loss(outputs['features'], batch['target_features'])
        return emotion_loss + feature_loss
    
    def validate(self, val_dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                outputs = self.model(batch['input_text'])
                loss = self.calculate_loss(outputs, batch)
                total_loss += loss.item()
        return total_loss / len(val_dataloader)

def main():
    # Initialize configuration
    config = Config()
    logger.info("Initializing Chappie with configuration...")
    
    # Create model
    model = ChappieIntegratedModel(config)
    trainer = ChappieTrainer(model, config)
    
    # Example usage
    text = "I am feeling happy today!"
    logger.info(f"Processing text: {text}")
    
    try:
        # Wrap the input text in a dictionary
        outputs = model(text, {'input_text': text})
        emotion_scores = outputs['emotion_scores']
        features = outputs['features']
        
        logger.info(f"Processed successfully!")
        logger.info(f"Emotion Scores Shape: {emotion_scores.shape}")
        logger.info(f"Features Shape: {features.shape}")
        
        # Save initial checkpoint
        model.save_checkpoint(0, trainer.optimizer, 0.0)
        
        # Export the trained model
        model.export_model(os.path.join(config.checkpoint_dir, "chappie_model.pt"))
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise
    
    logger.info(f"Models and tokenizers are cached in: {config.cache_dir}")
    logger.info(f"Checkpoints are saved in: {config.checkpoint_dir}")

if __name__ == "__main__":
    main()
