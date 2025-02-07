import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import Dict, Any
import logging
from collections import deque

logger = logging.getLogger(__name__)

class EnhancedStateEncoder(nn.Module):
    """Enhanced state encoder with temporal awareness"""
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
        return self.encoder(x)

class DynamicEmotionProcessor(nn.Module):
    """Dynamic emotion processor with attention gates"""
    def __init__(self, input_size: int, num_emotions: int):
        super().__init__()
        self.attention_gate = nn.Sequential(
            nn.Linear(input_size, 1),
            nn.Sigmoid()
        )
        self.emotion_net = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.GELU(),
            nn.LayerNorm(input_size * 2),
            nn.Linear(input_size * 2, num_emotions)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.attention_gate(x)
        return self.emotion_net(x * attention)

class HierarchicalSelfReflection(nn.Module):
    """Hierarchical self-reflection with multi-layer attention"""
    def __init__(self, hidden_size: int, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_size, num_heads=8, dropout=0.1, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, norm in zip(self.layers, self.norms):
            residual = x
            x, _ = attn(x, x, x)
            x = norm(residual + x)
        return x

class NeuralMemoryBank(nn.Module):
    """Differentiable neural memory with content-based addressing"""
    def __init__(self, hidden_size: int, memory_size: int = 1024):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(memory_size, hidden_size))
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.selector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(x, self.memory.T)
        weights = F.softmax(scores, dim=-1)
        read = torch.matmul(weights, self.memory)
        update_gate = torch.sigmoid(scores).mean(dim=(0,1))
        read_avg = read.mean(dim=(0,1))
        new_memory = self.memory * (1 - update_gate.unsqueeze(-1)) + read_avg.unsqueeze(0) * update_gate.unsqueeze(-1)
        self.memory.data = self.memory_norm(new_memory)
        return read

class DecisionEngine(nn.Module):
    """Parallel decision engine with dynamic routing"""
    def __init__(self, hidden_size: int, num_decisions: int = 5):
        super().__init__()
        self.decision_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, 1)
            ) for _ in range(num_decisions)
        ])
        self.router = nn.Linear(hidden_size, num_decisions)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        routing_weights = F.softmax(self.router(x), dim=-1)
        decisions = torch.cat([
            head(x) * weight.unsqueeze(-1) 
            for head, weight in zip(self.decision_heads, routing_weights.unbind(-1))
        ], dim=-1)
        return {
            'decisions': decisions,
            'routing_weights': routing_weights
        }

class TextCNN(nn.Module):
    """Convolutional Neural Network for text classification"""
    def __init__(self, hidden_size: int, num_filters: int = 128, kernel_sizes: list = [3,4,5], num_classes: int = 2):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_size, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_size) -> transpose to (batch, hidden_size, seq_len)
        x = x.transpose(1, 2)
        conv_results = [F.relu(conv(x)) for conv in self.convs]
        pooled = [F.max_pool1d(r, kernel_size=r.shape[2]).squeeze(2) for r in conv_results]
        cat = torch.cat(pooled, dim=1)
        logits = self.fc(cat)
        return logits

class EnhancedChappie(nn.Module):
    """Enhanced Chappie model with consciousness mechanisms, integrated with a CNN for text classification."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        try:
            self.config = config
            model_config = AutoConfig.from_pretrained(config["model_name"])
            self.transformer = AutoModel.from_pretrained(config["model_name"], config=model_config)
            self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
            
            # Enhanced components (for auxiliary outputs)
            self.state_encoder = EnhancedStateEncoder(model_config.hidden_size, config["state_size"])
            self.emotion_processor = DynamicEmotionProcessor(model_config.hidden_size, config["num_emotions"])
            self.self_reflection = HierarchicalSelfReflection(model_config.hidden_size)
            self.memory_bank = NeuralMemoryBank(model_config.hidden_size)
            self.decision_engine = DecisionEngine(model_config.hidden_size)
            
            # CNN classifier for text classification
            self.cnn_classifier = TextCNN(model_config.hidden_size, num_filters=128, kernel_sizes=[3,4,5], num_classes=2)
            
            # Caching system (if needed)
            self.cache = deque(maxlen=config.get("cache_size", 100))
            self.cache_proj = nn.Linear(model_config.hidden_size, model_config.hidden_size // 2)
            
            self._init_weights()
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "output" in name or "head" in name or "classifier" in name or "fc" in name:
                    nn.init.normal_(module.weight, std=0.02)
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def update_cache(self, embeddings: torch.Tensor):
        compressed = self.cache_proj(embeddings.detach())
        self.cache.append(compressed.mean(dim=1))
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None) -> Dict[str, Any]:
        try:
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # shape: (batch, seq_len, hidden)
            
            # Process auxiliary consciousness modules
            state = self.state_encoder(hidden_states)
            emotions = self.emotion_processor(hidden_states)
            reflected = self.self_reflection(hidden_states)
            memory_context = self.memory_bank(reflected)
            decisions = self.decision_engine(memory_context)
            self.update_cache(memory_context)
            
            # Use the CNN classifier for final classification
            logits = self.cnn_classifier(hidden_states)  # expected shape: (batch, num_classes)
            logits = logits.view(hidden_states.size(0), 2)  # ensure shape is (batch, 2)
            
            if labels is not None:
                loss = F.cross_entropy(logits, labels)
                return {
                    "loss": loss,
                    "logits": logits,
                    "state": state,
                    "emotions": emotions,
                    "decisions": decisions,
                    "memory_context": memory_context
                }
            else:
                # During evaluation, return only the logits
                return logits
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            raise

    def generate_response(self, input_text: str, max_length: int = 100) -> str:
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(next(self.transformer.parameters()).device)
        try:
            generated = self.transformer.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            return self.tokenizer.decode(generated[0], skip_special_tokens=True)
        except AttributeError:
            logger.error("The underlying transformer model does not support the generate() method. Consider using a generative model.")
            return ""
