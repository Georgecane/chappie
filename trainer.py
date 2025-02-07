import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset
from model import EnhancedChappie
import numpy as np
import evaluate

# Load the CoLA metric once using evaluate.
cola_metric = evaluate.load("glue", "cola")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return cola_metric.compute(predictions=predictions, references=labels)

def main():
    # Load the GLUE CoLA dataset.
    dataset = load_dataset("glue", "cola")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def preprocess_function(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    # Model configuration.
    config = {
        "model_name": "bert-base-uncased",
        "state_size": 768,
        "num_emotions": 5,
        "cache_size": 100
    }
    model = EnhancedChappie(config)

    # Training arguments: using eval_strategy instead of evaluation_strategy.
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,  # Adjust epochs as needed.
        weight_decay=0.01,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()

    model.save_pretrained("./enhanced_chappie")
    tokenizer.save_pretrained("./enhanced_chappie")

if __name__ == "__main__":
    main()
