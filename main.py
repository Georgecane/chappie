import os
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertForSequenceClassification, BertTokenizer
from torch.utils.data import Dataset, DataLoader, random_split
import mlcroissant as mlc

# Load pre-trained models and tokenizers
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Add pad token for GPT-2 tokenizer
gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))

# Set the cache directory
cache_dir = 'D:\Datasets_cache'
os.environ['MLCROISSANT_CACHE_DIR'] = cache_dir

# Load dataset using mlcroissant
dataset = mlc.Dataset(jsonld="https://huggingface.co/api/datasets/HuggingFaceFW/fineweb/croissant")
records = dataset.records("CC-MAIN-2014-23")

class SentimentDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.data = [record['text'] for record in records]  # Access 'text' from records
        self.labels = [0] * len(self.data)  # Adjust this based on your dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        tokens = self.tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True, padding='max_length')
        return tokens.squeeze(), self.labels[index]

sentiment_dataset = SentimentDataset(records, gpt2_tokenizer)  # Pass records instead of dataset

# Split dataset into training and testing sets
train_size = int(0.8 * len(sentiment_dataset))
test_size = len(sentiment_dataset) - train_size
train_data, test_data = random_split(sentiment_dataset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
test_loader = DataLoader(test_data, batch_size=4)

class LargeNeuralNetworkModel(nn.Module):
    def __init__(self):
        super(LargeNeuralNetworkModel, self).__init__()
        self.gpt2_model = gpt2_model
        self.bert_model = bert_model
        self.linear = nn.Linear(50258, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids):
        gpt2_output = self.gpt2_model(input_ids)[0]
        return gpt2_output

    def compute_sentiment_score(self, input_ids):
        bert_output = self.bert_model(input_ids)[0]
        sentiment_score = self.sigmoid(bert_output[:, 0]).item()
        return sentiment_score

    def train_LNNM(self, input_ids, sentiment_score):
        optimizer = torch.optim.Adam(self.parameters())
        criterion = nn.MSELoss()

        gpt2_output = self.gpt2_model(input_ids)[0]

        gpt2_output_vector = gpt2_output.mean(dim=1).squeeze()
        predicted_sentiment_score = self.linear(gpt2_output_vector)

        loss = criterion(predicted_sentiment_score, torch.tensor([sentiment_score]).float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def select_output(self, gpt2_output, sentiment_score):
        gpt2_output_text = gpt2_tokenizer.decode(gpt2_output[0], skip_special_tokens=True)
        if sentiment_score >= 0.5:
            return "Positive sentiment: " + gpt2_output_text
        else:
            return "Negative sentiment: " + gpt2_output_text

lnnm_model = LargeNeuralNetworkModel()

# Ensure model parameters are set to require gradients
for param in lnnm_model.parameters():
    param.requires_grad = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lnnm_model.to(device)

if torch.cuda.device_count() > 1:
    lnnm_model = nn.DataParallel(lnnm_model)

epochs = 10

for epoch in range(epochs):
    epoch_loss = 0
    for input_ids, _ in train_loader:
        input_ids = input_ids.to(device)
        loss = lnnm_model.module.train_LNNM(input_ids, 0) if torch.cuda.device_count() > 1 else lnnm_model.train_LNNM(input_ids, 0)
        epoch_loss += loss.item()
        del input_ids  # Free memory
        torch.cuda.empty_cache()  # Clear cache
    print(f"Epoch: {epoch+1}, Loss: {epoch_loss/len(train_loader)}")

# Save the model
torch.save(lnnm_model.module.state_dict() if torch.cuda.device_count() > 1 else lnnm_model.state_dict(), "trained_lnnm_model.pt")

# Load the trained model
lnnm_model.load_state_dict(torch.load("trained_lnnm_model.pt"))

# Provide input text
input_text = "I am so happy today!"

# Predict sentiment
input_ids = gpt2_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True, padding='max_length').to(device)
output = lnnm_model.module.select_output(lnnm_model(input_ids), lnnm_model.module.compute_sentiment_score(input_ids)) if torch.cuda.device_count() > 1 else lnnm_model.select_output(lnnm_model(input_ids), lnnm_model.compute_sentiment_score(input_ids))
print(output)
