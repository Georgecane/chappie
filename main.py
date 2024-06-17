import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertForSequenceClassification, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import datasets
import logging

gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))


dataset = datasets.load_dataset("keirp/common_crawl_sample")

class SentimentDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset["train"]["text"]
        self.labels = [0] * len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


sentiment_dataset = SentimentDataset(dataset)


train_data, test_data = train_test_split(sentiment_dataset, test_size=0.2, random_state=42)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16)


class LargeNeuralNetworkModel(nn.Module):
    def __init__(self):
        super(LargeNeuralNetworkModel, self).__init__()
        self.gpt2_model = gpt2_model
        self.bert_model = bert_model
        self.linear = nn.Linear(50258, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_text):

        input_ids = gpt2_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True, padding='max_length')
        gpt2_output = self.gpt2_model(input_ids)[0]


        bert_input_ids = bert_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True, padding='max_length')
        bert_output = self.bert_model(bert_input_ids)[0]
        sentiment_score = self.sigmoid(bert_output[:, 0]).item()


        selected_output = self.select_output(gpt2_output, sentiment_score)
        return selected_output

    def train_LNNM(self, input_text, sentiment_score):

        optimizer = torch.optim.Adam(self.parameters())
        criterion = nn.MSELoss()


        input_ids = gpt2_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True, padding='max_length')
        print(input_ids.shape)
        gpt2_output = self.gpt2_model(input_ids)[0]


        gpt2_output_vector = gpt2_output.mean(dim=1).squeeze()

        predicted_sentiment_score = self.linear(gpt2_output_vector)


        loss = criterion(predicted_sentiment_score, torch.tensor([sentiment_score]).float())

        optimizer.zero_grad()
        optimizer.step()
        return loss

    def select_output(self, gpt2_output, sentiment_score):
        gpt2_output_text = gpt2_tokenizer.decode(gpt2_output[0], skip_special_tokens=True)
        if sentiment_score >= 0.5:
            return "Positive sentiment: " + gpt2_output_text
        else:
            return "Negative sentiment: " + gpt2_output_text

lnnm_model = LargeNeuralNetworkModel()
epochs = 10 

lnnm_model.requires_grad_ = True

for epoch in range(epochs):
    for input_text, _ in train_loader:
        loss = lnnm_model.train_LNNM(input_text, 0)
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")


torch.save(lnnm_model.state_dict(), "trained_lnnm_model.pt")

lnnm_model.load_state_dict(torch.load("trained_lnnm_model.pt"))

input_text = "I am so happy today!"

output = lnnm_model(input_text)
print(output)
