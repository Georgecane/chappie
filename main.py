import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define ResNet Block
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip_connection = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip_connection(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

# Define GPT-like Encoder
class GPTEncoder(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size):
        super(GPTEncoder, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_size, nhead=1),
            num_layers=num_layers
        )
        self.fc = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # Transformer expects input as (sequence_length, batch_size, input_size)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # Return to (batch_size, sequence_length, input_size)
        x = self.fc(x[:, -1, :])  # Use the output of the last time step
        return x

# Define the final model that uses ResNet and GPT
class CombinedModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, vocab_size):
        super(CombinedModel, self).__init__()
        self.resnet = nn.Sequential(
            ResNetBlock(input_size, 64),
            ResNetBlock(64, 128),
            ResNetBlock(128, 256)
        )
        self.gpt_encoder = GPTEncoder(input_size=256, num_layers=num_layers, hidden_size=hidden_size)
        self.fc_out = nn.Linear(256 + hidden_size, vocab_size)  # Output layer for vocab_size

    def forward(self, x):
        resnet_features = self.resnet(x)
        # ResNet output shape: (batch_size, 256, seq_len)
        # We want to use the features from the last time step of ResNet
        resnet_last_step = resnet_features[:, :, -1]  # Shape: (batch_size, 256)
        gpt_features = self.gpt_encoder(resnet_features.permute(2, 0, 1))  # Permute for GPT input

        # Combine features
        combined_features = torch.cat((resnet_last_step, gpt_features), dim=1)  # Ensure dimensions match
        out = self.fc_out(combined_features)
        return out

# Set parameters
input_size = 256  # Input size
hidden_size = 50  # Hidden size in GPT
num_layers = 2  # Number of layers in GPT
vocab_size = 100  # Dictionary size for text generation

# Create the model
model = CombinedModel(input_size, hidden_size, num_layers, vocab_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Generate synthetic data
def generate_synthetic_data(size, seq_len, input_size):
    inputs = torch.randn(size, input_size, seq_len)  # Random float values
    targets = torch.randint(0, vocab_size, (size,))
    return inputs, targets

# Save the model
def save_model(model, epoch, path='model_checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, path)
    print(f'Model saved to {path} at epoch {epoch + 1}')

# Train the model
def train_model(model, train_loader, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Save the model after each epoch
        save_model(model, epoch)

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Test the model
def test_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    print(f'Average Test Loss: {avg_loss:.4f}')

# Generate text
def generate_text(model, start_sequence, max_length=50):
    model.eval()
    input_seq = torch.tensor(start_sequence).unsqueeze(0).float()  # Convert to float and add batch dimension
    generated = list(start_sequence)

    for _ in range(max_length):
        with torch.no_grad():
            output = model(input_seq)
            _, predicted = torch.max(output, dim=1)
            next_token = predicted.item()
            generated.append(next_token)
            input_seq = torch.cat((input_seq[:, 1:], torch.tensor([[next_token]]).float()), dim=1)

    return generated

# Set training and test data parameters
batch_size = 8  # Increase batch size
epochs = 10
seq_len = 20  # Sequence length

# Generate training data
train_inputs, train_targets = generate_synthetic_data(100, seq_len, input_size)
train_dataset = TensorDataset(train_inputs, train_targets)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Generate test data
test_inputs, test_targets = generate_synthetic_data(20, seq_len, input_size)
test_dataset = TensorDataset(test_inputs, test_targets)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Train the model
train_model(model, train_loader, epochs)

# Test the model
test_model(model, test_loader)

# Generate text with user input
start_sequence = [1] * seq_len  # Example input sequence for text generation
generated_text = generate_text(model, start_sequence, max_length=50)
print('Generated Text:', generated_text)
