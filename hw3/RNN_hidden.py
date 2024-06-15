import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import io
import matplotlib.pyplot as plt
from tqdm import tqdm

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Shakespeare training dataset
train_data_url = 'shakespeare_train1.txt'
with io.open(train_data_url, 'r', encoding='utf-8') as f:
    train_text = f.read()

# Load Shakespeare validation dataset
valid_data_url = 'shakespeare_valid.txt'
with io.open(valid_data_url, 'r', encoding='utf-8') as f:
    valid_text = f.read()

# Characters’ collection for training data
train_vocab = set(train_text)
# Construct character dictionary for training data
train_vocab_to_int = {c: i for i, c in enumerate(train_vocab)}
int_to_vocab_train = {i: c for c, i in train_vocab_to_int.items()}

# Encode training data, shape = [number of characters]
train_data = np.array([train_vocab_to_int[c] for c in train_text], dtype=np.int32)

# Characters’ collection for validation data
valid_vocab = set(valid_text)
# Construct character dictionary for validation data
valid_vocab_to_int = {c: i for i, c in enumerate(valid_vocab)}

# Encode validation data, shape = [number of characters]
valid_data = np.array([valid_vocab_to_int[c] for c in valid_text], dtype=np.int32)

# Parameters
batch_size = 64
vocab_size = len(train_vocab)
embedding_dim = 64
hidden_size = 128
num_layers = 1

# Create input and target sequences for training data
train_input_data = train_data[:-1]
train_target_data = train_data[1:]

# Convert training data to PyTorch tensors
train_input_data = torch.from_numpy(train_input_data).to(device)
train_target_data = torch.from_numpy(train_target_data).to(device)

# Create DataLoader for training data
train_dataset = torch.utils.data.TensorDataset(train_input_data, train_target_data)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create input and target sequences for validation data
valid_input_data = valid_data[:-1]
valid_target_data = valid_data[1:]

# Convert validation data to PyTorch tensors
valid_input_data = torch.from_numpy(valid_input_data).to(device)
valid_target_data = torch.from_numpy(valid_target_data).to(device)

# Create DataLoader for validation data
valid_dataset = torch.utils.data.TensorDataset(valid_input_data, valid_target_data)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super(CharRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the recurrent weights and biases
        self.Wxh = nn.Parameter(torch.randn(embedding_dim, hidden_size)).to(device)  # Move to device
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size)).to(device)  # Move to device
        self.bh = nn.Parameter(torch.zeros(hidden_size)).to(device)  # Move to device

        # Define the output weights and biases
        self.Why = nn.Parameter(torch.randn(hidden_size, vocab_size)).to(device)  # Move to device
        self.by = nn.Parameter(torch.zeros(vocab_size)).to(device)  # Move to device

    def forward(self, x):
        # Initialize hidden state
        h_t = torch.zeros(x.size(0), self.hidden_size).to(device)  # Move to device

        # Embedding layer
        x = self.embedding(x)

        # Unsqueeze time dimension
        x = x.unsqueeze(1)

        outputs = []
        for t in range(x.size(1)):  # Fix the indexing here
            # Input at time step t
            x_t = x[:, t, :]

            # RNN update rule
            h_t = torch.tanh(x_t @ self.Wxh + h_t @ self.Whh + self.bh)

            # Output at time step t
            y_t = h_t @ self.Why + self.by

            outputs.append(y_t)

        # Stack outputs along the time dimension
        output = torch.stack(outputs, dim=1)

        # Squeeze time dimension back
        output = output.squeeze(1)

        return output

# Custom loss function for BPC
class BPCLoss(nn.Module):
    def forward(self, output, target):
        cross_entropy = nn.CrossEntropyLoss(reduction='sum')(output, target)
        bpc = cross_entropy / (target.size(0) * np.log(2))
        return bpc

# Function to calculate accuracy
def calculate_accuracy(output, target):
    _, predicted = torch.max(output, -1)
    correct = (predicted == target)
    accuracy = torch.sum(correct).item() / target.numel()
    return accuracy

# Define different configurations
hidden_sizes = [64, 128, 256]
sequence_lengths = [50, 100, 150]

# Dictionary to store training losses for each configuration
training_losses_dict = {}

# Iterate over different hidden sizes and sequence lengths
for hidden_size in hidden_sizes:
    for seq_length in sequence_lengths:
        # Update model and dataloaders for the current configuration
        model = CharRNN(vocab_size, embedding_dim, hidden_size, num_layers).to(device)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Instantiate the model, loss function, and optimizer
        model = CharRNN(vocab_size, embedding_dim, hidden_size, num_layers).to(device)
        criterion = BPCLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # Training loop with tqdm and learning curve
        num_epochs = 10
        train_losses = []

        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            # Training phase with tqdm
            model.train()
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
            for batch_input, batch_target in progress_bar:
                optimizer.zero_grad()
                output = model(batch_input)
                batch_target = batch_target.long()
                loss = criterion(output.view(-1, vocab_size), batch_target.view(-1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                progress_bar.set_postfix({'Loss': total_loss / num_batches})

            # Calculate average loss for training data
            average_loss_train = total_loss / num_batches
            train_losses.append(average_loss_train)

        # Store training losses for the current configuration
        key = f"HiddenSize_{hidden_size}_SeqLength_{seq_length}"
        training_losses_dict[key] = train_losses

# Plot the training losses for different configurations
plt.figure(figsize=(12, 8))
for key, losses in training_losses_dict.items():
    plt.plot(losses, label=key)

plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss vs. Different Parameters')
plt.legend()
plt.savefig('training_loss_comparison.png')
plt.show()
