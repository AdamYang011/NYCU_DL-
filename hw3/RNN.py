import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Load data
def load_data(data_path):
    with open(data_path, 'r', encoding='utf8') as f:
        text = f.read()

    # Character's collection
    vocab = list(set(text))

    # Construct character dictionary
    vocab_to_int = {c: i for i, c in enumerate(vocab)}
    int_to_vocab = {i: c for i, c in enumerate(vocab)}

    # Encode data, shape = [number of characters]
    data = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

    return data, vocab_to_int, int_to_vocab

# Construct RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

# Training function
def train(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        loss_total = 0
        # Initialize hidden state at the beginning of each epoch
        hidden = model.init_hidden(batch_size)

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            hidden = hidden.detach()  # detach hidden state to prevent backprop through time
            output, hidden = model(inputs, hidden)
            loss = criterion(output.view(-1, output_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            loss_total += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss_total / len(train_loader)}')

# Validation function
def validate(model, valid_loader, criterion):
    model.eval()
    hidden = model.init_hidden(batch_size)
    loss_total = 0
    with torch.no_grad():
        for inputs, targets in valid_loader:
            output, hidden = model(inputs, hidden)
            loss = criterion(output.view(-1, output_size), targets.view(-1))
            loss_total += loss.item()

    return loss_total / len(valid_loader)

# Hyperparameters
train_data, vocab_to_int, int_to_vocab = load_data('shakespeare_train.txt')
valid_data, _, _ = load_data('shakespeare_valid.txt')
input_size = len(vocab_to_int)
hidden_size = 128
output_size = len(vocab_to_int)
learning_rate = 0.01
batch_size = 64
epochs = 10

# Load data and create DataLoader
train_dataset = TensorDataset(torch.from_numpy(train_data[:-1]), torch.from_numpy(train_data[1:]))
valid_dataset = TensorDataset(torch.from_numpy(valid_data[:-1]), torch.from_numpy(valid_data[1:]))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, optimizer, and criterion
model = SimpleRNN(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training
train(model, train_loader, criterion, optimizer, epochs=epochs)

# Validation
validation_loss = validate(model, valid_loader, criterion)
print(f'Validation Loss: {validation_loss}')
