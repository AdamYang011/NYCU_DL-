import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import io
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

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

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)

        # Define the output weights and biases
        self.Why = nn.Parameter(torch.randn(hidden_size, vocab_size)).to(device)
        self.by = nn.Parameter(torch.zeros(vocab_size)).to(device)

    def forward(self, x):
        # Add an extra dimension to the input tensor
        x = x.unsqueeze(1)

        # Initialize hidden state and cell state
        h_t = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c_t = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Embedding layer
        x = self.embedding(x)

        # LSTM layer
        lstm_out, _ = self.lstm(x, (h_t, c_t))

        # Output layer
        output = lstm_out @ self.Why + self.by

        return output


# Assuming 'device' is defined as the desired device (e.g., 'cuda' or 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom loss function for BPC
class BPCLoss(nn.Module):
    def forward(self, output, target):
        cross_entropy = nn.CrossEntropyLoss(reduction='sum')(output, target)
        bpc = cross_entropy / (target.size(0) * np.log(2))
        return bpc

# Function to calculate accuracy
def calculate_accuracy(output, target):
    _, predicted = torch.max(output, -1)
    correct = (predicted.squeeze() == target)
    '''print("correct:", torch.sum(correct).item())
    print("target.numel ", target.numel())
    print("predicted.numel ", predicted.numel())
    print("Shape of predicted:", predicted.shape)
    print("Shape of target:", target.shape)
    print("Shape of correct:", correct.shape)'''
    accuracy = torch.sum(correct).item() / target.numel()
    #print("accuracy: ",accuracy)
    return accuracy

def print_first_batch_output(model, dataloader, vocab_size):
    model.eval()
    with torch.no_grad():
        batch_input, batch_target = next(iter(dataloader))
        output = model(batch_input)
        output = output.view(-1, vocab_size).cpu().numpy()
        predicted_indices = np.argmax(output, axis=1)
        predicted_text = ''.join([int_to_vocab_train[idx] for idx in predicted_indices])
        print(f"Output for the first batch:\n{predicted_text}\n")

def generate_text(model, seed_sequence, length=100, temperature=1.0):
    model.eval()
    with torch.no_grad():
        current_sequence = seed_sequence
        input_data = torch.tensor([train_vocab_to_int[c] for c in current_sequence], dtype=torch.long).to(device)
        input_data = input_data.unsqueeze(0).unsqueeze(0).to(torch.float32)  # Add batch and sequence dimensions

        h_t = torch.zeros(model.num_layers, 1, model.hidden_size).to(device)
        c_t = torch.zeros(model.num_layers, 1, model.hidden_size).to(device)

        for _ in range(length):
            # Ensure input_data has the correct size
            input_data = input_data[:, :, :model.embedding.embedding_dim]

            # Pad input_data if needed
            if input_data.size(-1) < model.embedding.embedding_dim:
                pad_size = model.embedding.embedding_dim - input_data.size(-1)
                input_data = F.pad(input_data, (0, pad_size), 'constant', 0)

            output, (h_t, c_t) = model.lstm(input_data, (h_t, c_t))

            # Use Gumbel-Softmax for temperature-scaled sampling
            scaled_logits = output[0, -1, :] / temperature
            probabilities = nn.functional.gumbel_softmax(scaled_logits, tau=1.0, hard=True).cpu().numpy()

            # Choose the index based on the probabilities
            next_char_index = np.random.choice(vocab_size, p=probabilities)

            # Sample the next character from the output distribution
            next_char_index = np.random.choice(np.arange(vocab_size), p=probabilities)
            next_char = int_to_vocab_train[next_char_index]

            current_sequence += next_char

            # Update input_data for the next iteration
            input_data = torch.tensor([train_vocab_to_int[c] for c in next_char], dtype=torch.long).to(device)
            input_data = input_data.unsqueeze(0).unsqueeze(0).to(torch.float32)  # Add batch and sequence dimensions

        return current_sequence

# Instantiate the model, loss function, and optimizer
model = CharLSTM(vocab_size, embedding_dim, hidden_size, num_layers).to(device)
criterion = BPCLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop with tqdm and learning curve
num_epochs = 10
train_bpcs = []  # To store training BPC for each epoch
train_accuracies = []  # To store training accuracy for each epoch
valid_bpcs = []  # To store validation BPC for each epoch
valid_accuracies = []  # To store validation accuracy for each epoch
train_losses = []
valid_losses = []
checkpoint_epochs = [0, 2, 4, 6, 9]

# Define a seed sequence for text generation
seed_sequence = "MENENIUS: Come, come, you have been too rough, something too rough; You must return and mend it."

def generate_and_print_text(model, seed_sequence, length=100, temperature=1.0):
    generated_text = generate_text(model, seed_sequence, length, temperature)
    print(f"Generated text:\n{generated_text}\n")

for epoch in range(num_epochs):
    total_loss = 0
    total_accuracy = 0
    num_batches = 0

    # Training phase with tqdm
    model.train()
    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
    for batch_input, batch_target in progress_bar:
        optimizer.zero_grad()
        output = model(batch_input)
        # Cast target to torch.long
        batch_target = batch_target.long()
        loss = criterion(output.view(-1, vocab_size), batch_target.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        accuracy = calculate_accuracy(output, batch_target)
        total_accuracy += accuracy
        num_batches += 1
        progress_bar.set_postfix({'Loss': total_loss / num_batches, 'Accuracy': total_accuracy / num_batches})

    # Calculate average loss and accuracy for training data
    average_loss_train = total_loss / num_batches
    average_accuracy_train = total_accuracy / num_batches
    train_losses.append(average_loss_train)
    train_accuracies.append(average_accuracy_train)

    # Calculate and store BPC for training data
    train_bpc = average_loss_train / (batch_size * np.log(2))
    train_bpcs.append(train_bpc)

    # Validation phase
    total_loss_valid = 0
    total_accuracy_valid = 0
    num_batches_valid = 0
    model.eval()
    with torch.no_grad():
        for batch_input_valid, batch_target_valid in valid_dataloader:
            output_valid = model(batch_input_valid)
            # Cast target to torch.long
            batch_target_valid = batch_target_valid.long()
            loss_valid = criterion(output_valid.view(-1, vocab_size), batch_target_valid.view(-1))
            total_loss_valid += loss_valid.item()
            accuracy_valid = calculate_accuracy(output_valid, batch_target_valid)
            total_accuracy_valid += accuracy_valid
            num_batches_valid += 1

    # Calculate average loss and accuracy for validation data
    average_loss_valid = total_loss_valid / num_batches_valid
    average_accuracy_valid = total_accuracy_valid / num_batches_valid
    valid_losses.append(average_loss_valid)
    valid_accuracies.append(average_accuracy_valid)

    # Calculate and store BPC for validation data
    valid_bpc = average_loss_valid / (batch_size * np.log(2))
    valid_bpcs.append(valid_bpc)

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {average_loss_train:.4f}, Train Accuracy: {average_accuracy_train:.4f}, '
          f'Valid Loss: {average_loss_valid:.4f}, Valid Accuracy: {average_accuracy_valid:.4f}')

    if epoch in checkpoint_epochs:
        print(f'Epoch [{epoch+1}/{num_epochs}], '
                f'Train Loss: {average_loss_train:.4f}, Train Accuracy: {average_accuracy_train:.4f}, '
                f'Train BPC: {train_bpc:.4f}, '
                f'Valid Loss: {average_loss_valid:.4f}, Valid Accuracy: {average_accuracy_valid:.4f}, '
                f'Valid BPC: {valid_bpc:.4f}')
        
        print_first_batch_output(model, train_dataloader, vocab_size)

# Save the trained model if needed
torch.save(model.state_dict(), 'char_lstm_model.pth')

# Plot the learning curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_bpcs, label='Training BPC')
ax1.plot(valid_bpcs, label='Validation BPC')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('BPC')
ax1.legend()
plt.savefig('learning_curve_BPC_LSTM.png')
plt.show()

ax2.plot(train_accuracies, label='Training Accuracy')
ax2.plot(valid_accuracies, label='Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
plt.savefig('learning_curve_Accuracy_LSTM.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.title('Learning Curve - Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('learning_curve_loss_LSTM.png')
plt.show()


