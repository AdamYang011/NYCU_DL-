import numpy as np

# Load the Ionosphere dataset (assuming you have it in a CSV file)
# Replace 'your_dataset.csv' with the actual file name or path to your dataset
data = np.genfromtxt('./DL_HW1/ionosphere_data.csv', delimiter=',', dtype='str')

# Extract features (first 34 columns) and labels (last column)
X = data[:, :-1].astype(np.float32)
y = np.where(data[:, -1] == 'g', 1, 0).astype(np.float32)  # Convert labels to binary: 1 for 'g' and 0 for 'b'

# Custom function to split the dataset into training and testing sets
def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    num_samples = X.shape[0]
    num_test_samples = int(test_size * num_samples)
    
    # Shuffle the data
    indices = np.random.permutation(num_samples)
    X = X[indices]
    y = y[indices]
    
    # Split into training and testing sets
    X_train, X_test = X[:-num_test_samples], X[-num_test_samples:]
    y_train, y_test = y[:-num_test_samples], y[-num_test_samples:]
    
    return X_train, X_test, y_train, y_test

# Split the dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42)

# Define sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Initialize weights and biases for the neural network
np.random.seed(42)  # For reproducibility
input_dim = X_train.shape[1]
output_dim = 1
hidden_dim = 16
learning_rate = 0.01

weights_input_hidden = np.random.randn(input_dim, hidden_dim)
bias_hidden = np.zeros((1, hidden_dim))
weights_hidden_output = np.random.randn(hidden_dim, output_dim)
bias_output = np.zeros((1, output_dim))

# Training the neural network
epochs = 1000

for epoch in range(epochs):
    # Forward propagation
    hidden_activation = np.dot(X_train, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_activation)
    
    final_activation = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_activation)
    
    # Compute cross-entropy loss
    loss = -np.mean(y_train * np.log(final_output) + (1 - y_train) * np.log(1 - final_output))
    
    # Backpropagation
    d_output = final_output - y_train
    d_hidden = np.dot(d_output, weights_hidden_output.T) * sigmoid_derivative(hidden_activation)

    # Update weights and biases
    weights_hidden_output -= learning_rate * np.dot(hidden_output.T, d_output)
    bias_output -= learning_rate * np.sum(d_output, axis=0, keepdims=True)
    weights_input_hidden -= learning_rate * np.dot(X_train.T, d_hidden)
    bias_hidden -= learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Evaluate the trained model on the test set
hidden_activation_test = np.dot(X_test, weights_input_hidden) + bias_hidden
hidden_output_test = sigmoid(hidden_activation_test)

final_activation_test = np.dot(hidden_output_test, weights_hidden_output) + bias_output
final_output_test = sigmoid(final_activation_test)

predictions = np.round(final_output_test).flatten()
accuracy = np.mean(predictions == y_test)
print(f"Accuracy on test set: {accuracy}")
