import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('./DL_HW1/ionosphere_data.csv', header=None)

# Extract features (first 34 columns) and labels (last column)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].apply(lambda x: 1 if x == 'g' else 0).values

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

# Split the data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features (optional, but recommended)
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_train.mean()) / X_train.std()

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.loss_history = []  # Initialize an empty list to store loss values

        # Initialize weights and biases for the hidden layer and output layer
        self.W_hidden = np.random.randn(self.input_dim, self.hidden_dim)
        self.b_hidden = np.zeros((1, self.hidden_dim))
        self.W_output = np.random.randn(self.hidden_dim, self.output_dim)
        self.b_output = np.zeros((1, self.output_dim))

    def display_architecture(self):
        print("Neural Network Architecture:")
        print(f"Input layer dimension: {self.input_dim}")
        print(f"Hidden layer dimension: {self.hidden_dim}")
        print(f"Output layer dimension: {self.output_dim}")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        return -np.sum(np.log(y_pred[np.arange(m), y_true] + 1e-15)) / m

    def train(self, X, y, learning_rate=0.01, epochs=1000):
        for epoch in range(epochs):
            # Forward propagation
            hidden_output = self.sigmoid(np.dot(X, self.W_hidden) + self.b_hidden)
            output = self.softmax(np.dot(hidden_output, self.W_output) + self.b_output)

            # Compute loss
            loss = self.cross_entropy_loss(output, y)

            # Append the loss to the loss_history list
            self.loss_history.append(loss)

            # Backpropagation
            grad_output = output
            grad_output[range(X.shape[0]), y] -= 1
            grad_output /= X.shape[0]

            grad_W_output = np.dot(hidden_output.T, grad_output)
            grad_b_output = np.sum(grad_output, axis=0, keepdims=True)

            grad_hidden = np.dot(grad_output, self.W_output.T)
            grad_hidden *= self.sigmoid_derivative(hidden_output)

            grad_W_hidden = np.dot(X.T, grad_hidden)
            grad_b_hidden = np.sum(grad_hidden, axis=0, keepdims=True)

            # Update weights and biases
            self.W_hidden -= learning_rate * grad_W_hidden
            self.b_hidden -= learning_rate * grad_b_hidden
            self.W_output -= learning_rate * grad_W_output
            self.b_output -= learning_rate * grad_b_output

            # Print the loss every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        hidden_output = self.sigmoid(np.dot(X, self.W_hidden) + self.b_hidden)
        output = self.softmax(np.dot(hidden_output, self.W_output) + self.b_output)
        return np.argmax(output, axis=1)


# Define the neural network
input_dim = X_train.shape[1]
hidden_dim = 128  # Adjust as needed
output_dim = 2  # Binary classification

# Create and train the neural network
model = NeuralNetwork(input_dim, hidden_dim, output_dim)
model.display_architecture()  # Display the architecture
model.train(X_train, y_train, learning_rate=0.1, epochs=1000)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = np.mean(y_pred == y_test)
print(f'Test Accuracy: {accuracy}')

# Plot the learning curve
plt.plot(range(1, len(model.loss_history) + 1), model.loss_history, marker='.')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.grid(True)
plt.show()
