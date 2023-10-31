import numpy as np
import matplotlib.pyplot as plt
from PCA import PCA

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.loss_history = [] 
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

            if epoch == 10 or epoch == 9999:
                #print(f'Epoch {epoch}, Loss: {loss}')
                
                # Record hidden layer activations
                hidden_activations = self.sigmoid(np.dot(X, self.W_hidden) + self.b_hidden)
                
                # Apply PCA for dimensionality reduction
                pca = PCA(n_components=2)
                pca.fit(hidden_activations)
                hidden_activations_pca = pca.transform(hidden_activations)

                # Create a scatter plot of the latent features
                plt.figure(figsize=(8, 6))
                plt.scatter(hidden_activations_pca[y == 0, 0], hidden_activations_pca[y == 0, 1], label='Class 0', alpha=0.5, color='red')
                plt.scatter(hidden_activations_pca[y == 1, 0], hidden_activations_pca[y == 1, 1], label='Class 1', alpha=0.5, color='blue')
                plt.xlabel('Principal Component 1')
                plt.ylabel('Principal Component 2')
                plt.legend()
                plt.title(f'PCA of Latent Features in the Hidden Layer (Epoch {epoch})')
                plt.savefig(f'latent_features_epoch_{epoch}_{self.hidden_dim}.png')
                plt.show()
            # Print the loss every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        hidden_output = self.sigmoid(np.dot(X, self.W_hidden) + self.b_hidden)
        output = self.softmax(np.dot(hidden_output, self.W_output) + self.b_output)
        return np.argmax(output, axis=1)