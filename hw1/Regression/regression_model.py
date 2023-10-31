import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, losses):
        self.weights_input_hidden = np.full((input_size, hidden_size), 0.5)
        self.bias_hidden = np.full((1, hidden_size), 0.5)
        self.weights_hidden_output = np.full((hidden_size, output_size), 0.5)
        self.bias_output = np.full((1, output_size), 0.5)
        self.learning_rate = learning_rate
        self.losses = losses

    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.leaky_relu(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.leaky_relu(self.final_input)
        return self.final_output

    def backward(self, X, y, output):
        error = y - output
        delta_output = error
        error_hidden = delta_output.dot(self.weights_hidden_output.T)
        delta_hidden = error_hidden * self.leaky_relu_derivative(self.hidden_output)

        X = X.reshape(1, -1)
        self.weights_hidden_output += self.learning_rate * self.hidden_output.T.dot(delta_output)
        self.bias_output += self.learning_rate * np.sum(delta_output, axis=0, keepdims=True)
        self.weights_input_hidden += self.learning_rate * X.T.dot(delta_hidden)
        self.bias_hidden += self.learning_rate * np.sum(delta_hidden, axis=0)

    def train(self, train_data, epochs):
        for i in range(epochs):
            total_loss = 0
            for j in range(train_data.shape[0]):
                input_data = train_data[j, :16]  
                target = train_data[j, 16]  
                output = self.forward(input_data)
                loss = 0.5 * np.square(target - output).sum()  
                total_loss += loss
                self.backward(input_data, target, output)

            average_loss = total_loss / train_data.shape[0]
            RMSE = average_loss ** 0.5
            self.losses.append(RMSE)

            if i % 1000 == 0:
                print(f"Epoch {i}, Loss: {total_loss / train_data.shape[0]}")