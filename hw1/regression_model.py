import numpy as np

# Define sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
	return np.maximum(0.0, x)

def relu_derivative(x):
    np.where(x >= 0, 1, 0)

# Define the neural network architecture
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        #print(self.weights_input_hidden)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.bias_output = np.zeros((1, output_size))

    def forward(self, inputs):
        self.hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = relu(self.hidden_input)
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        return self.output
    
    def backward(self, inputs, targets, learning_rate):
        error = -(targets - self.output)
        d_output = error
        error_hidden = d_output.dot(self.weights_hidden_output.T)
        d_hidden = error_hidden * relu_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += inputs.reshape(-1,1).dot(d_hidden) * learning_rate
        self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    def train(self, train_data, learning_rate):
        for data_point in train_data:
            inputs = data_point[:self.input_size]
            targets = data_point[self.input_size]
            self.forward(inputs)
            self.backward(inputs, targets, learning_rate)

    def predict(self, test_data):
        predictions = []
        for data_point in test_data:
            inputs = data_point[:self.input_size]
            output = self.forward(inputs)
            predictions.append(output)
        return np.array(predictions)

# Root Mean Square Error (RMS) function
def calculate_rms(predictions, targets):
    #print("predictions: ",predictions)
    #print("targets: ", targets)
    error = predictions - targets
    #print("error: ", error)
    mse = np.mean(error ** 2)
    rms = np.sqrt(mse)
    return rms