import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from classification_model import NeuralNetwork

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


if __name__ == "__main__":
    data = pd.read_csv('./DL_HW1/ionosphere_data.csv', header=None)

    # Extract features (first 34 columns) and labels (last column)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].replace({'g': 1, 'b': 0}).values

    # Split the data into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the features (optional, but recommended)
    X_train = (X_train - X_train.mean()) / X_train.std()
    X_test = (X_test - X_train.mean()) / X_train.std()

    # Define the neural network
    input_dim = X_train.shape[1]
    hidden_dim = 128  
    output_dim = 2

    model = NeuralNetwork(input_dim, hidden_dim, output_dim)
    model.display_architecture() 
    model.train(X_train, y_train, learning_rate=0.1, epochs=10000)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Predict on the training set
    y_train_pred = model.predict(X_train)

    # Calculate the training error rate
    train_error_rate = 1 - np.mean(y_train_pred == y_train)
    print(f'Training Error Rate: {train_error_rate}')

    # Predict on the test set
    y_test_pred = model.predict(X_test)

    # Calculate the test error rate
    test_error_rate = 1 - np.mean(y_test_pred == y_test)
    print(f'Test Error Rate: {test_error_rate}')

    # Evaluate the model
    accuracy = np.mean(y_pred == y_test)
    print(f'Test Accuracy: {accuracy}')

    # Plot the learning curve
    plt.plot(range(1, len(model.loss_history) + 1), model.loss_history, marker='.')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.grid(True)
    plt.savefig(f'classification_learning_curve_{hidden_dim}.png')
    plt.show()
