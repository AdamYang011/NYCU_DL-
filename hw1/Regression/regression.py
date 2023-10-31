import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from regression_model import NeuralNetwork

losses = []

def load_dataset(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    return data

def shuffle_dataset(data):
    np.random.shuffle(data)
    return data

def encode_categorical_features(data, orientation_dict, glazing_dict):
    encoded_orientation = np.zeros((data.shape[0], len(orientation_dict)), dtype=int)
    for i in range(data.shape[0]):
        encoded_orientation[i, orientation_dict[str(int(data[i, 5]))]] = 1
    
    original_data = data[:, :5]
    glazing_area_data = data[:, 6].reshape(-1, 1)
    predict_catagory_data = data[:, -2].reshape(-1, 1) 

    encoded_glazing = np.zeros((data.shape[0], len(glazing_dict)), dtype=int)
    for i in range(data.shape[0]):
        encoded_glazing[i, glazing_dict[str(int(data[i, 7]))]] = 1

    data = np.delete(data, [5, 7], axis=1)
    data = np.concatenate((original_data, encoded_orientation, glazing_area_data, encoded_glazing, predict_catagory_data), axis=1)
    return data

def split_dataset(data, train_ratio=0.75):
    train_size = int(train_ratio * data.shape[0])
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

def plotting_learning_curve():
    plt.plot(range(epochs), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.savefig(f'regression_learning_curve.png')
    plt.show()

def plot_label_predict(target, predictions, str):
    plt.figure(figsize=(10, 6))
    plt.plot(target, label="Labels")
    plt.plot(predictions, color='red', label="Predict")
    plt.xlabel("Data Points")
    plt.ylabel("Values")
    plt.title(f"Regression Result with {str} Labels")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{str}_regression_result.png')
    plt.show()

def calculate_rmse(targets, predictions):
    error = targets - predictions
    mse = (error ** 2).mean()
    rmse = np.sqrt(mse)
    return rmse



if __name__ == "__main__":
    dataset_file_path = './DL_HW1/energy_efficiency_data.csv'
    data = load_dataset(dataset_file_path)

    orientation_dict = {'2': 0, '3': 1, '4': 2, '5': 3}
    glazing_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}

    shuffled_data = shuffle_dataset(data)
    encoded_data = encode_categorical_features(shuffled_data, orientation_dict, glazing_dict)

    train_data, test_data = split_dataset(encoded_data, train_ratio=0.75)

    input_size = 16
    hidden_size = 8
    output_size = 1
    learning_rate = 0.00001
    model = NeuralNetwork(input_size, hidden_size, output_size, learning_rate,losses)

    epochs = 10000
    model.train(train_data, epochs)

    plotting_learning_curve()

    test_input = test_data[:, :16]
    test_target = test_data[:, 16]
    test_predictions = model.forward(test_input)
    plot_label_predict(test_target, test_predictions, "testing")

    train_input = train_data[:, :16]
    train_target = train_data[:, 16]
    train_predictions = model.forward(train_input)
    plot_label_predict(train_target, train_predictions, "training")

    train_rmse = calculate_rmse(train_target, train_predictions)
    test_rmse = calculate_rmse(test_target, test_predictions)

    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Testing RMSE: {test_rmse:.2f}")
