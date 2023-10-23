import numpy as np
import pandas as pd
from regression_model import NeuralNetwork, calculate_rms

#df = pd.read_csv("./DL_HW1/energy_efficiency_data.csv")
#print(df.head())

# Load the dataset from a CSV file
def load_dataset(file_path):
    # Assuming the CSV file has a header row and contains the data
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

    return data

# Function to shuffle the dataset
def shuffle_dataset(data):
    np.random.shuffle(data)
    return data

# Function to encode categorical features into one-hot vectors
def encode_categorical_features(data, orientation_dict, glazing_dict):
    # Encode orientation (north, south, east, west)
    encoded_orientation = np.zeros((data.shape[0], len(orientation_dict)), dtype=int)
    print("data shape: ",data.shape[0])
    for i in range(data.shape[0]):
        k = str(int(data[i, 5]))
        #if i < 5:
        #    print(data[i, 5])
        #    print(k)
        #    print(orientation_dict[k])
        #    print(i, orientation_dict[k])
        encoded_orientation[i, orientation_dict[str(int(data[i, 5]))]] = 1
    
    original_data = data[:, :5]
    glazing_area_data = data[:, 6].reshape(-1, 1)  # 轉換為列向量
    predict_catagory_data = data[:, -2].reshape(-1, 1)  # 轉換為列向量
    print(data[:, -2])
    # Encode glazing area distribution (uniform, north, south, east, west)
    encoded_glazing = np.zeros((data.shape[0], len(glazing_dict)), dtype=int)
    for i in range(data.shape[0]):
        k = str(int(data[i, 7]))
        #if i < 5:
        #    print(data[i, 7])
        #    print(k)
        #    print(glazing_dict[k])
        #    print(i, glazing_dict[k])
        #    print("\n")
        encoded_glazing[i, glazing_dict[str(int(data[i, 7]))]] = 1

    # Replace the original columns with the encoded one-hot vectors
    data = np.delete(data, [5, 7], axis=1)
    data = np.concatenate((original_data, encoded_orientation,glazing_area_data ,encoded_glazing,predict_catagory_data), axis=1)
    print(data[0])

    return data

# Function to split the dataset into training and testing sets
def split_dataset(data, train_ratio=0.75):
    train_size = int(train_ratio * data.shape[0])
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data


# Main code for training and testing the neural network
if __name__ == "__main__":
    # Load your dataset
    dataset_file_path = './DL_HW1/energy_efficiency_data.csv'
    data = load_dataset(dataset_file_path)

    # Define dictionaries for categorical feature encoding
    orientation_dict = {'2': 0, '3': 1, '4': 2, '5': 3}
    glazing_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6}

    # Shuffle the dataset
    #shuffled_data = shuffle_dataset(data)

    # Encode categorical features
    encoded_data = encode_categorical_features(data, orientation_dict, glazing_dict)

    # Split the dataset into training and testing sets
    '''train_data, test_data = split_dataset(encoded_data, train_ratio=0.75)

    # Define the neural network architecture
    input_size = data.shape[1] - 2  # Remove the two columns that were one-hot encoded
    hidden_size = 8  # Adjust the number of hidden units as needed
    output_size = 1  # Assuming a single output for heating load prediction

    # Create the neural network
    nn = NeuralNetwork(input_size, hidden_size, output_size)

    # Train the neural network
    learning_rate = 0.01
    epochs = 1000
    nn.train(train_data, learning_rate, epochs)

    # Test the neural network
    predictions = nn.predict(test_data)

    # Calculate RMS error
    targets = test_data[:, input_size:]
    rms_error = calculate_rms(predictions, targets)

    print("Root Mean Square Error (RMS):", rms_error)'''


