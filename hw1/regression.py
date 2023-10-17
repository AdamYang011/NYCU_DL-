import numpy as np
import pandas as pd

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
    for i in range(data.shape[0]):
        k = str(int(data[i, 5]))
        print(data[i, 5])
        print(k)
        print(orientation_dict[k])
        print(i, orientation_dict[k])
        encoded_orientation[i, orientation_dict[str(int(data[i, 5]))]] = 1

    # Encode glazing area distribution (uniform, north, south, east, west)
    encoded_glazing = np.zeros((data.shape[0], len(glazing_dict)), dtype=int)
    for i in range(data.shape[0]):
        k = str(int(data[i, 7]))
        print(data[i, 7])
        print(k)
        print(glazing_dict[k])
        print(i, glazing_dict[k])
        encoded_glazing[i, glazing_dict[str(int(data[i, 7]))]] = 1

    # Replace the original columns with the encoded one-hot vectors
    data = np.delete(data, [6, 7], axis=1)
    data = np.concatenate((data, encoded_orientation, encoded_glazing), axis=1)

    return data

# Function to split the dataset into training and testing sets
def split_dataset(data, train_ratio=0.75):
    train_size = int(train_ratio * data.shape[0])
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

# Main code for data preprocessing
if __name__ == "__main__":
    # Load your dataset
    dataset_file_path = './DL_HW1/energy_efficiency_data.csv'
    data = load_dataset(dataset_file_path)
    # Assuming data is loaded as a numpy array with appropriate dimensions

    # Define dictionaries for categorical feature encoding
    orientation_dict = {'2': 0, '3': 1, '4': 2, '5': 3}
    glazing_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6}

    # Shuffle the dataset
    shuffled_data = shuffle_dataset(data)

    # Encode categorical features
    encoded_data = encode_categorical_features(shuffled_data, orientation_dict, glazing_dict)

    # Split the dataset into training and testing sets
    train_data, test_data = split_dataset(encoded_data, train_ratio=0.75)
    print(train_data[0])
    # Further preprocessing as needed...
