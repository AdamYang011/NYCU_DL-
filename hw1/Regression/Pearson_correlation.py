import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
dataset_file_path = './DL_HW1/energy_efficiency_data.csv'
df = pd.read_csv(dataset_file_path)

# Calculate Pearson correlation coefficients
correlation_matrix = df.corr()

# Assuming the target variable is 'energy_load', you can extract the correlations
correlations = correlation_matrix['Cooling Load']

# Sort the correlations in descending order to see the most important features
sorted_correlations = correlations.abs().sort_values(ascending=False)

# Print the sorted correlations
print("Pearson Correlations:\n", sorted_correlations)

# Plot a bar chart to visualize the correlations
plt.figure(figsize=(10, 6))
sorted_correlations.plot(kind='bar')
plt.title("Pearson Correlations with Energy Load")
plt.ylabel("Correlation Coefficient")
plt.show()