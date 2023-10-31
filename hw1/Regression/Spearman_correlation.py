import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def cal_spearman_correlations(data, feature_names, target_column):
    spearman_correlations = pd.DataFrame()

    for feature_name in feature_names:
        feature = data[feature_name]
        rank_feature = feature.rank()
        rank_target = data[target_column].rank()
        rank_diff = rank_feature - rank_target
        n = len(feature)
        correlation = 1 - 6 * (sum(rank_diff ** 2)) / (n * (n ** 2 - 1))
        spearman_correlations[feature_name] = [correlation]

    return spearman_correlations

def plot_spearman_heatmap(correlations, target_name, save_filename):
    plt.figure(figsize=(10, 6))
    plt.imshow(correlations.values, cmap='coolwarm', aspect='auto')
    plt.xticks(np.arange(correlations.shape[1]), correlations.columns, rotation=45, ha="right")
    plt.yticks([])
    plt.colorbar(label="Spearman Correlation")
    plt.title(f"Spearman Correlations Heatmap ({target_name})")
    plt.savefig(save_filename)
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv('./DL_HW1/energy_efficiency_data.csv')

    target1_name = data.columns[-2]
    target2_name = data.columns[-1]

    feature_names = data.columns[:8]

    correlations_heat = cal_spearman_correlations(data, feature_names, target1_name)
    plot_spearman_heatmap(correlations_heat, target1_name, "Spearman_Correlations_heat.png")
    print(f"Spearman Correlations ({target1_name}):\n{correlations_heat.to_string(index=False)}")

    correlations_cool = cal_spearman_correlations(data, feature_names, target2_name)
    plot_spearman_heatmap(correlations_cool, target2_name, "Spearman_Correlations_cool.png")
