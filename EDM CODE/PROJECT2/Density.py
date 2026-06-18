import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ===== DATASET =====
surface_data = {
    'Exp.value': [4.137, 3.752, 2.458, 3.795, 4.225, 3.782, 2.692, 3.320, 3.561, 3.885,
                  3.431, 3.354, 2.777, 3.645, 3.284, 3.279, 3.859, 3.774, 4.577, 4.487,
                  2.209, 2.285, 4.026, 3.937, 3.331, 3.415, 4.829, 3.638, 3.925, 4.910,
                  3.725, 4.017, 5.060, 4.944, 2.963, 3.042, 4.722, 4.812, 3.895, 3.805,
                  3.531, 3.612],
    'Ensemble': [4.174,3.775,2.457,3.778,4.174,3.775,2.716,3.080,3.530,3.868,
3.426,3.316,3.080,3.501,3.501,3.299,3.856,3.856,4.513,4.513,
2.229,2.272,4.050,4.050,3.422,3.422,3.999,3.999,4.474,4.474,
3.999,3.974,4.883,4.961,3.004,3.004,4.747,4.747,3.877,3.842,
3.574,3.574]
}

srr_data = {
    'Exp.value': [5.866, 5.556, 4.389, 5.695, 5.957, 5.653, 4.919, 5.382, 5.442, 5.737,
                  5.544, 5.561, 5.024, 5.535, 5.643, 5.172, 5.736, 5.644, 5.901, 5.797,
                  4.290, 4.393, 5.728, 5.683, 5.338, 5.447, 6.589, 5.517, 5.752, 6.668,
                  5.562, 5.850, 6.577, 6.467, 4.848, 4.950, 6.258, 6.331, 5.773, 5.684,
                  5.519, 5.614],
    'Ensemble': [5.918,5.631,4.388,5.686,5.918,5.631,4.953,5.214,5.434,5.735,
5.540,5.526,5.214,5.573,5.573,5.202,5.729,5.729,5.829,5.829,
4.309,4.382,5.753,5.753,5.440,5.440,5.821,5.821,6.220,6.220,
5.821,5.813,6.467,6.485,4.899,4.899,6.277,6.277,5.759,5.725,
5.563,5.563]
}

kerf_data = {
    'Exp.value': [1.378, 1.277, 0.943, 1.289, 1.400, 1.286, 1.004, 0.965, 1.227, 1.311,
                  1.194, 1.135, 1.026, 1.249, 1.219, 1.155, 1.305, 1.283, 1.489, 1.467,
                  0.879, 0.898, 1.347, 1.325, 1.169, 1.191, 1.556, 1.249, 1.322, 1.576,
                  1.272, 1.347, 1.615, 1.584, 1.074, 1.093, 1.528, 1.551, 1.314, 1.291,
                  1.222, 1.241],
    'Ensemble': [1.378,1.277,0.943,1.289,1.400,1.286,1.004,0.965,1.227,1.311,
1.194,1.135,1.026,1.249,1.219,1.155,1.305,1.283,1.489,1.467,
0.879,0.898,1.347,1.325,1.169,1.191,1.556,1.249,1.322,1.576,
1.272,1.347,1.615,1.584,1.074,1.093,1.528,1.551,1.314,1.291,
1.222,1.241]
}

# ===== DATAFRAME =====
datasets = {
    "ESR": pd.DataFrame(surface_data),
    "SRR": pd.DataFrame(srr_data),
    "KG": pd.DataFrame(kerf_data)
}

# ===== Colors SCHEME =====
colors = ['#2ca02c', '#9467bd']  # Green: Actual, Purple: Predicted


# ===== PLOT & SUMMARY =====
def plot_and_summarize(df, metric_name):
    plt.figure(figsize=(6, 4))

    sns.kdeplot(df['Exp.value'], label='Actual', color=colors[0], linewidth=2, fill=True, alpha=0.5, bw_adjust=0.5)
    sns.kdeplot(df['Ensemble'], label='Predicted', color=colors[1], linewidth=2, fill=True, alpha=0.5, bw_adjust=0.5)

    # ===== TITLE & AXIS =====
    plt.title(f"{metric_name}: Actual vs Predicted Density", fontsize=14, fontweight='bold')
    plt.xlabel(metric_name, fontsize=14, fontweight='bold')
    plt.ylabel('Density', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.legend(fontsize=10, title_fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.show()

    # ===== SUMMARY =====
    print(f"\n=== {metric_name} Statistical Summary ===")
    print(f"Actual   - Mean: {df['Exp.value'].mean():.4f}, Std: {df['Exp.value'].std():.4f}")
    print(f"Predicted- Mean: {df['Ensemble'].mean():.4f}, Std: {df['Ensemble'].std():.4f}")

    # Kolmogorov-Smirnov test
    ks_stat, p_value = stats.ks_2samp(df['Exp.value'], df['Ensemble'])
    print(f"Kolmogorov-Smirnov Test: Stat={ks_stat:.4f}, p-value={p_value:.4f}")
    if p_value > 0.05:
        print("Distributions are statistically similar (p > 0.05)")
    else:
        print("Distributions are statistically different (p ≤ 0.05)")

# ===== APPLY FUNCTION =====
for name, df in datasets.items():
    plot_and_summarize(df, name)
