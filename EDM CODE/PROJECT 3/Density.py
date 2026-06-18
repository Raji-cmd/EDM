import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ===== DATASET =====
df = pd.read_excel('EDM3.xlsx')

datasets = {
    "SU": pd.read_excel('EDM3.xlsx', sheet_name='Surface Undulation Su'),
    "SVR": pd.read_excel('EDM3.xlsx', sheet_name='Substance Vaporization Rate SVR'),
    "KC": pd.read_excel('EDM3.xlsx', sheet_name='Kerf Clearence KC')
}

# ===== Colors SCHEME =====
colors = ['#2ca02c', '#9467bd']  # Green: Actual, Purple: Predicted


# ===== PLOT & SUMMARY =====
def plot_and_summarize(df, metric_name):
    plt.figure(figsize=(6, 4))

    sns.kdeplot(df['Exp.Value'], label='Actual', color=colors[0], linewidth=2, fill=True, alpha=0.5, bw_adjust=0.5)
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
    print(f"Actual   - Mean: {df['Exp.Value'].mean():.4f}, Std: {df['Exp.Value'].std():.4f}")
    print(f"Predicted- Mean: {df['Ensemble'].mean():.4f}, Std: {df['Ensemble'].std():.4f}")

    # Kolmogorov-Smirnov test
    ks_stat, p_value = stats.ks_2samp(df['Exp.Value'], df['Ensemble'])
    print(f"Kolmogorov-Smirnov Test: Stat={ks_stat:.4f}, p-value={p_value:.4f}")
    if p_value > 0.05:
        print("Distributions are statistically similar (p > 0.05)")
    else:
        print("Distributions are statistically different (p ≤ 0.05)")

# ===== APPLY FUNCTION =====
for name, df in datasets.items():
    plot_and_summarize(df, name)
