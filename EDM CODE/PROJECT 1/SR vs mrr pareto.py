import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# ===== DATA =====

data = {
    "Pulse_On": [12, 12, 8, 8, 12, 12, 8, 8, 12, 12, 16, 16, 8, 12, 12, 16, 12, 12, 16, 16, 8, 8, 14, 14, 12, 12, 8, 8,
                 12, 12, 8, 12, 12, 16, 16, 16, 16, 16, 12, 12, 10, 10],
    "Pulse_Off": [12, 12, 14, 14, 12, 12, 10, 10, 14, 14, 10, 10, 10, 13, 13, 10, 12, 12, 14, 14, 14, 14, 12, 12, 12,
                  12, 10, 10, 11, 11, 10, 12, 11, 10, 14, 14, 14, 14, 12, 12, 12, 12],
    "Servo_V": [55, 60, 70, 50, 55, 60, 70, 70, 60, 70, 70, 70, 70, 60, 60, 70, 60, 60, 50, 50, 50, 70, 60, 60, 65, 65,
                50, 50, 60, 60, 50, 60, 60, 50, 70, 70, 50, 50, 50, 45, 60, 60],
    "Current": [4, 3.5, 5, 5, 4, 3.5, 5, 5, 4, 5, 5, 5, 5, 4, 4, 3.5, 4.5, 4.5, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 4, 4, 3,
                4, 4, 5, 3, 3, 5, 5, 4, 4.5, 4, 4],
    "Angle": [60, 60, 30, 90, 60, 60, 90, 30, 60, 90, 30, 90, 30, 60, 60, 90, 60, 60, 90, 90, 90, 90, 60, 60, 60, 60,
              30, 30, 75, 75, 30, 60, 60, 90, 30, 30, 30, 30, 30, 45, 60, 60],
    "Surface Roughness": [4.137, 3.752, 2.458, 3.795, 4.225, 3.782, 2.692, 3.320, 3.561, 3.885, 3.431, 3.354, 2.777,
                          3.645, 3.284, 3.279, 3.859, 3.774, 4.577, 4.487, 2.209, 2.285, 4.026, 3.937, 3.331, 3.415,
                          4.829, 3.638, 3.925, 4.910, 3.725, 4.017, 5.060, 4.944, 2.963, 3.042, 4.722, 4.812, 3.895,
                          3.805, 3.531, 3.612],
    "MRR": [5.866, 5.556, 4.389, 5.695, 5.957, 5.653, 4.919, 5.382, 5.442, 5.737, 5.544, 5.561, 5.024, 5.535, 5.643,
            5.172, 5.736, 5.644, 5.901, 5.797, 4.290, 4.393, 5.728, 5.683, 5.338, 5.447, 6.589, 5.517, 5.752, 6.668,
            5.562, 5.850, 6.577, 6.467, 4.848, 4.950, 6.258, 6.331, 5.773, 5.684, 5.519, 5.614]
}

df = pd.DataFrame(data)
SR = df["Surface Roughness"].values
mrr = df["MRR"].values

# ===== PARETO FRONTIER IDENTIFICATION =====
def identify_pareto(scores):
    is_efficient = np.ones(scores.shape[0], dtype=bool)
    for i, c in enumerate(scores):
        if is_efficient[i]:
            is_efficient[is_efficient] = (
                np.any(scores[is_efficient] < c, axis=1) |
                np.any(scores[is_efficient] > c, axis=1)
            )
            is_efficient[i] = True
    return is_efficient

scores = np.column_stack([SR, -mrr])
pareto_mask = identify_pareto(scores)

pareto_points = np.column_stack([SR[pareto_mask], mrr[pareto_mask]])
pareto_points = pareto_points[pareto_points[:, 0].argsort()]
best_quality_idx = np.argmin(SR)
best_productivity_idx = np.argmax(mrr)
balanced_idx = np.argmin(np.sqrt((SR - SR.min())**2 + (mrr - mrr.max())**2))


# =============== Plotting Pareto Front ===================
plt.figure(figsize=(10, 8))
plt.scatter(SR, mrr, color='gray', alpha=0.6, s=50, label="Experimental Points")

plt.scatter(pareto_points[:, 0], pareto_points[:, 1],
            color='blue', s=80, edgecolors='black', linewidth=1,
            label="Pareto-optimal Solutions")

plt.plot(pareto_points[:, 0], pareto_points[:, 1], 'b--', lw=1.6)
special = [
    (best_quality_idx, "Best Quality", 's', 'red'),
    (best_productivity_idx, "Best Productivity", '^', 'green'),
    (balanced_idx, "Balanced", 'D', 'purple')
]

for idx, name, marker, color in special:
    x, y = SR[idx], mrr[idx]
    plt.scatter(x, y, s=150, marker=marker, color=color,
                edgecolors='black', linewidth=2)
    plt.annotate(f"{name}\nSR={x:.2f}, MRR={y:.2f}",
                 xy=(x, y), xytext=(20, -40),
                 textcoords='offset points', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8),
                 arrowprops=dict(arrowstyle='->'))

plt.xlabel("Surface Roughness (µm)", fontsize=14, fontweight="bold")
plt.ylabel("Material Removal Rate (mm³/min)", fontsize=14, fontweight="bold")
plt.xticks(fontsize=12, fontweight="bold")
plt.yticks(fontsize=12, fontweight="bold")
plt.title("Pareto Frontier for EDM Process\nTrade-off between SR and MRR"
          , fontsize=16, fontweight="bold")
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

# ===== SUMMARY =====
print("\n=========== Summary ===========")
print(f"Total Pareto-optimal solutions = {pareto_mask.sum()}")

best_quality = df.iloc[best_quality_idx]
best_productivity = df.iloc[best_productivity_idx]
balanced = df.iloc[balanced_idx]

summary = pd.DataFrame({
    "Parameter": df.columns,
    "Best Quality (Min SR)": best_quality.values,
    "Best Productivity (Max MRR)": best_productivity.values,
    "Balanced": balanced.values
})

print(summary.to_string(index=False))
