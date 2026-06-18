import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# INPUT PARAMETERS
# =====================================================

data = {
    "Pulse_On": [12,12,8,8,12,12,8,8,12,12,16,16,8,12,12,16,12,12,16,16,8,8,14,14,12,12,8,8,12,12,8,12,12,16,16,16,16,16,12,12,10,10],
    "Pulse_Off": [12,12,14,14,12,12,10,10,14,14,10,10,10,13,13,10,12,12,14,14,14,14,12,12,12,12,10,10,11,11,10,12,11,10,14,14,14,14,12,12,12,12],
    "Servo_V": [55,60,70,50,55,60,70,70,60,70,70,70,70,60,60,70,60,60,50,50,50,70,60,60,65,65,50,50,60,60,50,60,60,50,70,70,50,50,50,45,60,60],
    "Current": [4,3.5,5,5,4,3.5,5,5,4,5,5,5,5,4,4,3.5,4.5,4.5,3,3,3,3,4,4,4,4,3,3,4,4,3,4,4,5,3,3,5,5,4,4.5,4,4],
    "Angle": [60,60,30,90,60,60,90,30,60,90,30,90,30,60,60,90,60,60,90,90,90,90,60,60,60,60,30,30,75,75,30,60,60,90,30,30,30,30,30,45,60,60]
}

df = pd.DataFrame(data)

# =====================================================
# ENSEMBLE PREDICTIONS
# =====================================================

SU = np.array([
4.129,3.796,2.507,3.664,4.129,3.796,2.767,3.053,3.490,3.748,
3.425,3.314,3.053,3.553,3.553,3.349,3.898,3.898,4.500,4.500,
2.428,2.317,4.031,4.031,3.460,3.460,4.044,4.044,4.355,4.355,
4.044,3.940,4.595,4.895,3.010,3.010,4.719,4.719,3.865,3.917,
3.611,3.611
])

SVR = np.array([
5.879,5.630,4.483,5.593,5.879,5.630,5.023,5.209,5.423,5.611,
5.515,5.490,5.209,5.584,5.584,5.307,5.784,5.784,5.828,5.828,
4.525,4.415,5.777,5.777,5.483,5.483,5.895,5.895,6.118,6.118,
5.895,5.788,6.263,6.442,4.900,4.900,6.247,6.247,5.767,5.827,
5.561,5.561
])

KC = np.array([
1.367,1.282,0.955,1.259,1.367,1.282,1.000,0.994,1.203,1.275,
1.188,1.152,0.994,1.234,1.234,1.180,1.304,1.304,1.476,1.476,
0.932,0.915,1.336,1.336,1.207,1.207,1.356,1.356,1.442,1.442,
1.356,1.325,1.494,1.566,1.089,1.089,1.528,1.528,1.308,1.330,
1.242,1.242
])

df["SU_Ensemble"] = SU
df["SVR_Ensemble"] = SVR
df["KC_Ensemble"] = KC

# =====================================================
# PARETO FRONT IDENTIFICATION
# SU -> MINIMIZE
# SVR -> MAXIMIZE
# =====================================================

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


scores = np.column_stack([SU, -SVR])

pareto_mask = identify_pareto(scores)

pareto_points = np.column_stack([
    SU[pareto_mask],
    SVR[pareto_mask]
])

pareto_points = pareto_points[
    pareto_points[:, 0].argsort()
]

# =====================================================
# SPECIAL SOLUTIONS
# =====================================================

best_quality_idx = np.argmin(SU)

best_productivity_idx = np.argmax(SVR)

balanced_idx = np.argmin(
    np.sqrt(
        (SU - SU.min())**2 +
        (SVR - SVR.max())**2
    )
)

# =====================================================
# PLOT
# =====================================================

plt.figure(figsize=(10, 8))

plt.scatter(
    SU,
    SVR,
    color='lightgray',
    s=60,
    alpha=0.8,
    label='All Solutions'
)

plt.scatter(
    pareto_points[:, 0],
    pareto_points[:, 1],
    color='#4C78A8',
    edgecolors='black',
    linewidth=1,
    s=90,
    label='Pareto-optimal Solutions'
)

plt.plot(
    pareto_points[:, 0],
    pareto_points[:, 1],
    '--',
    color='#4C78A8',
    linewidth=2
)

special = [
    (best_quality_idx, "Best Quality", 's', '#E45756'),
    (best_productivity_idx, "Best Productivity", '^', '#54A24B'),
    (balanced_idx, "Balanced", 'D', '#B279A2')
]

for idx, name, marker, color in special:

    x = SU[idx]
    y = SVR[idx]

    plt.scatter(
        x,
        y,
        marker=marker,
        s=180,
        color=color,
        edgecolors='black',
        linewidth=2
    )

    plt.annotate(
        f"{name}\nSU={x:.3f}\nSVR={y:.3f}",
        xy=(x, y),
        xytext=(20, -40),
        textcoords='offset points',
        fontsize=10,
        bbox=dict(
            facecolor='white',
            alpha=0.85
        ),
        arrowprops=dict(
            arrowstyle='->'
        )
    )

plt.xlabel(
    "SU (µm)",
    fontsize=14,
    fontweight='bold'
)

plt.ylabel(
    "SVR",
    fontsize=14,
    fontweight='bold'
)

plt.title(
    "Pareto Frontier Based on Ensemble Predictions\n(Minimize SU and Maximize SVR)",
    fontsize=16,
    fontweight='bold'
)

plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

plt.grid(True, linestyle='--', alpha=0.4)

plt.legend(fontsize=11)

plt.tight_layout()
plt.show()

# =====================================================
# SUMMARY TABLE
# =====================================================

print("\n" + "=" * 70)
print("PARETO OPTIMIZATION SUMMARY")
print("=" * 70)

print(f"\nTotal Pareto-optimal solutions : {pareto_mask.sum()}")

summary_cols = [
    "Pulse_On",
    "Pulse_Off",
    "Servo_V",
    "Current",
    "Angle",
    "SU_Ensemble",
    "SVR_Ensemble",
    "KC_Ensemble"
]

summary = pd.DataFrame({
    "Parameter": summary_cols,
    "Best Quality (Min SU)": df.loc[best_quality_idx, summary_cols].values,
    "Best Productivity (Max SVR)": df.loc[best_productivity_idx, summary_cols].values,
    "Balanced": df.loc[balanced_idx, summary_cols].values
})

print("\n")
print(summary.to_string(index=False))

print("\nBest Quality Experiment :", best_quality_idx + 1)
print("Best Productivity Experiment :", best_productivity_idx + 1)
print("Balanced Experiment :", balanced_idx + 1)