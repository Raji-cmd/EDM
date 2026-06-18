import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# === Data ===
data = {
    "Pulse On": [12, 12, 8, 8, 12, 12, 8, 8, 12, 12, 16, 16, 8, 12, 12, 16, 12, 12, 16, 16, 8, 8, 14, 14, 12, 12, 8, 8,
                 12, 12, 8, 12, 12, 16, 16, 16, 16, 16, 12, 12, 10, 10],
    "Pulse Off": [12, 12, 14, 14, 12, 12, 10, 10, 14, 14, 10, 10, 10, 13, 13, 10, 12, 12, 14, 14, 14, 14, 12, 12, 12,
                  12, 10, 10, 11, 11, 10, 12, 11, 10, 14, 14, 14, 14, 12, 12, 12, 12],
    "Servo Voltage": [55, 60, 70, 50, 55, 60, 70, 70, 60, 70, 70, 70, 70, 60, 60, 70, 60, 60, 50, 50, 50, 70, 60, 60,
                      65, 65, 50, 50, 60, 60, 50, 60, 60, 50, 70, 70, 50, 50, 50, 45, 60, 60],
    "Current": [4, 3.5, 5, 5, 4, 3.5, 5, 5, 4, 5, 5, 5, 5, 4, 4, 3.5, 4.5, 4.5, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 4, 4, 3,
                4, 4, 5, 3, 3, 5, 5, 4, 4.5, 4, 4],
    "Angle": [60, 60, 30, 90, 60, 60, 90, 30, 60, 90, 30, 90, 30, 60, 60, 90, 60, 60, 90, 90, 90, 90, 60, 60, 60, 60,
              30, 30, 75, 75, 30, 60, 60, 90, 30, 30, 30, 30, 30, 45, 60, 60],
    "Su": [4.174, 3.775, 2.457, 3.778, 4.174, 3.775, 2.716, 3.080, 3.530, 3.868,
            3.426, 3.316, 3.080, 3.501, 3.501, 3.299, 3.856, 3.856, 4.513, 4.513,
            2.229, 2.272, 4.050, 4.050, 3.422, 3.422, 3.999, 3.999, 4.474, 4.474,
            3.999, 3.974, 4.883, 4.961, 3.004, 3.004, 4.747, 4.747, 3.877, 3.842,
            3.574, 3.574],
    "SVR": [5.918, 5.631, 4.388, 5.686, 5.918, 5.631, 4.953, 5.214, 5.434, 5.735,
            5.540, 5.526, 5.214, 5.573, 5.573, 5.202, 5.729, 5.729, 5.829, 5.829,
            4.309, 4.382, 5.753, 5.753, 5.440, 5.440, 5.821, 5.821, 6.220, 6.220,
            5.821, 5.813, 6.467, 6.485, 4.899, 4.899, 6.277, 6.277, 5.759, 5.725,
            5.563, 5.563],
    "KC": [1.378, 1.277, 0.943, 1.289, 1.400, 1.286, 1.004, 0.965, 1.227, 1.311,
           1.194, 1.135, 1.026, 1.249, 1.219, 1.155, 1.305, 1.283, 1.489, 1.467,
           0.879, 0.898, 1.347, 1.325, 1.169, 1.191, 1.556, 1.249, 1.322, 1.576,
           1.272, 1.347, 1.615, 1.584, 1.074, 1.093, 1.528, 1.551, 1.314, 1.291,
           1.222, 1.241]
}

df = pd.DataFrame(data)

# === Correct Feature Names ===
X = df[["Pulse On", "Pulse Off", "Servo Voltage", "Current", "Angle"]]

# === Correct Target Mapping ===
targets = {
    'Su': df['Su'],
    'SVR': df['SVR'],
    'KC': df['KC']
}

# === Train RF model & plot feature importance for each target ===
for target_name, y in targets.items():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)

    plt.figure(figsize=(8, 6))
    sns.barplot(
        x=importance.values,
        y=importance.index,
        palette="viridis",
        hue=importance.index,
        legend=False
    )

    plt.title(f"Random Forest Feature Importance\n{target_name}", fontsize=16, fontweight='bold')
    plt.xlabel("Importance Score", fontsize=14, fontweight='bold')
    plt.ylabel("Feature", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
