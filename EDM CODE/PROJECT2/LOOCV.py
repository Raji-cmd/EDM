import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut

# ===== DATA =====
df = pd.DataFrame({
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
    "ESR": [4.174, 3.775, 2.457, 3.778, 4.174, 3.775, 2.716, 3.080, 3.530, 3.868,
            3.426, 3.316, 3.080, 3.501, 3.501, 3.299, 3.856, 3.856, 4.513, 4.513,
            2.229, 2.272, 4.050, 4.050, 3.422, 3.422, 3.999, 3.999, 4.474, 4.474,
            3.999, 3.974, 4.883, 4.961, 3.004, 3.004, 4.747, 4.747, 3.877, 3.842,
            3.574, 3.574],
    "SRR": [5.918, 5.631, 4.388, 5.686, 5.918, 5.631, 4.953, 5.214, 5.434, 5.735,
            5.540, 5.526, 5.214, 5.573, 5.573, 5.202, 5.729, 5.729, 5.829, 5.829,
            4.309, 4.382, 5.753, 5.753, 5.440, 5.440, 5.821, 5.821, 6.220, 6.220,
            5.821, 5.813, 6.467, 6.485, 4.899, 4.899, 6.277, 6.277, 5.759, 5.725,
            5.563, 5.563],
    "KG": [1.378, 1.277, 0.943, 1.289, 1.400, 1.286, 1.004, 0.965, 1.227, 1.311,
           1.194, 1.135, 1.026, 1.249, 1.219, 1.155, 1.305, 1.283, 1.489, 1.467,
           0.879, 0.898, 1.347, 1.325, 1.169, 1.191, 1.556, 1.249, 1.322, 1.576,
           1.272, 1.347, 1.615, 1.584, 1.074, 1.093, 1.528, 1.551, 1.314, 1.291,
           1.222, 1.241]
})


X = df[['Pulse_On', 'Pulse_Off', 'Servo_V', 'Current', 'Angle']]
y1 = df['ESR']
y2 = df['SRR']
y3 = df['KG']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

loo = LeaveOneOut()

# ===== LOO PREDICTION =====
preds = { "rough": [], "srr": [], "kerf": [] }
trues = { "rough": [], "srr": [], "kerf": [] }

for train_idx, test_idx in loo.split(X_scaled):

    model_r = RandomForestRegressor(n_estimators=200, random_state=42)
    model_r.fit(X_scaled[train_idx], y1.iloc[train_idx])
    preds["rough"].append(model_r.predict(X_scaled[test_idx])[0])
    trues["rough"].append(y1.iloc[test_idx].values[0])

    model_m = RandomForestRegressor(n_estimators=200, random_state=42)
    model_m.fit(X_scaled[train_idx], y2.iloc[train_idx])
    preds["srr"].append(model_m.predict(X_scaled[test_idx])[0])
    trues["srr"].append(y2.iloc[test_idx].values[0])

    model_k = RandomForestRegressor(n_estimators=200, random_state=42)
    model_k.fit(X_scaled[train_idx], y3.iloc[train_idx])
    preds["kerf"].append(model_k.predict(X_scaled[test_idx])[0])
    trues["kerf"].append(y3.iloc[test_idx].values[0])

# ===== TITLE AND AXIS =====
def plot_single(y_true, y_pred, title):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, s=80)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Actual", fontsize=14, fontweight='bold')
    plt.ylabel("Predicted", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

plot_single(trues["rough"], preds["rough"], "LOO CV: ESR")
plot_single(trues["srr"], preds["srr"], "LOO CV: SRR")
plot_single(trues["kerf"], preds["kerf"], "LOO CV: KG")
