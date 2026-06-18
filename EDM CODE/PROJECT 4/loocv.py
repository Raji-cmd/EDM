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
    "SW": [
        4.165, 3.817, 2.510, 3.730, 4.165, 3.817, 2.768, 3.095, 3.536, 3.824,
        3.442, 3.363, 3.095, 3.504, 3.504, 3.348, 3.833, 3.833, 4.466, 4.466,
        2.342, 2.300, 4.018, 4.018, 3.409, 3.409, 4.005, 4.005, 4.497, 4.497,
        4.005, 3.937, 4.777, 4.886, 3.037, 3.037, 4.698, 4.698, 3.914, 3.912,
        3.579, 3.579
    ],

    "MVR": [
        5.910, 5.647, 4.468, 5.631, 5.910, 5.647, 5.002, 5.206, 5.427, 5.687,
        5.539, 5.507, 5.206, 5.553, 5.553, 5.245, 5.728, 5.728, 5.811, 5.811,
        4.386, 4.402, 5.758, 5.758, 5.423, 5.423, 5.833, 5.833, 6.198, 6.198,
        5.833, 5.789, 6.339, 6.427, 4.917, 4.917, 6.246, 6.246, 5.767, 5.769,
        5.570, 5.570
    ],

    "KD": [
        1.366, 1.279, 0.961, 1.279, 1.366, 1.279, 1.018, 1.004, 1.236, 1.289,
        1.193, 1.151, 1.004, 1.227, 1.227, 1.179, 1.278, 1.278, 1.473, 1.473,
        0.928, 0.899, 1.328, 1.328, 1.180, 1.180, 1.349, 1.349, 1.457, 1.457,
        1.349, 1.310, 1.517, 1.568, 1.095, 1.095, 1.537, 1.537, 1.319, 1.319,
        1.219, 1.219
    ]

})


X = df[['Pulse_On', 'Pulse_Off', 'Servo_V', 'Current', 'Angle']]
y1 = df['SW']
y2 = df['MVR']
y3 = df['KD']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

loo = LeaveOneOut()

# ===== LOO PREDICTION =====
preds = { "sw": [], "mvr": [], "kd": [] }
trues = { "sw": [], "mvr": [], "kd": [] }

for train_idx, test_idx in loo.split(X_scaled):

    model_r = RandomForestRegressor(n_estimators=200, random_state=42)
    model_r.fit(X_scaled[train_idx], y1.iloc[train_idx])
    preds["sw"].append(model_r.predict(X_scaled[test_idx])[0])
    trues["sw"].append(y1.iloc[test_idx].values[0])

    model_m = RandomForestRegressor(n_estimators=200, random_state=42)
    model_m.fit(X_scaled[train_idx], y2.iloc[train_idx])
    preds["mvr"].append(model_m.predict(X_scaled[test_idx])[0])
    trues["mvr"].append(y2.iloc[test_idx].values[0])

    model_k = RandomForestRegressor(n_estimators=200, random_state=42)
    model_k.fit(X_scaled[train_idx], y3.iloc[train_idx])
    preds["kd"].append(model_k.predict(X_scaled[test_idx])[0])
    trues["kd"].append(y3.iloc[test_idx].values[0])

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

plot_single(trues["sw"], preds["sw"], "LOO CV:SW")
plot_single(trues["mvr"], preds["mvr"], "LOO CV: MVR")
plot_single(trues["kd"], preds["kd"], "LOO CV: KD")
