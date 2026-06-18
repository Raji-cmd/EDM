import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut

# ===== DATA =====
data = {
    "Pulse_On": [12,12,8,8,12,12,8,8,12,12,16,16,8,12,12,16,12,12,16,16,8,8,14,14,12,12,8,8,12,12,8,12,12,16,16,16,16,16,12,12,10,10],
    "Pulse_Off": [12,12,14,14,12,12,10,10,14,14,10,10,10,13,13,10,12,12,14,14,14,14,12,12,12,12,10,10,11,11,10,12,11,10,14,14,14,14,12,12,12,12],
    "Servo_V": [55,60,70,50,55,60,70,70,60,70,70,70,70,60,60,70,60,60,50,50,50,70,60,60,65,65,50,50,60,60,50,60,60,50,70,70,50,50,50,45,60,60],
    "Current": [4,3.5,5,5,4,3.5,5,5,4,5,5,5,5,4,4,3.5,4.5,4.5,3,3,3,3,4,4,4,4,3,3,4,4,3,4,4,5,3,3,5,5,4,4.5,4,4],
    "Angle": [60,60,30,90,60,60,90,30,60,90,30,90,30,60,60,90,60,60,90,90,90,90,60,60,60,60,30,30,75,75,30,60,60,90,30,30,30,30,30,45,60,60],
    "SU": [4.174,3.775,2.457,3.778,4.174,3.775,2.716,3.080,3.530,3.868,
3.426,3.316,3.080,3.501,3.501,3.299,3.856,3.856,4.513,4.513,
2.229,2.272,4.050,4.050,3.422,3.422,3.999,3.999,4.474,4.474,
3.999,3.974,4.883,4.961,3.004,3.004,4.747,4.747,3.877,3.842,
3.574,3.574],
    "SVR": [5.918,5.631,4.388,5.686,5.918,5.631,4.953,5.214,5.434,5.735,
5.540,5.526,5.214,5.573,5.573,5.202,5.729,5.729,5.829,5.829,
4.309,4.382,5.753,5.753,5.440,5.440,5.821,5.821,6.220,6.220,
5.821,5.813,6.467,6.485,4.899,4.899,6.277,6.277,5.759,5.725,
5.563,5.563],
    "KC": [1.378,1.277,0.943,1.289,1.400,1.286,1.004,0.965,1.227,1.311,
1.194,1.135,1.026,1.249,1.219,1.155,1.305,1.283,1.489,1.467,
0.879,0.898,1.347,1.325,1.169,1.191,1.556,1.249,1.322,1.576,
1.272,1.347,1.615,1.584,1.074,1.093,1.528,1.551,1.314,1.291,
1.222,1.241]
}
df = pd.DataFrame(data)
# ===== INPUTS =====
X = df[['Pulse_On', 'Pulse_Off', 'Servo_V', 'Current', 'Angle']]

# ===== OUTPUTS =====
y1 = df['SU']
y2 = df['SVR']
y3 = df['KC']

# ===== SCALING =====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===== LOO =====
loo = LeaveOneOut()

preds = {
    "SU": [],
    "SVR": [],
    "KC": []
}

trues = {
    "SU": [],
    "SVR": [],
    "KC": []
}

# ===== LOO CROSS VALIDATION =====
for train_idx, test_idx in loo.split(X_scaled):

    # SU
    model_su = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model_su.fit(
        X_scaled[train_idx],
        y1.iloc[train_idx]
    )

    preds["SU"].append(
        model_su.predict(X_scaled[test_idx])[0]
    )

    trues["SU"].append(
        y1.iloc[test_idx].values[0]
    )

    # SVR
    model_svr = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model_svr.fit(
        X_scaled[train_idx],
        y2.iloc[train_idx]
    )

    preds["SVR"].append(
        model_svr.predict(X_scaled[test_idx])[0]
    )

    trues["SVR"].append(
        y2.iloc[test_idx].values[0]
    )

    # KC
    model_kc = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model_kc.fit(
        X_scaled[train_idx],
        y3.iloc[train_idx]
    )

    preds["KC"].append(
        model_kc.predict(X_scaled[test_idx])[0]
    )

    trues["KC"].append(
        y3.iloc[test_idx].values[0]
    )

# ===== PLOT FUNCTION =====
def plot_single(y_true, y_pred, title):

    plt.figure(figsize=(6, 6))

    sns.scatterplot(
        x=y_true,
        y=y_pred,
        s=80
    )

    mn = min(min(y_true), min(y_pred))
    mx = max(max(y_true), max(y_pred))

    plt.plot(
        [mn, mx],
        [mn, mx],
        'r--',
        linewidth=2
    )

    plt.title(
        title,
        fontsize=16,
        fontweight='bold'
    )

    plt.xlabel(
        "Actual",
        fontsize=14,
        fontweight='bold'
    )

    plt.ylabel(
        "Predicted",
        fontsize=14,
        fontweight='bold'
    )

    plt.xticks(
        fontsize=12,
        fontweight='bold'
    )

    plt.yticks(
        fontsize=12,
        fontweight='bold'
    )

    plt.grid(
        True,
        linestyle='--',
        alpha=0.5
    )

    plt.tight_layout()
    plt.show()

# ===== PLOTS =====
plot_single(trues["SU"], preds["SU"], "LOO CV: Surface Undulation")

plot_single(trues["SVR"], preds["SVR"], "LOO CV: Substance Vaporization Rate")

plot_single(trues["KC"], preds["KC"], "LOO CV: Kerf Clearance")