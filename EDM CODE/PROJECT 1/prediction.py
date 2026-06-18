import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# === Data ===
data = {
    "Pulse_On": [12,12,8,8,12,12,8,8,12,12,16,16,8,12,12,16,12,12,16,16,8,8,14,14,12,12,8,8,12,12,8,12,12,16,16,16,16,16,12,12,10,10],
    "Pulse_Off": [12,12,14,14,12,12,10,10,14,14,10,10,10,13,13,10,12,12,14,14,14,14,12,12,12,12,10,10,11,11,10,12,11,10,14,14,14,14,12,12,12,12],
    "Servo_V": [55,60,70,50,55,60,70,70,60,70,70,70,70,60,60,70,60,60,50,50,50,70,60,60,65,65,50,50,60,60,50,60,60,50,70,70,50,50,50,45,60,60],
    "Current": [4,3.5,5,5,4,3.5,5,5,4,5,5,5,5,4,4,3.5,4.5,4.5,3,3,3,3,4,4,4,4,3,3,4,4,3,4,4,5,3,3,5,5,4,4.5,4,4],
    "Angle": [60,60,30,90,60,60,90,30,60,90,30,90,30,60,60,90,60,60,90,90,90,90,60,60,60,60,30,30,75,75,30,60,60,90,30,30,30,30,30,45,60,60],
    "Surface Roughness": [4.137,3.752,2.458,3.795,4.225,3.782,2.692,3.320,3.561,3.885,3.431,3.354,2.777,3.645,3.284,3.279,3.859,3.774,4.577,4.487,2.209,2.285,4.026,3.937,3.331,3.415,4.829,3.638,3.925,4.910,3.725,4.017,5.060,4.944,2.963,3.042,4.722,4.812,3.895,3.805,3.531,3.612],
    "Material Removal Rate": [5.866,5.556,4.389,5.695,5.957,5.653,4.919,5.382,5.442,5.737,5.544,5.561,5.024,5.535,5.643,5.172,5.736,5.644,5.901,5.797,4.290,4.393,5.728,5.683,5.338,5.447,6.589,5.517,5.752,6.668,5.562,5.850,6.577,6.467,4.848,4.950,6.258,6.331,5.773,5.684,5.519,5.614],
    "Kerf width": [1.378,1.277,0.943,1.289,1.400,1.286,1.004,0.965,1.227,1.311,1.194,1.135,1.026,1.249,1.219,1.155,1.305,1.283,1.489,1.467,0.879,0.898,1.347,1.325,1.169,1.191,1.556,1.249,1.322,1.576,1.272,1.347,1.615,1.584,1.074,1.093,1.528,1.551,1.314,1.291,1.222,1.241]
}

df = pd.DataFrame(data)
exp_nos = list(range(1, len(df) + 1))

# === Inputs and Targets ===
X = df[['Pulse_On', 'Pulse_Off', 'Servo_V', 'Current', 'Angle']]

targets = {
    'Surface Roughness': df['Surface Roughness'],
    'Material Removal Rate': df['Material Removal Rate'],
    'Kerf width': df['Kerf width']
}

# === Models ===
models = {
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
}

# === Training & Prediction ===
def get_predictions(X, y, test_size=0.2):
    predictions = {}
    for name, model in models.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if name in ['Neural Network', 'SVR', 'ElasticNet', 'Ridge']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        full_pred = np.zeros(len(X))
        full_pred[X_test.index] = y_pred
        full_pred[X_train.index] = y_train.values
        predictions[name] = full_pred

    return predictions

# === Metrics ===
def calculate_metrics(y_true, predictions):
    metrics = {}
    for name, pred in predictions.items():
        rmse = np.sqrt(mean_squared_error(y_true, pred))
        mae = mean_absolute_error(y_true, pred)
        r2 = r2_score(y_true, pred)
        metrics[name] = {'RMSE': rmse, 'MAE': mae, 'R²': r2}
    return metrics

colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']

# === Run for Each Target ===
for target_name, y in targets.items():
    predictions = get_predictions(X, y)
    metrics = calculate_metrics(y, predictions)

    plt.figure(figsize=(12, 6))
    plt.plot(exp_nos, y, 'ko-', linewidth=3, markersize=8, label='Experimental')

    for i, (name, pred) in enumerate(predictions.items()):
        plt.plot(exp_nos, pred, 's-', color=colors[i], markersize=6, linewidth=2, label=name)

    plt.title(f'{target_name} Prediction',fontsize=16, fontweight='bold')
    plt.xlabel('Experiment',fontsize=14, fontweight='bold')
    plt.ylabel(target_name,fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"\n{target_name} Model Metrics:")
    print(f"{'Model':<20} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    for model, metric in metrics.items():
        print(f"{model:<20} {metric['RMSE']:>10.4f} {metric['MAE']:>10.4f} {metric['R²']:>10.4f}")
