import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score


# ===== DATA =====
data = {
    "Pulse_On": [12,12,8,8,12,12,8,8,12,12,16,16,8,12,12,16,12,12,16,16,8,8,14,14,12,12,8,8,12,12,8,12,12,16,16,16,16,16,12,12,10,10],
    "Pulse_Off": [12,12,14,14,12,12,10,10,14,14,10,10,10,13,13,10,12,12,14,14,14,14,12,12,12,12,10,10,11,11,10,12,11,10,14,14,14,14,12,12,12,12],
    "Servo_V": [55,60,70,50,55,60,70,70,60,70,70,70,70,60,60,70,60,60,50,50,50,70,60,60,65,65,50,50,60,60,50,60,60,50,70,70,50,50,50,45,60,60],
    "Current": [4,3.5,5,5,4,3.5,5,5,4,5,5,5,5,4,4,3.5,4.5,4.5,3,3,3,3,4,4,4,4,3,3,4,4,3,4,4,5,3,3,5,5,4,4.5,4,4],
    "Angle": [60,60,30,90,60,60,90,30,60,90,30,90,30,60,60,90,60,60,90,90,90,90,60,60,60,60,30,30,75,75,30,60,60,90,30,30,30,30,30,45,60,60],
    "Surface Roughness": [4.137,3.752,2.458,3.795,4.225,3.782,2.692,3.320,3.561,3.885,3.431,3.354,2.777,3.645,3.284,3.279,3.859,3.774,4.577,4.487,2.209,2.285,4.026,3.937,3.331,3.415,4.829,3.638,3.925,4.910,3.725,4.017,5.060,4.944,2.963,3.042,4.722,4.812,3.895,3.805,3.531,3.612],
    "MRR": [5.866,5.556,4.389,5.695,5.957,5.653,4.919,5.382,5.442,5.737,5.544,5.561,5.024,5.535,5.643,5.172,5.736,5.644,5.901,5.797,4.290,4.393,5.728,5.683,5.338,5.447,6.589,5.517,5.752,6.668,5.562,5.850,6.577,6.467,4.848,4.950,6.258,6.331,5.773,5.684,5.519,5.614],
    "Kerf": [1.378,1.277,0.943,1.289,1.400,1.286,1.004,0.965,1.227,1.311,1.194,1.135,1.026,1.249,1.219,1.155,1.305,1.283,1.489,1.467,0.879,0.898,1.347,1.325,1.169,1.191,1.556,1.249,1.322,1.576,1.272,1.347,1.615,1.584,1.074,1.093,1.528,1.551,1.314,1.291,1.222,1.241]
}

df = pd.DataFrame(data)
input_features = ['Pulse_On', 'Pulse_Off', 'Servo_V', 'Current', 'Angle']
targets = ["Surface Roughness", "MRR", "Kerf"]

# ===== MODEL CONFIGURATION =====
model_configs = {
    "Surface Roughness": {
        "XGB": XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            random_state=42
        ),
        "SVR": SVR(kernel='rbf', C=50, epsilon=0.1),
        "RF": RandomForestRegressor(
            n_estimators=300, max_depth=10, max_features='sqrt',
            random_state=42
        ),
        "NN": MLPRegressor(
            hidden_layer_sizes=(8,4,2), activation='relu',
            alpha=0.001, learning_rate_init=0.01,
            max_iter=2000, random_state=42
        )
    },
    "MRR": {
        "XGB": XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            random_state=42
        ),
        "SVR": SVR(kernel='rbf', C=50, epsilon=0.1),
        "RF": RandomForestRegressor(
            n_estimators=300, max_depth=10, max_features='sqrt',
            random_state=42
        ),
        "NN": MLPRegressor(
            hidden_layer_sizes=(16, 10, 6), activation='relu',
            alpha=0.001, learning_rate_init=0.01,
            max_iter=2000, random_state=42
        )
    },
    "Kerf": {
        "XGB": XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            random_state=42
        ),
        "SVR": SVR(kernel='rbf', C=40, epsilon=0.05),
        "RF": RandomForestRegressor(
            n_estimators=300, max_depth=10, max_features='sqrt',
            random_state=42
        ),
        "NN": MLPRegressor(
            hidden_layer_sizes=(12, 6, 3), activation='relu',
            alpha=0.001, learning_rate_init=0.01,
            max_iter=2000, random_state=42
        )
    }
}

results = {}

# ===== TRAINING =====
for target in targets:
    X = df[input_features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    row = {}

    for name, model in model_configs[target].items():
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        r2 = round(r2_score(y, y_pred), 3)
        row[name] = {"hyperparameters": model.get_params(), "R²": r2}

        print(f"{target} - {name}: R² = {r2}")

    results[target] = row
    print("-" * 30)

# ===== SAVE IN EXCEL =====
flattened_rows = []
for target, model_dict in results.items():
    row = {"Output": target}
    for model_name, metrics in model_dict.items():
        row[f"{model_name} Hyperparameter"] = str(metrics["hyperparameters"])
        row[f"{model_name} R²"] = metrics["R²"]
    flattened_rows.append(row)

final_excel_df = pd.DataFrame(flattened_rows)
final_excel_df.to_excel("Hyperparameter_All_Data.xlsx", index=False)

print("\nResults saved to Hyperparameter_All_Data.xlsx")
print(final_excel_df)
