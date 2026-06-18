import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np


# ===== DATA =====
data = {
    "Pulse_On": [12,12,8,8,12,12,8,8,12,12,16,16,8,12,12,16,12,12,16,16,8,8,14,14,12,12,8,8,12,12,8,12,12,16,16,16,16,16,12,12,10,10],
    "Pulse_Off": [12,12,14,14,12,12,10,10,14,14,10,10,10,13,13,10,12,12,14,14,14,14,12,12,12,12,10,10,11,11,10,12,11,10,14,14,14,14,12,12,12,12],
    "Servo_V": [55,60,70,50,55,60,70,70,60,70,70,70,70,60,60,70,60,60,50,50,50,70,60,60,65,65,50,50,60,60,50,60,60,50,70,70,50,50,50,45,60,60],
    "Current": [4,3.5,5,5,4,3.5,5,5,4,5,5,5,5,4,4,3.5,4.5,4.5,3,3,3,3,4,4,4,4,3,3,4,4,3,4,4,5,3,3,5,5,4,4.5,4,4],
    "Angle": [60,60,30,90,60,60,90,30,60,90,30,90,30,60,60,90,60,60,90,90,90,90,60,60,60,60,30,30,75,75,30,60,60,90,30,30,30,30,30,45,60,60],
    "ESR": [4.174,3.775,2.457,3.778,4.174,3.775,2.716,3.080,3.530,3.868,
3.426,3.316,3.080,3.501,3.501,3.299,3.856,3.856,4.513,4.513,
2.229,2.272,4.050,4.050,3.422,3.422,3.999,3.999,4.474,4.474,
3.999,3.974,4.883,4.961,3.004,3.004,4.747,4.747,3.877,3.842,
3.574,3.574],
    "SRR": [5.918,5.631,4.388,5.686,5.918,5.631,4.953,5.214,5.434,5.735,
5.540,5.526,5.214,5.573,5.573,5.202,5.729,5.729,5.829,5.829,
4.309,4.382,5.753,5.753,5.440,5.440,5.821,5.821,6.220,6.220,
5.821,5.813,6.467,6.485,4.899,4.899,6.277,6.277,5.759,5.725,
5.563,5.563],
    "KG": [1.378,1.277,0.943,1.289,1.400,1.286,1.004,0.965,1.227,1.311,
1.194,1.135,1.026,1.249,1.219,1.155,1.305,1.283,1.489,1.467,
0.879,0.898,1.347,1.325,1.169,1.191,1.556,1.249,1.322,1.576,
1.272,1.347,1.615,1.584,1.074,1.093,1.528,1.551,1.314,1.291,
1.222,1.241]
}

df = pd.DataFrame(data)

input_features = ['Pulse_On', 'Pulse_Off', 'Servo_V', 'Current', 'Angle']
targets = ["ESR", "SRR", "KG"]

# ===== MODEL CONFIGURATION =====
model_configs = {
    "ESR": {
        "XGB": XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42
        ),
        "SVR": SVR(kernel='rbf', C=50, epsilon=0.1),
        "KNN": KNeighborsRegressor(n_neighbors=5)
    },

    "SRR": {
        "XGB": XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42
        ),
        "SVR": SVR(kernel='rbf', C=50, epsilon=0.1),
        "KNN": KNeighborsRegressor(n_neighbors=4)
    },

    "KG": {
        "XGB": XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42
        ),
        "SVR": SVR(kernel='rbf', C=40, epsilon=0.05),
        "KNN": KNeighborsRegressor(n_neighbors=3)
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

    # ===== ML MODELS =====
    for name, model in model_configs[target].items():

        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)

        r2 = round(r2_score(y, y_pred), 3)

        row[name] = {
            "hyperparameters": model.get_params(),
            "R²": r2
        }

        print(f"{target} - {name}: R² = {r2}")

    # ===== LSTM MODEL =====
    X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

    lstm_model = Sequential([
        LSTM(32, activation='relu', input_shape=(1, X_scaled.shape[1])),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    lstm_model.compile(optimizer='adam', loss='mse')

    lstm_model.fit(
        X_lstm,
        y,
        epochs=200,
        batch_size=8,
        verbose=0
    )

    y_pred_lstm = lstm_model.predict(X_lstm, verbose=0).flatten()

    r2_lstm = round(r2_score(y, y_pred_lstm), 3)

    row["LSTM"] = {
        "hyperparameters": {
            "LSTM_units": 32,
            "Dense_units": 16,
            "epochs": 200,
            "batch_size": 8,
            "optimizer": "adam"
        },
        "R²": r2_lstm
    }

    print(f"{target} - LSTM: R² = {r2_lstm}")

    results[target] = row

    print("-" * 40)

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