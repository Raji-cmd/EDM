import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ================= DATA =================
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

exp_nos = np.arange(1, len(df) + 1)

X = df[['Pulse_On', 'Pulse_Off', 'Servo_V', 'Current', 'Angle']]

targets = ["SU", "SVR", "KC"]

# ================= LSTM =================
def create_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ================= MLP NN =================
def create_nn(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ================= PREDICTION =================
def get_predictions(X, y):

    predictions = {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ================= KNN =================
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    knn_pred = knn.predict(X_test_scaled)

    # ================= RANDOM FOREST =================
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)

    # ================= NN (MLP) =================
    nn = create_nn(X_train_scaled.shape[1])
    nn.fit(
        X_train_scaled, y_train,
        epochs=80,
        batch_size=8,
        verbose=0,
        validation_split=0.2
    )
    nn_pred = nn.predict(X_test_scaled, verbose=0).flatten()

    # ================= LSTM =================
    X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

    lstm = create_lstm((X_train_lstm.shape[1], 1))

    lstm.fit(
        X_train_lstm, y_train,
        epochs=100,
        batch_size=8,
        verbose=0,
        validation_split=0.2,
        callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]
    )

    lstm_pred = lstm.predict(X_test_lstm, verbose=0).flatten()

    # ================= FULL LENGTH ALIGN =================
    for name, pred in zip(
        ['KNN', 'RF', 'NN', 'LSTM'],
        [knn_pred, rf_pred, nn_pred, lstm_pred]
    ):
        full_pred = np.zeros(len(y))
        full_pred[X_test.index] = pred
        full_pred[X_train.index] = y_train.values  # keep train actual

        predictions[name] = full_pred

    return predictions

# ================= METRICS =================
def calculate_metrics(y_true, predictions):

    metrics = {}

    for name, pred in predictions.items():

        metrics[name] = {
            'RMSE': np.sqrt(mean_squared_error(y_true, pred)),
            'MAE': mean_absolute_error(y_true, pred),
            'R²': r2_score(y_true, pred)
        }

    return metrics

# ================= COLORS =================
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# ================= RUN =================
for target in targets:

    y = df[target]

    predictions = get_predictions(X, y)
    metrics = calculate_metrics(y, predictions)

    plt.figure(figsize=(12, 6))

    plt.plot(exp_nos, y, 'ko-', linewidth=3, label='Experimental')

    for i, (name, pred) in enumerate(predictions.items()):
        plt.plot(exp_nos, pred, 's-', color=colors[i], label=name)

    plt.title(f'{target} Prediction', fontsize=16, fontweight='bold')
    plt.xlabel('Experiment', fontsize=14, fontweight='bold')
    plt.ylabel(target, fontsize=14, fontweight='bold')

    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"\n{target}")
    print(f"{'Model':<10}{'RMSE':>10}{'MAE':>10}{'R²':>10}")

    for m, v in metrics.items():
        print(f"{m:<10}{v['RMSE']:>10.4f}{v['MAE']:>10.4f}{v['R²']:>10.4f}")