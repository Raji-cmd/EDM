import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ===== DATA =====
datasets = {
    "SU": pd.read_excel('EDM3.xlsx', sheet_name='Surface Undulation Su'),
    "SVR": pd.read_excel('EDM3.xlsx', sheet_name='Substance Vaporization Rate SVR'),
    "KC": pd.read_excel('EDM3.xlsx', sheet_name='Kerf Clearence KC')
}

num_samples = len(next(iter(datasets.values())))
X = np.arange(num_samples).reshape(-1, 1)

# ===== TRAIN AND PLOT  =====
def train_and_plot_loss(y_ensemble, target_name):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    es = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)

    history = model.fit(
        X, y_ensemble,
        epochs=500,
        batch_size=4,
        verbose=0,
        callbacks=[es]
    )
    loss = np.array(history.history['loss'])
    smooth_loss = np.convolve(loss, np.ones(10)/10, mode='valid')

    # ===== TITLE AND AXIS =====
    plt.figure(figsize=(6, 3.8))
    plt.plot(smooth_loss, linewidth=1.5)
    plt.xlabel("Epochs", fontsize=14, fontweight="bold")
    plt.ylabel("Mean Square Error", fontsize=14, fontweight="bold")
    plt.title(f"{target_name} - Training Loss", fontsize=16, fontweight="bold")
    plt.xticks(fontweight="bold", fontsize=12)
    plt.yticks(fontweight="bold", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# ===== SUMMARY =====
for name, data_dict in datasets.items():
    y_ensemble = np.array(data_dict["Ensemble"])
    train_and_plot_loss(y_ensemble, name)
