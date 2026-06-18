import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ===== DATA =====
surface_data = {
    'Exp.value': [4.137, 3.752, 2.458, 3.795, 4.225, 3.782, 2.692, 3.320, 3.561, 3.885,
                  3.431, 3.354, 2.777, 3.645, 3.284, 3.279, 3.859, 3.774, 4.577, 4.487,
                  2.209, 2.285, 4.026, 3.937, 3.331, 3.415, 4.829, 3.638, 3.925, 4.910,
                  3.725, 4.017, 5.060, 4.944, 2.963, 3.042, 4.722, 4.812, 3.895, 3.805,
                  3.531, 3.612],
    'Ensemble': [4.174,3.775,2.457,3.778,4.174,3.775,2.716,3.080,3.530,3.868,
3.426,3.316,3.080,3.501,3.501,3.299,3.856,3.856,4.513,4.513,
2.229,2.272,4.050,4.050,3.422,3.422,3.999,3.999,4.474,4.474,
3.999,3.974,4.883,4.961,3.004,3.004,4.747,4.747,3.877,3.842,
3.574,3.574]
}

srr_data = {
    'Exp.value': [5.866, 5.556, 4.389, 5.695, 5.957, 5.653, 4.919, 5.382, 5.442, 5.737,
                  5.544, 5.561, 5.024, 5.535, 5.643, 5.172, 5.736, 5.644, 5.901, 5.797,
                  4.290, 4.393, 5.728, 5.683, 5.338, 5.447, 6.589, 5.517, 5.752, 6.668,
                  5.562, 5.850, 6.577, 6.467, 4.848, 4.950, 6.258, 6.331, 5.773, 5.684,
                  5.519, 5.614],
    'Ensemble': [5.918,5.631,4.388,5.686,5.918,5.631,4.953,5.214,5.434,5.735,
5.540,5.526,5.214,5.573,5.573,5.202,5.729,5.729,5.829,5.829,
4.309,4.382,5.753,5.753,5.440,5.440,5.821,5.821,6.220,6.220,
5.821,5.813,6.467,6.485,4.899,4.899,6.277,6.277,5.759,5.725,
5.563,5.563]
}

kerf_data = {
    'Exp.value': [1.378, 1.277, 0.943, 1.289, 1.400, 1.286, 1.004, 0.965, 1.227, 1.311,
                  1.194, 1.135, 1.026, 1.249, 1.219, 1.155, 1.305, 1.283, 1.489, 1.467,
                  0.879, 0.898, 1.347, 1.325, 1.169, 1.191, 1.556, 1.249, 1.322, 1.576,
                  1.272, 1.347, 1.615, 1.584, 1.074, 1.093, 1.528, 1.551, 1.314, 1.291,
                  1.222, 1.241],
    'Ensemble': [1.378,1.277,0.943,1.289,1.400,1.286,1.004,0.965,1.227,1.311,
1.194,1.135,1.026,1.249,1.219,1.155,1.305,1.283,1.489,1.467,
0.879,0.898,1.347,1.325,1.169,1.191,1.556,1.249,1.322,1.576,
1.272,1.347,1.615,1.584,1.074,1.093,1.528,1.551,1.314,1.291,
1.222,1.241]
}

datasets = {
    "ESR": surface_data,
    "SRR": srr_data,
    "KG": kerf_data
}

num_samples = len(surface_data["Exp.value"])
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
