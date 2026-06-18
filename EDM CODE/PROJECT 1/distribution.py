import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ===== DATA =====
data = {
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
    "ESR": [4.137, 3.752, 2.458, 3.795, 4.225, 3.782, 2.692, 3.320, 3.561, 3.885, 3.431, 3.354, 2.777,
                          3.645, 3.284, 3.279, 3.859, 3.774, 4.577, 4.487, 2.209, 2.285, 4.026, 3.937, 3.331, 3.415,
                          4.829, 3.638, 3.925, 4.910, 3.725, 4.017, 5.060, 4.944, 2.963, 3.042, 4.722, 4.812, 3.895,
                          3.805, 3.531, 3.612],
    "SRR": [5.866, 5.556, 4.389, 5.695, 5.957, 5.653, 4.919, 5.382, 5.442, 5.737, 5.544, 5.561, 5.024, 5.535, 5.643,
            5.172, 5.736, 5.644, 5.901, 5.797, 4.290, 4.393, 5.728, 5.683, 5.338, 5.447, 6.589, 5.517, 5.752, 6.668,
            5.562, 5.850, 6.577, 6.467, 4.848, 4.950, 6.258, 6.331, 5.773, 5.684, 5.519, 5.614],
    "KG": [1.378, 1.277, 0.943, 1.289, 1.400, 1.286, 1.004, 0.965, 1.227, 1.311, 1.194, 1.135, 1.026, 1.249,
                   1.219, 1.155, 1.305, 1.283, 1.489, 1.467, 0.879, 0.898, 1.347, 1.325, 1.169, 1.191, 1.556, 1.249,
                   1.322, 1.576, 1.272, 1.347, 1.615, 1.584, 1.074, 1.093, 1.528, 1.551, 1.314, 1.291, 1.222, 1.241]
}
# ===== DATAFRAME =====
df = pd.DataFrame(data)
targets = ['ESR', 'SRR', 'KG']

# ===== COLOR SCHEME =====
colors = {
    'ESR': ('#FFD6D6', '#C0392B'),   # Light Red → Dark Red
    'SRR': ('#D4F1F4', '#2471A3'),   # Light Cyan → Steel Blue
    'KG': ('#FFF2CC', '#B7950B')     # Light Yellow → Golden Brown

}

# ===== TITLE AND AXIS PLOT =====
for target in targets:
    train, test = train_test_split(df[target], test_size=0.2, random_state=42)

    plt.figure(figsize=(6, 4))
    plt.hist(train, bins=8, alpha=0.7, color=colors[target][0], edgecolor='black', label='Train')
    plt.hist(test, bins=8, alpha=0.7, color=colors[target][1], edgecolor='black', label='Test')
    plt.title(f'{target} Distribution', fontsize=16, fontweight='bold')
    plt.xlabel(target, fontsize=14, fontweight='bold')
    plt.ylabel('Frequency', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
