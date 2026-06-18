import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingRegressor,RandomForestRegressor
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
# ==========================
# LOAD EXCEL DATA
# ==========================
df = pd.read_excel("EDM3.xlsx")


# ===== LEARNING CURVE FUNCTION =====
def plot_learning_curve(data, title, ylabel):
    X = np.column_stack([
        data['KNN'],
        data['LSTM'],
        data['RandomForest'],
        data['NeuralNet']
    ])
    y = np.array(data['Exp.Value'])

    rf = RandomForestRegressor(random_state=0)
    nn = MLPRegressor(random_state=0, max_iter=1000)
    knn = KNeighborsRegressor(n_neighbors=5)
    enet = ElasticNet()

    ensemble = VotingRegressor([
        ('rf', rf),
        ('knn', knn),
        ('enet', enet)
    ])

    train_sizes, train_scores, test_scores = learning_curve(
        ensemble, X, y, cv=5, scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=-1
    )

    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation Score')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='g')
    plt.xlabel("Training Examples", fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.title(f"Learning Curve - Ensemble {title}", fontsize=16, fontweight='bold')
    plt.legend(loc="best", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()

import pandas as pd

# Read each sheet
su_data = pd.read_excel("EDM3.xlsx", sheet_name="Surface Undulation Su")
svr_data = pd.read_excel("EDM3.xlsx", sheet_name="Substance Vaporization Rate SVR")
kc_data  = pd.read_excel("EDM3.xlsx", sheet_name="Kerf Clearence KC")

plot_learning_curve(su_data, "SU", "Mean Squared Error")
plot_learning_curve(svr_data, "SVR", "Mean Squared Error")
plot_learning_curve(kc_data, "KC", "Mean Squared Error")