import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import learning_curve
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

# ==========================
# LOAD EXCEL DATA
# ==========================
df = pd.read_excel("EDM2.xlsx")


# ===== LEARNING CURVE FUNCTION =====
def plot_learning_curve(data, title, ylabel):
    X = np.column_stack([
        data['XGBoost'],
        data['SVR'],
        data['KNN'],
        data['LSTM']
    ])
    y = np.array(data['Exp.Value'])

    xgb = XGBRegressor(random_state=0)
    svr = SVR()
    knn = KNeighborsRegressor(n_neighbors=5)

    ensemble = VotingRegressor([
        ('xgb', xgb),
        ('svr', svr),
        ('knn', knn)
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
esr_data = pd.read_excel("EDM2.xlsx", sheet_name="External Surface Roughness")
srr_data = pd.read_excel("EDM2.xlsx", sheet_name="Substance Removal Rate SRR")
kg_data  = pd.read_excel("EDM2.xlsx", sheet_name="Kerf Gap KG")

plot_learning_curve(esr_data, "External Surface Roughness", "Mean Squared Error")
plot_learning_curve(srr_data, "SRR", "Mean Squared Error")
plot_learning_curve(kg_data, "Kerf Gap", "Mean Squared Error")