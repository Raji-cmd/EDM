from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
# ===== DATA(ACTUAL) =====
df = pd.read_excel('EDM4.xlsx')

datasets = {
    "SW": pd.read_excel('EDM4.xlsx', sheet_name='Surface Waviness SW'),
    "MVR": pd.read_excel('EDM4.xlsx', sheet_name='Material Vaporization Rate MVR'),
    "KD": pd.read_excel('EDM4.xlsx', sheet_name='Kerf Distance KD')
}
SW_df = pd.read_excel("EDM4.xlsx", sheet_name="Surface Waviness SW")
MVR_df = pd.read_excel("EDM4.xlsx", sheet_name="Material Vaporization Rate MVR")
KD_df  = pd.read_excel("EDM4.xlsx", sheet_name="Kerf Distance KD")
datasets = {
    "SW": {
        "Actual": SW_df["Exp.Value"].values,
        "Predicted": SW_df["Ensemble"].values
    },

    "MVR": {
        "Actual": MVR_df["Exp.Value"].values,
        "Predicted": MVR_df["Ensemble"].values
    },

    "KD": {
        "Actual": KD_df["Exp.Value"].values,
        "Predicted": KD_df["Ensemble"].values
    }
}
SW_actual = datasets["SW"]["Actual"]
SW_pred   = datasets["SW"]["Predicted"]

MVR_actual = datasets["MVR"]["Actual"]
MVR_pred   = datasets["MVR"]["Predicted"]

KD_actual = datasets["KD"]["Actual"]
KD_pred   = datasets["KD"]["Predicted"]
# ===== METRICES CALCULATION FUNCTION =====
def get_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# ===== METRICES FOR EACH TARGET =====
SW_metrics = get_metrics(SW_actual, SW_pred)
MVR_metrics = get_metrics(MVR_actual, MVR_pred)
KD_metrics = get_metrics(KD_actual, KD_pred)

# ===== PRINT RESULT =====
print("Ensemble Performance Metrics (RMSE, MAE, R²):")
print(f"{'Target':<25}{'RMSE':>10}{'MAE':>10}{'R²':>10}")
print(f"{'SW':<25}{SW_metrics[0]:>10.4f}{SW_metrics[1]:>10.4f}{SW_metrics[2]:>10.4f}")
print(f"{'MVR':<25}{MVR_metrics[0]:>10.4f}{MVR_metrics[1]:>10.4f}{MVR_metrics[2]:>10.4f}")
print(f"{'KD':<25}{KD_metrics[0]:>10.4f}{KD_metrics[1]:>10.4f}{KD_metrics[2]:>10.4f}")
