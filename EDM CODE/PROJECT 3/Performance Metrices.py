from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
# ===== DATA(ACTUAL) =====
df = pd.read_excel('EDM3.xlsx')

datasets = {
    "SU": pd.read_excel('EDM3.xlsx', sheet_name='Surface Undulation Su'),
    "SVR": pd.read_excel('EDM3.xlsx', sheet_name='Substance Vaporization Rate SVR'),
    "KC": pd.read_excel('EDM3.xlsx', sheet_name='Kerf Clearence KC')
}
su_df = pd.read_excel("EDM3.xlsx", sheet_name="Surface Undulation Su")
svr_df = pd.read_excel("EDM3.xlsx", sheet_name="Substance Vaporization Rate SVR")
kc_df  = pd.read_excel("EDM3.xlsx", sheet_name="Kerf Clearence KC")
datasets = {
    "SU": {
        "Actual": su_df["Exp.Value"].values,
        "Predicted": su_df["Ensemble"].values
    },

    "SVR": {
        "Actual": svr_df["Exp.Value"].values,
        "Predicted": svr_df["Ensemble"].values
    },

    "KC": {
        "Actual": kc_df["Exp.Value"].values,
        "Predicted": kc_df["Ensemble"].values
    }
}
su_actual = datasets["SU"]["Actual"]
su_pred   = datasets["SU"]["Predicted"]

svr_actual = datasets["SVR"]["Actual"]
svr_pred   = datasets["SVR"]["Predicted"]

kc_actual = datasets["KC"]["Actual"]
kc_pred   = datasets["KC"]["Predicted"]
# ===== METRICES CALCULATION FUNCTION =====
def get_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# ===== METRICES FOR EACH TARGET =====
su_metrics = get_metrics(su_actual, su_pred)
svr_metrics = get_metrics(svr_actual, svr_pred)
kc_metrics = get_metrics(kc_actual, kc_pred)

# ===== PRINT RESULT =====
print("Ensemble Performance Metrics (RMSE, MAE, R²):")
print(f"{'Target':<25}{'RMSE':>10}{'MAE':>10}{'R²':>10}")
print(f"{'SU':<25}{su_metrics[0]:>10.4f}{su_metrics[1]:>10.4f}{su_metrics[2]:>10.4f}")
print(f"{'SVR':<25}{svr_metrics[0]:>10.4f}{svr_metrics[1]:>10.4f}{svr_metrics[2]:>10.4f}")
print(f"{'KC':<25}{kc_metrics[0]:>10.4f}{kc_metrics[1]:>10.4f}{kc_metrics[2]:>10.4f}")
