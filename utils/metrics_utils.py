import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calc_mape(y_true, y_pred, eps=1e-8):
    denominator = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denominator)) * 100


def calc_smape(y_true, y_pred, eps=1e-8):
    denominator = np.abs(y_true) + np.abs(y_pred)
    denominator = np.where(denominator < eps, eps, denominator)
    return np.mean(2.0 * np.abs(y_pred - y_true) / denominator) * 100


def compute_metrics(y_true, y_pred, target_names):
    """
    对每个输出目标计算回归指标
    """
    rows = []

    for i, target_name in enumerate(target_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        mae = mean_absolute_error(yt, yp)
        mse = mean_squared_error(yt, yp)
        rmse = np.sqrt(mse)

        try:
            r2 = r2_score(yt, yp)
        except Exception:
            r2 = np.nan

        mape = calc_mape(yt, yp)
        smape = calc_smape(yt, yp)

        rows.append({
            "目标": target_name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "MAPE(%)": mape,
            "SMAPE(%)": smape
        })

    metrics_df = pd.DataFrame(rows)

    overall_row = pd.DataFrame([{
        "目标": "整体平均",
        "MAE": metrics_df["MAE"].mean(),
        "MSE": metrics_df["MSE"].mean(),
        "RMSE": metrics_df["RMSE"].mean(),
        "R2": metrics_df["R2"].mean(),
        "MAPE(%)": metrics_df["MAPE(%)"].mean(),
        "SMAPE(%)": metrics_df["SMAPE(%)"].mean()
    }])

    metrics_df = pd.concat([metrics_df, overall_row], ignore_index=True)
    return metrics_df


def build_prediction_detail(original_df, y_true, y_pred, target_names):
    """
    构建测试集预测详情表
    """
    result_df = original_df.reset_index(drop=True).copy()

    for i, target_name in enumerate(target_names):
        result_df[f"真实_{target_name}"] = y_true[:, i]
        result_df[f"预测_{target_name}"] = y_pred[:, i]
        result_df[f"误差_{target_name}"] = y_pred[:, i] - y_true[:, i]
        result_df[f"绝对误差_{target_name}"] = np.abs(y_pred[:, i] - y_true[:, i])

    return result_df