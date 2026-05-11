import os
import sys
import torch
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.network import BuildingLoadModel
from utils.data_utils import FEATURE_COLS, TARGET_COLS, load_processors


MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pth")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "model", "preprocessor.pkl")
Y_SCALER_PATH = os.path.join(BASE_DIR, "model", "y_scaler.pkl")


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    input_dim = checkpoint["input_dim"]
    output_dim = checkpoint["output_dim"]

    model = BuildingLoadModel(
        input_dim=input_dim,
        output_dim=output_dim
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def predict_single(data_dict):
    """
    单条数据预测
    """
    model = load_model()
    preprocessor, y_scaler = load_processors(PREPROCESSOR_PATH, Y_SCALER_PATH)

    df = pd.DataFrame([data_dict])
    X = df[FEATURE_COLS]

    X_processed = preprocessor.transform(X)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    X_tensor = torch.tensor(X_processed, dtype=torch.float32)

    with torch.no_grad():
        pred_scaled = model(X_tensor).numpy()

    pred = y_scaler.inverse_transform(pred_scaled)

    result = {}
    for target_name, value in zip(TARGET_COLS, pred[0]):
        result[target_name] = float(value)

    return result


def predict_from_excel(input_excel_path, output_excel_path=None):
    """
    批量预测
    Excel 至少需要包含 FEATURE_COLS 中的字段
    """
    model = load_model()
    preprocessor, y_scaler = load_processors(PREPROCESSOR_PATH, Y_SCALER_PATH)

    df = pd.read_excel(input_excel_path)
    df.columns = df.columns.astype(str).str.strip()

    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"输入文件缺少必要特征列: {missing}")

    X = df[FEATURE_COLS].copy()
    X["所在层"] = X["所在层"].astype(str).str.strip()

    X_processed = preprocessor.transform(X)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    X_tensor = torch.tensor(X_processed, dtype=torch.float32)

    with torch.no_grad():
        pred_scaled = model(X_tensor).numpy()

    pred = y_scaler.inverse_transform(pred_scaled)

    for i, target_name in enumerate(TARGET_COLS):
        df[f"预测_{target_name}"] = pred[:, i]

    if output_excel_path is not None:
        df.to_excel(output_excel_path, index=False)

    return df


if __name__ == "__main__":
    # 单条样本预测示例
    sample = {
        "所在层": "2层",
        "楼层建筑面积": 1329.12,
        "楼板平均厚度": 100,
        "框架梁总长度": 463.20,
        "框架柱总数量": 39,
        "纵墙总面积": 592.2,
        "横墙总面积": 216,
        "层高": 4.5
    }

    result = predict_single(sample)

    print("========== 输入样本 ==========")
    for k, v in sample.items():
        print(f"{k}: {v}")

    print("\n========== 预测结果 ==========")
    for k, v in result.items():
        print(f"{k}: {v:.4f}")