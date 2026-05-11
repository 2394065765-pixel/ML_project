import os
import sys
import copy
import random
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.network import BuildingLoadModel
from utils.data_utils import (
    FEATURE_COLS,
    TARGET_COLS,
    find_available_data_file,
    load_dataset,
    split_dataset,
    fit_processors,
    transform_dataset
)
from utils.metrics_utils import compute_metrics, build_prediction_detail
from utils.plot_utils import (
    safe_filename,
    plot_loss_curve,
    plot_actual_vs_predicted,
    plot_residual_histogram,
    plot_residual_scatter,
    plot_metric_bar
)


# =========================
# 路径设置
# =========================
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "log")
MODEL_DIR = os.path.join(BASE_DIR, "model")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

LOG_PATH = os.path.join(LOG_DIR, "train.log")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
Y_SCALER_PATH = os.path.join(MODEL_DIR, "y_scaler.pkl")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# =========================
# 训练参数
# =========================
RANDOM_SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 1000
PATIENCE = 120
VAL_SIZE = 0.1


# =========================
# 随机种子
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 日志配置
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_loader(X, y, batch_size=16, shuffle=False):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def evaluate_scaled_loss(model, X, y, criterion):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(DEVICE)
        pred = model(X_tensor)
        loss = criterion(pred, y_tensor).item()
    return loss


def predict_scaled(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        pred = model(X_tensor).cpu().numpy()
    return pred


def evaluate_and_save_reports(model, name, df_subset, X_subset, y_subset_scaled, y_scaler):
    """
    评估某个数据集(train/val/test)，保存指标
    """
    y_pred_scaled = predict_scaled(model, X_subset)

    y_true = y_scaler.inverse_transform(y_subset_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    metrics_df = compute_metrics(y_true, y_pred, TARGET_COLS)
    metrics_path = os.path.join(REPORT_DIR, f"metrics_{name}.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    logger.info(f"========== {name.upper()} 集评估结果 ==========")
    logger.info("\n" + metrics_df.to_string(index=False))

    # 如果是测试集，额外导出详情和图片
    if name == "test":
        detail_df = build_prediction_detail(df_subset, y_true, y_pred, TARGET_COLS)
        detail_path = os.path.join(REPORT_DIR, "test_predictions.xlsx")
        detail_df.to_excel(detail_path, index=False)

        # 每个目标单独画图
        for i, target_name in enumerate(TARGET_COLS):
            safe_name = safe_filename(target_name)

            plot_actual_vs_predicted(
                y_true[:, i],
                y_pred[:, i],
                target_name,
                os.path.join(FIGURE_DIR, f"{safe_name}_actual_vs_predicted.png")
            )

            residuals = y_pred[:, i] - y_true[:, i]

            plot_residual_histogram(
                residuals,
                target_name,
                os.path.join(FIGURE_DIR, f"{safe_name}_residual_hist.png")
            )

            plot_residual_scatter(
                y_pred[:, i],
                residuals,
                target_name,
                os.path.join(FIGURE_DIR, f"{safe_name}_residual_scatter.png")
            )

        # 指标柱状图
        for metric_name in ["MAE", "RMSE", "R2"]:
            plot_metric_bar(
                metrics_df,
                metric_name,
                os.path.join(FIGURE_DIR, f"metric_{metric_name}.png")
            )

    return metrics_df


def main():
    logger.info("当前设备: %s", DEVICE)

    data_path = find_available_data_file(DATA_DIR)
    logger.info("使用数据文件: %s", data_path)

    # 1. 读取数据
    df = load_dataset(data_path)
    logger.info("总样本数: %d", len(df))
    logger.info("特征列: %s", FEATURE_COLS)
    logger.info("标签列: %s", TARGET_COLS)

    if len(df) < 20:
        logger.warning("样本量较少，模型可能容易过拟合，评价结果仅供参考。")

    # 2. 数据划分
    train_df, val_df, test_df = split_dataset(
        df,
        random_seed=RANDOM_SEED,
        val_size=VAL_SIZE
    )

    logger.info("训练集样本数: %d", len(train_df))
    logger.info("验证集样本数: %d", len(val_df))
    logger.info("测试集样本数: %d", len(test_df))

    # 3. 只在训练集上拟合处理器
    X_train, y_train, preprocessor, y_scaler = fit_processors(
        train_df,
        PREPROCESSOR_PATH,
        Y_SCALER_PATH
    )
    X_val, y_val = transform_dataset(val_df, preprocessor, y_scaler)
    X_test, y_test = transform_dataset(test_df, preprocessor, y_scaler)

    input_dim = X_train.shape[1]
    output_dim = len(TARGET_COLS)

    logger.info("模型输入维度: %d", input_dim)
    logger.info("模型输出维度: %d", output_dim)

    # 4. DataLoader
    train_loader = create_loader(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)

    # 5. 初始化模型
    model = BuildingLoadModel(input_dim=input_dim, output_dim=output_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    history_train = []
    history_val = []

    logger.info("========== 开始训练 ==========")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)
            sample_count += batch_X.size(0)

        train_loss = running_loss / sample_count
        val_loss = evaluate_scaled_loss(model, X_val, y_val, criterion)

        history_train.append(train_loss)
        history_val.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 50 == 0 or epoch == 0:
            logger.info(
                "Epoch [%d/%d] - Train Loss: %.6f - Val Loss: %.6f",
                epoch + 1, EPOCHS, train_loss, val_loss
            )

        if wait >= PATIENCE:
            logger.info("触发 Early Stopping，提前结束训练。")
            break

    # 6. 恢复最佳模型
    model.load_state_dict(best_state)

    # 7. 保存模型
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_cols": FEATURE_COLS,
        "target_cols": TARGET_COLS,
        "input_dim": input_dim,
        "output_dim": output_dim
    }, MODEL_PATH)

    logger.info("最佳模型已保存: %s", MODEL_PATH)

    # 8. 训练过程损失曲线
    plot_loss_curve(
        history_train,
        history_val,
        os.path.join(FIGURE_DIR, "loss_curve.png")
    )

    # 9. 输出评估结果
    evaluate_and_save_reports(model, "train", train_df, X_train, y_train, y_scaler)
    evaluate_and_save_reports(model, "val", val_df, X_val, y_val, y_scaler)
    evaluate_and_save_reports(model, "test", test_df, X_test, y_test, y_scaler)

    logger.info("========== 训练与评估完成 ==========")
    logger.info("图像输出目录: %s", FIGURE_DIR)
    logger.info("报告输出目录: %s", REPORT_DIR)


if __name__ == "__main__":
    main()