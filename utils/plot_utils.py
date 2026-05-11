import re
import numpy as np
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def safe_filename(name):
    return re.sub(r'[\\/:*?"<>|]+', "_", str(name))


def plot_loss_curve(train_losses, val_losses, save_path):
    fig = plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss Curve")
    plt.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_actual_vs_predicted(y_true, y_pred, target_name, save_path):
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)

    min_v = min(np.min(y_true), np.min(y_pred))
    max_v = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")

    plt.xlabel("真实值")
    plt.ylabel("预测值")
    plt.title(f"{target_name} - 真实值 vs 预测值")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_residual_histogram(residuals, target_name, save_path):
    fig = plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=20)
    plt.xlabel("残差（预测值 - 真实值）")
    plt.ylabel("频数")
    plt.title(f"{target_name} - 残差直方图")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_residual_scatter(y_pred, residuals, target_name, save_path):
    fig = plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(y=0, linestyle="--")
    plt.xlabel("预测值")
    plt.ylabel("残差")
    plt.title(f"{target_name} - 残差散点图")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_metric_bar(metrics_df, metric_name, save_path):
    """
    绘制各目标的某一个指标柱状图
    """
    plot_df = metrics_df[metrics_df["目标"] != "整体平均"].copy()

    fig = plt.figure(figsize=(8, 5))
    plt.bar(plot_df["目标"].astype(str), plot_df[metric_name])
    plt.xlabel("预测目标")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} 指标对比")
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)