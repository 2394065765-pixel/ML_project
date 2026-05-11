import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURE_COLS = [
    "所在层",
    "楼层建筑面积",
    "楼板平均厚度",
    "框架梁总长度",
    "框架柱总数量",
    "纵墙总面积",
    "横墙总面积",
    "层高"
]

TARGET_COLS = [
    "整层构建总自重",
    "整体合计恒载"
]

SPLIT_COL = "数据集划分"


def load_dataset(excel_path):
    """
    读取并清洗 Excel 数据
    """
    df = pd.read_excel(excel_path)
    df.columns = df.columns.astype(str).str.strip()

    required_cols = FEATURE_COLS + TARGET_COLS
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Excel 中缺少必要字段: {missing}")

    keep_cols = required_cols.copy()
    if SPLIT_COL in df.columns:
        keep_cols.append(SPLIT_COL)

    df = df[keep_cols].copy()

    # 处理文本列
    df["所在层"] = df["所在层"].astype(str).str.strip()

    # 数值列转数字
    numeric_cols = [col for col in FEATURE_COLS if col != "所在层"] + TARGET_COLS
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 处理划分列
    if SPLIT_COL in df.columns:
        df[SPLIT_COL] = df[SPLIT_COL].astype(str).str.strip().str.lower()

    # 删除关键字段为空的样本
    df = df.dropna(subset=FEATURE_COLS + TARGET_COLS).reset_index(drop=True)

    return df


def split_dataset(df, random_seed=42, val_size=0.1):
    """
    优先使用 Excel 中已有的 数据集划分(train/test)
    然后再从 train 中划出一部分作为 val
    """
    if SPLIT_COL in df.columns and df[SPLIT_COL].isin(["train", "test"]).any():
        train_val_df = df[df[SPLIT_COL] == "train"].copy()
        test_df = df[df[SPLIT_COL] == "test"].copy()

        if len(train_val_df) == 0 or len(test_df) == 0:
            raise ValueError("数据集划分列存在，但 train/test 数据为空，请检查 Excel。")
    else:
        train_val_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=random_seed
        )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=random_seed
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )


def build_preprocessor():
    """
    所在层 -> OneHot 编码
    其余数值特征 -> 标准化
    """
    categorical_features = ["所在层"]

    numeric_features = [
        "楼层建筑面积",
        "楼板平均厚度",
        "框架梁总长度",
        "框架柱总数量",
        "纵墙总面积",
        "横墙总面积",
        "层高"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features)
        ]
    )

    return preprocessor


def fit_processors(train_df, preprocessor_path, y_scaler_path):
    """
    只在训练集上拟合预处理器，避免数据泄露
    """
    preprocessor = build_preprocessor()

    X_train = preprocessor.fit_transform(train_df[FEATURE_COLS])
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()

    y_train = train_df[TARGET_COLS].values.astype(np.float32)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)

    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(y_scaler, y_scaler_path)

    return X_train.astype(np.float32), y_train_scaled.astype(np.float32), preprocessor, y_scaler


def transform_dataset(df, preprocessor, y_scaler=None):
    """
    使用已拟合好的处理器处理数据
    """
    X = preprocessor.transform(df[FEATURE_COLS])
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = X.astype(np.float32)

    y = None
    if all(col in df.columns for col in TARGET_COLS):
        y = df[TARGET_COLS].values.astype(np.float32)
        if y_scaler is not None:
            y = y_scaler.transform(y).astype(np.float32)

    return X, y


def load_processors(preprocessor_path, y_scaler_path):
    preprocessor = joblib.load(preprocessor_path)
    y_scaler = joblib.load(y_scaler_path)
    return preprocessor, y_scaler


def find_available_data_file(data_dir):
    """
    自动寻找可用数据文件
    优先使用带划分的数据
    """
    candidates = [
        "building_data_with_split.xlsx",
        "building_data_cleaned.xlsx",
        "building_data.xlsx"
    ]

    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"在 {data_dir} 下未找到数据文件，请放入以下任意文件：{candidates}"
    )