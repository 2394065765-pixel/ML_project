import os
import sys
import torch
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
try:
    from flask_cors import CORS
except ModuleNotFoundError:
    def CORS(app):
        return app
import joblib
import warnings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.network import BuildingLoadModel
from utils.data_utils import FEATURE_COLS, TARGET_COLS

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pth")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "model", "preprocessor.pkl")
Y_SCALER_PATH = os.path.join(BASE_DIR, "model", "y_scaler.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
preprocessor = None
y_scaler = None


def load_model_and_processors():
    global model, preprocessor, y_scaler
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}，请先运行 train.py 训练模型")
    
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"预处理器文件不存在: {PREPROCESSOR_PATH}")
    
    if not os.path.exists(Y_SCALER_PATH):
        raise FileNotFoundError(f"标签缩放器文件不存在: {Y_SCALER_PATH}")
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        
        input_dim = checkpoint['input_dim']
        output_dim = checkpoint['output_dim']
        
        model = BuildingLoadModel(input_dim=input_dim, output_dim=output_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            y_scaler = joblib.load(Y_SCALER_PATH)
        
        print("✓ 模型和处理器加载成功！")
        print(f"✓ 设备: {DEVICE}")
        print(f"✓ 输入维度: {input_dim}")
        print(f"✓ 输出维度: {output_dim}")
    except Exception as e:
        print(f"✗ 加载失败: {str(e)}")
        print("\n可能的原因：")
        print("1. sklearn 版本不兼容（训练时和现在的版本不同）")
        print("2. 模型文件损坏")
        print("\n解决方案：")
        print("请运行 'python src/train.py' 重新训练模型")
        raise


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        if not data:
            return jsonify({
                "success": False,
                "error": "请求数据为空"
            }), 400
        
        for col in FEATURE_COLS:
            if col not in data:
                return jsonify({
                    "success": False,
                    "error": f"缺少特征列: {col}"
                }), 400
        
        # 关键修复：正确处理不同类型的数据
        input_data = {}
        for col in FEATURE_COLS:
            value = data[col]
            
            # "所在层" 是分类变量，需要转换为字符串
            if col == "所在层":
                try:
                    # 先转为整数，再转为字符串（确保格式一致）
                    input_data[col] = str(int(float(value)))
                except (ValueError, TypeError):
                    return jsonify({
                        "success": False,
                        "error": f"特征 '{col}' 的值 '{value}' 无效，请输入整数"
                    }), 400
            else:
                # 其他特征是数值型，转换为浮点数
                try:
                    input_data[col] = float(value)
                except (ValueError, TypeError):
                    return jsonify({
                        "success": False,
                        "error": f"特征 '{col}' 的值 '{value}' 无法转换为数字"
                    }), 400
        
        # 创建 DataFrame 并指定列顺序
        df_input = pd.DataFrame([input_data], columns=FEATURE_COLS)
        
        # 确保数据类型正确
        df_input["所在层"] = df_input["所在层"].astype(str)
        for col in FEATURE_COLS:
            if col != "所在层":
                df_input[col] = df_input[col].astype(np.float64)
        
        print(f"输入数据: {df_input.to_dict('records')[0]}")
        print(f"数据类型: {df_input.dtypes.to_dict()}")
        
        # 进行预测
        X_processed = preprocessor.transform(df_input)
        
        X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            pred_scaled = model(X_tensor).cpu().numpy()
        
        pred_original = y_scaler.inverse_transform(pred_scaled)
        
        result = {
            "success": True,
            "predictions": {
                TARGET_COLS[0]: float(pred_original[0][0]),
                TARGET_COLS[1]: float(pred_original[0][1])
            }
        }
        
        print(f"预测结果: {result['predictions']}")
        
        return jsonify(result)
    
    except AttributeError as e:
        error_msg = str(e)
        if '_name_to_fitted_passthrough' in error_msg or 'ColumnTransformer' in error_msg:
            return jsonify({
                "success": False,
                "error": "模型版本不兼容，请重新训练模型：python src/train.py"
            }), 500
        return jsonify({
            "success": False,
            "error": f"处理错误: {error_msg}"
        }), 500
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"预测错误详情:\n{error_detail}")
        return jsonify({
            "success": False,
            "error": f"预测失败: {str(e)}"
        }), 500


if __name__ == '__main__':
    print("=" * 50)
    print("建筑恒载智能预测系统")
    print("=" * 50)
    load_model_and_processors()
    print("=" * 50)
    print("服务器启动中...")
    print("访问地址: http://localhost:5000")
    print("按 Ctrl+C 停止服务器")
    print("=" * 50)
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
