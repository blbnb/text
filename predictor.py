# predictor.py - 预测模块
import numpy as np
import joblib
import os
from datetime import datetime
from typing import Dict, Any

class Predictor:
    """预测类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def predict(self, model_type: str, features: Dict[str, float]):
        """使用模型进行预测"""
        try:
            # 检查模型是否存在
            model_path = os.path.join(self.config['MODELS_DIR'], f"{model_type}_model.pkl")
            scaler_path = os.path.join(self.config['MODELS_DIR'], 'scaler.pkl')
            
            if not os.path.exists(model_path):
                return {
                    'status': 'error',
                    'message': f'模型 {model_type} 未找到，请先训练模型'
                }
            
            if not os.path.exists(scaler_path):
                return {
                    'status': 'error',
                    'message': 'Scaler未找到，请先训练模型'
                }
            
            # 加载模型和scaler
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            # 准备特征数据
            feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            
            X = []
            for feature in feature_cols:
                if feature in features:
                    X.append(float(features[feature]))
                else:
                    return {
                        'status': 'error',
                        'message': f'缺少特征: {feature}'
                    }
            
            X = np.array(X).reshape(1, -1)
            
            # 标准化特征
            X_scaled = scaler.transform(X)
            
            # 进行预测
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0].tolist()
            
            # 添加特征重要性信息
            important_features = []
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                for i, feature in enumerate(feature_cols):
                    important_features.append({
                        'feature': feature,
                        'importance': float(importances[i]) if i < len(importances) else 0.0,
                        'value': float(X[0][i])
                    })
            
            return {
                'status': 'success',
                'prediction': int(prediction),
                'probability': probability,
                'prediction_label': '有糖尿病' if prediction == 1 else '无糖尿病',
                'model_type': model_type,
                'important_features': important_features,
                'explanation': '该患者有较高的糖尿病风险，建议进一步检查' if prediction == 1 else '该患者糖尿病风险较低，建议保持健康生活方式',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }