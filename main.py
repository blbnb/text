# app_fixed.py - 完整版，包含图片API
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# 机器学习库
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)

# 模型库（移除 CatBoost）
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
)
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

app = Flask(__name__)
CORS(app)

# 配置
CONFIG = {
    'MODELS_DIR': 'models',
    'IMAGES_DIR': 'static/images',
    'DATA_FILE': 'diabetes.csv',
    'TEST_SIZE': 0.2,
    'RANDOM_STATE': 42,
    'CV_FOLDS': 5,
    'ENABLE_PCA': True,
    'PCA_COMPONENTS': 0.95,
    'ENABLE_FEATURE_SELECTION': True,
    'SELECT_FEATURES': 6,
    'OUTLIER_HANDLING': 'winsorize',
    'SCALING_METHOD': 'standard'
}

# 全局变量
models: Dict[str, Any] = {}
scalers: Dict[str, Any] = {}

def create_directories():
    """创建必要的目录"""
    for directory in [CONFIG['MODELS_DIR'], CONFIG['IMAGES_DIR']]:
        if not os.path.exists(directory):
            os.makedirs(directory)

def save_training_plots(X_train, y_train, X_test, y_test, model, model_name):
    """保存模型训练后的图片"""
    try:
        # 设置样式
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name} - 模型性能分析', fontsize=16, fontweight='bold')
        
        # 1. 特征重要性（如果模型支持）
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                feature_names = [f'Feature {i+1}' for i in range(len(importances))]
                
                axes[0, 0].bar(range(len(importances)), importances[indices])
                axes[0, 0].set_xlabel('特征')
                axes[0, 0].set_ylabel('重要性分数')
                axes[0, 0].set_title('特征重要性')
                axes[0, 0].set_xticks(range(len(importances)))
                axes[0, 0].set_xticklabels([feature_names[i] for i in indices], rotation=45)
        except:
            axes[0, 0].text(0.5, 0.5, '特征重要性不可用', 
                           ha='center', va='center', fontsize=12)
            axes[0, 0].set_title('特征重要性')
        
        # 2. 预测概率分布
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        axes[0, 1].hist(y_pred_proba[y_test == 0], alpha=0.5, label='无糖尿病', bins=20)
        axes[0, 1].hist(y_pred_proba[y_test == 1], alpha=0.5, label='有糖尿病', bins=20)
        axes[0, 1].set_xlabel('预测概率')
        axes[0, 1].set_ylabel('频率')
        axes[0, 1].set_title('预测概率分布')
        axes[0, 1].legend()
        axes[0, 1].axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
        
        # 3. ROC曲线
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC曲线 (AUC = {roc_auc:.3f})')
        axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        axes[1, 0].set_xlim([0.0, 1.0])
        axes[1, 0].set_ylim([0.0, 1.05])
        axes[1, 0].set_xlabel('假阳性率')
        axes[1, 0].set_ylabel('真阳性率')
        axes[1, 0].set_title('ROC曲线')
        axes[1, 0].legend(loc="lower right")
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 混淆矩阵
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_xlabel('预测标签')
        axes[1, 1].set_ylabel('真实标签')
        axes[1, 1].set_title('混淆矩阵')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(CONFIG['IMAGES_DIR'], f'{model_name}_{timestamp}.png')
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已保存训练图片: {image_path}")
        return image_path
        
    except Exception as e:
        print(f"保存训练图片时出错: {str(e)}")
        return None

def save_model_comparison_plot(comparison_results):
    """保存模型比较图"""
    try:
        plt.figure(figsize=(12, 8))
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 准备数据
        models_list = []
        accuracies = []
        
        for model_name, result in comparison_results.items():
            if result['status'] == 'success' and result['accuracy'] is not None:
                models_list.append(model_name)
                accuracies.append(result['accuracy'])
        
        if not models_list:
            return None
        
        # 创建条形图
        colors = plt.cm.Set3(np.linspace(0, 1, len(models_list)))
        bars = plt.barh(models_list, accuracies, color=colors)
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{acc:.3f}', va='center', fontsize=10)
        
        plt.xlabel('准确率')
        plt.title('模型性能比较', fontsize=14, fontweight='bold')
        plt.xlim([0, 1.05])
        plt.grid(True, axis='x', alpha=0.3)
        
        # 添加平均线
        avg_accuracy = np.mean(accuracies)
        plt.axvline(x=avg_accuracy, color='red', linestyle='--', alpha=0.7, 
                   label=f'平均准确率: {avg_accuracy:.3f}')
        plt.legend()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(CONFIG['IMAGES_DIR'], f'model_comparison_{timestamp}.png')
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已保存模型比较图片: {image_path}")
        return image_path
        
    except Exception as e:
        print(f"保存模型比较图时出错: {str(e)}")
        return None

def save_data_visualization(df):
    """保存数据可视化图片"""
    try:
        plt.figure(figsize=(15, 10))
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. 糖尿病分布
        plt.subplot(2, 3, 1)
        diabetes_counts = df['Outcome'].value_counts()
        plt.pie(diabetes_counts.values, labels=['无糖尿病', '有糖尿病'], 
                autopct='%1.1f%%', startangle=90, colors=['lightblue', 'salmon'])
        plt.title('糖尿病分布')
        
        # 2. 年龄分布
        plt.subplot(2, 3, 2)
        plt.hist(df['Age'], bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('年龄')
        plt.ylabel('频率')
        plt.title('年龄分布')
        
        # 3. BMI分布
        plt.subplot(2, 3, 3)
        plt.hist(df['BMI'], bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('BMI')
        plt.ylabel('频率')
        plt.title('BMI分布')
        
        # 4. 葡萄糖与糖尿病关系
        plt.subplot(2, 3, 4)
        diabetic = df[df['Outcome'] == 1]
        non_diabetic = df[df['Outcome'] == 0]
        
        plt.boxplot([non_diabetic['Glucose'], diabetic['Glucose']], 
                   labels=['无糖尿病', '有糖尿病'])
        plt.ylabel('葡萄糖水平')
        plt.title('葡萄糖水平与糖尿病')
        
        # 5. 相关性热图
        plt.subplot(2, 3, 5)
        correlation = df.corr()
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        sns.heatmap(correlation, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=.5, fmt='.2f')
        plt.title('特征相关性热图')
        
        # 6. 特征散点图
        plt.subplot(2, 3, 6)
        plt.scatter(df['Glucose'], df['BMI'], c=df['Outcome'], 
                   cmap='viridis', alpha=0.6, edgecolors='w', linewidth=0.5)
        plt.xlabel('葡萄糖')
        plt.ylabel('BMI')
        plt.title('葡萄糖 vs BMI (按糖尿病状态着色)')
        plt.colorbar(label='糖尿病 (0=无, 1=有)')
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(CONFIG['IMAGES_DIR'], f'data_visualization_{timestamp}.png')
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已保存数据可视化图片: {image_path}")
        return image_path
        
    except Exception as e:
        print(f"保存数据可视化时出错: {str(e)}")
        return None

def load_and_preprocess_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """加载和预处理数据"""
    try:
        if data_path is None:
            data_path = CONFIG['DATA_FILE']
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件 {data_path} 不存在")
        
        df = pd.read_csv(data_path)
        print(f"成功加载数据: {df.shape[0]} 行, {df.shape[1]} 列")
        
        # 数据清洗
        df_clean = clean_data(df)
        
        return df_clean
        
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        raise

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """数据清洗"""
    df_clean = df.copy()
    
    # 检查并处理缺失值
    missing_before = df_clean.isnull().sum().sum()
    print(f"缺失值数量（清理前）: {missing_before}")
    
    if missing_before > 0:
        # 用中位数填充缺失值
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                median_value = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_value)
    
    # 检查零值（在某些特征中可能表示缺失）
    zero_value_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for feature in zero_value_features:
        if feature in df_clean.columns:
            zero_count = (df_clean[feature] == 0).sum()
            if zero_count > 0:
                print(f"特征 {feature} 有 {zero_count} 个零值")
                
                # 将零值替换为中位数
                median_value = df_clean[feature].replace(0, np.nan).median()
                df_clean[feature] = df_clean[feature].replace(0, median_value)
    
    return df_clean

def get_model_config(model_type: str) -> Dict[str, Any]:
    """获取模型配置（移除 CatBoost）"""
    model_configs = {
        'logistic_regression': {
            'class': LogisticRegression,
            'params': {
                'solver': 'liblinear',
                'penalty': 'l2',
                'C': 1.0,
                'max_iter': 1000,
                'random_state': CONFIG['RANDOM_STATE']
            }
        },
        'decision_tree': {
            'class': DecisionTreeClassifier,
            'params': {
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': CONFIG['RANDOM_STATE']
            }
        },
        'random_forest': {
            'class': RandomForestClassifier,
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': CONFIG['RANDOM_STATE']
            }
        },
        'svm': {
            'class': SVC,
            'params': {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'probability': True,
                'random_state': CONFIG['RANDOM_STATE']
            }
        },
        'gradient_boosting': {
            'class': GradientBoostingClassifier,
            'params': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': CONFIG['RANDOM_STATE']
            }
        },
        'xgboost': {
            'class': XGBClassifier,
            'params': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': CONFIG['RANDOM_STATE'],
                'eval_metric': 'logloss',
                'verbosity': 0
            }
        },
        'lightgbm': {
            'class': LGBMClassifier,
            'params': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': CONFIG['RANDOM_STATE'],
                'verbosity': -1
            }
        },
        'neural_network': {
            'class': MLPClassifier,
            'params': {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'solver': 'adam',
                'max_iter': 1000,
                'random_state': CONFIG['RANDOM_STATE']
            }
        },
        'knn': {
            'class': KNeighborsClassifier,
            'params': {
                'n_neighbors': 5,
                'weights': 'distance'
            }
        },
        'naive_bayes': {
            'class': GaussianNB,
            'params': {}
        },
        'adaboost': {
            'class': AdaBoostClassifier,
            'params': {
                'n_estimators': 50,
                'learning_rate': 1.0,
                'random_state': CONFIG['RANDOM_STATE']
            }
        },
        'extra_trees': {
            'class': ExtraTreesClassifier,
            'params': {
                'n_estimators': 100,
                'random_state': CONFIG['RANDOM_STATE']
            }
        },
        'bagging': {
            'class': BaggingClassifier,
            'params': {
                'n_estimators': 10,
                'random_state': CONFIG['RANDOM_STATE']
            }
        },
        'lda': {
            'class': LinearDiscriminantAnalysis,
            'params': {}
        },
        'qda': {
            'class': QuadraticDiscriminantAnalysis,
            'params': {}
        }
    }
    
    return model_configs.get(model_type, model_configs['logistic_regression'])

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': list(models.keys())
    })

@app.route('/api/data/overview', methods=['GET'])
def get_data_overview():
    """获取数据集概览"""
    try:
        df = load_and_preprocess_data()
        
        overview = {
            'total_records': int(df.shape[0]),
            'total_features': int(df.shape[1]),
            'features': df.columns.tolist(),
            'diabetes_ratio': float(df['Outcome'].mean()),
            'data_types': df.dtypes.astype(str).to_dict(),
            'head': df.head().to_dict(orient='records'),
            'missing_values': df.isnull().sum().to_dict(),
            'statistical_summary': df.describe().to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({'status': 'success', 'data': overview})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/model/train', methods=['POST'])
def train_models():
    """训练多个模型"""
    try:
        data = request.json or {}
        selected_models = data.get('models', ['logistic_regression', 'decision_tree', 'random_forest'])
        
        df = load_and_preprocess_data()
        
        # 保存数据可视化图片
        save_data_visualization(df)
        
        # 准备特征和目标
        feature_cols = [col for col in df.columns if col != 'Outcome']
        X = df[feature_cols]
        y = df['Outcome']
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=CONFIG['TEST_SIZE'], random_state=CONFIG['RANDOM_STATE'], stratify=y
        )
        
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        training_images = []
        
        for model_type in selected_models:
            try:
                model_config = get_model_config(model_type)
                model_class = model_config['class']
                model_params = model_config['params']
                
                # 创建模型
                model = model_class(**model_params)
                
                # 训练模型
                model.fit(X_train_scaled, y_train)
                
                # 预测
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                # 保存训练图片
                image_path = save_training_plots(X_train_scaled, y_train, X_test_scaled, y_test, model, model_type)
                if image_path:
                    training_images.append({
                        'model': model_type,
                        'image_path': image_path
                    })
                
                # 保存模型
                model_filename = f"{model_type}_model.pkl"
                model_path = os.path.join(CONFIG['MODELS_DIR'], model_filename)
                joblib.dump(model, model_path)
                
                # 保存 scaler
                scaler_path = os.path.join(CONFIG['MODELS_DIR'], 'scaler.pkl')
                joblib.dump(scaler, scaler_path)
                
                models[model_type] = model
                scalers['default'] = scaler
                
                results[model_type] = {
                    'status': 'success',
                    'accuracy': float(accuracy),
                    'model_path': model_path,
                    'image_path': image_path
                }
                
            except Exception as e:
                results[model_type] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return jsonify({
            'status': 'success',
            'message': f'成功训练 {len(results)} 个模型',
            'results': results,
            'training_images': training_images,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/model/predict', methods=['POST'])
def predict():
    """使用模型进行预测"""
    try:
        data = request.json
        
        if not data:
            return jsonify({'status': 'error', 'message': '请求数据为空'}), 400
        
        model_type = data.get('model_type', 'logistic_regression')
        features = data.get('features', {})
        
        # 检查模型是否存在
        model_path = os.path.join(CONFIG['MODELS_DIR'], f"{model_type}_model.pkl")
        scaler_path = os.path.join(CONFIG['MODELS_DIR'], 'scaler.pkl')
        
        if not os.path.exists(model_path):
            return jsonify({
                'status': 'error',
                'message': f'模型 {model_type} 未找到，请先训练模型'
            }), 404
        
        if not os.path.exists(scaler_path):
            return jsonify({
                'status': 'error',
                'message': 'Scaler未找到，请先训练模型'
            }), 404
        
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
                return jsonify({'status': 'error', 'message': f'缺少特征: {feature}'}), 400
        
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
        
        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'probability': probability,
            'prediction_label': '有糖尿病' if prediction == 1 else '无糖尿病',
            'model_type': model_type,
            'important_features': important_features,
            'explanation': '该患者有较高的糖尿病风险，建议进一步检查' if prediction == 1 else '该患者糖尿病风险较低，建议保持健康生活方式',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/model/compare', methods=['GET'])
def compare_all_models():
    """比较所有可用模型"""
    try:
        df = load_and_preprocess_data()
        
        feature_cols = [col for col in df.columns if col != 'Outcome']
        X = df[feature_cols]
        y = df['Outcome']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=CONFIG['TEST_SIZE'], random_state=CONFIG['RANDOM_STATE'], stratify=y
        )
        
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 所有可用模型（移除 CatBoost）
        all_models = [
            'logistic_regression', 'decision_tree', 'random_forest',
            'svm', 'gradient_boosting', 'xgboost', 'lightgbm',
            'neural_network', 'knn', 'naive_bayes', 'adaboost',
            'extra_trees', 'bagging', 'lda', 'qda'
        ]
        
        comparison = {}
        for model_type in all_models:
            try:
                model_config = get_model_config(model_type)
                model_class = model_config['class']
                model_params = model_config['params']
                
                model = model_class(**model_params)
                model.fit(X_train_scaled, y_train)
                
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                comparison[model_type] = {
                    'accuracy': float(accuracy),
                    'status': 'success'
                }
                
            except Exception as e:
                comparison[model_type] = {
                    'accuracy': None,
                    'status': 'error',
                    'error': str(e)
                }
        
        # 保存比较图片
        comparison_image_path = save_model_comparison_plot(comparison)
        
        # 找到最佳模型
        successful_models = {k: v for k, v in comparison.items() if v['status'] == 'success' and v['accuracy'] is not None}
        if successful_models:
            best_model = max(successful_models.items(), key=lambda x: x[1]['accuracy'])
            best_model_name = best_model[0]
            best_accuracy = best_model[1]['accuracy']
        else:
            best_model_name = None
            best_accuracy = None
        
        return jsonify({
            'status': 'success',
            'comparison': comparison,
            'best_model': best_model_name,
            'best_accuracy': best_accuracy,
            'model_count': len(comparison),
            'successful_models': len(successful_models),
            'comparison_image_path': comparison_image_path,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ============== 图片相关API ==============

@app.route('/api/plots/list', methods=['GET'])
def list_plots():
    """列出所有可用的图片"""
    try:
        images = []
        if os.path.exists(CONFIG['IMAGES_DIR']):
            for filename in os.listdir(CONFIG['IMAGES_DIR']):
                if filename.endswith('.png'):
                    filepath = os.path.join(CONFIG['IMAGES_DIR'], filename)
                    try:
                        stat = os.stat(filepath)
                        images.append({
                            'name': filename,
                            'path': f'/api/plots/image/{filename}',
                            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                            'size': stat.st_size
                        })
                    except Exception as e:
                        print(f"处理图片 {filename} 时出错: {str(e)}")
                        continue
        
        # 按创建时间排序
        images.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({
            'status': 'success',
            'count': len(images),
            'images': images,
            'directory': CONFIG['IMAGES_DIR']
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/plots/image/<filename>', methods=['GET'])
def get_image(filename):
    """获取图片文件"""
    try:
        # 安全检查：防止目录遍历攻击
        if '..' in filename or filename.startswith('/'):
            return jsonify({'status': 'error', 'message': '无效的文件名'}), 400
        
        image_path = os.path.join(CONFIG['IMAGES_DIR'], filename)
        
        if not os.path.exists(image_path):
            return jsonify({'status': 'error', 'message': '图片不存在'}), 404
        
        # 使用send_file返回图片
        return send_file(image_path, mimetype='image/png')
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/plots/delete/<filename>', methods=['DELETE'])
def delete_image(filename):
    """删除图片文件"""
    try:
        # 安全检查：防止目录遍历攻击
        if '..' in filename or filename.startswith('/'):
            return jsonify({'status': 'error', 'message': '无效的文件名'}), 400
        
        image_path = os.path.join(CONFIG['IMAGES_DIR'], filename)
        
        if not os.path.exists(image_path):
            return jsonify({'status': 'error', 'message': '图片不存在'}), 404
        
        os.remove(image_path)
        return jsonify({
            'status': 'success', 
            'message': f'已删除图片: {filename}',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/plots/<plot_type>', methods=['GET'])
def get_plot_by_type(plot_type):
    """根据类型获取图片"""
    try:
        if not os.path.exists(CONFIG['IMAGES_DIR']):
            return jsonify({'status': 'error', 'message': '图片目录不存在'}), 404
        
        if plot_type == 'latest_training':
            # 获取最新的训练图片
            images = [f for f in os.listdir(CONFIG['IMAGES_DIR']) 
                     if f.endswith('.png') and not f.startswith('model_comparison_') 
                     and not f.startswith('data_visualization_')]
            if not images:
                return jsonify({'status': 'error', 'message': '没有找到训练图片'}), 404
            
            latest_image = max(images, key=lambda x: os.path.getctime(os.path.join(CONFIG['IMAGES_DIR'], x)))
            return send_file(os.path.join(CONFIG['IMAGES_DIR'], latest_image), mimetype='image/png')
            
        elif plot_type == 'comparison':
            # 获取最新的比较图片
            images = [f for f in os.listdir(CONFIG['IMAGES_DIR']) 
                     if f.startswith('model_comparison_') and f.endswith('.png')]
            if not images:
                return jsonify({'status': 'error', 'message': '没有找到比较图片'}), 404
            
            latest_image = max(images, key=lambda x: os.path.getctime(os.path.join(CONFIG['IMAGES_DIR'], x)))
            return send_file(os.path.join(CONFIG['IMAGES_DIR'], latest_image), mimetype='image/png')
            
        elif plot_type == 'data_visualization':
            # 获取最新的数据可视化图片
            images = [f for f in os.listdir(CONFIG['IMAGES_DIR']) 
                     if f.startswith('data_visualization_') and f.endswith('.png')]
            if not images:
                return jsonify({'status': 'error', 'message': '没有找到数据可视化图片'}), 404
            
            latest_image = max(images, key=lambda x: os.path.getctime(os.path.join(CONFIG['IMAGES_DIR'], x)))
            return send_file(os.path.join(CONFIG['IMAGES_DIR'], latest_image), mimetype='image/png')
            
        else:
            return jsonify({'status': 'error', 'message': '不支持的图片类型'}), 400
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ============== 静态文件服务 ==============

@app.route('/static/<path:filename>')
def serve_static(filename):
    """提供静态文件服务"""
    return send_from_directory('static', filename)

if __name__ == '__main__':
    # 确保必要的目录存在
    create_directories()
    
    print("=" * 60)
    print("糖尿病预测系统 - 完整版")
    print("=" * 60)
    print("包含图片API和可视化功能")
    print(f"数据文件: {CONFIG['DATA_FILE']}")
    print(f"模型目录: {CONFIG['MODELS_DIR']}")
    print(f"图片目录: {CONFIG['IMAGES_DIR']}")
    print("=" * 60)
    
    # 检查数据文件是否存在
    if not os.path.exists(CONFIG['DATA_FILE']):
        print(f"警告: 数据文件 {CONFIG['DATA_FILE']} 不存在")
        print("请确保 diabetes.csv 文件位于当前目录")
    
    print("可用模型:")
    for model_name in get_model_config('').keys():
        print(f"  - {model_name}")
    
    print("\nAPI端点:")
    print("  GET  /api/data/overview      - 数据概览")
    print("  POST /api/model/train        - 训练模型")
    print("  POST /api/model/predict      - 预测")
    print("  GET  /api/model/compare      - 比较模型")
    print("  GET  /api/plots/list         - 列出所有图片")
    print("  GET  /api/plots/image/<name> - 获取图片")
    print("  GET  /api/plots/<type>       - 获取类型图片")
    print("  DELETE /api/plots/delete/<name> - 删除图片")
    print("  GET  /api/health             - 健康检查")
    print("  GET  /static/<path>          - 静态文件")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)