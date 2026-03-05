# model_trainer.py - 模型训练模块
import numpy as np
import joblib
import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix

# 导入模型库
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

class ModelTrainer:
    """模型训练类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """获取模型配置"""
        model_configs = {
            'logistic_regression': {
                'class': LogisticRegression,
                'params': {
                    'solver': 'liblinear',
                    'penalty': 'l2',
                    'C': 1.0,
                    'max_iter': 1000,
                    'random_state': self.config['RANDOM_STATE']
                }
            },
            'decision_tree': {
                'class': DecisionTreeClassifier,
                'params': {
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': self.config['RANDOM_STATE']
                }
            },
            'random_forest': {
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': self.config['RANDOM_STATE']
                }
            },
            'svm': {
                'class': SVC,
                'params': {
                    'kernel': 'rbf',
                    'C': 1.0,
                    'gamma': 'scale',
                    'probability': True,
                    'random_state': self.config['RANDOM_STATE']
                }
            },
            'gradient_boosting': {
                'class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3,
                    'random_state': self.config['RANDOM_STATE']
                }
            },
            'xgboost': {
                'class': XGBClassifier,
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'random_state': self.config['RANDOM_STATE'],
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
                    'random_state': self.config['RANDOM_STATE'],
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
                    'random_state': self.config['RANDOM_STATE']
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
                    'random_state': self.config['RANDOM_STATE']
                }
            },
            'extra_trees': {
                'class': ExtraTreesClassifier,
                'params': {
                    'n_estimators': 100,
                    'random_state': self.config['RANDOM_STATE']
                }
            },
            'bagging': {
                'class': BaggingClassifier,
                'params': {
                    'n_estimators': 10,
                    'random_state': self.config['RANDOM_STATE']
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
    
    def prepare_data(self, df: pd.DataFrame):
        """准备训练数据"""
        feature_cols = [col for col in df.columns if col != 'Outcome']
        X = df[feature_cols]
        y = df['Outcome']
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['TEST_SIZE'], 
            random_state=self.config['RANDOM_STATE'], stratify=y
        )
        
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    def save_training_plots(self, X_train, y_train, X_test, y_test, model, model_name):
        """保存模型训练后的图片"""
        try:
            if not os.path.exists(self.config['IMAGES_DIR']):
                os.makedirs(self.config['IMAGES_DIR'])
            
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
            image_path = os.path.join(self.config['IMAGES_DIR'], f'{model_name}_{timestamp}.png')
            plt.savefig(image_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"已保存训练图片: {image_path}")
            return image_path
            
        except Exception as e:
            print(f"保存训练图片时出错: {str(e)}")
            return None
    
    def save_model_comparison_plot(self, comparison_results):
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
            image_path = os.path.join(self.config['IMAGES_DIR'], f'model_comparison_{timestamp}.png')
            plt.savefig(image_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"已保存模型比较图片: {image_path}")
            return image_path
            
        except Exception as e:
            print(f"保存模型比较图时出错: {str(e)}")
            return None
    
    def train_models(self, df: pd.DataFrame, selected_models: List[str]):
        """训练多个模型"""
        results = {}
        training_images = []
        
        # 准备数据
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = self.prepare_data(df)
        
        for model_type in selected_models:
            try:
                model_config = self.get_model_config(model_type)
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
                image_path = self.save_training_plots(X_train_scaled, y_train, X_test_scaled, y_test, model, model_type)
                if image_path:
                    training_images.append({
                        'model': model_type,
                        'image_path': image_path
                    })
                
                # 保存模型
                model_filename = f"{model_type}_model.pkl"
                model_path = os.path.join(self.config['MODELS_DIR'], model_filename)
                joblib.dump(model, model_path)
                
                # 保存 scaler
                scaler_path = os.path.join(self.config['MODELS_DIR'], 'scaler.pkl')
                joblib.dump(scaler, scaler_path)
                
                self.models[model_type] = model
                self.scalers['default'] = scaler
                
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
        
        return results, training_images
    
    def compare_all_models(self, df: pd.DataFrame):
        """比较所有可用模型"""
        # 准备数据
        X_train_scaled, X_test_scaled, y_train, y_test, _ = self.prepare_data(df)
        
        # 所有可用模型
        all_models = [
            'logistic_regression', 'decision_tree', 'random_forest',
            'svm', 'gradient_boosting', 'xgboost', 'lightgbm',
            'neural_network', 'knn', 'naive_bayes', 'adaboost',
            'extra_trees', 'bagging', 'lda', 'qda'
        ]
        
        comparison = {}
        for model_type in all_models:
            try:
                model_config = self.get_model_config(model_type)
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
        comparison_image_path = self.save_model_comparison_plot(comparison)
        
        # 找到最佳模型
        successful_models = {k: v for k, v in comparison.items() if v['status'] == 'success' and v['accuracy'] is not None}
        if successful_models:
            best_model = max(successful_models.items(), key=lambda x: x[1]['accuracy'])
            best_model_name = best_model[0]
            best_accuracy = best_model[1]['accuracy']
        else:
            best_model_name = None
            best_accuracy = None
        
        return {
            'comparison': comparison,
            'best_model': best_model_name,
            'best_accuracy': best_accuracy,
            'model_count': len(comparison),
            'successful_models': len(successful_models),
            'comparison_image_path': comparison_image_path
        }