# data_processor.py - 数据处理模块
import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

class DataProcessor:
    """数据处理类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """加载数据"""
        try:
            if data_path is None:
                data_path = self.config['DATA_FILE']
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"数据文件 {data_path} 不存在")
            
            df = pd.read_csv(data_path)
            print(f"成功加载数据: {df.shape[0]} 行, {df.shape[1]} 列")
            
            return df
            
        except Exception as e:
            print(f"加载数据时出错: {str(e)}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
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
    
    def get_data_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取数据概览"""
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
        return overview
    
    def save_data_visualization(self, df: pd.DataFrame) -> str:
        """保存数据可视化图片"""
        try:
            if not os.path.exists(self.config['IMAGES_DIR']):
                os.makedirs(self.config['IMAGES_DIR'])
            
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
            image_path = os.path.join(self.config['IMAGES_DIR'], f'data_visualization_{timestamp}.png')
            plt.savefig(image_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"已保存数据可视化图片: {image_path}")
            return image_path
            
        except Exception as e:
            print(f"保存数据可视化时出错: {str(e)}")
            return None