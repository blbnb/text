# app_fixed.py - 主应用程序
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import os
from datetime import datetime

from data_processor import DataProcessor
from model_trainer import ModelTrainer
from predictor import Predictor

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

# 初始化处理器
data_processor = DataProcessor(CONFIG)
model_trainer = ModelTrainer(CONFIG)
predictor = Predictor(CONFIG)

def create_directories():
    """创建必要的目录"""
    for directory in [CONFIG['MODELS_DIR'], CONFIG['IMAGES_DIR']]:
        if not os.path.exists(directory):
            os.makedirs(directory)

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': list(model_trainer.models.keys())
    })

@app.route('/api/data/overview', methods=['GET'])
def get_data_overview():
    """获取数据集概览"""
    try:
        df = data_processor.load_data()
        df_clean = data_processor.clean_data(df)
        overview = data_processor.get_data_overview(df_clean)
        
        return jsonify({'status': 'success', 'data': overview})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/data/visualize', methods=['POST'])
def visualize_data():
    """数据可视化"""
    try:
        df = data_processor.load_data()
        df_clean = data_processor.clean_data(df)
        image_path = data_processor.save_data_visualization(df_clean)
        
        if image_path:
            return jsonify({
                'status': 'success',
                'message': '数据可视化图片已保存',
                'image_path': image_path,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'status': 'error', 'message': '保存图片失败'}), 500
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/model/train', methods=['POST'])
def train_models():
    """训练多个模型"""
    try:
        data = request.json or {}
        selected_models = data.get('models', ['logistic_regression', 'decision_tree', 'random_forest'])
        
        # 加载并处理数据
        df = data_processor.load_data()
        df_clean = data_processor.clean_data(df)
        
        # 保存数据可视化图片
        data_processor.save_data_visualization(df_clean)
        
        # 训练模型
        results, training_images = model_trainer.train_models(df_clean, selected_models)
        
        return jsonify({
            'status': 'success',
            'message': f'成功训练 {len([r for r in results.values() if r["status"] == "success"])} 个模型',
            'results': results,
            'training_images': training_images,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/model/predict', methods=['POST'])
def make_prediction():
    """使用模型进行预测"""
    try:
        data = request.json
        
        if not data:
            return jsonify({'status': 'error', 'message': '请求数据为空'}), 400
        
        model_type = data.get('model_type', 'logistic_regression')
        features = data.get('features', {})
        
        # 进行预测
        prediction_result = predictor.predict(model_type, features)
        
        if prediction_result['status'] == 'success':
            return jsonify(prediction_result)
        else:
            return jsonify(prediction_result), 404 if prediction_result['message'].find('未找到') != -1 else 500
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/model/compare', methods=['GET'])
def compare_all_models():
    """比较所有可用模型"""
    try:
        df = data_processor.load_data()
        df_clean = data_processor.clean_data(df)
        
        # 比较模型
        comparison_result = model_trainer.compare_all_models(df_clean)
        
        return jsonify({
            'status': 'success',
            **comparison_result,
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
    print("糖尿病预测系统 - 模块化重构版")
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
    
    print("\nAPI端点:")
    print("  GET  /api/data/overview      - 数据概览")
    print("  POST /api/data/visualize     - 数据可视化")
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