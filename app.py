# app.py 后端完整代码
import string

from PIL.Image import Image
from PIL.ImageDraw import ImageDraw
from PIL.ImageFont import ImageFont
from emails import Message
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, get_flashed_messages, session, send_file
from flask_mail import Mail
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import bcrypt
import torch
import torch.nn as nn
import numpy as np
import logging
import os

from numpy.random import random
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from sklearn.preprocessing import StandardScaler
import pandas as pd
from datetime import datetime
import re
import pickle
import io
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config.update({
    'SECRET_KEY': 'your_secret_key',
    'SQLALCHEMY_DATABASE_URI': 'sqlite:///users.db',
    'SQLALCHEMY_TRACK_MODIFICATIONS': False,
    'UPLOAD_FOLDER': 'uploads',
    'ALLOWED_EXTENSIONS': {'xls', 'xlsx'}
})

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    predictions = db.relationship('PredictionHistory', backref='user', lazy=True)
    optimizations = db.relationship('OptimizationHistory', backref='user', lazy=True)

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    data_type = db.Column(db.String(50), nullable=False)
    input_data = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class OptimizationHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    result_path = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Model Configuration
# MODEL_CONFIGS = {
#     'type1': {
#         'name': 'Co-flow',
#         'input_dim': 8,
#         'model_path': 'module/mlp_model_Co-flow.pth',
#         'train_file': 'train_file/Diameter-C-train.xlsx',
#         'features': ['D1', 'D2', 'D3', 'Q1', 'Q2', 's', 'μ1', 'μ2']
#     },
#     'type2': {
#         'name': 'T-junction',
#         'input_dim': 8,
#         'model_path': 'module/mlp_model_T-junction.pth',
#         'train_file': 'train_file/Diameter-T-train.xlsx',
#         'features': ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
#     },
#     'type3': {
#         'name': 'Flow-focusing',
#         'input_dim': 9,
#         'model_path': 'module/mlp_model_Flow-focusing.pth',
#         'train_file': 'train_file/Diameter-F-train.xlsx',
#         'features': ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9']
#     }
# }

class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

        for layer in [self.fc1, self.fc2, self.fc3]:
            torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)# ...保持原有模型结构不变...

class ModelManager:
    def __init__(self):
        self.scalers = {}
        self.models = {}

    def load_model(self, model_type: str):
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Invalid model type: {model_type}")

        if model_type not in self.models:
            config = MODEL_CONFIGS[model_type]

            scaler = StandardScaler()
            try:
                train_data = pd.read_excel(config['train_file'])
                X_train = train_data[config['features']].values
                scaler.fit(X_train)
                self.scalers[model_type] = scaler
            except Exception as e:
                app.logger.error(f"Failed to fit scaler for {config['name']}: {str(e)}")
                raise

            try:
                model = MLP(config['input_dim'])
                model.load_state_dict(torch.load(config['model_path'], map_location=torch.device('cpu')))
                model.eval()
                self.models[model_type] = model
            except Exception as e:
                app.logger.error(f"Failed to load model for {config['name']}: {str(e)}")
                raise

        return self.models[model_type], self.scalers[model_type]

    # ...保持原有模型加载逻辑...

    def load_coflow_model(self):
        if 'coflow' not in self.models:
            self.models['coflow'], self.scalers['coflow'] = self.load_model('type1')
        return self.models['coflow'], self.scalers['coflow']

model_manager = ModelManager()

# 新增优化路由


@app.route('/optimize', methods=['GET', 'POST'])
@login_required
def optimize():
    if request.method == 'POST':
        # 文件验证和处理逻辑
        if 'file' not in request.files:
            flash('请选择文件', 'danger')
            return redirect(url_for('optimize'))

        file = request.files['file']
        if file.filename == '':
            flash('未选择文件', 'danger')
            return redirect(url_for('optimize'))

        if not allowed_file(file.filename):
            flash('仅支持Excel文件', 'danger')
            return redirect(url_for('optimize'))

        try:
            # 读取和处理Excel文件
            df = pd.read_excel(file)
            required_columns = ['D1', 'D2', 'D3', 'σ', 'μ1', 'μ2', 'D']

            # 数据验证
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                flash(f'缺失必要列: {", ".join(missing)}', 'danger')
                return redirect(url_for('optimize'))

            # 加载模型
            model, scaler = model_manager.load_coflow_model()

            # 处理每一行数据
            # 处理每一行数据
            results = []
            for _, row in df.iterrows():
                best_params, error = genetic_optimization(row, model, scaler)
            # 处理每一行数据

                best_params, error = genetic_optimization(row, model, scaler)
                # 修复字典展开语法
                result_row = {
                **row.to_dict(),
                'Q1_pred': best_params[0],
                'Q2_pred': best_params[1],
                'Absolute_Error': error,
                }
                results.append(result_row)

            # 保存结果（移到循环外）
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filename = secure_filename(file.filename)
            result_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                       f'optim_{current_user.id}_{datetime.now().timestamp()}.xlsx')
            pd.DataFrame(results).to_excel(result_path, index=False)

            # 保存历史记录
            optim_record = OptimizationHistory(
                user_id=current_user.id,
                filename=filename,
                result_path=result_path
            )
            db.session.add(optim_record)
            db.session.commit()

            return send_file(result_path, as_attachment=True)

        except Exception as e:  # 修复except缩进
            logging.error(f"优化失败: {str(e)}")
            flash('文件处理失败，请检查格式是否正确', 'danger')

    # 获取历史记录
    optim_hist = OptimizationHistory.query.filter_by(user_id=current_user.id).all()
    return render_template('optimize.html', optim_hist=optim_hist)

def genetic_optimization(row, model, scaler):
    """遗传算法参数反推核心逻辑"""
    # 算法参数配置
    class GAConfig:
        population_size = 300     # 种群规模
        generations = 100        # 迭代次数
        mutation_rate = 0.2       # 变异概率
        crossover_rate = 0.9      # 交叉概率
        elite_ratio = 0.1        # 精英保留比例
        target_value = float(row['D'])  # 从输入行获取目标值

    # 获取固定参数
    fixed_params = row[['D1', 'D2', 'D3', 'σ', 'μ1', 'μ2']].values.astype(float)

    # 定义参数范围 (Q1和Q2的搜索范围)
    param_ranges = np.array([
        [1, 5000],   # Q1范围
        [1, 5000]    # Q2范围
    ])

    # 初始化种群
    def initialize_population():
        population = np.empty((GAConfig.population_size, 2))
        population[:, 0] = np.random.uniform(param_ranges[0][0], param_ranges[0][1], GAConfig.population_size)
        population[:, 1] = np.random.uniform(param_ranges[1][0], param_ranges[1][1], GAConfig.population_size)
        return population

    # 两点交叉
    def crossover(parent1, parent2):
        if np.random.rand() < GAConfig.crossover_rate:
            # 随机选择两个交叉点
            cx1, cx2 = sorted(np.random.choice(range(1, 2), 2, replace=False))
            child1 = np.concatenate([parent1[:cx1], parent2[cx1:cx2], parent1[cx2:]])
            child2 = np.concatenate([parent2[:cx1], parent1[cx1:cx2], parent2[cx2:]])
            return child1, child2
        return parent1.copy(), parent2.copy()

    # 非均匀变异
    def mutate(individual, generation):
        mutated = individual.copy()
        for i in range(2):
            if np.random.rand() < GAConfig.mutation_rate:
                # 随着代数增加减小变异幅度
                delta = np.random.normal(0, (param_ranges[i][1]-param_ranges[i][0])*(1-generation/GAConfig.generations))
                mutated[i] = np.clip(mutated[i] + delta, param_ranges[i][0], param_ranges[i][1])
        return mutated

    # 计算适应度
    def calculate_fitness(population):
        # 构造完整输入数据
        full_input = np.hstack((
            np.tile(fixed_params, (len(population), 1)),
            population
        ))

        # 标准化
        scaled_input = scaler.transform(full_input)
        tensor_input = torch.tensor(scaled_input, dtype=torch.float32)

        with torch.no_grad():
            predictions = model(tensor_input).numpy().flatten()

        # 适应度为误差的倒数
        errors = np.abs(predictions - GAConfig.target_value)
        return 1 / (1 + errors)

    # 进化流程
    population = initialize_population()
    best_error = float('inf')
    best_params = None

    for gen in range(GAConfig.generations):
        # 计算适应度
        fitness = calculate_fitness(population)

        # 更新最佳解
        current_best_idx = np.argmax(fitness)
        current_error = 1/fitness[current_best_idx] - 1
        if current_error < best_error:
            best_error = current_error
            best_params = population[current_best_idx]

        # 选择精英
        elite_size = int(GAConfig.population_size * GAConfig.elite_ratio)
        elite_indices = np.argsort(fitness)[-elite_size:]
        elites = population[elite_indices]

        # 轮盘赌选择
        parents = []
        prob = fitness / fitness.sum()
        for _ in range(GAConfig.population_size - elite_size):
            idx = np.random.choice(len(population), p=prob)
            parents.append(population[idx])

        # 交叉和变异
        children = []
        for i in range(0, len(parents)-1, 2):
            child1, child2 = crossover(parents[i], parents[i+1])
            children.append(mutate(child1, gen))
            children.append(mutate(child2, gen))

        # 生成新一代
        population = np.vstack([np.array(children), elites])

    return best_params, best_error



# 预测历史记录模型

# 客服常见问题模型
class CustomerServiceQuestion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String(255), nullable=False)
    answer = db.Column(db.Text, nullable=False)

# 用户上传数据模型
class UserUpload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    data = db.Column(db.Text, nullable=False)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)

# 用户收藏历史记录模型
class UserFavorite(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    history_id = db.Column(db.Integer, db.ForeignKey('prediction_history.id'), nullable=False)
    history = db.relationship('PredictionHistory', backref=db.backref('favorites', lazy=True))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# 初始化模型管理器
model_manager = ModelManager()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        password2 = request.form.get('password2')

        if User.query.filter_by(username=username).first():
            flash('用户名已存在，请选择其他用户名。', 'danger')
            return redirect(url_for('register'))

        elif password != password2:
            flash('前后密码不一致，请重新输入。', 'danger')
            return redirect(url_for('register'))

        # 检查密码长度
        elif len(password) < 6:
            flash('密码长度至少为 6 位，且密码必须包含数字、字母或特殊字符中的至少两种。', 'danger')
            return redirect(url_for('register'))

        # 检查密码复杂度（至少包含数字、字母或字符中的两种）
        has_digit = re.search(r'\d', password) is not None
        has_letter = re.search(r'[a-zA-Z]', password) is not None
        has_special_char = re.search(r'[^a-zA-Z0-9]', password) is not None

        # 至少满足两种条件
        if sum([has_digit, has_letter, has_special_char]) < 2:
            flash('密码长度至少为 6 位，且密码必须包含数字、字母或特殊字符中的至少两种。', 'danger')
            return redirect(url_for('register'))

        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        new_user = User(username=username, password=hashed)
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('注册成功，请登录。', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Registration error: {str(e)}")
            flash('注册失败，请稍后再试。', 'danger')
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        # 验证码验证逻辑
        user_captcha = request.form.get('captcha', '').strip().lower()

        # 验证验证码
        stored_captcha = session.get('captcha', '').lower()
        if not stored_captcha or user_captcha != stored_captcha:
            flash('验证码错误。', 'danger')
            session.pop('captcha', None)
            return redirect(url_for('login'))

        user = User.query.filter_by(username=username).first()

        if user and bcrypt.checkpw(password.encode('utf-8'), user.password):
            login_user(user)
            flash('登录成功。', 'success')
            #  登录成功后清除验证码
            session.pop('captcha', None)
            return redirect(url_for('index'))
        else:
            flash('用户名或密码错误。', 'danger')
            #  登录失败后清除验证码
            session.pop('captcha', None)
            return redirect(url_for('login'))

    return render_template('login.html')

# 新增验证码生成路由
@app.route('/captcha')
def captcha():
    # 生成随机4位验证码（字母+数字）
    captcha_text = ''.join(random.choices(string.ascii_letters + string.digits, k=4))

    # 创建图片对象（包含干扰线、噪点等安全特性）
    image = Image.new('RGB', (120, 40), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    for i, char in enumerate(captcha_text):
        draw.text((10 + i*25 + random.randint(-3,3), 5 + random.randint(-3,3)),
                  char, font=font, fill=(random.randint(0,150), random.randint(0,150), random.randint(0,150)))

    for _ in range(5):
        x1 = random.randint(0, 120)
        y1 = random.randint(0, 40)
        x2 = random.randint(0, 120)
        y2 = random.randint(0, 40)
        draw.line((x1, y1, x2, y2), fill=(0, 0, 0), width=1)

    for _ in range(200):
        x = random.randint(0, 120)
        y = random.randint(0, 40)
        draw.point((x, y), fill=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))

    buf = io.BytesIO()
    image.save(buf, 'png')
    buf.seek(0)

    session['captcha'] = captcha_text
    return send_file(buf, mimetype='image/png')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('已退出登录。', 'success')
    get_flashed_messages()
    return redirect(url_for('index'))

@app.route('/history')
@login_required
def history():
    history = PredictionHistory.query.filter_by(user_id=current_user.id).all()
    return render_template('history.html', history=history)

@app.route('/customer_service', methods=['GET'])
@login_required
def customer_service():
    questions = CustomerServiceQuestion.query.all()
    return render_template('customer_service.html', questions=questions)

@app.route('/customer_service_admin', methods=['GET', 'POST'])
@login_required
def customer_service_admin():
    if current_user.username != 'maker' or '1':
        flash('你没有权限访问该页面。', 'danger')
        return redirect(url_for('index'))

    if request.method == 'POST':
        question = request.form.get('question')
        answer = request.form.get('answer')
        question_id = request.form.get('question_id')

        if question_id:
            # 修改问题
            cs_question = CustomerServiceQuestion.query.get(question_id)
            cs_question.question = question
            cs_question.answer = answer
        else:
            # 添加问题
            cs_question = CustomerServiceQuestion(question=question, answer=answer)
            db.session.add(cs_question)

        db.session.commit()
        flash('问题已更新。', 'success')
        return redirect(url_for('customer_service_admin'))

    questions = CustomerServiceQuestion.query.all()
    return render_template('customer_service_admin.html', questions=questions)

@app.route('/upload_data', methods=['POST'])
@login_required
def upload_data():
    data = request.form.get('data')
    if data:
        upload = UserUpload(user_id=current_user.id, data=data)
        db.session.add(upload)
        db.session.commit()
        flash('数据上传成功。', 'success')
    else:
        flash('请提供有效数据。', 'danger')
    return redirect(url_for('index'))

@app.route('/favorite_history/<int:history_id>', methods=['POST'])
@login_required
def favorite_history(history_id):
    history = PredictionHistory.query.get(history_id)
    if history and history.user_id == current_user.id:
        favorite = UserFavorite(user_id=current_user.id, history_id=history_id)
        db.session.add(favorite)
        db.session.commit()
        flash('历史记录已收藏。', 'success')
    else:
        flash('无法收藏该历史记录。', 'danger')
    return redirect(url_for('history'))

@app.route('/export_history', methods=['GET'])
@login_required
def export_history():
    # 查询用户的历史记录
    history = PredictionHistory.query.filter_by(user_id=current_user.id).all()

    # 准备数据
    data = []
    for record in history:
        data.append({
            '数据类型': record.data_type,
            '输入数据': record.input_data,
            '预测结果': record.prediction,
            '时间': record.timestamp
        })

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 创建输出流
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')

    # 将DataFrame写入Excel
    df.to_excel(writer, sheet_name='历史记录', index=False)

    # 完成写入并关闭writer
    writer.close()

    # 将指针移至输出流的开始位置
    output.seek(0)

    # 构造响应
    response = send_file(
        output,
        as_attachment=True,
        download_name='历史记录.xlsx',
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

    return response

# 配置邮件
app.config['MAIL_SERVER'] = 'smtp.qq.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = '2354669103@qq.com'#发送者的邮箱
app.config['MAIL_PASSWORD'] = 'lrzetsahwwftebbc'#发送者的授权码
mail = Mail(app)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/feedback', methods=['POST'])
@login_required
def feedback():
    feedback_content = request.form.get('feedback')

    if feedback_content:
        msg = Message('用户反馈', sender='2354669103@qq.com', recipients=['1686795835@qq.com'])#recipients是接受邮件的地址
        msg.body = f'用户 {current_user.username} 反馈：{feedback_content} '

        try:
            mail.send(msg)
            flash('反馈已提交，感谢你的支持。', 'success')
        except Exception as e:
            app.logger.error(f"反馈发送失败: {str(e)}")
            flash('反馈提交失败，请稍后再试。', 'danger')
    else:
        flash('请提供有效反馈内容。', 'danger')
    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        data_type = data.get('dataType')
        if data_type not in MODEL_CONFIGS:
            return jsonify({'error': f'Invalid data type: {data_type}'}), 400

        input_data = np.array(data.get('inputData')).reshape(1, -1)

        if not isinstance(input_data, np.ndarray) or input_data.shape[1] != MODEL_CONFIGS[data_type]['input_dim']:
            return jsonify({'error': 'Invalid input dimensions'}), 400

        if np.isnan(input_data).any() or np.isinf(input_data).any():
            return jsonify({'error': 'Input contains invalid values (NaN or inf)'}), 400

        model, scaler = model_manager.load_model(data_type)

        input_scaled = scaler.transform(input_data)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        with torch.no_grad():
            output = model(input_tensor)
            prediction = output.item()

        if np.isnan(prediction):
            raise ValueError("Model produced NaN prediction")

        # 保存预测历史记录
        history_record = PredictionHistory(
            user_id=current_user.id,
            data_type=data_type,
            input_data=str(data.get('inputData')),
            prediction=prediction
        )
        db.session.add(history_record)
        db.session.commit()

        app.logger.info(f"Successful prediction for {MODEL_CONFIGS[data_type]['name']}: {prediction}")
        return jsonify({'prediction': prediction})

    except ValueError as ve:
        app.logger.error(f"Value error in prediction: {str(ve)}")
        return jsonify({'error': '输入数据或模型预测结果无效，请检查输入。'}), 400
    except FileNotFoundError as fnfe:
        app.logger.error(f"File not found error in prediction: {str(fnfe)}")
        return jsonify({'error': '模型或训练数据文件未找到，请联系管理员。'}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error in prediction: {str(e)}")
        return jsonify({'error': '预测过程中发生未知错误，请稍后再试。'}), 500# 保持其他原有路由和功能不变...

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True)
