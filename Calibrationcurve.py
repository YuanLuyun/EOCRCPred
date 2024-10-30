import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.svm import FastKernelSurvivalSVM
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from xgbse import XGBSEKaplanNeighbors
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import matplotlib.backends.backend_pdf
import warnings
from pycox.models import CoxPH
from lifelines import KaplanMeierFitter
from sklearn.calibration import calibration_curve
# 忽略警告信息
warnings.filterwarnings("ignore")

# ==================== 数据预处理部分 ====================
# 加载数据并去掉 'Patient_ID' 列
data = pd.read_csv('data_encoded7408.csv')
if 'Patient_ID' in data.columns:
    data = data.drop(columns=['Patient_ID'])  # 删除 Patient_ID 列

# 构建生存数据
X = data.drop(columns=['OS_month', 'Survival_status'])
y = Surv.from_dataframe('Survival_status', 'OS_month', data)
y_time = data['OS_month'].values
y_event = data['Survival_status'].values

# 自动识别数值型特征和独热编码特征
numerical_features = ['Age', 'Grade', 'TNM_Stage', 'T', 'N', 'CEA', 'No.of_resected_LNs', 'Tumor_Deposits',
                      'Tumor_size', 'Median_household_income']
categorical_features = [col for col in X.columns if col not in numerical_features]

# 定义数值型特征的归一化处理
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', 'passthrough', categorical_features)
    ])

# 应用预处理
X_processed = preprocessor.fit_transform(X)

# 按照 8:2 划分训练集与测试集
X_train, X_test, y_train_time, y_test_time, y_train_event, y_test_event = train_test_split(
    X_processed, y_time, y_event, test_size=0.2, random_state=42
)

# 构建生存数据对象
y_train = Surv.from_arrays(event=y_train_event == 1, time=y_train_time)
y_test = Surv.from_arrays(event=y_test_event == 1, time=y_test_time)

# ==================== 外部验证集数据处理 ====================
external_data = pd.read_csv('external_validation_set.csv')
if 'Patient_ID' in external_data.columns:
    external_data = external_data.drop(columns=['Patient_ID'])

X_external = external_data.drop(columns=['OS_month', 'Survival_status'])
y_external_time = external_data['OS_month'].values
y_external_event = external_data['Survival_status'].values
y_external = Surv.from_arrays(event=y_external_event == 1, time=y_external_time)
X_external_processed = preprocessor.transform(X_external)

# ==================== DeepSurv 模型定义和训练 ====================
class CoxPHModel(nn.Module):
    def __init__(self, in_features, num_nodes, dropout):
        super(CoxPHModel, self).__init__()
        layers = []
        prev_nodes = in_features
        for nodes in num_nodes:
            layers.append(nn.Linear(prev_nodes, nodes))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_nodes = nodes
        layers.append(nn.Linear(prev_nodes, 1))  # 输出层
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 初始化DeepSurv网络并应用权重初始化
net = CoxPHModel(in_features=X_train.shape[1], num_nodes=[16, 16], dropout=0.1)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

net.apply(initialize_weights)

# 设置训练参数
optimizer = optim.Adam(net.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)

# 定义 EarlyStopping 类
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def step(self, loss):
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False
modelcoxPHdeepsurv = CoxPH(net, optimizer)
early_stopping = EarlyStopping(patience=10)

# 准备训练数据集
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_time_tensor = torch.tensor(y_train_time, dtype=torch.float32)
y_train_event_tensor = torch.tensor(y_train_event, dtype=torch.float32)
train_ds = TensorDataset(X_train_tensor, y_train_time_tensor, y_train_event_tensor)
train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)

# 训练DeepSurv模型
# 训练模型
epochs = 100
for epoch in range(epochs):
    modelcoxPHdeepsurv.net.train()
    total_loss = 0
    for x_batch, y_time_batch, y_event_batch in train_dl:
        optimizer.zero_grad()
        output = modelcoxPHdeepsurv.net(x_batch)
        loss = modelcoxPHdeepsurv.loss(output, y_time_batch, y_event_batch)
        if torch.isnan(loss):
            print(f"Epoch {epoch + 1}/{epochs}, Loss: nan detected")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelcoxPHdeepsurv.net.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_dl)
    print(f'Epoch {epoch + 1}, Average Loss: {average_loss:.4f}')
    scheduler.step(average_loss)
    if early_stopping.step(average_loss):
        print("Early stopping triggered.")
        break

# 添加计算风险分数并转换为事件发生概率的部分
def predict_event_prob_deepsurv(model, X, time_horizons, baseline_survival):
    """DeepSurv 模型事件发生概率计算"""
    with torch.no_grad():
        risk_scores = model(torch.tensor(X, dtype=torch.float32)).numpy().squeeze()
    # 根据基线生存函数计算生存概率
    survival_matrix = np.zeros((X.shape[0], len(time_horizons)))
    for i, risk_score in enumerate(risk_scores):
        for j, t in enumerate(time_horizons):
            survival_matrix[i, j] = baseline_survival.loc[t].values ** np.exp(risk_score)
    # 将生存概率转换为事件发生概率
    return 1 - survival_matrix

# ==================== 模型事件发生概率计算函数 ====================
def compute_event_prob(model, X, time_horizons, y_train=None, baseline_survival=None):
    """根据模型类型计算事件发生概率"""

    if isinstance(model, RandomSurvivalForest):
        # RandomSurvivalForest 事件发生概率计算
        y_pred_prob = model.predict_survival_function(X, return_array=True)
        # 获取对应的时间点索引
        time_points = np.linspace(0, y_pred_prob.shape[1] - 1, y_pred_prob.shape[1])
        # 根据 time_horizons 找到对应的索引
        time_indices = [np.argmin(np.abs(time_points - t)) for t in time_horizons]
        # 提取对应时间点的生存概率，并将其转换为事件发生概率
        return np.array([[1 - y_pred_prob[i][time_idx] for time_idx in time_indices] for i in range(len(y_pred_prob))])

    elif isinstance(model, CoxPHSurvivalAnalysis):
        # CoxPHSurvivalAnalysis 事件发生概率计算
        survival_functions = model.predict_survival_function(X)
        return np.array([[1 - fn(t) for t in time_horizons] for fn in survival_functions])

    elif isinstance(model, FastKernelSurvivalSVM):
        # FastKernelSurvivalSVM 事件发生概率计算

        if y_train is None:
            raise ValueError("y_train cannot be None for FastKernelSurvivalSVM")
        # 1. 获取SVM的风险得分
        risk_scores = model.predict(X)
        # 2. 使用Kaplan-Meier估计基线生存函数 S_0(t)
        kmf = KaplanMeierFitter()
        kmf.fit(y_train['time'], event_observed=y_train['event'])
        baseline_survival = kmf.survival_function_
        # 3. 计算生存概率
        survival_matrix = np.zeros((X.shape[0], len(time_horizons)))
        for i, risk_score in enumerate(risk_scores):
            for j, t in enumerate(time_horizons):
                survival_matrix[i, j] = baseline_survival.loc[t].values ** np.exp(risk_score)
        # 将生存概率转换为事件发生概率
        return 1 - survival_matrix

    elif isinstance(model, XGBSEKaplanNeighbors):
        # XGBSEKaplanNeighbors 事件发生概率计算
        return 1 - model.predict(X, time_bins=time_horizons)

    elif isinstance(model, GradientBoostingSurvivalAnalysis):
        # GradientBoostingSurvivalAnalysis 事件发生概率计算
        survival_functions = model.predict_survival_function(X)
        return np.array([[1 - fn(t) for t in time_horizons] for fn in survival_functions])



    elif isinstance(model, CoxPHModel):  # 修改后的 DeepSurv 预测逻辑

        # DeepSurv 需要传入 baseline_survival

        if baseline_survival is None:
            raise ValueError("baseline_survival cannot be None for CoxPHModel (DeepSurv)")

        return predict_event_prob_deepsurv(model, X, time_horizons, baseline_survival)


kmf = KaplanMeierFitter()
kmf.fit(y_train_time, event_observed=y_train_event)
baseline_survival = kmf.survival_function_
# 定义时间点
time_horizons = [12, 36, 60]
# 在计算DeepSurv模型的事件发生概率时传入 baseline_survival
event_prob_deepsurv_train = compute_event_prob(net, X_train, time_horizons, baseline_survival=baseline_survival)
event_prob_deepsurv_test = compute_event_prob(net, X_test, time_horizons, baseline_survival=baseline_survival)
event_prob_deepsurv_external = compute_event_prob(net, X_external_processed, time_horizons, baseline_survival=baseline_survival)

# ==================== 训练其他模型并计算事件发生概率 ====================
# 定义模型
models = {
    'RSF': RandomSurvivalForest(n_estimators=1625, max_depth=6, min_samples_split=2, min_samples_leaf=4, n_jobs=-1, random_state=42),
    'CoxPH': CoxPHSurvivalAnalysis(),
    'S-SVM': FastKernelSurvivalSVM(kernel="rbf", alpha=0.01, gamma=0.01, tol=1e-4, max_iter=1000, random_state=42),
    'XGBSE': XGBSEKaplanNeighbors(),
    'GBSA': GradientBoostingSurvivalAnalysis(),
    'DeepSurv': net  # 包含DeepSurv模型
}

# 训练模型
for name, model in models.items():
    if name != 'DeepSurv':  # DeepSurv 模型已经训练好，其他模型需要训练
        model.fit(X_train, y_train)

# 计算训练集的事件发生概率
# 计算训练集的事件发生概率
model_probs_train = {
    name: compute_event_prob(model, X_train, time_horizons,
                             y_train=y_train if name == 'S-SVM' else None,
                             baseline_survival=baseline_survival if name == 'DeepSurv' else None)
    for name, model in models.items()
}

# 计算测试集的事件发生概率
model_probs_test = {
    name: compute_event_prob(model, X_test, time_horizons,
                             y_train=y_train if name == 'S-SVM' else None,
                             baseline_survival=baseline_survival if name == 'DeepSurv' else None)
    for name, model in models.items()
}

# 计算外部验证集的事件发生概率
model_probs_external = {
    name: compute_event_prob(model, X_external_processed, time_horizons,
                             y_train=y_train if name == 'S-SVM' else None,
                             baseline_survival=baseline_survival if name == 'DeepSurv' else None)
    for name, model in models.items()
}

# 打印前几个事件发生概率结果
print("Training Set Event Probabilities:")
for name, probs in model_probs_train.items():
    print(f"Model: {name}")
    print(probs[:5])  # 打印前5个结果
    print("\n")

print("Test Set Event Probabilities:")
for name, probs in model_probs_test.items():
    print(f"Model: {name}")
    print(probs[:5])  # 打印前5个结果
    print("\n")

print("External Validation Set Event Probabilities:")
for name, probs in model_probs_external.items():
    print(f"Model: {name}")
    print(probs[:5])  # 打印前5个结果
    print("\n")

# ==================== 校准曲线绘制函数 ====================
def plot_calibration_curve(y_true, probs, time_horizon, ax, model_name, label):
    """绘制校准曲线"""
    # 按预测概率排序
    sorted_indices = np.argsort(probs)
    probs_sorted = probs[sorted_indices]
    y_true_sorted = y_true.iloc[sorted_indices]

    # 分成 10 组并计算平均值
    n_bins = 10
    bin_size = len(probs) // n_bins
    prob_true, prob_pred = [], []

    for i in range(n_bins):
        start, end = i * bin_size, (i + 1) * bin_size if i < n_bins - 1 else len(probs)
        indices = range(start, end)

        prob_pred_group = np.mean(probs_sorted[indices])
        prob_pred.append(prob_pred_group)

        y_time_group = y_true_sorted['time'].iloc[indices]
        y_event_group = y_true_sorted['event'].iloc[indices]

        kmf = KaplanMeierFitter()
        kmf.fit(y_time_group, event_observed=y_event_group)
        prob_true_group = kmf.predict(time_horizon)
        prob_true.append(1 - prob_true_group)

    # 绘制校准曲线
    ax.plot(prob_pred, prob_true, marker='o', label=model_name)
    ax.set_aspect('equal', 'box')
    # 设置横轴和纵轴标题
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('True Probability', fontsize=12)

# ==================== 校准曲线绘制并保存 ====================
pdf_filename_calibration = "Fig.5.pdf"
pdf_calibration = matplotlib.backends.backend_pdf.PdfPages(pdf_filename_calibration)

fig_calibration, axes_calibration = plt.subplots(3, 3, figsize=(15, 15))

datasets = ['Training cohort', 'Test cohort', 'External validation cohort']
model_probs_list = [model_probs_train, model_probs_test, model_probs_external]
y_trues = [pd.DataFrame({'time': y_train['time'], 'event': y_train['event']}),
           pd.DataFrame({'time': y_test['time'], 'event': y_test['event']}),
           pd.DataFrame({'time': y_external['time'], 'event': y_external['event']})]

# 生成子图标签（A-I）
labels = [chr(65 + i) for i in range(9)]  # 生成 ['A', 'B', 'C', ..., 'I']
# 遍历数据集并绘制校准曲线
for idx, (y_true, model_probs, dataset_name) in enumerate(zip(y_trues, model_probs_list, datasets)):
    for jdx, time_horizon in enumerate(time_horizons):
        ax = axes_calibration[idx, jdx]
        time_point = time_horizons[jdx]

        # 计算当前子图的标签索引
        label_index = idx * len(time_horizons) + jdx
        label = labels[label_index]  # 不需要担心索引超出范围，因为长度匹配

        # 绘制每个模型的校准曲线
        for model_name, probs in model_probs.items():
            probs_at_time = probs.iloc[:, jdx].values if isinstance(probs, pd.DataFrame) else probs[:, jdx]

            # 调用绘图函数，传递标签参数
            plot_calibration_curve(y_true, probs_at_time, time_point, ax, model_name, label)
        # 设置各列的特定坐标范围
        if jdx == 0:
            ax.set_xlim(0, 0.13)
            ax.set_ylim(0, 0.13)
        elif jdx == 1:
            ax.set_xlim(0, 0.45)
            ax.set_ylim(0, 0.45)
        elif jdx == 2:
            ax.set_xlim(0, 0.8)
            ax.set_ylim(0, 0.8)
        # 添加理想校准线
        perfect_line, = ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')

        # 设置标题和图例
        ax.set_title(f"{dataset_name} - {time_point} months")
        handles, legend_labels = ax.get_legend_handles_labels()
        ax.legend(handles, legend_labels, loc='upper left', fontsize=10)
        ax.text(-0.15, 1.15, label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

# 调整布局并保存PDF
fig_calibration.tight_layout(pad=4.0)
pdf_calibration.savefig(fig_calibration)
pdf_calibration.close()

print(f"所有校准曲线已经保存到 {pdf_filename_calibration}")
