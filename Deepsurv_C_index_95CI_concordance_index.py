import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from torch.utils.data import TensorDataset, DataLoader
from pycox.models import CoxPH
import torch.nn as nn
import numpy as np
from lifelines.utils import concordance_index
import warnings
import os

# ==================== 数据预处理部分 ====================

# 忽略警告信息
warnings.filterwarnings("ignore")

# 加载 data_encoded7408.csv 数据并去掉 'Patient_ID' 列
data = pd.read_csv('data_encoded7408.csv')
if 'Patient_ID' in data.columns:
    data = data.drop(columns=['Patient_ID'])  # 删除 Patient_ID 列

# 加载 external_validation_set.csv 数据并去掉 'Patient_ID' 列
external_data = pd.read_csv('external_validation_set.csv')

# 构建生存数据
y_time = data['OS_month'].values
y_event = data['Survival_status'].values
X = data.drop(columns=['OS_month', 'Survival_status'])

# 自动识别数值型特征和独热编码特征
numerical_features = ['Age', 'Grade', 'TNM_Stage', 'T', 'N', 'CEA',
                      'No.of_resected_LNs', 'Tumor_Deposits',
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

# 将数据转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train_time = torch.tensor(y_train_time, dtype=torch.float32)
y_train_event = torch.tensor(y_train_event, dtype=torch.float32)
y_test_time = torch.tensor(y_test_time, dtype=torch.float32)
y_test_event = torch.tensor(y_test_event, dtype=torch.float32)

# ==================== 模型定义与训练部分 ====================

# 定义神经网络的层结构
num_nodes = [16, 16]  # 两个隐藏层，每层16个神经元
dropout = 0.1  # dropout 率

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

# 初始化网络并应用权重初始化
net = CoxPHModel(in_features=X_train.shape[1], num_nodes=num_nodes, dropout=dropout)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

net.apply(initialize_weights)  # 应用权重初始化

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
            return False  # 不停止训练
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # 停止训练
            return False

# 使用CoxPH模型封装网络
model = CoxPH(net, optimizer)

# 准备训练数据集
batch_size = 256
train_ds = TensorDataset(X_train, y_train_time, y_train_event)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

early_stopping = EarlyStopping(patience=10)

# 训练过程
epochs = 100
for epoch in range(epochs):
    model.net.train()
    total_loss = 0
    for x_batch, y_time_batch, y_event_batch in train_dl:
        optimizer.zero_grad()
        output = model.net(x_batch)
        loss = model.loss(output, y_time_batch, y_event_batch)

        if torch.isnan(loss):
            print(f"Epoch {epoch + 1}/{epochs}, Loss: nan detected")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.net.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_dl)
    print(f'Epoch {epoch + 1}, Average Loss: {average_loss:.3f}')

    scheduler.step(average_loss)  # 调整学习率

    if early_stopping.step(average_loss):
        print("Early stopping triggered.")
        break

# ==================== C-Index计算部分 ====================

# 定义自助法计算C-index置信区间的函数（整体C-index）
def bootstrap_cindex(y_time, y_event, risk_scores, n_bootstraps=100, random_state=42):
    rng = np.random.default_rng(random_state)
    c_indices = []
    n = len(risk_scores)

    for _ in range(n_bootstraps):
        sample_indices = rng.integers(0, n, n)
        sample_y_time = y_time[sample_indices]
        sample_y_event = y_event[sample_indices].astype(bool)  # 转换为布尔类型
        sample_risk_scores = risk_scores[sample_indices]

        # 确保样本中至少有一个事件发生和一个删失
        if np.sum(sample_y_event) == 0 or np.sum(~sample_y_event) == 0:
            continue

        c_index = concordance_index(sample_y_time, -sample_risk_scores, sample_y_event)
        c_indices.append(c_index)

    if len(c_indices) == 0:
        raise ValueError("所有自助样本均无效，无法计算置信区间。")

    lower = np.percentile(c_indices, 2.5)
    upper = np.percentile(c_indices, 97.5)
    return lower, upper

# 定义自助法计算C-index置信区间的函数（特定时间点）
def bootstrap_cindex_at_timepoints(y_time, y_event, risk_scores, timepoint, n_bootstraps=100, random_state=42):
    rng = np.random.default_rng(random_state)
    c_indices = []
    n = len(risk_scores)

    for _ in range(n_bootstraps):
        sample_indices = rng.integers(0, n, n)
        sample_y_time = y_time[sample_indices]
        sample_y_event = y_event[sample_indices].astype(bool)  # 转换为布尔类型
        sample_risk_scores = risk_scores[sample_indices]

        # 筛选在特定时间点前发生事件的个体
        mask = sample_y_time <= timepoint
        filtered_y_time = sample_y_time[mask]
        filtered_y_event = sample_y_event[mask]
        filtered_risk_scores = sample_risk_scores[mask]

        # 确保样本中至少有一个事件发生和一个删失
        if np.sum(filtered_y_event) == 0 or np.sum(~filtered_y_event) == 0:
            continue

        c_index = concordance_index(filtered_y_time, -filtered_risk_scores, filtered_y_event)
        c_indices.append(c_index)

    if len(c_indices) == 0:
        raise ValueError(f"所有自助样本在时间点 {timepoint} 月均无效，无法计算置信区间。")

    lower = np.percentile(c_indices, 2.5)
    upper = np.percentile(c_indices, 97.5)
    return lower, upper

# 定义计算整体C-index及其置信区间的函数
def compute_overall_cindex_with_ci(model, X, y_time, y_event, n_bootstraps=100):
    model.net.eval()  # 切换到评估模式
    with torch.no_grad():
        risk_scores = model.net(X).squeeze().numpy()

    # 将 y_event 转换为布尔类型
    y_event_bool = y_event.numpy().astype(bool)

    # 计算整体C-Index
    c_index = concordance_index(y_time.numpy(), -risk_scores, y_event_bool)

    # 计算置信区间
    ci_lower, ci_upper = bootstrap_cindex(y_time.numpy(), y_event_bool, risk_scores, n_bootstraps)

    return c_index, ci_lower, ci_upper

# 定义计算特定时间点C-index及其置信区间的函数
def compute_cindex_at_timepoints_with_ci(model, X, y_time, y_event, timepoints, n_bootstraps=100):
    model.net.eval()  # 切换到评估模式
    with torch.no_grad():
        risk_scores = model.net(X).squeeze().numpy()

    c_indices = {}
    for t in timepoints:
        try:
            ci_lower, ci_upper = bootstrap_cindex_at_timepoints(
                y_time.numpy(),
                y_event.numpy(),
                risk_scores,
                timepoint=t,
                n_bootstraps=n_bootstraps
            )
            # 计算C-Index
            mask = y_time.numpy() <= t
            filtered_y_time = y_time.numpy()[mask]
            filtered_y_event = y_event.numpy().astype(bool)[mask]
            filtered_risk_scores = risk_scores[mask]
            c_index = concordance_index(filtered_y_time, -filtered_risk_scores, filtered_y_event)
            c_indices[t] = (c_index, ci_lower, ci_upper)
        except ValueError as e:
            c_indices[t] = (np.nan, np.nan, np.nan)
            print(e)

    return c_indices

# 计算C-index及其置信区间
timepoints = [12, 36, 60]

# 计算训练集的整体C-Index及置信区间
c_index_train_overall, ci_train_overall_lower, ci_train_overall_upper = compute_overall_cindex_with_ci(
    model, X_train, y_train_time, y_train_event, n_bootstraps=100
)

# 计算训练集在特定时间点的C-Index及置信区间
c_indices_train_timepoints = compute_cindex_at_timepoints_with_ci(
    model, X_train, y_train_time, y_train_event, timepoints, n_bootstraps=100
)

# 计算测试集的整体C-Index及置信区间
c_index_test_overall, ci_test_overall_lower, ci_test_overall_upper = compute_overall_cindex_with_ci(
    model, X_test, y_test_time, y_test_event, n_bootstraps=100
)

# 计算测试集在特定时间点的C-Index及置信区间
c_indices_test_timepoints = compute_cindex_at_timepoints_with_ci(
    model, X_test, y_test_time, y_test_event, timepoints, n_bootstraps=100
)

# 处理外部验证集数据
external_X_processed = preprocessor.transform(external_data.drop(columns=['OS_month', 'Survival_status']))
external_X = torch.tensor(external_X_processed, dtype=torch.float32)
external_y_time = torch.tensor(external_data['OS_month'].values, dtype=torch.float32)
external_y_event = torch.tensor(external_data['Survival_status'].values, dtype=torch.float32)

# 计算外部验证集的整体C-Index及置信区间
c_index_external_overall, ci_external_overall_lower, ci_external_overall_upper = compute_overall_cindex_with_ci(
    model, external_X, external_y_time, external_y_event, n_bootstraps=100
)

# 计算外部验证集在特定时间点的C-Index及置信区间
c_indices_external_timepoints = compute_cindex_at_timepoints_with_ci(
    model, external_X, external_y_time, external_y_event, timepoints, n_bootstraps=100
)

# 打印结果
print(f'Overall C-Index - Train: {c_index_train_overall:.3f} ({ci_train_overall_lower:.3f}, {ci_train_overall_upper:.3f})')
print(f'Overall C-Index - Test: {c_index_test_overall:.3f} ({ci_test_overall_lower:.3f}, {ci_test_overall_upper:.3f})')
print(f'Overall C-Index - External Validation: {c_index_external_overall:.3f} ({ci_external_overall_lower:.3f}, {ci_external_overall_upper:.3f})')

for t in timepoints:
    c, lower, upper = c_indices_train_timepoints[t]
    print(f'Train C-Index at {t} months: {c:.3f} ({lower:.3f}, {upper:.3f})')

for t in timepoints:
    c, lower, upper = c_indices_test_timepoints[t]
    print(f'Test C-Index at {t} months: {c:.3f} ({lower:.3f}, {upper:.3f})')

for t in timepoints:
    c, lower, upper = c_indices_external_timepoints[t]
    print(f'External C-Index at {t} months: {c:.3f} ({lower:.3f}, {upper:.3f})')

# ==================== 结果保存部分 ====================

# 将 C-index 及置信区间结果存入列表
results = [
    "DeepSurv"  # 第一行标签
]

# 添加训练集的整体C-Index
results.append(f"{c_index_train_overall:.3f} ({ci_train_overall_lower:.3f}, {ci_train_overall_upper:.3f})")
# 添加测试集的整体C-Index
results.append(f"{c_index_test_overall:.3f} ({ci_test_overall_lower:.3f}, {ci_test_overall_upper:.3f})")
# 添加外部验证集的整体C-Index
results.append(f"{c_index_external_overall:.3f} ({ci_external_overall_lower:.3f}, {ci_external_overall_upper:.3f})")
# 添加训练集在特定时间点的C-Index
for t in timepoints:
    c_index, ci_lower, ci_upper = c_indices_train_timepoints[t]
    results.append(f"{c_index:.3f} ({ci_lower:.3f}, {ci_upper:.3f})")

# 添加测试集在特定时间点的C-Index
for t in timepoints:
    c_index, ci_lower, ci_upper = c_indices_test_timepoints[t]
    results.append(f"{c_index:.3f} ({ci_lower:.3f}, {ci_upper:.3f})")

# 添加外部验证集在特定时间点的C-Index
for t in timepoints:
    c_index, ci_lower, ci_upper = c_indices_external_timepoints[t]
    results.append(f"{c_index:.3f} ({ci_lower:.3f}, {ci_upper:.3f})")

# 创建 DataFrame 用于存储结果
df_results = pd.DataFrame({'Deepsurv': results})

# 定义CSV文件路径
file_path = 'c-index95.csv'

# 检查文件是否存在
if os.path.exists(file_path):
    # 读取现有的CSV文件，不包含header
    existing_df = pd.read_csv(file_path, header=None)
else:
    # 如果文件不存在，创建一个空的DataFrame
    existing_df = pd.DataFrame()

# 确保DataFrame至少有7列（索引为0到6）
while existing_df.shape[1] < 7:
    existing_df[f'Unnamed:{existing_df.shape[1]}'] = np.nan

# 确保DataFrame有足够的行来存储结果
required_rows = len(results)
current_rows = existing_df.shape[0]
if current_rows < required_rows:
    # 计算需要添加的行数
    additional_rows = required_rows - current_rows
    # 创建空行
    empty_rows = pd.DataFrame([[np.nan] * existing_df.shape[1]] * additional_rows, columns=existing_df.columns)
    # 追加空行到现有DataFrame
    existing_df = pd.concat([existing_df, empty_rows], ignore_index=True)

# 将结果写入第七列（索引为6）
existing_df.iloc[:len(results), 6] = df_results['Deepsurv']

# 保存回CSV文件，不包含索引和表头
existing_df.to_csv(file_path, index=False, header=False)

print("\nC-index及其置信区间结果已成功添加到 'c-index95.csv' 的第七列。")
