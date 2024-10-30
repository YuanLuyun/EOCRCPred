import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from pycox.models import CoxPH
import torch.nn as nn
import numpy as np
from sksurv.metrics import brier_score
from sksurv.metrics import integrated_brier_score
from sklearn.utils import resample
import torchtuples as tt
import warnings
import os
from sksurv.util import Surv
from sklearn.metrics import brier_score_loss

# 忽略警告信息
warnings.filterwarnings("ignore")

# 加载数据并去掉 'Patient_ID' 列
data = pd.read_csv('data_encoded7408.csv')
external_data = pd.read_csv('external_validation_set.csv')

if 'Patient_ID' in data.columns:
    data = data.drop(columns=['Patient_ID'])
if 'Patient_ID' in external_data.columns:
    external_data = external_data.drop(columns=['Patient_ID'])

# 构建生存数据
y_time = data['OS_month'].values.astype('float32')
y_event = data['Survival_status'].values.astype(bool)
X = data.drop(columns=['OS_month', 'Survival_status'])

# 特征归一化
numerical_features = ['Age', 'Grade', 'TNM_Stage', 'T', 'N', 'CEA',
                      'No.of_resected_LNs', 'Tumor_Deposits',
                      'Tumor_size', 'Median_household_income']
categorical_features = [col for col in X.columns if col not in numerical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', 'passthrough', categorical_features)
    ])
X_processed = preprocessor.fit_transform(X).astype('float32')

# 数据划分
X_train, X_test, y_train_time, y_test_time, y_train_event, y_test_event = train_test_split(
    X_processed, y_time, y_event, test_size=0.2, random_state=42
)
# 将生存数据转换为 Surv 对象
y_train = Surv.from_arrays(event=y_train_event, time=y_train_time)
y_test = Surv.from_arrays(event=y_test_event, time=y_test_time)
# 定义神经网络
class Net(nn.Module):
    def __init__(self, in_features, num_nodes, dropout):
        super(Net, self).__init__()
        layers = []
        prev_nodes = in_features
        for nodes in num_nodes:
            layers.append(nn.Linear(prev_nodes, nodes))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(nodes))
            layers.append(nn.Dropout(dropout))
            prev_nodes = nodes
        layers.append(nn.Linear(prev_nodes, 1))  # 输出层
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

in_features = X_train.shape[1]
net = Net(in_features=in_features, num_nodes=[16, 16], dropout=0.1)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

net.apply(initialize_weights)

# 创建模型
model = CoxPH(net, tt.optim.Adam)

# 准备数据
train = (X_train, (y_train_time, y_train_event))
val = (X_test, (y_test_time, y_test_event))

# 训练模型
batch_size = 256
epochs = 100
callbacks = [tt.callbacks.EarlyStopping(patience=10)]
verbose = True

model.fit(*train, batch_size, epochs, callbacks, verbose, val_data=val)
# 计算基准风险函数
model.compute_baseline_hazards()
# 外部验证集
external_y_time = external_data['OS_month'].values.astype('float32')
external_y_event = external_data['Survival_status'].values.astype(bool)
external_X = external_data.drop(columns=['OS_month', 'Survival_status'])
external_X_processed = preprocessor.transform(external_X).astype('float32')
# 将外部验证集的生存数据转换为 Surv 对象
y_external = Surv.from_arrays(event=external_y_event, time=external_y_time)

# 评估模式下进行预测
with torch.no_grad():
    model.net.eval()  # 切换到评估模式
    # 获取预测的生存函数
    surv_train = model.predict_surv_df(X_train)
    surv_test = model.predict_surv_df(X_test)
    surv_external = model.predict_surv_df(external_X_processed)


# 定义计算Brier Score的函数，使用brier_score_loss
def calculate_brier_score(y_true, surv_func, times):
    brier_scores = {}
    for time_point in times:
        # 从生存函数中获取在 time_point 时间点的预测生存概率
        predicted_survival = surv_func.loc[time_point].values
        # 计算事件发生的预测概率
        predicted_probabilities = 1 - predicted_survival
        # 提取真实事件和时间
        event = y_true['event']
        time = y_true['time']
        # 计算实际事件发生状态
        actual_outcome = (time <= time_point) & event
        actual_outcome = actual_outcome.astype(int)
        # 计算Brier Score
        bsl = brier_score_loss(actual_outcome, predicted_probabilities)
        brier_scores[time_point] = bsl
    return brier_scores

# 定义时间点
specific_times = [12, 36, 60]

# 计算各时间点的Brier Score
brier_train = calculate_brier_score(y_train, surv_train, specific_times)
brier_test = calculate_brier_score(y_test, surv_test, specific_times)
brier_external = calculate_brier_score(y_external, surv_external, specific_times)

# 计算Integrated Brier Score的函数
def compute_integrated_brier_score(y_train, y_true, surv_func):
    # y_true 包含事件和生存时间
    event = y_true['event']
    time = y_true['time']

    # 动态计算最大时间和最小时间
    max_time = time.max()
    min_time = time.min()

    # 确保 times 是从 Pandas DataFrame 的 index 取得的
    times = surv_func.index.values
    # 限制 times 在 min_time 和 max_time 范围之间
    times = times[(times >= min_time) & (times < max_time)]

    preds = surv_func.loc[times].values.T  # 这里确保操作的是 Pandas DataFrame

    # 构建 structured 数组
    y_true_structured = np.array([(e, t) for e, t in zip(event, time)],
                                 dtype=[('event', 'bool'), ('time', 'float32')])

    # 计算 integrated Brier score
    ibs = integrated_brier_score(y_train, y_true_structured, preds, times)
    return ibs

# 计算训练集 Integrated Brier Score
integrated_brier_train = compute_integrated_brier_score(y_train, y_train, surv_train)

# 计算测试集 Integrated Brier Score
integrated_brier_test = compute_integrated_brier_score(y_train, y_test, surv_test)

# 计算外部验证集 Integrated Brier Score
integrated_brier_external = compute_integrated_brier_score(y_train, y_external, surv_external)


def bootstrap_confidence_interval(X, y_true, surv_func, n_iterations, times):
    np.random.seed(42)
    n_samples = X.shape[0]
    scores = {time: [] for time in times}
    integrated_scores = []

    for i in range(n_iterations):
        # 重采样索引
        indices = np.random.choice(n_samples, n_samples, replace=True)
        # 重采样 X 和 y_true
        y_resample_time = y_true['time'][indices]
        y_resample_event = y_true['event'][indices]
        # 构建重采样的 Surv 对象
        y_resample_surv = Surv.from_arrays(event=y_resample_event, time=y_resample_time)

        # 使用传入的 surv_func 对应的重采样索引来获取新的生存函数
        surv_func_resample = surv_func.iloc[:, indices]

        # 使用 brier_score_loss 计算 Brier Score
        brier_scores = calculate_brier_score(y_resample_surv, surv_func_resample, times)
        for time_point in times:
            scores[time_point].append(brier_scores[time_point])

        # 计算 Integrated Brier Score
        ibs = compute_integrated_brier_score(y_train, y_resample_surv, surv_func_resample)
        integrated_scores.append(ibs)

    # 计算置信区间
    confidence_intervals = {time: (np.percentile(scores[time], 2.5), np.percentile(scores[time], 97.5)) for time in times}
    integrated_confidence_interval = (np.percentile(integrated_scores, 2.5), np.percentile(integrated_scores, 97.5))
    return confidence_intervals, integrated_confidence_interval
# 定义自助法迭代次数
n_iterations = 100  # 你可以根据需要调整这个值
# 计算训练集的置信区间
ci_train, ci_integrated_train = bootstrap_confidence_interval(
    X_train, y_train, surv_train, n_iterations, specific_times
)

# 计算测试集的置信区间
ci_test, ci_integrated_test = bootstrap_confidence_interval(
    X_test, y_test, surv_test, n_iterations, specific_times
)

# 计算外部验证集的置信区间
ci_external, ci_integrated_external = bootstrap_confidence_interval(
    external_X_processed, y_external, surv_external, n_iterations, specific_times
)


# 你的结果数据
results = [
    f"{brier_train[12]:.3f} ({ci_train[12][0]:.3f}, {ci_train[12][1]:.3f})",
    f"{brier_train[36]:.3f} ({ci_train[36][0]:.3f}, {ci_train[36][1]:.3f})",
    f"{brier_train[60]:.3f} ({ci_train[60][0]:.3f}, {ci_train[60][1]:.3f})",
    f"{brier_test[12]:.3f} ({ci_test[12][0]:.3f}, {ci_test[12][1]:.3f})",
    f"{brier_test[36]:.3f} ({ci_test[36][0]:.3f}, {ci_test[36][1]:.3f})",
    f"{brier_test[60]:.3f} ({ci_test[60][0]:.3f}, {ci_test[60][1]:.3f})",
    f"{brier_external[12]:.3f} ({ci_external[12][0]:.3f}, {ci_external[12][1]:.3f})",
    f"{brier_external[36]:.3f} ({ci_external[36][0]:.3f}, {ci_external[36][1]:.3f})",
    f"{brier_external[60]:.3f} ({ci_external[60][0]:.3f}, {ci_external[60][1]:.3f})",
    f"{integrated_brier_train:.3f} ({ci_integrated_train[0]:.3f}, {ci_integrated_train[1]:.3f})",
    f"{integrated_brier_test:.3f} ({ci_integrated_test[0]:.3f}, {ci_integrated_test[1]:.3f})",
    f"{integrated_brier_external:.3f} ({ci_integrated_external[0]:.3f}, {ci_integrated_external[1]:.3f})"
]

# 标签数据
labels = [
    "Train at 12 months",
    "Train at 36 months",
    "Train at 60 months",
    "Test at 12 months",
    "Test at 36 months",
    "Test at 60 months",
    "External at 12 months",
    "External at 36 months",
    "External at 60 months",
    "Train integrated",
    "Test integrated",
    "External integrated"
]

# 打印带有标签的结果
for label, result in zip(labels, results):
    print(f"{label}: {result}")

# 指定CSV文件
csv_file = 'brierscore95.csv'

# 检查文件是否为空或不存在
if not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0:
    # 如果文件为空或不存在，创建新的DataFrame并写入标签和结果
    df = pd.DataFrame({"Label": ["Model"] + labels, "Result": ["DeepSurv"] + results})
else:
    # 如果文件存在且不为空，读取文件
    df = pd.read_csv(csv_file, header=None)

    # 确保 DataFrame 的行数足够长，至少能够容纳所有的 labels 和 results
    required_rows = max(len(labels) + 1, len(df))
    if len(df) < required_rows:
        additional_rows = required_rows - len(df)
        new_rows = pd.DataFrame([[None] * df.shape[1]] * additional_rows)
        df = pd.concat([df, new_rows], ignore_index=True)

    # 写入数据
    if len(df.columns) < 7:
        df[6] = None  # 如果没有第7列，则创建第7列

    df.iloc[0, 6] = 'DeepSurv'  # 在第7列的第一行写入"DeepSurv"
    df.iloc[1:len(labels)+1, 6] = results  # 在第7列写入结果

# 将修改后的数据写回CSV文件，不带列名
df.to_csv(csv_file, index=False, header=False)