import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgbse import XGBSEKaplanNeighbors
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
import warnings
import os

# 忽略警告信息
warnings.filterwarnings("ignore")


# 定义计算C-index及其置信区间的函数
def bootstrap_c_index(y_true, risk_scores, n_bootstrap=100, random_state=42):
    rng = np.random.RandomState(random_state)
    indices = np.arange(len(y_true))
    c_indices = []

    for _ in range(n_bootstrap):
        sample_indices = rng.choice(indices, size=len(indices), replace=True)
        events = y_true['Survival_status'][sample_indices]
        if len(np.unique(events)) < 2:
            continue  # 跳过没有事件和删失的数据
        c_idx = concordance_index_censored(
            y_true['Survival_status'][sample_indices],
            y_true['OS_month'][sample_indices],
            risk_scores[sample_indices]
        )[0]
        c_indices.append(c_idx)

    if not c_indices:
        raise ValueError("未生成有效的自助样本。")

    lower = np.percentile(c_indices, 2.5)
    upper = np.percentile(c_indices, 97.5)
    mean = np.mean(c_indices)
    return mean, lower, upper


# 加载内部数据集并删除 'Patient_ID'
data = pd.read_csv('data_encoded7408.csv')
if 'Patient_ID' in data.columns:
    data = data.drop(columns=['Patient_ID'])

# 加载外部验证数据集并删除 'Patient_ID'
external_data = pd.read_csv('external_validation_set.csv')

# 构建生存数据
y = Surv.from_dataframe('Survival_status', 'OS_month', data)
X = data.drop(columns=['OS_month', 'Survival_status'])

y_external = Surv.from_dataframe('Survival_status', 'OS_month', external_data)
X_external = external_data.drop(columns=['OS_month', 'Survival_status'])

# 划分训练集和测试集（80-20）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 初始化并训练模型
xgbse = XGBSEKaplanNeighbors()
xgbse.fit(X_train, y_train)

# 预测内部训练集、内部测试集和外部验证集的生存概率
predicted_survival_train = xgbse.predict(X_train)
predicted_survival_test = xgbse.predict(X_test)
predicted_survival_external = xgbse.predict(X_external)

# 提取时间点
time_points = [float(col) for col in predicted_survival_test.columns]
time_bins = [12, 36, 60]
time_indices = []
for t in time_bins:
    idx = np.argmin(np.abs(np.array(time_points) - t))
    time_indices.append(idx)

# 将生存概率转换为NumPy数组
predicted_survival_train_np = predicted_survival_train.values
predicted_survival_test_np = predicted_survival_test.values
predicted_survival_external_np = predicted_survival_external.values

# 定义一个列表来存储C-index结果
c_index_results = []


# 定义一个函数来计算并存储C-index及其置信区间
def compute_c_index(group_name, y_true, predicted_survival_np):
    # 计算整体C-index
    risk_scores_overall = 1 - predicted_survival_np[:, -1]  # 使用最后一个时间点的生存概率
    c_index_overall, lower_overall, upper_overall = bootstrap_c_index(
        y_true, risk_scores_overall
    )
    print(f"{group_name} 整体 C-index: {c_index_overall:.3f} (95% CI: {lower_overall:.3f} - {upper_overall:.3f})")
    c_index_results.append(f"{c_index_overall:.3f} ({lower_overall:.3f} , {upper_overall:.3f})")

    # 计算特定时间点的C-index
    for i, time_point in enumerate(time_bins):
        surv_probs_at_time = predicted_survival_np[:, time_indices[i]]
        risk_scores = 1 - surv_probs_at_time
        c_index, lower, upper = bootstrap_c_index(
            y_true, risk_scores
        )
        print(f"{group_name} {time_point} 个月的 C-index: {c_index:.3f} (95% CI: {lower:.3f} - {upper:.3f})")
        c_index_results.append(f"{c_index:.3f} ({lower:.3f} , {upper:.3f})")


# 计算并存储内部训练集的C-index
compute_c_index("内部训练集", y_train, predicted_survival_train_np)

# 计算并存储内部测试集的C-index
compute_c_index("内部测试集", y_test, predicted_survival_test_np)

# 计算并存储外部验证集的C-index
compute_c_index("外部验证集", y_external, predicted_survival_external_np)

# 准备要写入CSV的数据
# 生成C-index结果的顺序：训练集整体C-index，测试集整体C-index，验证集整体C-index，
# 训练集12、36、60个月C-index，测试集12、36、60个月C-index，验证集12、36、60个月C-index
# 共12个结果

# 创建要写入的列表，第一行为'XGBSE'
csv_data = [['', '', '', '', 'XGBSE']]

# 按顺序添加C-index及其置信区间
for c_idx in c_index_results:
    csv_data.append(['', '', '', '', c_idx])

# 检查是否存在c-index95.csv文件
file_path = 'c-index95.csv'
if os.path.exists(file_path):
    # 读取现有的CSV文件，不包含header
    df_existing = pd.read_csv(file_path, header=None)
else:
    # 如果文件不存在，创建一个空的DataFrame
    df_existing = pd.DataFrame()

# 确保现有DataFrame有足够的行
required_rows = len(csv_data)
current_rows = df_existing.shape[0]
if current_rows < required_rows:
    # 补充空行
    additional_rows = required_rows - current_rows
    for _ in range(additional_rows):
        df_existing = df_existing.append(pd.Series([np.nan] * max(5, df_existing.shape[1])), ignore_index=True)

# 确保至少有5列
if df_existing.shape[1] < 5:
    for _ in range(5 - df_existing.shape[1]):
        df_existing[f'Unnamed:{df_existing.shape[1]}'] = np.nan

# 将新的数据添加到现有DataFrame的第五列
for i, row in enumerate(csv_data):
    if i < df_existing.shape[0]:
        df_existing.at[i, 4] = row[4]
    else:
        # 如果现有DataFrame行数不足，扩展DataFrame
        df_existing = df_existing.append(pd.Series([np.nan] * df_existing.shape[1]), ignore_index=True)
        df_existing.at[i, 4] = row[4]

# 保存回CSV文件，不包含索引和表头
df_existing.to_csv(file_path, index=False, header=False)

print("\nC-index及其置信区间结果已成功添加到 'c-index95.csv' 的第5列。")
