import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
import warnings
import os

# 忽略警告信息
warnings.filterwarnings("ignore")

# 加载 data_encoded7408.csv 数据并去掉 'Patient_ID' 列
data = pd.read_csv('data_encoded7408.csv')
if 'Patient_ID' in data.columns:
    data = data.drop(columns=['Patient_ID'])  # 删除 Patient_ID 列

# 加载 external_validation_set.csv 数据并去掉 'Patient_ID' 列
external_data = pd.read_csv('external_validation_set.csv')


# 构建生存数据
y = Surv.from_dataframe('Survival_status', 'OS_month', data)
X = data.drop(columns=['OS_month', 'Survival_status'])

# 外部验证集生存数据的构建
y_external = Surv.from_dataframe('Survival_status', 'OS_month', external_data)
X_external = external_data.drop(columns=['OS_month', 'Survival_status'])

# 按照 8:2 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练Cox比例风险模型
coxph = CoxPHSurvivalAnalysis()
coxph.fit(X_train, y_train)


# C-index 计算函数
def calculate_c_index(y_true, y_pred):
    return concordance_index_censored(y_true["Survival_status"], y_true["OS_month"], y_pred)[0]


# 使用 bootstrapping 计算置信区间
def bootstrap_ci(y_true, predictions, n_bootstraps=100):
    indices = np.arange(len(predictions))
    c_indices = []

    # Bootstrapping
    for _ in range(n_bootstraps):
        sample_indices = np.random.choice(indices, size=len(indices), replace=True)
        sample_y_true = y_true[sample_indices]
        sample_predictions = predictions[sample_indices]
        c_index = calculate_c_index(sample_y_true, sample_predictions)
        c_indices.append(c_index)

    # 计算置信区间
    c_indices = np.array(c_indices)
    lower_bound = np.percentile(c_indices, 2.5)
    upper_bound = np.percentile(c_indices, 97.5)
    return lower_bound, upper_bound


# 计算训练集、测试集和外部验证集的 C-index 及其置信区间
train_predictions = coxph.predict(X_train)
train_ci_lower, train_ci_upper = bootstrap_ci(y_train, train_predictions)
c_index_train_coxph = calculate_c_index(y_train, train_predictions)

test_predictions = coxph.predict(X_test)
test_ci_lower, test_ci_upper = bootstrap_ci(y_test, test_predictions)
c_index_test_coxph = calculate_c_index(y_test, test_predictions)

external_predictions = coxph.predict(X_external)
external_ci_lower, external_ci_upper = bootstrap_ci(y_external, external_predictions)
c_index_external = calculate_c_index(y_external, external_predictions)

# 将 C-index 及置信区间结果存入列表
results = [
    f"{c_index_train_coxph:.3f} ({train_ci_lower:.3f}, {train_ci_upper:.3f})",  # 训练集 C-index
    f"{c_index_test_coxph:.3f} ({test_ci_lower:.3f}, {test_ci_upper:.3f})",  # 测试集 C-index
    f"{c_index_external:.3f} ({external_ci_lower:.3f}, {external_ci_upper:.3f})"  # 外部验证集 C-index
]

# 打印训练集、测试集和外部验证集的 C-index 及置信区间
print(f"Train C-index: {c_index_train_coxph:.3f} (95% CI: {train_ci_lower:.3f}, {train_ci_upper:.3f})")
print(f"Test C-index: {c_index_test_coxph:.3f} (95% CI: {test_ci_lower:.3f}, {test_ci_upper:.3f})")
print(f"External C-index: {c_index_external:.3f} (95% CI: {external_ci_lower:.3f}, {external_ci_upper:.3f})")

# 特定时间点的计算
time_points = [12, 36, 60]

# 在特定时间点筛选数据并计算 C-index 和置信区间
for time_point in time_points:
    # 训练集
    train_mask = y_train['OS_month'] >= time_point
    y_train_filtered, X_train_filtered = y_train[train_mask], X_train[train_mask]
    if len(y_train_filtered) > 0:
        train_at_time_pred = coxph.predict(X_train_filtered)
        c_index_train_at_time = calculate_c_index(y_train_filtered, train_at_time_pred)
        ci_train_lower, ci_train_upper = bootstrap_ci(y_train_filtered, train_at_time_pred)
        results.append(f"{c_index_train_at_time:.3f} ({ci_train_lower:.3f}, {ci_train_upper:.3f})")
        print(
            f"Train C-index at {time_point} months: {c_index_train_at_time:.3f} (95% CI: {ci_train_lower:.3f}, {ci_train_upper:.3f})")
    else:
        results.append("NaN (NaN, NaN)")
        print(f"Train C-index at {time_point} months: NaN (No data)")

    # 测试集
    test_mask = y_test['OS_month'] >= time_point
    y_test_filtered, X_test_filtered = y_test[test_mask], X_test[test_mask]
    if len(y_test_filtered) > 0:
        test_at_time_pred = coxph.predict(X_test_filtered)
        c_index_test_at_time = calculate_c_index(y_test_filtered, test_at_time_pred)
        ci_test_lower, ci_test_upper = bootstrap_ci(y_test_filtered, test_at_time_pred)
        results.append(f"{c_index_test_at_time:.3f} ({ci_test_lower:.3f}, {ci_test_upper:.3f})")
        print(
            f"Test C-index at {time_point} months: {c_index_test_at_time:.3f} (95% CI: {ci_test_lower:.3f}, {ci_test_upper:.3f})")
    else:
        results.append("NaN (NaN, NaN)")
        print(f"Test C-index at {time_point} months: NaN (No data)")

    # 外部验证集
    external_mask = y_external['OS_month'] >= time_point
    y_external_filtered, X_external_filtered = y_external[external_mask], X_external[external_mask]
    if len(y_external_filtered) > 0:
        external_at_time_pred = coxph.predict(X_external_filtered)
        c_index_external_at_time = calculate_c_index(y_external_filtered, external_at_time_pred)
        ci_external_lower, ci_external_upper = bootstrap_ci(y_external_filtered, external_at_time_pred)
        results.append(f"{c_index_external_at_time:.3f} ({ci_external_lower:.3f}, {ci_external_upper:.3f})")
        print(
            f"External C-index at {time_point} months: {c_index_external_at_time:.3f} (95% CI: {ci_external_lower:.3f}, {ci_external_upper:.3f})")
    else:
        results.append("NaN (NaN, NaN)")
        print(f"External C-index at {time_point} months: NaN (No data)")

# 创建 DataFrame
df_results = pd.DataFrame({'CoxPH': results})

# 将结果写入 CSV 文件的第四列
file_path = 'c-index95.csv'

try:
    # 读取现有的 CSV 文件，不包含 header
    existing_df = pd.read_csv(file_path, header=None)

    # 确保现有 DataFrame 有至少 3 列
    while existing_df.shape[1] < 3:
        existing_df[f'Unnamed:{existing_df.shape[1]}'] = np.nan

    # 将 'XGBSE' 写入第一行的第3列（索引为2）
    existing_df.at[0, 2] = 'CoxPH'  # 根据您的最新要求，这里应该写入 'CoxPH'，而不是 'XGBSE'

    # 将 C-index 结果从第二行开始写入第3列
    for i, c_idx in enumerate(df_results['CoxPH'], start=1):
        if i < existing_df.shape[0]:
            existing_df.at[i, 2] = c_idx
        else:
            # 如果现有 DataFrame 行数不足，添加新行
            new_row = [np.nan] * existing_df.shape[1]
            new_row[2] = c_idx
            existing_df = existing_df.append(pd.Series(new_row), ignore_index=True)
except FileNotFoundError:
    # 如果文件不存在，创建一个新的 DataFrame
    new_rows = [['', '', '', 'CoxPH']]  # 第一行，第3列写入 'CoxPH'
    for c_idx in df_results['CoxPH']:
        new_rows.append(['', '', '', c_idx])
    existing_df = pd.DataFrame(new_rows)

# 保存回 CSV 文件，不包含索引和表头
existing_df.to_csv(file_path, index=False, header=False)

print("\nC-index及其置信区间结果已成功添加到 'c-index95.csv' 的第3列。")
