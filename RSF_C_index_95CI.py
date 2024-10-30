import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.utils import resample
from joblib import Parallel, delayed

# 加载和预处理数据
def load_and_prepare_data(file_path, drop_columns=None):
    data = pd.read_csv(file_path)
    if drop_columns:
        data = data.drop(columns=drop_columns)
    return data

# 分割数据集并提取生存时间和事件状态
def split_data(data_encoded, test_size=0.2, random_state=42):
    y = Surv.from_dataframe('Survival_status', 'OS_month', data_encoded)
    X = data_encoded.drop(['OS_month', 'Survival_status'], axis=1)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
# 计算置信区间，并支持特定时间点的 C-index 计算
def calculate_c_index_with_ci(model, X, y, time=None, n_bootstrap=100, alpha=0.95, n_jobs=-1):
    def bootstrap_c_index(X, y):
        while True:
            X_resampled, y_resampled = resample(X, y, random_state=None)

            # 确保至少有一个事件样本
            if np.sum(y_resampled['Survival_status'] == 1) > 0:
                break

        # 如果有 time 参数，截断生存时间
        if time is not None:
            y_resampled_at_time = np.array(
                [(min(y_i['OS_month'], time),
                  0 if y_i['OS_month'] > time else y_i['Survival_status']) for y_i in y_resampled],
                dtype=[('OS_month', 'f8'), ('Survival_status', '?')]
            )
        else:
            y_resampled_at_time = y_resampled

        return concordance_index_censored(
            y_resampled_at_time["Survival_status"], y_resampled_at_time["OS_month"], model.predict(X_resampled)
        )[0]

    # 并行计算多个 Bootstrap 样本的 C-index
    c_indices = Parallel(n_jobs=n_jobs)(delayed(bootstrap_c_index)(X, y) for _ in range(n_bootstrap))

    # 移除 NaN 并计算置信区间
    c_indices = np.array(c_indices)
    c_indices = c_indices[~np.isnan(c_indices)]

    # 计算 C-index 的下界和上界
    lower_bound = np.percentile(c_indices, (1 - alpha) / 2 * 100)
    upper_bound = np.percentile(c_indices, (1 + alpha) / 2 * 100)

    # 返回平均 C-index 及其置信区间
    return np.mean(c_indices), lower_bound, upper_bound

# 保存 C-index 和置信区间结果的函数
def save_c_index_to_csv(filename, labels, c_indices, lower_bounds, upper_bounds):
    # 尝试读取现有的 CSV 文件
    try:
        existing_df = pd.read_csv(filename, header=None)  # 不读取 header，直接读取所有行
    except FileNotFoundError:
        # 如果文件不存在，创建一个空的 DataFrame
        existing_df = pd.DataFrame()

    # 创建一个新的列来保存 C-index 结果
    rows = [["", "RSF"]]  # 第一行空行，写入标记
    for label, c_index, lower, upper in zip(labels, c_indices, lower_bounds, upper_bounds):
        formatted_result = f"{c_index:.3f} ({lower:.3f} , {upper:.3f})"
        rows.append([label, formatted_result])  # 每个 label 和结果作为一行，label 在第一列，结果在第二列

        # 将新的结果转换为 DataFrame，仅保留两列
        new_df = pd.DataFrame(rows, columns=[0, 1])  # 第一列是标签，第二列是 C-index 结果

        # 将新数据直接插入现有 DataFrame 的第一列和第二列，不管行数是否匹配
        # 如果现有文件不为空，则替换前两列，忽略行数差异
        if not existing_df.empty:
            existing_df.iloc[:new_df.shape[0], :2] = new_df.values  # 替换现有行的前两列
        else:
            # 如果文件为空，直接使用新创建的 DataFrame
            existing_df = new_df

        # 保存修改后的 DataFrame，覆盖文件
        existing_df.to_csv(filename, index=False, header=False)
# 计算 C-index 并存储结果
def evaluate_and_save_c_indices(rsf, X_train, y_train, X_test, y_test, X_external, y_external):
    print("进入 evaluate_and_save_c_indices 函数...")
    labels = [
        "Train C-index", "Test C-index", "External C-index",
        "Train C-index at 12 months", "Train C-index at 36 months", "Train C-index at 60 months",
        "Test C-index at 12 months", "Test C-index at 36 months", "Test C-index at 60 months",
        "External C-index at 12 months", "External C-index at 36 months", "External C-index at 60 months"
    ]

    c_indices = []
    lower_bounds = []
    upper_bounds = []

    # 计算 Train, Test, External 的 C-index
    for X, y, label in [(X_train, y_train, "Train"), (X_test, y_test, "Test"), (X_external, y_external, "External")]:
        c_index, lower, upper = calculate_c_index_with_ci(rsf, X, y)
        c_indices.append(c_index)
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        print(f"{label} C-index: {c_index:.3f} (95% CI: {lower:.3f} - {upper:.3f})")

    # 计算特定时间点（12, 36, 60个月）的 C-index
    time_points = [12, 36, 60]
    for time in time_points:
        for X, y, label in [(X_train, y_train, "Train"), (X_test, y_test, "Test"), (X_external, y_external, "External")]:
            c_index, lower, upper = calculate_c_index_with_ci(rsf, X, y, time)
            print(f"{label} C-index 计算结果: c_index={c_index}, lower={lower}, upper={upper}")
            c_indices.append(c_index)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
            print(f"{label} C-index at {time} months: {c_index:.3f} (95% CI: {lower:.3f} - {upper:.3f})")

    # 保存结果到 CSV
    save_c_index_to_csv('c-index95.csv', labels, c_indices, lower_bounds, upper_bounds)

# 主程序
if __name__ == "__main__":
    # 加载数据
    train_data = load_and_prepare_data('data_encoded7408.csv', drop_columns=['Patient_ID'])
    external_data = load_and_prepare_data('external_validation_set.csv')

    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = split_data(train_data)

    # 提取外部验证集
    y_external = Surv.from_dataframe('Survival_status', 'OS_month', external_data)
    X_external = external_data.drop(['OS_month', 'Survival_status'], axis=1)

    # 训练随机生存森林模型
    rsf = RandomSurvivalForest(n_estimators=1625, max_depth=6, min_samples_split=2, min_samples_leaf=4, random_state=42)
    rsf.fit(X_train, y_train)
    print("模型训练完成。")
    print("开始计算并保存 C-index...")
    # 计算并保存 C-index
    evaluate_and_save_c_indices(rsf, X_train, y_train, X_test, y_test, X_external, y_external)