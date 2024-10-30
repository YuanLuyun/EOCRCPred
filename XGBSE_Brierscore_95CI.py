import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgbse import XGBSEKaplanNeighbors
from sksurv.util import Surv
from sklearn.metrics import brier_score_loss
from sklearn.utils import resample
from joblib import Parallel, delayed
import os

# 读取原始数据，并去除不需要的列
data = pd.read_csv('data_encoded7408.csv')
data_encoded = data.drop(columns=['Patient_ID'])

# 提取生存时间和事件状态并划分训练集和验证集
y = Surv.from_dataframe('Survival_status', 'OS_month', data_encoded)
X = data_encoded.drop(['OS_month', 'Survival_status'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 读取外部验证集数据
data_external = pd.read_csv('external_validation_set.csv')
y_external = Surv.from_dataframe('Survival_status', 'OS_month', data_external)
X_external = data_external.drop(['OS_month', 'Survival_status'], axis=1)

# 使用 XGBSE 模型进行拟合
xgbse_model = XGBSEKaplanNeighbors()
xgbse_model.fit(X_train, y_train)

# 定义预测的时间点
specific_times = np.array([12, 36, 60])  # 定义时间点，单位是“月”

# 计算生存概率，使用 predict 来预测特定时间点的生存概率
predicted_survival_train = xgbse_model.predict(X_train, time_bins=specific_times)
predicted_survival_test = xgbse_model.predict(X_test, time_bins=specific_times)
predicted_survival_external = xgbse_model.predict(X_external, time_bins=specific_times)

# 定义计算Brier Score的函数
def calculate_brier_score(y_true, predicted_survival_probabilities, specific_times):
    brier_scores = {}
    for idx, time in enumerate(specific_times):
        # 打印 predicted_survival_probabilities 的类型
        # 使用 .iloc 进行 pandas DataFrame 切片
        predicted_probabilities = predicted_survival_probabilities.iloc[:, idx]  # 提取指定时间点的预测概率
        actual_outcome = (y_true["OS_month"] <= time) & (y_true["Survival_status"] == 1)
        actual_outcome = actual_outcome.astype(int)
        brier_scores[time] = brier_score_loss(actual_outcome, 1 - predicted_probabilities)
    return brier_scores

# 计算各时间点的Brier Score
brier_train = calculate_brier_score(y_train, predicted_survival_train, specific_times)
brier_test = calculate_brier_score(y_test, predicted_survival_test, specific_times)
brier_external = calculate_brier_score(y_external, predicted_survival_external, specific_times)

# 计算Integrated Brier Score的函数
def integrated_brier_score(y_true, predicted_survival_probabilities, max_time, n_steps=100):
    times = np.linspace(0, max_time, n_steps)
    brier_scores = np.zeros(n_steps)

    for i, time in enumerate(times):
        # 直接从 predicted_survival_probabilities 中获取对应时间点的生存概率
        # 这里我们假设 predicted_survival_probabilities 的列代表不同的时间点
        predicted_probabilities = predicted_survival_probabilities.iloc[:, i] if i < \
                                                                                 predicted_survival_probabilities.shape[
                                                                                     1] else predicted_survival_probabilities.iloc[
                                                                                             :, -1]

        # 计算实际事件发生的情况
        actual_outcome = (y_true["OS_month"] <= time) & (y_true["Survival_status"] == 1)
        actual_outcome = actual_outcome.astype(int)

        # 计算Brier分数
        brier_scores[i] = brier_score_loss(actual_outcome, 1 - predicted_probabilities)

    # 使用积分方法计算Integrated Brier Score
    integrated_brier = np.trapz(brier_scores, times) / max_time
    return integrated_brier


# 计算最大时间
max_time_train = y_train["OS_month"].max()
max_time_test = y_test["OS_month"].max()
max_time_external = y_external["OS_month"].max()

# 计算Integrated Brier Score
integrated_brier_train = integrated_brier_score(y_train, predicted_survival_train, max_time_train)
integrated_brier_test = integrated_brier_score(y_test, predicted_survival_test, max_time_test)
integrated_brier_external = integrated_brier_score(y_external, predicted_survival_external, max_time_external)

# 计算Brier评分的95%置信区间，使用并行加速
def bootstrap_confidence_interval(y_true, X, n_iterations, max_time, specific_times):
    scores = {time: [] for time in specific_times}
    integrated_scores = []

    # 定义 compute_resample 函数
    def compute_resample(seed):
        X_resampled, y_resampled = resample(X, y_true, random_state=seed)
        # 使用已经拟合的模型进行预测，修正这里的问题
        predicted_survival_resampled = xgbse_model.predict(X_resampled, time_bins=specific_times)
        # 计算Brier分数
        brier_scores = calculate_brier_score(y_resampled, predicted_survival_resampled, specific_times)
        integrated_brier = integrated_brier_score(y_resampled, predicted_survival_resampled, max_time)

        return brier_scores, integrated_brier

    # 使用 joblib 进行并行计算
    results = Parallel(n_jobs=-1)(delayed(compute_resample)(i) for i in range(n_iterations))

    for res_brier, res_integrated in results:
        for time in specific_times:
            scores[time].append(res_brier[time])
        integrated_scores.append(res_integrated)

    # 计算每个时间点的置信区间
    confidence_intervals = {time: (np.percentile(scores[time], 2.5), np.percentile(scores[time], 97.5)) for time in specific_times}
    integrated_confidence_interval = (np.percentile(integrated_scores, 2.5), np.percentile(integrated_scores, 97.5))

    return confidence_intervals, integrated_confidence_interval

# 计算训练集、测试集和外部验证集的置信区间
ci_train, ci_integrated_train = bootstrap_confidence_interval(y_train, X_train, 100, max_time_train, specific_times)
ci_test, ci_integrated_test = bootstrap_confidence_interval(y_test, X_test, 100, max_time_test, specific_times)
ci_external, ci_integrated_external = bootstrap_confidence_interval(y_external, X_external, 100, max_time_external, specific_times)

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
    df = pd.DataFrame({"Label": labels, "Result": ["XGBSE"] + results})
else:
    df = pd.read_csv(csv_file, header=None)
    required_rows = max(len(labels), len(results) + 1)

    if len(df) < required_rows:
        additional_rows = required_rows - len(df)
        new_rows = pd.DataFrame([[None, None]] * additional_rows)
        df = pd.concat([df, new_rows], ignore_index=True)

    if len(df.columns) < 5:
        df[4] = None

    df.iloc[0, 4] = 'XGBSE'
    df.iloc[1:len(results) + 1, 4] = results

# 保存至CSV文件
df.to_csv(csv_file, index=False, header=False)
