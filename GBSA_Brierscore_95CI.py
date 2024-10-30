import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
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

# 使用基于梯度提升的生存分析模型
gbsa = GradientBoostingSurvivalAnalysis()
gbsa.fit(X_train, y_train)

# 定义预测的时间点
specific_times = np.array([12, 36, 60])  # 定义时间点，单位是“月”

# 计算生存概率
def calculate_survival_probabilities(model, X, specific_times):
    surv_funcs = model.predict_survival_function(X)
    surv_probs = []
    for idx, fn in enumerate(surv_funcs):
        # 在指定的时间点评估生存函数
        probs = fn(specific_times)
        # 只打印前10个样本的概率
        if idx < 10:
            print(f"Sample {idx + 1} survival probabilities at times {specific_times}: {probs}")
        surv_probs.append(probs)
    surv_probs = np.array(surv_probs)  # 转换为 (n_samples, n_times)
    return surv_probs

# 计算训练集、测试集和外部验证集的生存概率
print("Calculating survival probabilities for training set:")
surv_probs_train = calculate_survival_probabilities(gbsa, X_train, specific_times)

print("Calculating survival probabilities for test set:")
surv_probs_test = calculate_survival_probabilities(gbsa, X_test, specific_times)

print("Calculating survival probabilities for external validation set:")
surv_probs_external = calculate_survival_probabilities(gbsa, X_external, specific_times)

# 定义计算Brier Score的函数
def calculate_brier_score(y_true, surv_probs, specific_times):
    brier_scores = {}

    for idx, time in enumerate(specific_times):
        # 将生存概率转换为事件发生的概率
        predicted_probabilities = 1 - surv_probs[:, idx]

        # 只打印前10个样本的预测概率
        if idx < 10:
            print(f"Predicted probabilities at time {time} for the first 10 samples: {predicted_probabilities[:10]}")

        # 正确地访问 y_true 的字段
        actual_outcome = (y_true["OS_month"] <= time) & (y_true["Survival_status"] == 1)
        actual_outcome = actual_outcome.astype(int)
        print(f"Actual outcomes at time {time} for the first 10 samples: {actual_outcome[:10]}")

        brier_scores[time] = brier_score_loss(actual_outcome, predicted_probabilities)
        if idx < 10:
            print(f"Brier score at time {time}: {brier_scores[time]}")
    return brier_scores


# 计算各时间点的Brier Score
brier_train = calculate_brier_score(y_train, surv_probs_train, specific_times)
brier_test = calculate_brier_score(y_test, surv_probs_test, specific_times)
brier_external = calculate_brier_score(y_external, surv_probs_external, specific_times)

# 计算Integrated Brier Score的函数
def integrated_brier_score(y_true, surv_probs, max_time, n_steps=100):
    times = np.linspace(0, max_time, n_steps)
    brier_scores = np.zeros(n_steps)

    for i, time in enumerate(times):
        # 将生存概率转换为事件发生的概率
        predicted_probabilities = 1 - surv_probs[:, i] if i < surv_probs.shape[1] else 1 - surv_probs[:, -1]

        # 计算实际的结果（事件是否发生）
        actual_outcome = (y_true["OS_month"] <= time) & (y_true["Survival_status"] == 1)
        actual_outcome = actual_outcome.astype(int)

        # 计算 Brier Score
        brier_scores[i] = brier_score_loss(actual_outcome, predicted_probabilities)

    # 使用积分方法计算Integrated Brier Score
    integrated_brier = np.trapz(brier_scores, times) / max_time
    return integrated_brier

# 计算最大时间
max_time_train = y_train["OS_month"].max()
max_time_test = y_test["OS_month"].max()
max_time_external = y_external["OS_month"].max()

# 计算Integrated Brier Score
integrated_brier_train = integrated_brier_score(y_train, surv_probs_train, max_time_train)
integrated_brier_test = integrated_brier_score(y_test, surv_probs_test, max_time_test)
integrated_brier_external = integrated_brier_score(y_external, surv_probs_external, max_time_external)

# 计算Brier评分的95%置信区间，使用并行加速
def bootstrap_confidence_interval(y_true, X, n_iterations, max_time, specific_times, model):
    scores = {time: [] for time in specific_times}
    integrated_scores = []

    def compute_resample(seed):
        X_resampled, y_resampled = resample(X, y_true, random_state=seed)
        model.fit(X_resampled, y_resampled)
        surv_probs_resampled = calculate_survival_probabilities(model, X_resampled, specific_times)

        brier_scores = calculate_brier_score(y_resampled, surv_probs_resampled, specific_times)
        integrated_brier = integrated_brier_score(y_resampled, surv_probs_resampled, max_time)
        return brier_scores, integrated_brier

    results = Parallel(n_jobs=-1)(delayed(compute_resample)(i) for i in range(n_iterations))

    for res_brier, res_integrated in results:
        for time in specific_times:
            scores[time].append(res_brier[time])
        integrated_scores.append(res_integrated)

    confidence_intervals = {time: (np.percentile(scores[time], 2.5), np.percentile(scores[time], 97.5)) for time in specific_times}
    integrated_confidence_interval = (np.percentile(integrated_scores, 2.5), np.percentile(integrated_scores, 97.5))

    return confidence_intervals, integrated_confidence_interval

# 计算训练集、测试集和外部验证集的置信区间
ci_train, ci_integrated_train = bootstrap_confidence_interval(y_train, X_train, 100, max_time_train, specific_times, gbsa)
ci_test, ci_integrated_test = bootstrap_confidence_interval(y_test, X_test, 100, max_time_test, specific_times, gbsa)
ci_external, ci_integrated_external = bootstrap_confidence_interval(y_external, X_external, 100, max_time_external, specific_times, gbsa)

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

    if len(df.columns) < 6:
        df[5] = None

    df.iloc[0, 5] = 'GBSA'
    df.iloc[1:len(results) + 1, 5] = results

# 保存至CSV文件
df.to_csv(csv_file, index=False, header=False)
