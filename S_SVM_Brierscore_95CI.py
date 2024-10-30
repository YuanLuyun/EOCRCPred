import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.svm import FastKernelSurvivalSVM
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

# 使用指定的参数初始化生存支持向量机模型
svm = FastKernelSurvivalSVM(kernel="rbf", alpha=0.01, gamma=0.01, tol=1e-4, max_iter=1000, random_state=42)

# 训练 SVM 模型
svm.fit(X_train, y_train)

# 计算 SVM 的预测结果
predicted_survival_train = svm.predict(X_train)
predicted_survival_test = svm.predict(X_test)
predicted_survival_external = svm.predict(X_external)
# 打印训练集的生存预测值
print("Predicted Survival Values - Train Set:")
print(predicted_survival_train[:5])  # 打印前5个结果

# 打印测试集的生存预测值
print("Predicted Survival Values - Test Set:")
print(predicted_survival_test[:5])  # 打印前5个结果

# 打印外部验证集的生存预测值
print("Predicted Survival Values - External Validation Set:")
print(predicted_survival_external[:5])  # 打印前5个结果

# 定义计算Brier Score的函数
def calculate_brier_score(y_true, predicted_survival_values, specific_times):
    brier_scores = {}
    for time in specific_times:
        predicted_probabilities = np.clip(predicted_survival_values, 0, 1)
        actual_outcome = (y_true["OS_month"] <= time) & (y_true["Survival_status"] == 1)
        actual_outcome = actual_outcome.astype(int)
        brier_scores[time] = brier_score_loss(actual_outcome, predicted_probabilities)
    return brier_scores

# 定义时间点
specific_times = [12, 36, 60]

# 计算各时间点的Brier Score
brier_train = calculate_brier_score(y_train, predicted_survival_train, specific_times)
brier_test = calculate_brier_score(y_test, predicted_survival_test, specific_times)
brier_external = calculate_brier_score(y_external, predicted_survival_external, specific_times)

# 计算Integrated Brier Score的函数
def integrated_brier_score(y_true, predicted_survival_values, max_time, n_steps=100):
    times = np.linspace(0, max_time, n_steps)
    brier_scores = np.zeros(n_steps)
    predicted_probabilities = np.clip(predicted_survival_values, 0, 1)
    for i, time in enumerate(times):
        actual_outcome = (y_true["OS_month"] <= time) & (y_true["Survival_status"] == 1)
        actual_outcome = actual_outcome.astype(int)
        brier_scores[i] = brier_score_loss(actual_outcome, predicted_probabilities)
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

# 定义计算Brier评分的95%置信区间的函数
# 定义计算Brier评分的95%置信区间的函数，并行化使用joblib
# 定义计算Brier评分和Integrated Brier Score的95%置信区间的函数，并行化使用joblib
def bootstrap_brier_and_integrated_confidence_intervals(y_true, X, model, specific_times, max_time, n_iterations=100,
                                                        n_jobs=-1):
    scores = {time: [] for time in specific_times}
    integrated_scores = []

    # 定义每次迭代中需要进行的重采样和计算Brier分数及Integrated Brier Score的任务
    def compute_resample(i):
        # 重采样数据
        X_resampled, y_resampled = resample(X, y_true, random_state=i)
        model.fit(X_resampled, y_resampled)
        predicted_survival_values_resampled = model.predict(X)

        # 计算重采样后的Brier分数
        brier_scores_resampled = calculate_brier_score(y_true, predicted_survival_values_resampled, specific_times)

        # 计算重采样后的Integrated Brier Score
        integrated_brier_resampled = integrated_brier_score(y_true, predicted_survival_values_resampled, max_time)

        return brier_scores_resampled, integrated_brier_resampled

    # 并行化执行n_iterations次的重采样和计算
    results = Parallel(n_jobs=n_jobs)(delayed(compute_resample)(i) for i in range(n_iterations))

    # 将结果存入相应的时间点和Integrated Brier Score
    for res_brier, res_integrated in results:
        for time in specific_times:
            scores[time].append(res_brier[time])
        integrated_scores.append(res_integrated)

    # 计算每个时间点的95%置信区间
    confidence_intervals = {time: (np.percentile(scores[time], 2.5), np.percentile(scores[time], 97.5)) for time in
                            specific_times}

    # 计算Integrated Brier Score的95%置信区间
    integrated_confidence_interval = (np.percentile(integrated_scores, 2.5), np.percentile(integrated_scores, 97.5))

    return confidence_intervals, integrated_confidence_interval

# 调用bootstrap函数计算置信区间
ci_train, ci_integrated_train = bootstrap_brier_and_integrated_confidence_intervals(y_train, X_train, svm, specific_times, max_time_train)
ci_test, ci_integrated_test = bootstrap_brier_and_integrated_confidence_intervals(y_test, X_test, svm, specific_times, max_time_test)
ci_external, ci_integrated_external = bootstrap_brier_and_integrated_confidence_intervals(y_external, X_external, svm, specific_times, max_time_external)

# 你的结果数据，带有置信区间
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
    df = pd.DataFrame({"Label": labels, "Result": ["S-SVM"] + results})
else:
    # 如果文件存在且不为空，读取文件
    df = pd.read_csv(csv_file, header=None)

    # 确保 DataFrame 的行数足够长，至少能够容纳所有的 labels 和 results
    required_rows = max(len(labels), len(results) + 1)

    # 如果当前行数不足，扩展 DataFrame
    if len(df) < required_rows:
        additional_rows = required_rows - len(df)
        new_rows = pd.DataFrame([[None] * df.shape[1]] * additional_rows)  # 扩展所有列
        df = pd.concat([df, new_rows], ignore_index=True)

    # 写入数据
    if len(df.columns) < 4:
        df[3] = None  # 如果没有第4列，则创建第4列

    df.iloc[0, 3] = 'S-SVM'  # 在第4列的第一行写入"S-SVM"
    df.iloc[1:len(results) + 1, 3] = results  # 在第4列的第二行开始写入结果

# 将修改后的数据写回CSV文件，不带列名
df.to_csv(csv_file, index=False, header=False)
