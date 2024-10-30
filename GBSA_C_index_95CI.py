import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sklearn.utils import resample


# 加载和预处理数据
data = pd.read_csv('data_encoded7408.csv').drop(columns=['Patient_ID'])
external_data = pd.read_csv('external_validation_set.csv')

# 构建生存数据
y = Surv.from_dataframe('Survival_status', 'OS_month', data)
X = data.drop(columns=['OS_month', 'Survival_status'])

y_external = Surv.from_dataframe('Survival_status', 'OS_month', external_data)
X_external = external_data.drop(columns=['OS_month', 'Survival_status'])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用基于梯度提升的生存分析模型
gbsa = GradientBoostingSurvivalAnalysis()
gbsa.fit(X_train, y_train)


# 计算置信区间的函数
def bootstrap_c_index(model, X, y, n_bootstrap=100, alpha=0.95):
    c_indices = []
    for _ in range(n_bootstrap):
        X_resampled, y_resampled = resample(X, y, random_state=None)
        risk_scores = model.predict(X_resampled)
        c_index = concordance_index_censored(y_resampled['Survival_status'], y_resampled['OS_month'], risk_scores)[0]
        c_indices.append(c_index)

    lower_bound = np.percentile(c_indices, (1 - alpha) / 2 * 100)
    upper_bound = np.percentile(c_indices, (1 + alpha) / 2 * 100)
    return np.mean(c_indices), lower_bound, upper_bound


# 计算总体的 C-index 和置信区间
def compute_overall_c_index(gbsa, X_train, y_train, X_test, y_test, X_external, y_external):
    overall_results = []
    for X, y, label in [(X_train, y_train, "Train"), (X_test, y_test, "Test"), (X_external, y_external, "External")]:
        mean_c_index, lower, upper = bootstrap_c_index(gbsa, X, y)
        overall_results.append(f"{mean_c_index:.3f} ({lower:.3f} , {upper:.3f})")
        print(f"{label} 总体 C-index: {mean_c_index:.3f} (95% CI: {lower:.3f} - {upper:.3f})")
    return overall_results


# 计算特定时间点的 C-index 和置信区间
def bootstrap_time_dependent_c_index(model, X_train, X_test, y_train, y_test, time_points, n_bootstrap=100,
                                     alpha=0.95):
    results = {}

    for t in time_points:
        c_indices = []
        for _ in range(n_bootstrap):
            X_resampled, y_resampled = resample(X_test, y_test, random_state=None)
            surv_funcs_resampled = model.predict_survival_function(X_resampled)
            surv_probs_at_t = np.array([fn(t) for fn in surv_funcs_resampled])
            risk_scores_at_t = 1 - surv_probs_at_t
            c_index = concordance_index_ipcw(y_train, y_resampled, risk_scores_at_t, tau=t)[0]
            c_indices.append(c_index)

        lower_bound = np.percentile(c_indices, (1 - alpha) / 2 * 100)
        upper_bound = np.percentile(c_indices, (1 + alpha) / 2 * 100)
        results[t] = (np.mean(c_indices), lower_bound, upper_bound)

    return results


# 计算每个组的特定时间点的 C-index 和置信区间
def compute_time_dependent_c_index(gbsa, X_train, y_train, X_test, y_test, X_external, y_external, time_points):
    time_dependent_results = []
    for X_train, X_test, y_train, y_test, label in [
        (X_train, X_train, y_train, y_train, "Train"),
        (X_train, X_test, y_train, y_test, "Test"),
        (X_train, X_external, y_train, y_external, "External")]:

        time_dependent_c_indices = bootstrap_time_dependent_c_index(gbsa, X_train, X_test, y_train, y_test, time_points)

        for t in time_points:
            mean_c_index, lower, upper = time_dependent_c_indices[t]
            time_dependent_results.append(f"{mean_c_index:.3f} ({lower:.3f} , {upper:.3f})")
            print(f"{label} 在 {t} 个月时的 C-index: {mean_c_index:.3f} (95% CI: {lower:.3f} - {upper:.3f})")

    return time_dependent_results


# 保存 C-index 到 CSV 的函数
def save_c_index_to_csv(filename, overall_results, time_dependent_results):
    # 尝试读取现有的 CSV 文件
    try:
        existing_df = pd.read_csv(filename, header=None)  # 不读取 header
    except FileNotFoundError:
        existing_df = pd.DataFrame()  # 文件不存在时创建空的 DataFrame

    # 创建新的列用于保存结果，第一行写 "GBSA"
    results = ["GBSA"] + overall_results + time_dependent_results

    # 如果现有文件不为空，将结果保存到第六列
    if not existing_df.empty:
        # 如果现有列数少于6列，扩展列数
        while existing_df.shape[1] < 6:
            existing_df.insert(existing_df.shape[1], f'col_{existing_df.shape[1]}', np.nan)  # 插入空列直到有6列

        # 插入到第六列
        existing_df.iloc[:len(results), 5] = results
    else:
        # 如果文件为空，则新建 DataFrame 并将数据放入第六列
        new_df = pd.DataFrame(np.nan, index=np.arange(len(results)), columns=[0, 1, 2, 3, 4, 5])
        new_df.iloc[:, 5] = results
        existing_df = new_df
    # 保存到 CSV 文件
    existing_df.to_csv(filename, index=False, header=False)


# 主程序
if __name__ == "__main__":
    # 确定你关心的时间点
    time_points = [12, 36, 60]

    # 计算并获取每个组的总体 C-index 和置信区间
    overall_results = compute_overall_c_index(gbsa, X_train, y_train, X_test, y_test, X_external, y_external)

    # 计算并获取每个组在特定时间点的 C-index 和置信区间
    time_dependent_results = compute_time_dependent_c_index(gbsa, X_train, y_train, X_test, y_test, X_external,
                                                            y_external, time_points)

    # 保存到 CSV 文件的第六列
    save_c_index_to_csv('c-index95.csv', overall_results, time_dependent_results)
