import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
import matplotlib.pyplot as plt

# 加载数据并去掉 'Patient ID' 列
data = pd.read_csv('data_encoded8415.csv')
data = data.drop(columns=['Patient ID'])

# 构建生存数据
y = Surv.from_dataframe('Survival status', 'OS month', data)
X = data.drop(columns=['OS month', 'Survival status'])

# 按照 7:3 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 初始化随机生存森林模型
rsf = RandomSurvivalForest(n_estimators=1625, max_depth=6, min_samples_split=2, min_samples_leaf=4, n_jobs=-1,
                           random_state=42)
rsf.fit(X_train, y_train)

# 在训练集上计算 C-index
c_index_train = concordance_index_censored(y_train["Survival status"], y_train["OS month"], rsf.predict(X_train))[0]
# 在测试集上计算 C-index
c_index_test = concordance_index_censored(y_test["Survival status"], y_test["OS month"], rsf.predict(X_test))[0]

print(f"Train C-index: {c_index_train:.4f}")
print(f"Test C-index: {c_index_test:.4f}")

# 定义感兴趣的时间点（每12个月一个时间点）
time_points = np.arange(1, 96, 3)

# 计算测试集上的时间依赖性 AUC
predicted_risk_test = rsf.predict(X_test)
cumulative_auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, predicted_risk_test, time_points)

# 打印每个时间点的 AUC 值
for i, t in enumerate(time_points):
    print(f"AUC at {t} months: {cumulative_auc[i]:.4f}")

# 绘制时间依赖性 AUC 曲线
plt.plot(time_points, cumulative_auc, marker="o")
plt.xlabel("Time (months)")
plt.ylabel("Time-dependent AUC")
plt.title("Time-dependent AUC over time")
plt.grid(True)
plt.show()
