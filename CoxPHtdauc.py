import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

# 加载 data_encoded7408.csv 数据并去掉 'Patient_ID' 列
data = pd.read_csv('data_encoded7408.csv')
data = data.drop(columns=['Patient_ID'])  # 删除 Patient_ID 列

# 加载 external_validation_set.csv 数据并去掉 'Patient_ID' 列
external_data = pd.read_csv('external_validation_set.csv')
external_data = external_data.drop(columns=['Patient_ID'])  # 删除 Patient_ID 列

# 构建生存数据
y = Surv.from_dataframe('Survival_status', 'OS_month', data)
X = data.drop(columns=['OS_month', 'Survival_status'])

# 外部验证集生存数据的构建
y_external = Surv.from_dataframe('Survival_status', 'OS_month', external_data)
X_external = external_data.drop(columns=['OS_month', 'Survival_status'])

# 按照 9:1 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练Cox比例风险模型
coxph = CoxPHSurvivalAnalysis()
coxph.fit(X_train, y_train)

# 训练集上预测并计算C-index
c_index_train_coxph = concordance_index_censored(y_train["Survival_status"], y_train["OS_month"], coxph.predict(X_train))[0]
print(f"CoxPHSurvivalAnalysis - Training C-index: {c_index_train_coxph}")

# 测试集上预测并计算C-index
c_index_test_coxph = concordance_index_censored(y_test["Survival_status"], y_test["OS_month"], coxph.predict(X_test))[0]
print(f"CoxPHSurvivalAnalysis - Test C-index: {c_index_test_coxph}")

# 外部验证集上预测并计算C-index
predictions_external = coxph.predict(X_external)
c_index_external = concordance_index_censored(y_external['Survival_status'], y_external['OS_month'], predictions_external)[0]
print(f"CoxPHSurvivalAnalysis - External Validation C-index: {c_index_external}")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
import matplotlib.pyplot as plt

# 加载 data_encoded7408.csv 数据并去掉 'Patient_ID' 列
data = pd.read_csv('data_encoded7408.csv')
data = data.drop(columns=['Patient_ID'])  # 删除 Patient_ID 列

# 加载 external_validation_set.csv 数据并去掉 'Patient_ID' 列
external_data = pd.read_csv('external_validation_set.csv')
external_data = external_data.drop(columns=['Patient_ID'])  # 删除 Patient_ID 列

# 构建生存数据
y = Surv.from_dataframe('Survival_status', 'OS_month', data)
X = data.drop(columns=['OS_month', 'Survival_status'])

# 外部验证集生存数据的构建
y_external = Surv.from_dataframe('Survival_status', 'OS_month', external_data)
X_external = external_data.drop(columns=['OS_month', 'Survival_status'])

# 按照 9:1 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练Cox比例风险模型
coxph = CoxPHSurvivalAnalysis()
coxph.fit(X_train, y_train)

# 训练集上预测并计算C-index
c_index_train_coxph = concordance_index_censored(y_train["Survival_status"], y_train["OS_month"], coxph.predict(X_train))[0]
print(f"CoxPHSurvivalAnalysis - Training C-index: {c_index_train_coxph}")

# 测试集上预测并计算C-index
c_index_test_coxph = concordance_index_censored(y_test["Survival_status"], y_test["OS_month"], coxph.predict(X_test))[0]
print(f"CoxPHSurvivalAnalysis - Test C-index: {c_index_test_coxph}")

# 外部验证集上预测并计算C-index
predictions_external = coxph.predict(X_external)
c_index_external = concordance_index_censored(y_external['Survival_status'], y_external['OS_month'], predictions_external)[0]
print(f"CoxPHSurvivalAnalysis - External Validation C-index: {c_index_external}")

# 定义感兴趣的时间点（每3个月一个时间点）
time_points = np.arange(1, 120, 3)

# 计算测试集上的时间依赖性 AUC
predicted_risk_test = coxph.predict(X_test)
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

