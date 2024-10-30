import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
from sksurv.util import Surv
from sklearn.model_selection import train_test_split

# 读取原始数据
data = pd.read_csv('data_encoded7408.csv')
data_encoded = data.drop(columns=['Patient_ID'])

# 提取生存时间和事件状态并划分训练集和验证集
y = Surv.from_dataframe('Survival_status', 'OS_month', data_encoded)
X = data_encoded.drop(['OS_month', 'Survival_status'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 读取外部验证集数据
data_external = pd.read_csv('external_validation_set.csv')
data_external = data_external.drop(columns=['Patient_ID'])
y_external = Surv.from_dataframe('Survival_status', 'OS_month', data_external)
X_external = data_external.drop(['OS_month', 'Survival_status'], axis=1)

# 创建Kaplan-Meier拟合器
kmf_train = KaplanMeierFitter()
kmf_test = KaplanMeierFitter()
kmf_external = KaplanMeierFitter()

# 绘制生存曲线
fig, ax = plt.subplots(figsize=(10, 8))

# 拟合并绘制训练集生存曲线
kmf_train.fit(y_train['OS_month'], event_observed=y_train['Survival_status'], label="Training cohort")
kmf_train.plot_survival_function(ax=ax, color='blue')

# 拟合并绘制测试集生存曲线
kmf_test.fit(y_test['OS_month'], event_observed=y_test['Survival_status'], label="Test cohort")
kmf_test.plot_survival_function(ax=ax, color='green')

# 拟合并绘制外部验证集生存曲线
kmf_external.fit(y_external['OS_month'], event_observed=y_external['Survival_status'], label="External validation cohort")
kmf_external.plot_survival_function(ax=ax, color='red')

# 计算训练集 vs 测试集的 log-rank p 值
result_train_test = logrank_test(y_train['OS_month'], y_test['OS_month'],
                                 event_observed_A=y_train['Survival_status'],
                                 event_observed_B=y_test['Survival_status'])
p_value_train_test = result_train_test.p_value

# 计算训练集 vs 外部验证集的 log-rank p 值
result_train_external = logrank_test(y_train['OS_month'], y_external['OS_month'],
                                     event_observed_A=y_train['Survival_status'],
                                     event_observed_B=y_external['Survival_status'])
p_value_train_external = result_train_external.p_value

# 在图上添加 log-rank p 值
ax.text(0.1, 0.2, f'Training vs Test p = {p_value_train_test:.3f}', transform=ax.transAxes, fontsize=12, color='black')
ax.text(0.1, 0.15, f'Training vs External p = {p_value_train_external:.3f}', transform=ax.transAxes, fontsize=12, color='black')

# 设置图的标题和标签
ax.set_xlabel('Time (months)')
ax.set_ylabel('Survival probability')

# 添加风险表
add_at_risk_counts(kmf_train, kmf_test, kmf_external, ax=ax)

# 调整布局并显示图像
plt.tight_layout()

# 保存图像为 PDF
plt.savefig('Fig.3.pdf')

# 显示图像
plt.show()
