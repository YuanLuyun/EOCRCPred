import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.metrics import (
    balanced_accuracy_score, average_precision_score, matthews_corrcoef,
    brier_score_loss, f1_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

# 读取Cox回归结果和原始数据
cph_summary = pd.read_csv('cox.csv', index_col=0)
final_significant_vars = cph_summary[cph_summary['p'] < 0.05].index.tolist()
print("Significant variables:", final_significant_vars)
data = pd.read_csv('data.csv')


# 使用独热编码处理分类变量，并去除不需要的列
# 注意"Radiation"列现在已经是数据集的一部分，不需要额外处理
data_encoded = pd.get_dummies(data.drop(columns=['Patient ID']), drop_first=True)

# 处理缺失值
data_encoded = data_encoded.fillna(data_encoded.mean())


# 提取生存时间和事件状态
y = Surv.from_dataframe('Survival status', 'OS month', data_encoded)
X = data_encoded.drop(['OS month', 'Survival status'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 训练随机生存森林模型
rsf = RandomSurvivalForest(n_estimators=2000, max_depth=5, min_samples_split=5, min_samples_leaf=5, random_state=42)
rsf.fit(X_train, y_train)

# 预测和评估模型
predicted_survival_functions = rsf.predict_survival_function(X_test, return_array=False)
threshold = 36
predicted_probabilities = np.array([fn(threshold) for fn in predicted_survival_functions])
optimal_threshold = 0.8  # 敏感度较高的阈值
predicted_classes = (predicted_probabilities > optimal_threshold).astype(int)  # 大于阈值视为高风险（死亡）
actual_classes = (y_test['Survival status'] == 0).astype(int)

# 计算评估指标
ba = balanced_accuracy_score(actual_classes, predicted_classes)
ap = average_precision_score(actual_classes, predicted_probabilities)
mcc = matthews_corrcoef(actual_classes, predicted_classes)
brier = brier_score_loss(actual_classes, predicted_probabilities)
roc_auc = roc_auc_score(actual_classes, predicted_probabilities)

# 输出指标
print(f"Balanced Accuracy (BA): {ba}")
print(f"Average Precision (AP): {ap}")
print(f"Matthews Correlation Coefficient (MCC): {mcc}")
print(f"Brier Score: {brier}")
print(f"AUC: {roc_auc}")


fpr, tpr, _ = roc_curve(actual_classes, predicted_probabilities)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()