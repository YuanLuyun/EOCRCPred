import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
import matplotlib.pyplot as plt
import shap
import numpy as np
import matplotlib.ticker as ticker

# 定义自定义缩放函数，使接近0的值显示得更详细
def custom_transform(x):
    return np.sign(x) * np.log1p(np.abs(x))  # 采用对数缩放来压缩大值，扩展小值

def inverse_custom_transform(x):
    return np.sign(x) * (np.expm1(np.abs(x)))  # 逆变换，用于还原到原始尺度

# 加载数据
data = pd.read_csv('data_encoded7408.csv')
data = data.drop(columns=['Patient_ID'])
data.columns = data.columns.str.replace('_', ' ')

# 构建生存数据
y = Surv.from_dataframe('Survival status', 'OS month', data)
X = data.drop(columns=['OS month', 'Survival status'])

# 限制用于计算 SHAP 值的样本数量
sample_size = 7408
X_sample = X[:sample_size]

# 初始化随机生存森林模型
rsf = RandomSurvivalForest(n_estimators=1600, max_depth=3, min_samples_split=2, min_samples_leaf=9, n_jobs=-1, random_state=42)
rsf.fit(X, y)

# 计算 SHAP 值
explainer = shap.Explainer(rsf.predict, X_sample)
shap_values = explainer(X_sample)

# 将 SHAP 值转换为 DataFrame 以便计算每个特征的重要性
shap_values_df = pd.DataFrame(shap_values.values, columns=X_sample.columns)

# 计算每个特征的平均 SHAP 值绝对值，确定重要性
feature_importance = shap_values_df.abs().mean().sort_values(ascending=False)

# 获取前 10 个重要特征
top_10_features = feature_importance.index[:10]  # 按特征重要性排序获取前10个特征

# 提取前 10 个重要特征的 SHAP 值
shap_values_top_10 = shap_values_df[top_10_features].values

# 提取前 10 个重要特征的数据
X_sample_top_10 = X_sample[top_10_features]

# 对 shap_values_top_10 的值应用自定义的对数缩放
shap_values_top_10_transformed = custom_transform(shap_values_top_10)

# 使用缩放后的 SHAP 值创建 SHAP summary plot
shap.summary_plot(shap_values_top_10_transformed, X_sample_top_10, show=False)

# 获取当前图的轴对象
ax = plt.gca()

# 设置 x 轴标签为真实的 SHAP 值
def custom_format(x, pos):
    original_value = inverse_custom_transform(x)  # 使用逆变换将缩放值还原为原始值
    if abs(original_value) < 1:
        return f'{original_value:.2f}'  # 小值保留两位小数
    else:
        return f'{int(original_value)}'  # 大值显示为整数

# 自定义 x 轴的显示标签
ax.xaxis.set_major_formatter(ticker.FuncFormatter(custom_format))

# 保存图表为 PDF 文件
output_pdf_path = "Fig.7.pdf"
plt.gcf().savefig(output_pdf_path, format='pdf')

# 显示图形
plt.show()

# 关闭图形
plt.close()
