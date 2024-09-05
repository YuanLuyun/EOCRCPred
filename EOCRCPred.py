import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc

# 设置Streamlit页面布局
st.title("生存分析模型 - 风险评分预测")
st.write("输入患者特征，预测对应的生存风险评分。")

# 加载数据并去掉 'Patient ID' 列
data = pd.read_csv('data_encoded8415.csv')
data = data.drop(columns=['Patient ID'])

# 构建生存数据
y = Surv.from_dataframe('Survival status', 'OS month', data)
X = data.drop(columns=['OS month', 'Survival status'])

# 按照 9:1 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 初始化随机生存森林模型
rsf = RandomSurvivalForest(n_estimators=1625, max_depth=6, min_samples_split=2, min_samples_leaf=4, n_jobs=-1, random_state=42)
rsf.fit(X_train, y_train)

# Streamlit表单用于输入患者特征
st.sidebar.header("输入患者特征")

# 根据数据集中的特征创建输入项（示例：假设有三个特征 'Age'、'Tumor Size'、'Gender'）
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=50)
tumor_size = st.sidebar.slider("Tumor Size", min_value=0.1, max_value=10.0, value=5.0)
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])

# 将输入数据转换为模型可用的格式
input_data = pd.DataFrame({
    "Age": [age],
    "Tumor Size": [tumor_size],
    "Gender": [1 if gender == "Male" else 0]  # 假设 'Gender' 是 1 表示 Male，0 表示 Female
})

# 预测风险评分
if st.sidebar.button("生成风险评分"):
    predicted_risk = rsf.predict(input_data)
    st.write(f"预测的风险评分: {predicted_risk[0]:.4f}")



