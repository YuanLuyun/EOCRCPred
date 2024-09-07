import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

# 设置Streamlit页面布局
st.title("Postoperative EOCRC Prediction Model (EOCRCpred)")
st.write("Enter the following items to display the predicted postoperative survival risk")

# 加载数据并去掉 'Patient ID' 列
@st.cache_data
def load_data():
    data = pd.read_csv('data_encoded8415.csv')
    data = data.drop(columns=['Patient_ID'])
    return data

data = load_data()

# 构建生存数据
y = Surv.from_dataframe('Survival_status', 'OS_month', data)
X = data.drop(columns=['OS_month', 'Survival_status'])

# 按照 9:1 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 初始化随机生存森林模型
@st.cache_resource
def train_model():
    rsf = RandomSurvivalForest(
        n_estimators=1625, 
        max_depth=6, 
        min_samples_split=2, 
        min_samples_leaf=4, 
        n_jobs=-1, 
        random_state=42
    )
    rsf.fit(X_train, y_train)
    return rsf

rsf = train_model()

# 使用 Streamlit 的布局工具
with st.form("input_form"):
    st.write("Basic Information")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=1.0, max_value=3.0, step=1.0, value=1.0)
        t = st.selectbox("T", options=[0.0, 1.0, 2.0, 3.0, 4.0], index=0)
        resected_lns = st.selectbox("No. of Resected LNs", options=[0.0, 1.0, 2.0], index=0)

    with col2:
        grade = st.selectbox("Grade", options=[1.0, 2.0, 3.0, 4.0], index=0)
        n = st.selectbox("N", options=[0.0, 1.0, 2.0], index=0)
        tumor_deposits = st.selectbox("Tumor Deposits", options=[0.0, 1.0, 2.0], index=0)

    with col3:
        tnm_stage = st.selectbox("TNM Stage", options=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], index=0)
        cea = st.selectbox("CEA", options=[0.0, 1.0, 2.0], index=0)
        tumor_size = st.number_input("Tumor Size", min_value=0.0, max_value=10.0, step=0.1, value=0.0)

    income = st.selectbox("Median Household Income", options=[1.0, 2.0, 3.0, 4.0], index=0)

    st.write("Demographic Information")
    
    col4, col5, col6 = st.columns(3)
    with col4:
        sex = st.selectbox("Sex", options=["Male", "Female"], index=0)
    with col5:
        race = st.selectbox("Race", options=["White", "Black", "Other"], index=0)
    with col6:
        marital_status = st.selectbox("Marital status", options=["Single", "Married", "Divorced", "Widowed"], index=0)

    # 构建输入数据
    input_data = pd.DataFrame({
        "Age": [age],
        "Grade": [grade],
        "TNM_Stage": [tnm_stage],
        "T": [t],
        "N": [n],
        "CEA": [cea],
        "No.of_resected_LNs": [resected_lns],
        "Tumor_Deposits": [tumor_deposits],
        "Tumor_size": [tumor_size],
        "Median_household_income": [income],
        "Sex_Female": [1 if sex == "Female" else 0],
        "Race_Black": [1 if race == "Black" else 0],
        "Race_Other": [1 if race == "Other" else 0],
        "Marital_status_Married": [1 if marital_status == "Married" else 0],
        "Marital_status_Divorced": [1 if marital_status == "Divorced" else 0],
        "Marital_status_Widowed": [1 if marital_status == "Widowed" else 0]
    })

    submit = st.form_submit_button("Submit")

# 预测风险评分
if submit:
    input_data = input_data.reindex(columns=X_train.columns, fill_value=0)
    predicted_risk = rsf.predict(input_data)
    st.success(f"预测的风险评分: {predicted_risk[0]:.4f}")
