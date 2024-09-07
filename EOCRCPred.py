import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

# 设置Streamlit页面布局
st.title("Postoperative EOCRC Prediction Model (EOCRCpred)")
st.write("Enter the following items to display the predicted postoperative survival risk")

# 加载数据
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

# 定义有序变量的类别
ordered_var_categories = {
    'Age': ['less than 35', '35-44', '45-49'],
    'Grade': ['Well differentiated', 'Moderately differentiated', 'Poorly differentiated', 'Undifferentiated'],
    'TNM Stage': ['0', 'I', 'IIA', 'IIB', 'IIC', 'IIIA', 'IIIB', 'IIIC'],
    'T': ['Tis', 'T1', 'T2', 'T3', 'T4'],
    'N': ['N0', 'N1', 'N2'],
    'CEA': ['negative', 'Borderline', 'positive'],
    'No.of resected LNs': ['Zero', '1 to 3', '4+'],
    'Tumor Deposits': ['Zero', '1 to 2', '3+'],
    'Tumor size': ['less than 5', '5+'],
    'Median household income': ['less than $35,000', '$35,000-$54,999', '$55,000-$74,999', '$75,000+']
}

# 三列布局
col1, col2, col3 = st.columns(3)
with col1:
    age = st.selectbox("Age", options=ordered_var_categories['Age'], index=0)
    sex = st.selectbox("Sex", options=["Male", "Female"], index=0)
    race = st.selectbox("Race", options=["White", "Black", "Other"], index=0)
    marital_status = st.selectbox("Marital status", options=["Single", "Married", "Divorced", "Widowed"], index=0)
    income = st.selectbox("Median Household Income", options=ordered_var_categories['Median household income'], index=0)
    primary_site = st.selectbox("Primary Site", options=[
        "Sigmoid colon", "Rectum", "Descending colon", "Transverse colon"
    ], index=0)
    tumor_size = st.selectbox("Tumor Size", options=ordered_var_categories['Tumor size'], index=0)
with col2:
    grade = st.selectbox("Grade", options=ordered_var_categories['Grade'], index=0)
    histology = st.selectbox("Histology", options=["Non-specific adenocarcinoma", "Specific adenocarcinoma", "Other"], index=0)
    cea = st.selectbox("CEA", options=ordered_var_categories['CEA'], index=0)
    tnm_stage = st.selectbox("TNM Stage", options=ordered_var_categories['TNM Stage'], index=0)
    t = st.selectbox("T", options=ordered_var_categories['T'], index=0)
    n = st.selectbox("N", options=ordered_var_categories['N'], index=0)
    resection_type = st.selectbox("Resection type", options=[
        "Partial/subtotal colectomy", 
        "Hemicolectomy or greater", 
        "Total colectomy", 
        "Colectomy plus removal of other organs"
    ], index=0)
with col3:
    tumor_deposits = st.selectbox("Tumor Deposits", options=ordered_var_categories['Tumor Deposits'], index=0)
    resected_lns = st.selectbox("No. of Resected LNs", options=ordered_var_categories['No.of resected LNs'], index=0)
    surg_rad_seq = st.selectbox("Surg.Rad.Seq", options=[
        "Untreated", 
        "Postoperative", 
        "Preoperative", 
        "Preoperative+Postoperative", 
        "Sequence unknown"
    ], index=0)
    systemic_sur_seq = st.selectbox("Systemic.Sur.Seq", options=[
        "Untreated", 
        "Postoperative", 
        "Preoperative", 
        "Preoperative+Postoperative", 
        "Sequence unknown"
    ], index=0)
    chemotherapy = st.selectbox("Chemotherapy", options=["No", "Yes"], index=0)
    perineural_invasion = st.selectbox("Perineural Invasion", options=["No", "Yes"], index=0)

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
    "Primary_site_Rectum": [1 if primary_site == "Rectum" else 0],
    "Primary_site_Ascending_colon": [1 if primary_site == "Ascending colon" else 0],
    "Primary_site_Transverse_colon": [1 if primary_site == "Transverse colon" else 0],
    "Primary_site_Sigmoid_colon": [1 if primary_site == "Sigmoid colon" else 0],
    "Marital_status_Married": [1 if marital_status == "Married" else 0],
    "Marital_status_Divorced": [1 if marital_status == "Divorced" else 0],
    "Marital_status_Widowed": [1 if marital_status == "Widowed" else 0],
    "Histology_Specific_adenocarcinoma": [1 if histology == "Specific adenocarcinoma" else 0],
    "Histology_Other": [1 if histology == "Other" else 0],
    "Resection_type_Hemicolectomy_or_greater": [1 if resection_type == "Hemicolectomy or greater" else 0],
    "Resection_type_Total_colectomy": [1 if resection_type == "Total colectomy" else 0],
    "Resection_type_Colectomy_plus_removal_of_other_organs": [1 if resection_type == "Colectomy plus removal of other organs" else 0],
    "Surg.Rad.Seq_Postoperative": [1 if surg_rad_seq == "Postoperative" else 0],
    "Surg.Rad.Seq_Preoperative": [1 if surg_rad_seq == "Preoperative" else 0],
    "Surg.Rad.Seq_Preoperative+Postoperative": [1 if surg_rad_seq == "Preoperative+Postoperative" else 0],
    "Surg.Rad.Seq_Sequence_unknown": [1 if surg_rad_seq == "Sequence unknown" else 0],
    "Chemotherapy_Yes": [1 if chemotherapy == "Yes" else 0],
    "Systemic.Sur.Seq_Postoperative": [1 if systemic_sur_seq == "Postoperative" else 0],
    "Systemic.Sur.Seq_Preoperative": [1 if systemic_sur_seq == "Preoperative" else 0],
    "Systemic.Sur.Seq_Preoperative+Postoperative": [1 if systemic_sur_seq == "Preoperative+Postoperative" else 0],
    "Systemic.Sur.Seq_Sequence_unknown": [1 if systemic_sur_seq == "Sequence unknown" else 0],
    "Perineural_Invasion_Yes": [1 if perineural_invasion == "Yes" else 0]
})

# # 预测风险评分
# if st.button("Submit"):
#     input_data = input_data.reindex(columns=X_train.columns, fill_value=0)
#     predicted_risk = rsf.predict(input_data)
#     st.success(f"预测的风险评分: {predicted_risk[0]:.4f}")

import matplotlib.pyplot as plt
import numpy as np

# 预测风险评分
if st.button("Submit"):
    input_data = input_data.reindex(columns=X_train.columns, fill_value=0)
    predicted_risk = rsf.predict(input_data)
    # st.success(f"预测的风险评分: {predicted_risk[0]:.4f}")
    # 确保输入数据与训练时的数据格式一致


    # 打印 input_data 的信息以调试
    st.write("Input Data for Prediction:")
    st.write(input_data.head())  # 查看数据前几行
    st.write(f"Data shape: {input_data.shape}")

    # 确保数据类型一致
    st.write(f"Data types:\n{input_data.dtypes}")

    # 预测风险评分
    predicted_risk = rsf.predict(input_data)
    st.success(f"预测的风险评分: {predicted_risk[0]:.4f}")

    # 计算累积风险函数
    cumulative_hazard = rsf.predict_cumulative_hazard_function(input_data, return_array=True)
    
    # 输出累积风险曲线
    fig, ax = plt.subplots()
    time_points = np.linspace(0, cumulative_hazard[0][-1], num=100)  # 选择100个时间点
    ax.plot(time_points, cumulative_hazard[0][:100], label='Cumulative Hazard')
    ax.set_xlabel("Time (Months)")
    ax.set_ylabel("Cumulative Hazard")
    ax.set_title("Cumulative Hazard Curve")
    ax.legend()
    st.pyplot(fig)

    # 计算三分位数风险分层
    all_risks = rsf.predict(X_train)  # 计算训练集中的所有风险评分
    q1, q2 = np.percentile(all_risks, [33.33, 66.67])
    
    if predicted_risk < q1:
        risk_group = "Low Risk"
    elif predicted_risk < q2:
        risk_group = "Medium Risk"
    else:
        risk_group = "High Risk"
    
    st.write(f"该患者属于: {risk_group}")
    
    # 预测在12, 36, 60个月的风险矩阵
    time_points = [12, 36, 60]  # 选择的时间点
    risks_at_time_points = []
    
    for time in time_points:
        risk_at_time = rsf.predict_cumulative_hazard_function(input_data, times=[time], return_array=True)
        risks_at_time_points.append(risk_at_time[0][0])  # 取出对应时间点的风险值
    
    risk_matrix = pd.DataFrame({
        "Time (Months)": time_points,
        "Predicted Risk": risks_at_time_points
    })
    
    st.write("不同时间点的预测风险矩阵:")
    st.dataframe(risk_matrix)

