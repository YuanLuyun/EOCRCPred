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

# 三列布局
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=1.0, max_value=3.0, step=1.0, value=1.0)
    sex = st.selectbox("Sex", options=["Male", "Female"], index=0)
    race = st.selectbox("Race", options=["White", "Black", "Other"], index=0)
    marital_status = st.selectbox("Marital status", options=["Single", "Married", "Divorced", "Widowed"], index=0)
    income = st.selectbox("Median Household Income", options=[1.0, 2.0, 3.0, 4.0], index=0)
    primary_site = st.selectbox("Primary Site", options=[
        "Sigmoid colon", "Rectum", "Descending colon", "Transverse colon"
    ], index=0)

with col2:
    tumor_size = st.number_input("Tumor Size", min_value=0.0, max_value=10.0, step=0.1, value=0.0)
    grade = st.selectbox("Grade", options=[1.0, 2.0, 3.0, 4.0], index=0)
    histology = st.selectbox("Histology", options=["Non-specific adenocarcinoma", "Specific adenocarcinoma", "Other"], index=0)
    tnm_stage = st.selectbox("TNM Stage", options=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], index=0)
    cea = st.selectbox("CEA", options=[0.0, 1.0, 2.0], index=0)
    t = st.selectbox("T", options=[0.0, 1.0, 2.0, 3.0, 4.0], index=0)
    n = st.selectbox("N", options=[0.0, 1.0, 2.0], index=0)

with col3:
    resection_type = st.selectbox("Resection type", options=[
        "Partial/subtotal colectomy", 
        "Hemicolectomy or greater", 
        "Total colectomy", 
        "Colectomy plus removal of other organs"
    ], index=0)
    tumor_deposits = st.selectbox("Tumor Deposits", options=[0.0, 1.0, 2.0], index=0)
    resected_lns = st.selectbox("No. of Resected LNs", options=[0.0, 1.0, 2.0], index=0)
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

# 预测风险评分
if st.button("Submit"):
    input_data = input_data.reindex(columns=X_train.columns, fill_value=0)
    predicted_risk = rsf.predict(input_data)
    st.success(f"预测的风险评分: {predicted_risk[0]:.4f}")
