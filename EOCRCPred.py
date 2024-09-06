import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

# 设置Streamlit页面布局
st.title("Prediction model for post operative EOCRC (EOCRCpred model)")
st.write("Enter the following items to display the predicted EOCRC risk")

# 加载数据并去掉 'Patient ID' 列
data = pd.read_csv('data_encoded8415.csv')
data = data.drop(columns=['Patient_ID'])

# 构建生存数据
y = Surv.from_dataframe('Survival_status', 'OS_month', data)
X = data.drop(columns=['OS_month', 'Survival_status'])

# 按照 9:1 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 初始化随机生存森林模型
rsf = RandomSurvivalForest(n_estimators=1625, max_depth=6, min_samples_split=2, min_samples_leaf=4, n_jobs=-1, random_state=42)
rsf.fit(X_train, y_train)

# Streamlit表单用于输入患者特征
st.sidebar.header("输入患者特征")

# 基本特征输入项
age = st.sidebar.number_input("Age", min_value=1.0, max_value=3.0, step=1.0, value=1.0)
grade = st.sidebar.selectbox("Grade", options=[1.0, 2.0, 3.0, 4.0], index=0)
tnm_stage = st.sidebar.selectbox("TNM Stage", options=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], index=0)
t = st.sidebar.selectbox("T", options=[0.0, 1.0, 2.0, 3.0, 4.0], index=0)
n = st.sidebar.selectbox("N", options=[0.0, 1.0, 2.0], index=0)
cea = st.sidebar.selectbox("CEA", options=[0.0, 1.0, 2.0], index=0)
resected_lns = st.sidebar.selectbox("No. of Resected LNs", options=[0.0, 1.0, 2.0], index=0)
tumor_deposits = st.sidebar.selectbox("Tumor Deposits", options=[0.0, 1.0, 2.0], index=0)
tumor_size = st.sidebar.number_input("Tumor Size", min_value=0.0, max_value=10.0, step=0.1, value=0.0)
income = st.sidebar.selectbox("Median Household Income", options=[1.0, 2.0, 3.0, 4.0], index=0)

# 独热编码变量
sex_female = st.sidebar.selectbox("Sex (Female)", options=[0, 1], index=0)
race_black = st.sidebar.selectbox("Race (Black)", options=[0, 1], index=0)
race_other = st.sidebar.selectbox("Race (Other)", options=[0, 1], index=0)

# 添加更多的 Primary site 选项
primary_site_rectum = st.sidebar.selectbox("Primary Site (Rectum)", options=[0, 1], index=0)
primary_site_descending_colon = st.sidebar.selectbox("Primary Site (Descending colon)", options=[0, 1], index=0)
primary_site_transverse_colon = st.sidebar.selectbox("Primary Site (Transverse colon)", options=[0, 1], index=0)

# 添加更多的 Histology 和 Resection type
histology_specific = st.sidebar.selectbox("Histology (Specific adenocarcinoma)", options=[0, 1], index=0)
histology_other = st.sidebar.selectbox("Histology (Other)", options=[0, 1], index=0)
resection_type_hemicolectomy = st.sidebar.selectbox("Resection type (Hemicolectomy or greater)", options=[0, 1], index=0)
resection_type_total_colectomy = st.sidebar.selectbox("Resection type (Total colectomy)", options=[0, 1], index=0)
resection_type_other = st.sidebar.selectbox("Resection type (Colectomy plus removal of other organs)", options=[0, 1], index=0)

# 添加 Surg.Rad.Seq 和其他特征
surg_rad_seq_postoperative = st.sidebar.selectbox("Surg.Rad.Seq (Postoperative)", options=[0, 1], index=0)
surg_rad_seq_preoperative = st.sidebar.selectbox("Surg.Rad.Seq (Preoperative)", options=[0, 1], index=0)
surg_rad_seq_preop_postop = st.sidebar.selectbox("Surg.Rad.Seq (Preoperative+Postoperative)", options=[0, 1], index=0)
surg_rad_seq_unknown = st.sidebar.selectbox("Surg.Rad.Seq (Sequence unknown)", options=[0, 1], index=0)

# 添加 Systemic.Sur.Seq
systemic_sur_seq_postoperative = st.sidebar.selectbox("Systemic.Sur.Seq (Postoperative)", options=[0, 1], index=0)
systemic_sur_seq_preoperative = st.sidebar.selectbox("Systemic.Sur.Seq (Preoperative)", options=[0, 1], index=0)
systemic_sur_seq_preop_postop = st.sidebar.selectbox("Systemic.Sur.Seq (Preoperative+Postoperative)", options=[0, 1], index=0)
systemic_sur_seq_unknown = st.sidebar.selectbox("Systemic.Sur.Seq (Sequence unknown)", options=[0, 1], index=0)

# 其他特征
perineural_invasion = st.sidebar.selectbox("Perineural Invasion (Yes)", options=[0, 1], index=0)
marital_status_married = st.sidebar.selectbox("Marital status (Married)", options=[0, 1], index=0)
marital_status_divorced = st.sidebar.selectbox("Marital status (Divorced)", options=[0, 1], index=0)
marital_status_widowed = st.sidebar.selectbox("Marital status (Widowed)", options=[0, 1], index=0)

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
    "Sex_Female": [sex_female],
    "Race_Black": [race_black],
    "Race_Other": [race_other],
    "Primary_site_Rectum": [primary_site_rectum],
    "Primary_site_Descending_colon": [primary_site_descending_colon],
    "Primary_site_Transverse_colon": [primary_site_transverse_colon],
    "Histology_Specific_adenocarcinoma": [histology_specific],
    "Histology_Other": [histology_other],
    "Resection_type_Hemicolectomy_or_greater_": [resection_type_hemicolectomy],
    "Resection_type_Total_colectomy": [resection_type_total_colectomy],
    "Resection_type_Colectomy_plus_removal_of_other_organs": [resection_type_other],
    "Surg.Rad.Seq_Postoperative": [surg_rad_seq_postoperative],
    "Surg.Rad.Seq_Preoperative": [surg_rad_seq_preoperative],
    "Surg.Rad.Seq_Preoperative+Postoperative": [surg_rad_seq_preop_postop],
    "Surg.Rad.Seq_Sequence_unknown": [surg_rad_seq_unknown],
    "Systemic.Sur.Seq_Postoperative": [systemic_sur_seq_postoperative],
    "Systemic.Sur.Seq_Preoperative": [systemic_sur_seq_preoperative],
    "Systemic.Sur.Seq_Preoperative+Postoperative": [systemic_sur_seq_preop_postop],
    "Systemic.Sur.Seq_Sequence_unknown": [systemic_sur_seq_unknown],
    "Perineural_Invasion_Yes": [perineural_invasion],
    "Marital_status_Married": [marital_status_married],
    "Marital_status_Divorced": [marital_status_divorced],
    "Marital_status_Widowed": [marital_status_widowed]
})

 &#8203;:contentReference[oaicite:0]{index=0}&#8203;
