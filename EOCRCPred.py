import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

# 设置Streamlit页面布局
st.title("Postoperative EOCRC Prediction Model (EOCRCpred)")
# 插入CSS来调整输入框宽度
st.markdown(
    """
    <style>
    .stNumberInput > div > div > input, .stSelectbox > div > div {
        width: 100% !important;
        max-width: 800px;  /* 调整此处的宽度 */
    }
    </style>
    """,
    unsafe_allow_html=True
)


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

# 创建三列布局，左侧和右侧为空，中间为输入框
left, center, right = st.columns([1, 2, 1])

with center:
    # st.header("输入患者特征")

    # 基本特征输入项
    age = st.number_input("Age", min_value=1.0, max_value=3.0, step=1.0, value=1.0)
    grade = st.selectbox("Grade", options=[1.0, 2.0, 3.0, 4.0], index=0)
    tnm_stage = st.selectbox("TNM Stage", options=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], index=0)
    t = st.selectbox("T", options=[0.0, 1.0, 2.0, 3.0, 4.0], index=0)
    n = st.selectbox("N", options=[0.0, 1.0, 2.0], index=0)
    cea = st.selectbox("CEA", options=[0.0, 1.0, 2.0], index=0)
    resected_lns = st.selectbox("No. of Resected LNs", options=[0.0, 1.0, 2.0], index=0)
    tumor_deposits = st.selectbox("Tumor Deposits", options=[0.0, 1.0, 2.0], index=0)
    tumor_size = st.number_input("Tumor Size", min_value=0.0, max_value=10.0, step=0.1, value=0.0)
    income = st.selectbox("Median Household Income", options=[1.0, 2.0, 3.0, 4.0], index=0)

    # 合并 Sex 选项
    sex = st.selectbox("Sex", options=["Male", "Female"], index=0)
    sex_female = 1 if sex == "Female" else 0  # Male 作为基准类别

    # 合并 Race 选项
    race = st.selectbox("Race", options=["White", "Black", "Other"], index=0)
    race_black = 1 if race == "Black" else 0
    race_other = 1 if race == "Other" else 0  # White 作为基准类别

    # 合并 Primary site 选项
    primary_site = st.selectbox("Primary Site", options=["Ascending colon", "Rectum", "Descending colon", "Transverse colon"], index=0)
    primary_site_rectum = 1 if primary_site == "Rectum" else 0
    primary_site_descending_colon = 1 if primary_site == "Descending colon" else 0
    primary_site_transverse_colon = 1 if primary_site == "Transverse colon" else 0  # Ascending colon 作为基准类别

    # 合并 Histology 选项
    histology = st.selectbox("Histology", options=["Non-specific adenocarcinoma", "Specific adenocarcinoma", "Other"], index=0)
    histology_specific = 1 if histology == "Specific adenocarcinoma" else 0
    histology_other = 1 if histology == "Other" else 0  # Non-specific adenocarcinoma 作为基准类别

    # 合并 Resection type 选项
    resection_type = st.selectbox("Resection type", options=[
        "Partial/subtotal colectomy", 
        "Hemicolectomy or greater", 
        "Total colectomy", 
        "Colectomy plus removal of other organs"
    ], index=0)
    resection_type_hemicolectomy = 1 if resection_type == "Hemicolectomy or greater" else 0
    resection_type_total_colectomy = 1 if resection_type == "Total colectomy" else 0
    resection_type_other = 1 if resection_type == "Colectomy plus removal of other organs" else 0  # Partial/subtotal colectomy 作为基准类别

    # 合并 Surg.Rad.Seq 选项
    surg_rad_seq = st.selectbox("Surg.Rad.Seq", options=[
        "Untreated", 
        "Postoperative", 
        "Preoperative", 
        "Preoperative+Postoperative", 
        "Sequence unknown"
    ], index=0)
    surg_rad_seq_postoperative = 1 if surg_rad_seq == "Postoperative" else 0
    surg_rad_seq_preoperative = 1 if surg_rad_seq == "Preoperative" else 0
    surg_rad_seq_preop_postop = 1 if surg_rad_seq == "Preoperative+Postoperative" else 0
    surg_rad_seq_unknown = 1 if surg_rad_seq == "Sequence unknown" else 0  # Untreated 作为基准类别

    # 合并 Systemic.Sur.Seq 选项
    systemic_sur_seq = st.selectbox("Systemic.Sur.Seq", options=[
        "Untreated", 
        "Postoperative", 
        "Preoperative", 
        "Preoperative+Postoperative", 
        "Sequence unknown"
    ], index=0)
    systemic_sur_seq_postoperative = 1 if systemic_sur_seq == "Postoperative" else 0
    systemic_sur_seq_preoperative = 1 if systemic_sur_seq == "Preoperative" else 0
    systemic_sur_seq_preop_postop = 1 if systemic_sur_seq == "Preoperative+Postoperative" else 0
    systemic_sur_seq_unknown = 1 if systemic_sur_seq == "Sequence unknown" else 0  # Untreated 作为基准类别

    # 合并 Perineural Invasion 选项
    perineural_invasion = st.selectbox("Perineural Invasion", options=["No", "Yes"], index=0)
    perineural_invasion_yes = 1 if perineural_invasion == "Yes" else 0  # No 作为基准类别

    # 合并 Marital status 选项
    marital_status = st.selectbox("Marital status", options=["Single", "Married", "Divorced", "Widowed"], index=0)
    marital_status_married = 1 if marital_status == "Married" else 0
    marital_status_divorced = 1 if marital_status == "Divorced" else 0
    marital_status_widowed = 1 if marital_status == "Widowed" else 0  # Single 作为基准类别

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
        "Perineural_Invasion_Yes": [perineural_invasion_yes],
        "Marital_status_Married": [marital_status_married],
        "Marital_status_Divorced": [marital_status_divorced],
        "Marital_status_Widowed": [marital_status_widowed]
    })

    # 预测风险评分
    if st.button("Submit"):
        # 确保输入数据的特征列与训练数据对齐
        input_data = input_data.reindex(columns=X_train.columns, fill_value=0)

        # 进行风险评分预测
        predicted_risk = rsf.predict(input_data)

        # 显示预测结果
        st.success(f"预测的风险评分: {predicted_risk[0]:.4f}")
