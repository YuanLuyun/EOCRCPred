import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
import matplotlib.pyplot as plt

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
    'Age': ['＜35', '35-44', '45-49'],
    'Grade': ['Well differentiated', 'Moderately differentiated', 'Poorly differentiated', 'Undifferentiated'],
    'TNM Stage': ['0', 'I', 'IIA', 'IIB', 'IIC', 'IIIA', 'IIIB', 'IIIC'],
    'T': ['Tis', 'T1', 'T2', 'T3', 'T4'],
    'N': ['N0', 'N1', 'N2'],
    'CEA': ['negative', 'Borderline', 'positive'],
    'No.of resected LNs': ['0', '1-3', '≥4'],
    'Tumor Deposits': ['0', '1-2', '3+'],
    'Tumor size': ['＜5', '≥5'],
    'Median household income': ['＜$35,000', '$35,000-$54,999', '$55,000-$74,999', '≥$75,000+']
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

# 手动编码每个分类特征
input_data = pd.DataFrame({
    "Age": [ordered_var_categories['Age'].index(age)],  # 将选择的年龄范围转为数值
    "Grade": [ordered_var_categories['Grade'].index(grade)],  # 将 Grade 转为数值
    "TNM_Stage": [ordered_var_categories['TNM Stage'].index(tnm_stage)],  # TNM Stage 转为数值
    "T": [ordered_var_categories['T'].index(t)],  # T 转为数值
    "N": [ordered_var_categories['N'].index(n)],  # N 转为数值
    "CEA": [ordered_var_categories['CEA'].index(cea)],  # CEA 转为数值
    "No.of_resected_LNs": [ordered_var_categories['No.of resected LNs'].index(resected_lns)],  # LNs 数值化
    "Tumor_Deposits": [ordered_var_categories['Tumor Deposits'].index(tumor_deposits)],  # Tumor Deposits 数值化
    "Tumor_size": [ordered_var_categories['Tumor size'].index(tumor_size)],  # Tumor Size 数值化
    "Median_household_income": [ordered_var_categories['Median household income'].index(income)],  # 收入数值化
    "Sex_Female": [1 if sex == "Female" else 0],  # 性别编码
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
    # 确保 input_data 的列顺序与训练时一致
    input_data = input_data.reindex(columns=X_train.columns, fill_value=0)

    # 打印 NumPy 数组形式的输入数据
    input_data_array = input_data.to_numpy()

    # 预测风险评分并赋值给 predicted_risk
    predicted_risk = rsf.predict(input_data)

    # 预测累积风险函数
    cumulative_hazard_functions = rsf.predict_cumulative_hazard_function(input_data_array)

    # 获取所有时间点的累积风险值
    risks_matrix = []
    time_index = None

    for cumulative_hazard_func in cumulative_hazard_functions:
        risks = cumulative_hazard_func.y  # 累积风险值
        time_index = cumulative_hazard_func.x  # 对应的时间点
        risks_matrix.append(risks)

    # 显示分层标题
    st.markdown("### Risk Stratification")

    # 计算三分位数风险分层
    all_risks = rsf.predict(X_train)  # 计算训练集中的所有风险评分
    q1, q2 = np.percentile(all_risks, [33.33, 66.67])  # 33.33% 和 66.67% 作为分位数

    # 显示风险分层的详细信息
    st.write(f"Low Risk: below {q1:.4f} (green line)")
    st.write(f"Medium Risk: between {q1:.4f} and {q2:.4f} (orange line)")
    st.write(f"High Risk: above {q2:.4f} (red line)")

    # 显示患者的风险分层并使用颜色
    if predicted_risk[0] < q1:
        st.markdown(f"<span style='color: green;'>Current patient risk stratification: Low Risk</span>", unsafe_allow_html=True)
    elif predicted_risk[0] < q2:
        st.markdown(f"<span style='color: orange;'>Current patient risk stratification: Medium Risk</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='color: red;'>Current patient risk stratification: High Risk</span>", unsafe_allow_html=True)

    # # 计算三分位数风险分层
    # all_risks = rsf.predict(X_train)  # 计算训练集中的所有风险评分
    # q1, q2 = np.percentile(all_risks, [33.33, 66.67])

    # if predicted_risk[0] < q1:
    #     risk_group = "Low Risk"
    # elif predicted_risk[0] < q2:
    #     risk_group = "Medium Risk"
    # else:
    #     risk_group = "High Risk"

    # st.write(f"该患者属于: {risk_group}")

    # 计算 1、3、5 年的生存率
    time_points = [12, 36, 60]  # 12个月(1年), 36个月(3年), 60个月(5年)
    survival_rates = {}

    for time_point in time_points:
        # 获取在特定时间点的累积风险值
        survival_rate = 1 - cumulative_hazard_functions[0](time_point)
        # 将月份转换为年，并保存生存率
        survival_rates[f"{time_point // 12} -year survival rate"] = survival_rate

    # 显示 1, 3, 5 年的生存率
    st.markdown("### 1, 3, 5-year survival rates")
    for time_point, survival_rate in survival_rates.items():
        # 显示生存率，时间点显示为年
        st.write(f"{time_point}: {survival_rate:.4f}")

    # 输出累积风险曲线
    st.markdown("### Cumulative Hazard Curve")
    fig, ax = plt.subplots()
    ax.plot(time_index, risks_matrix[0], label='Cumulative Hazard')
    ax.set_xlabel("Time (Months)")
    ax.set_ylabel("Cumulative Hazard")
    ax.set_title("Cumulative Hazard Curve")
    ax.legend()
    st.pyplot(fig)
     # 将累积风险矩阵转置，使时间点作为行索引，并加上表头
    risk_matrix_df = pd.DataFrame(risks_matrix).T  # 转置矩阵
    risk_matrix_df.index = time_index  # 将时间点设为行索引
    risk_matrix_df.columns = ["Predicted Relative Risk Score"] * risk_matrix_df.shape[1]  # 将所有列名设置为“Risk Score”
    risk_matrix_df.index.name = "Time Point (month)"  # 设置行索引的表头为“Time point (month)”

    # 显示表格上方的标题
    st.markdown("### Cumulative Hazard Function Matrix")

    # 显示风险矩阵，并使表格宽度较小
    st.dataframe(risk_matrix_df, width=600)  # 将表格宽度设置为 600


