# 使用Streamlit的列布局工具
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

    # 其余特征
    income = st.selectbox("Median Household Income", options=[1.0, 2.0, 3.0, 4.0], index=0)
    sex = st.selectbox("Sex", options=["Male", "Female"], index=0)
    race = st.selectbox("Race", options=["White", "Black", "Other"], index=0)
    marital_status = st.selectbox("Marital status", options=["Single", "Married", "Divorced", "Widowed"], index=0)
    histology = st.selectbox("Histology", options=["Non-specific adenocarcinoma", "Specific adenocarcinoma", "Other"], index=0)
    resection_type = st.selectbox("Resection type", options=[
        "Partial/subtotal colectomy", 
        "Hemicolectomy or greater", 
        "Total colectomy", 
        "Colectomy plus removal of other organs"
    ], index=0)
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
    perineural_invasion = st.selectbox("Perineural Invasion", options=["No", "Yes"], index=0)

    submit = st.form_submit_button("Submit")
