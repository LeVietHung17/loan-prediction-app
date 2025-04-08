import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model & data
model = joblib.load("random_forest_model.pkl")
data = pd.read_csv("loan_data_set.csv")

st.set_page_config(page_title="Loan Prediction App", layout="wide")

tab1, tab2 = st.tabs(["📊 Phân tích & Mô hình", "🔮 Dự đoán khoản vay"])

# TAB 1: Phân tích dữ liệu
with tab1:
    st.title("📊 Phân tích dữ liệu")
    st.dataframe(data)

    st.subheader("Phân phối thu nhập người vay")
    fig, ax = plt.subplots()
    sns.histplot(data['ApplicantIncome'], bins=10, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Tỷ lệ phê duyệt khoản vay")
    fig2, ax2 = plt.subplots()
    sns.countplot(x='Loan_Status', data=data, ax=ax2)
    ax2.set_xticklabels(['Từ chối', 'Phê duyệt'])
    st.pyplot(fig2)

# TAB 2: Dự đoán
with tab2:
    st.title("🔮 Dự đoán khoản vay")

    gender = st.selectbox("Giới tính", ['Male', 'Female'])
    married = st.selectbox("Hôn nhân", ['Yes', 'No'])
    education = st.selectbox("Trình độ học vấn", ['Graduate', 'Not Graduate'])
    income = st.number_input("Thu nhập người vay", value=4000)
    loan_amount = st.number_input("Số tiền vay", value=100)

    if st.button("Dự đoán"):
        gender_num = 1 if gender == 'Male' else 0
        married_num = 1 if married == 'Yes' else 0
        education_num = 1 if education == 'Graduate' else 0

        input_data = np.array([[gender_num, married_num, education_num, income, loan_amount]])
        prediction = model.predict(input_data)[0]
        result = "✅ Phê duyệt" if prediction == 1 else "❌ Từ chối"
        st.success(f"Kết quả: {result}")
