import streamlit as st
import pandas as pd
import joblib

# ------------------ LOAD MODELS ------------------
log_model = joblib.load('models/logistic_model.pkl')

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Credit Risk App",
    layout="wide"
)

# ------------------ SIDEBAR ------------------
st.sidebar.title("👨‍💻 About Me")

st.sidebar.write("""
**Name:** Abhishek Shelke  
**Role:** Aspiring Data Analyst  

🎓 MSc Computer Science  
📍 Pune, India  

🔗 [GitHub](https://github.com/Redskull2525)  
🔗 [LinkedIn](https://www.linkedin.com/in/abhishek-s-b98895249)
""")

st.sidebar.markdown("---")

st.sidebar.title("📊 Project Info")
st.sidebar.write("""
This project predicts whether a customer is **High Risk or Low Risk**  
using Machine Learning models.

Models used:
- Logistic Regression    
""")

# ------------------ MAIN UI ------------------
st.title("💳 Credit Risk Prediction System")

st.write("Enter customer details below:")

# ------------------ MODEL SELECT ------------------
model_choice = st.selectbox(
    "Choose Model",
    ["Logistic Regression"]
)

# ------------------ INPUT FIELDS ------------------
col1, col2 = st.columns(2)

with col1:
    income = st.number_input("💰 Income", min_value=0)
    age = st.number_input("🎂 Age", min_value=18, max_value=100)

with col2:
    loan_amount = st.number_input("💵 Loan Amount", min_value=0)
    interest_rate = st.number_input("📈 Interest Rate", min_value=0.0)
    emp_length = st.number_input("💼 Employment Length", min_value=0)

# ------------------ PREDICTION ------------------
if st.button("🔍 Predict Risk"):

    input_data = pd.DataFrame({
        'person_income': [income],
        'person_age': [age],
        'loan_amnt': [loan_amount],
        'loan_int_rate': [interest_rate],
        'person_emp_length': [emp_length]
    })

    # Match training columns
    input_data = input_data.reindex(columns=log_model.feature_names_in_, fill_value=0)

    # Select model
    model_choice == "Logistic Regression":
    prediction = log_model.predict(input_data)
    prob = log_model.predict_proba(input_data)[0][1]

    # ------------------ OUTPUT ------------------
    st.subheader("📊 Result")

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk Customer (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk Customer (Probability: {prob:.2f})")
