import streamlit as st
import pandas as pd
import joblib  # Import joblib to load the model

# Load the model
model_dict = joblib.load('artifacts/model_data.joblib')
model = model_dict['model']  # Adjust this key as needed

# Custom CSS to enhance the appearance
st.markdown("""
    <style>
        .stApp {
            background-color: #f5f5f5;
            font-family: 'Arial', sans-serif;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 24px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .stTitle {
            color: #333333;
            font-size: 32px;
            font-weight: bold;
            text-align: center;
        }
        .stNumberInput, .stSelectbox {
            margin-bottom: 10px;
        }
        .stNumberInput label, .stSelectbox label {
            font-weight: bold;
            color: #333333;
        }
        .stMarkdown h3 {
            color: #333333;
            font-size: 24px;
            font-weight: bold;
            text-align: center;
        }
        .result-positive {
            color: green;
            font-weight: bold;
        }
        .result-negative {
            color: red;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit App Code
st.title("💳 Credit Card Defaulter")

st.sidebar.header("User Inputs")

# Age input
age = st.sidebar.number_input("👤 Age", min_value=21, max_value=79, value=30, step=1)

# Updated Gender Selector with Encoding Information
gender = st.sidebar.selectbox("⚥ Gender (1: Male, 2: Female)", options=[1, 2], index=0, format_func=lambda x: "Male" if x == 1 else "Female")

# Education Level Selector with custom labels
education_labels = {
    1: "Graduate school",
    2: "University",
    3: "High school",
    4: "Others"
}

education_level = st.sidebar.selectbox(
    "🎓 Education Level",
    options=list(education_labels.keys()),
    index=0,
    format_func=lambda x: f"{x}: {education_labels[x]}"
)

# Marital Status Selector with custom labels
marital_labels = {
    1: "Married",
    2: "Single",
    3: "Others"
}

marital_status = st.sidebar.selectbox(
    "💍 Marital Status",
    options=list(marital_labels.keys()),
    index=0,
    format_func=lambda x: f"{x}: {marital_labels[x]}"
)

# Credit Limit input
credit_limit = st.sidebar.number_input("💳 Credit Limit", min_value=10000, max_value=2000000, value=50000, step=10000)

# Main Page Inputs
st.subheader("Repayment Status")

# Repayment Status inputs
repayment_status_sept = st.number_input("Repayment Status (Sept)", min_value=-1, max_value=8, value=-1, step=1)
repayment_status_aug = st.number_input("Repayment Status (Aug)", min_value=-1, max_value=8, value=-1, step=1)
repayment_status_july = st.number_input("Repayment Status (July)", min_value=-1, max_value=8, value=-1, step=1)
repayment_status_june = st.number_input("Repayment Status (June)", min_value=-1, max_value=8, value=-1, step=1)
repayment_status_may = st.number_input("Repayment Status (May)", min_value=-1, max_value=8, value=-1, step=1)
repayment_status_april = st.number_input("Repayment Status (April)", min_value=-1, max_value=8, value=-1, step=1)

st.subheader("Bill Statements")

# Bill Statement inputs
bill_statement_sept = st.number_input("Bill Statement (Sept)", min_value=0, max_value=2000000, value=0, step=10000)
bill_statement_aug = st.number_input("Bill Statement (Aug)", min_value=0, max_value=2000000, value=0, step=10000)
bill_statement_july = st.number_input("Bill Statement (July)", min_value=0, max_value=2000000, value=0, step=10000)
bill_statement_june = st.number_input("Bill Statement (June)", min_value=0, max_value=2000000, value=0, step=10000)
bill_statement_may = st.number_input("Bill Statement (May)", min_value=0, max_value=2000000, value=0, step=10000)
bill_statement_april = st.number_input("Bill Statement (April)", min_value=0, max_value=2000000, value=0, step=10000)

st.subheader("Previous Payments")

# Previous Payment inputs
previous_payment_sept = st.number_input("Previous Payment (Sept)", min_value=0, max_value=2000000, value=0, step=10000)
previous_payment_aug = st.number_input("Previous Payment (Aug)", min_value=0, max_value=2000000, value=0, step=10000)
previous_payment_july = st.number_input("Previous Payment (July)", min_value=0, max_value=2000000, value=0, step=10000)
previous_payment_june = st.number_input("Previous Payment (June)", min_value=0, max_value=2000000, value=0, step=10000)
previous_payment_may = st.number_input("Previous Payment (May)", min_value=0, max_value=2000000, value=0, step=10000)
previous_payment_april = st.number_input("Previous Payment (April)", min_value=0, max_value=2000000, value=0, step=10000)

# Button to predict default
if st.button("✅ Calculate Default Risk"):
    # Gather input data into a DataFrame for the model
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Education_Level': [education_level],
        'Marital_Status': [marital_status],
        'Credit_Limit': [credit_limit],
        'Repayment_Status_Sept': [repayment_status_sept],
        'Repayment_Status_Aug': [repayment_status_aug],
        'Repayment_Status_July': [repayment_status_july],
        'Repayment_Status_June': [repayment_status_june],
        'Repayment_Status_May': [repayment_status_may],
        'Repayment_Status_April': [repayment_status_april],
        'Bill_Statement_Sept': [bill_statement_sept],
        'Bill_Statement_Aug': [bill_statement_aug],
        'Bill_Statement_July': [bill_statement_july],
        'Bill_Statement_June': [bill_statement_june],
        'Bill_Statement_May': [bill_statement_may],
        'Bill_Statement_April': [bill_statement_april],
        'Previous_Payment_Sept': [previous_payment_sept],
        'Previous_Payment_Aug': [previous_payment_aug],
        'Previous_Payment_July': [previous_payment_july],
        'Previous_Payment_June': [previous_payment_june],
        'Previous_Payment_May': [previous_payment_may],
        'Previous_Payment_April': [previous_payment_april],
    })

    # Reorder the DataFrame columns to match the order expected by the model
    input_data = input_data[model.get_booster().feature_names]

    # Use the model to predict whether the customer will default
    prediction = model.predict(input_data)

    # Display the result with color coding
    if prediction[0] == 1:
        st.markdown("<h3 class='result-negative'>Prediction: The customer will default (1)</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 class='result-positive'>Prediction: The customer will not default (0)</h3>", unsafe_allow_html=True)
