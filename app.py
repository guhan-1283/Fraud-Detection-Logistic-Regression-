import streamlit as st
import joblib
import pandas as pd
import time

model = joblib.load("fraud_detection.pkl")

st.title("Fraud Detection Web App")

st.markdown("Please enter the Transaction details and press the predict button")

transaction_type = st.selectbox("Transaction Type",['PAYMENT','TRANSFER','CASH_OUT','DEPOSIT'])
amount = st.number_input("Amount",min_value=0.0,value=1000.0)
oldbalanceorg = st.number_input("Old Balance (Sender)",min_value=0.0,value=10000.0)
newbalanceorg = st.number_input("New Balance (Sender)",min_value=0.0,value=9000.0)
oldbalancedest = st.number_input("Old Balance (Receiver)",min_value=0.0,value=0.0)
newbalancedest = st.number_input("New Balance (Receiver)",min_value=0.0,value=0.0)


if st.button("Predict"):

    input_data = pd.DataFrame([
        {
            'type':transaction_type,
            'amount':amount,
            'oldbalanceOrg':oldbalanceorg,
            'newbalanceOrig':newbalanceorg,
            'oldbalanceDest':oldbalancedest,
            'newbalanceDest':newbalancedest
        }
    ])


    with st.spinner("Wait for to Predict"):
        time.sleep(2.0)

        prediction = model.predict(input_data)[0]


        if prediction == 1:
            st.error("This Transaction is fraud")
        
        else:
            st.success("This is not Fraud Transaction")