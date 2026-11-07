import streamlit as st
import joblib
import pandas as pd

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Dashboard Title
st.title("Diabetes Prediction Dashboard")

st.write("Enter the patient's information to predict diabetes.")

# Patient Information
st.subheader("Patient Information")

glucose = st.number_input("Glucose", min_value=0.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0)
insulin = st.number_input("Insulin", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=1.0)

# Prediction
if st.button("Predict"):

    input_data = pd.DataFrame([[
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        pedigree,
        age
    ]], columns=[
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ])

    # Scale input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Display result
    if prediction == 1:
        st.error("Diabetes")
    else:
        st.success("No Diabetes")