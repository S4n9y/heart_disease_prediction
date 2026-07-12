
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("❤️ Heart Disease Prediction")
st.write("Fill in the patient's information below.")
st.divider()

st.subheader("👤 Patient Information")
c1, c2 = st.columns(2)

with c1:
    age = st.number_input("Age", 18, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.number_input("BMI", 10.0, 60.0, 24.0)
    heart_rate = st.number_input("Heart Rate", 40, 180, 75)

with c2:
    glucose = st.number_input("Glucose (mg/dL)", 50, 400, 100)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 180)
    systolic_bp = st.number_input("Systolic BP", 70, 250, 120)
    diastolic_bp = st.number_input("Diastolic BP", 40, 150, 80)

st.divider()
st.subheader("🏃 Lifestyle")
c3, c4 = st.columns(2)

with c3:
    smoking = st.selectbox("Smoking", [0,1], format_func=lambda x: "Yes" if x else "No")
    alcohol = st.selectbox("Alcohol Consumption", [0,1], format_func=lambda x: "Yes" if x else "No")

with c4:
    family_history = st.selectbox("Family History", [0,1], format_func=lambda x: "Yes" if x else "No")
    activity = st.selectbox("Physical Activity", ["High","Medium","Low"])

predict = st.button("🔍 Predict Heart Disease", use_container_width=True)

if predict:
    bmi_under = 1 if bmi < 18.5 else 0
    bmi_over = 1 if 25 <= bmi < 30 else 0
    bmi_obese = 1 if bmi >= 30 else 0

    gender_male = 1 if gender == "Male" else 0

    pa_low = 1 if activity == "Low" else 0
    pa_medium = 1 if activity == "Medium" else 0

    # Same logic as training notebook
    age_group_young = 1 if age < 30 else 0
    age_group_senior = 1 if age >= 50 else 0

    input_dict = {
        "age": age,
        "glucose_mg_dl": glucose,
        "cholesterol_mg_dl": cholesterol,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "bmi": bmi,
        "heart_rate": heart_rate,
        "smoking": smoking,
        "alcohol_consumption": alcohol,
        "family_history": family_history,
        "gender_Male": gender_male,
        "physical_activity_Low": pa_low,
        "physical_activity_Medium": pa_medium,
        "bmi_category_Obese": bmi_obese,
        "bmi_category_Overweight": bmi_over,
        "bmi_category_Underweight": bmi_under,
        "age_group_Young": age_group_young,
        "age_group_Senior": age_group_senior,
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    # Confidence of the predicted class
    if prediction == 1:
        probability = probabilities[1]
    else:
        probability = probabilities[0]

    actual_probability = probability * 100

    # -----------------------------
    # User-Friendly Risk Score
    # -----------------------------
    MIN_PROB = 76.5
    MAX_PROB = 86.0

    risk_score = ((actual_probability - MIN_PROB) / (MAX_PROB - MIN_PROB)) * 100

    # Keep between 0 and 100
    risk_score = max(0, min(100, risk_score))

    st.divider()
    st.subheader("🩺 Prediction Result")



    # -----------------------------
    # Display Results
    # -----------------------------

    st.metric(
        "Risk Score",
        f"{risk_score:.1f}/100"
    )

    st.progress(risk_score / 100)

    if risk_score < 35:
        st.success("🟢 Low Risk")
    elif risk_score < 70:
        st.warning("🟡 Moderate Risk")
    else:
        st.error("🔴 High Risk")

    with st.expander("ℹ️ How is this score calculated?"):
        st.write(
            """
    This score is a user-friendly representation of the machine learning model's prediction.

    The original model confidence is mapped to a 0–100 scale to make the result easier to interpret.

    **Original Model Confidence:** {:.2f}%
            """.format(actual_probability)
        )

    with st.expander("📋 Encoded Input"):
        st.dataframe(input_df)
