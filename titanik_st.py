import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Titanic Predictor", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("titanic_model.joblib")

model = load_model()

st.title("🚢 Titanic Survival Prediction")
st.write("Ma'lumotlarni kiriting va tirik qolish ehtimolini ko‘ring.")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Pclass", [1, 2, 3], index=2)
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0, step=1.0)
    embarked = st.selectbox("Embarked", ["S", "C", "Q"], index=0)

with col2:
    sibsp = st.number_input("SibSp", min_value=0, max_value=10, value=0, step=1)
    parch = st.number_input("Parch", min_value=0, max_value=10, value=0, step=1)
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0, step=1.0)

input_df = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked
}])

if st.button("Predict", type="primary"):
    proba = model.predict_proba(input_df)[0][1]

    st.subheader("Result")
    st.write(f"Survival probability: **{proba:.1%}**")
    st.progress(float(proba))

    if proba >= 0.5:
        st.success("Prediction: Survived ✅")
    else:
        st.error("Prediction: Did not survive ❌")
