import streamlit as st
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack

# ---------------- LOAD MODELS ----------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_title.pkl", "rb") as f:
    tfidf_title = pickle.load(f)

with open("tfidf_skills.pkl", "rb") as f:
    tfidf_skills = pickle.load(f)

# ---------------- UI ----------------
st.set_page_config(page_title="Job Trend Predictor", layout="centered")

st.title("📈 Job & Skill Demand Predictor")
st.write("Predict whether a job role & skill set is **Trending or Not**")

# ---------------- USER INPUT ----------------
job_title = st.text_input("💼 Job Title", "Software Engineer")

skills = st.text_area(
    "🛠 Skills (comma separated)",
    "Python, SQL, Machine Learning"
)

experience = st.slider("📊 Minimum Experience (years)", 0, 10, 2)

# ---------------- PREDICT ----------------
if st.button("🔮 Predict Demand"):

    # Vectorize inputs
    X_title = tfidf_title.transform([job_title])
    X_skills = tfidf_skills.transform([skills])

    extra_features = np.array([[experience, len(skills.split(","))]])

    X_final = hstack([X_title, X_skills, extra_features])

    prediction = model.predict(X_final)[0]
    confidence = model.predict_proba(X_final)[0][prediction]

    # ---------------- RESULT ----------------
    if prediction == 1:
        st.success(f"🔥 TRENDING JOB & SKILLS\nConfidence: {confidence:.2f}")
    else:
        st.warning(f"📉 NOT TRENDING\nConfidence: {confidence:.2f}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built using Machine Learning | Job Market Analysis Project")
