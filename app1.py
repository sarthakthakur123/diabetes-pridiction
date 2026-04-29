# app.py — Diabetes Prediction System
# BCA Capstone Project

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "Diabetes Prediction System",
    page_icon  = "🩺",
    layout     = "wide"
)

# ── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('diabetes_scaler.pkl')
    return model, scaler

model, scaler = load_model()

# ── Sidebar Navigation ────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/000000/caduceus.png", width=80)
st.sidebar.title("🩺 Diabetes Prediction")
st.sidebar.markdown("**BCA Capstone Project**")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "📊 EDA & Insights",
    "🔮 Predict Diabetes"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** Pima Indians Diabetes")
st.sidebar.markdown("**Best Model:** Random Forest")
st.sidebar.markdown("**CV Accuracy:** 82.38%")

# ══════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🩺 Comparative Analysis of ML Models")
    st.subheader("For Diabetes Prediction — BCA Capstone Project")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📁 Dataset", "Pima Indians")
    col2.metric("📊 Total Records", "768")
    col3.metric("🤖 Models Compared", "8")
    col4.metric("✅ Best Accuracy", "82.38%")

    st.markdown("---")
    st.markdown("## 📌 About This Project")
    st.markdown("""
    This project performs a **comparative analysis of 8 machine learning models**
    for predicting diabetes using the **Pima Indians Diabetes Dataset**.

    ### 🎯 Objectives
    - Compare 8 ML models on the same dataset
    - Apply SMOTE to fix class imbalance
    - Use GridSearchCV for hyperparameter tuning
    - Select the best model based on accuracy and F1-Score
    - Deploy a real-time prediction system

    ### 🔬 Models Used
    | Model | Type |
    |---|---|
    | Logistic Regression | Linear |
    | Decision Tree | Tree-based |
    | Random Forest | Ensemble |
    | KNN | Instance-based |
    | SVM | Kernel-based |
    | Naive Bayes | Probabilistic |
    | Gradient Boosting | Boosting |
    | XGBoost | Boosting |

    ### 🛠️ Techniques Applied
    - **Data Cleaning** — Median imputation for missing values
    - **SMOTE** — Synthetic Minority Oversampling Technique
    - **GridSearchCV** — Hyperparameter Optimization
    - **10-Fold Cross Validation** — Reliable performance estimation
    """)

    st.markdown("---")
    st.markdown("## 📊 Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Features:**
        - Pregnancies
        - Glucose
        - Blood Pressure
        - Skin Thickness
        - Insulin
        - BMI
        - Diabetes Pedigree Function
        - Age
        """)
    with col2:
        st.markdown("""
        **Target Variable:**
        - 0 = No Diabetes (500 patients)
        - 1 = Diabetic (268 patients)

        **Challenge:**
        - Class imbalance (65% vs 35%)
        - Missing values in 5 columns
        - Small dataset (768 rows)
        """)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════
elif page == "📊 EDA & Insights":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")

    # Sample data for visualization
    data = {
        'Pregnancies': [6,1,8,1,0,5,3,10,2,8],
        'Glucose':     [148,85,183,89,137,116,78,115,197,125],
        'BloodPressure':[72,66,64,66,40,74,50,0,70,96],
        'SkinThickness':[35,29,0,23,35,0,32,0,45,0],
        'Insulin':     [0,0,0,94,168,0,88,0,543,0],
        'BMI':         [33.6,26.6,23.3,28.1,43.1,25.6,31.0,35.3,30.5,0],
        'DiabetesPedigreeFunction':[0.627,0.351,0.672,0.167,2.288,0.201,0.248,0.134,0.158,0.232],
        'Age':         [50,31,32,21,33,30,26,29,53,54],
        'Outcome':     [1,0,1,0,1,0,1,0,1,1]
    }
    df = pd.DataFrame(data)

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔥 Correlations", "📦 Boxplots"])

    with tab1:
        st.subheader("Feature Distributions")
        features = [c for c in df.columns if c != 'Outcome']
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        colors = ['#378ADD','#1D9E75','#D85A30','#7F77DD',
                  '#BA7517','#D4537E','#0F6E56']
        for i, (col, color) in enumerate(zip(features, colors)):
            axes[i].hist(df[col], bins=10, color=color,
                         edgecolor='white', alpha=0.85)
            axes[i].set_title(col, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

        st.info("💡 **Key Insight:** Glucose has the highest correlation with diabetes outcome (0.49). Higher glucose levels strongly indicate diabetes.")

    with tab2:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 7))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='RdYlGn',
                    mask=mask, linewidths=0.5, ax=ax)
        ax.set_title('Feature Correlation Heatmap', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

        st.info("💡 **Key Insight:** Glucose, BMI and Age are the top 3 predictors of diabetes in this dataset.")

    with tab3:
        st.subheader("Feature Distribution by Outcome")
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        for i, col in enumerate(features):
            sns.boxplot(x='Outcome', y=col, data=df,
                        palette=['#1D9E75','#D85A30'], ax=axes[i])
            axes[i].set_title(col, fontweight='bold')
            axes[i].set_xticklabels(['No Diabetes','Diabetes'])
        plt.tight_layout()
        st.pyplot(fig)

        st.info("💡 **Key Insight:** Diabetic patients have significantly higher Glucose, BMI, Age and Insulin values compared to non-diabetic patients.")

# ══════════════════════════════════════════════════════════════
# PAGE 4 — PREDICTION
# ══════════════════════════════════════════════════════════════
elif page == "🔮 Predict Diabetes":
    st.title("🔮 Diabetes Risk Prediction")
    st.markdown("Enter patient details to predict diabetes risk.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("🤰 Pregnancies",
                       min_value=0, max_value=20, value=1, step=1)
        glucose     = st.slider("🩸 Glucose Level (mg/dL)",
                       min_value=50, max_value=250, value=110)
        blood_pressure = st.slider("💉 Blood Pressure (mmHg)",
                       min_value=30, max_value=130, value=70)
        skin_thickness = st.slider("📏 Skin Thickness (mm)",
                       min_value=5, max_value=100, value=20)

    with col2:
        insulin     = st.slider("💊 Insulin Level (μU/mL)",
                       min_value=0, max_value=900, value=80)
        bmi         = st.number_input("⚖️ BMI",
                       min_value=10.0, max_value=70.0, value=25.0, step=0.1)
        dpf         = st.number_input("🧬 Diabetes Pedigree Function",
                       min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age         = st.number_input("👤 Age (years)",
                       min_value=10, max_value=100, value=30, step=1)

    st.markdown("---")

    if st.button("🔍 Predict Now", use_container_width=True, type="primary"):
        input_data = np.array([[pregnancies, glucose, blood_pressure,
                                 skin_thickness, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        prediction   = model.predict(input_scaled)[0]
        probability  = model.predict_proba(input_scaled)[0]

        diabetic_prob    = round(probability[1] * 100, 2)
        non_diabetic_prob = round(probability[0] * 100, 2)

        st.markdown("---")
        st.markdown("## 🏥 Prediction Result")

        col1, col2, col3 = st.columns(3)
        col1.metric("Diabetic Risk",     f"{diabetic_prob}%")
        col2.metric("Non-Diabetic",      f"{non_diabetic_prob}%")
        col3.metric("Prediction",
                    "⚠️ Diabetic" if prediction == 1 else "✅ Not Diabetic")

        if prediction == 1:
            if diabetic_prob >= 75:
                st.error(f"""
                ### ⚠️ HIGH RISK — Diabetes Detected
                **Risk Probability: {diabetic_prob}%**

                This patient shows strong indicators of diabetes.
                Immediate medical consultation is strongly recommended.
                """)
            else:
                st.warning(f"""
                ### ⚠️ MODERATE RISK — Possible Diabetes
                **Risk Probability: {diabetic_prob}%**

                Some diabetic indicators present.
                Medical consultation is recommended.
                """)
        else:
            st.success(f"""
            ### ✅ LOW RISK — No Diabetes Detected
            **Healthy Probability: {non_diabetic_prob}%**

            No strong indicators of diabetes found.
            Maintain a healthy lifestyle and regular checkups.
            """)

        # Risk gauge chart
        st.markdown("### 📊 Risk Probability Chart")
        fig, ax = plt.subplots(figsize=(8, 3))
        bars = ax.barh(['Non-Diabetic','Diabetic'],
                        [non_diabetic_prob, diabetic_prob],
                        color=['#1D9E75','#D85A30'], edgecolor='white',
                        height=0.4)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Probability (%)')
        ax.set_title('Prediction Probability', fontweight='bold')
        for bar, val in zip(bars, [non_diabetic_prob, diabetic_prob]):
            ax.text(val+1, bar.get_y()+bar.get_height()/2,
                    f'{val}%', va='center', fontweight='bold', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)

        # Input summary
        st.markdown("### 📋 Patient Input Summary")
        summary = pd.DataFrame({
            'Feature': ['Pregnancies','Glucose','Blood Pressure',
                        'Skin Thickness','Insulin','BMI',
                        'Diabetes Pedigree','Age'],
            'Value'  : [pregnancies, glucose, blood_pressure,
                        skin_thickness, insulin, bmi, dpf, age],
            'Normal Range': ['0-10','70-99 mg/dL','60-80 mmHg',
                              '10-40 mm','2-25 μU/mL','18.5-24.9',
                              '0.0-1.0','--']
        })
        st.dataframe(summary, use_container_width=True)

        st.caption("⚠️ Disclaimer: This is an ML-based prediction tool for educational purposes only. Always consult a qualified medical professional for diagnosis.")