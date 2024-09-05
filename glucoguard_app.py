import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nhanes.load import load_NHANES_data, load_NHANES_metadata
from sklearn.impute import SimpleImputer

# Set page config
st.set_page_config(page_title="GlucoGuard Demo", page_icon="🛡️", layout="wide")

# Custom CSS to improve the app's appearance
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .st-bw {
        background-color: white;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .patient-card {
        background-color: white;
        padding: 0px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .risk-card {
        background-color: #e0f0ff;
        padding: 0px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .stMetric {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# Load and preprocess NHANES data
@st.cache_data
def load_nhanes_data():
    data = load_NHANES_data(year='2017-2018')
    metadata = load_NHANES_metadata(year='2017-2018')
    
    variable_mapping = {
        'Age': 'AgeInYearsAtScreening',
        'BMI': 'BodyMassIndexKgm2',
        'SystolicBloodPressure': 'SystolicBloodPres1StRdgMmHg',
        'Glucose': 'Glycohemoglobin',
        'DiabetesStatus': 'DoctorToldYouHaveDiabetes'
    }
    
    selected_vars = [var for var in variable_mapping.values() if var is not None]
    data = data[selected_vars].copy()
    data.columns = [key for key, value in variable_mapping.items() if value is not None]
    
    data['DiabetesStatus'] = data['DiabetesStatus'].map({'Yes': 1, 'No': 0, 'Borderline': 0})
    data = data.dropna()
    
    return data, metadata

# Load data
data, metadata = load_nhanes_data()

# Create fictitious patient data
fictitious_patients = {
    "Emily Johnson, 35": {"Age": 35, "BMI": 24.8, "SystolicBloodPressure": 118, "Glucose": 5.2},
    "Michael Chen, 42": {"Age": 42, "BMI": 27.5, "SystolicBloodPressure": 125, "Glucose": 5.6},
    "Sophia Rodriguez, 28": {"Age": 28, "BMI": 22.1, "SystolicBloodPressure": 110, "Glucose": 4.9},
    "William Taylor, 55": {"Age": 55, "BMI": 31.2, "SystolicBloodPressure": 140, "Glucose": 6.3},
    "Olivia Brown, 48": {"Age": 48, "BMI": 29.7, "SystolicBloodPressure": 132, "Glucose": 5.9},
    "James Wilson, 62": {"Age": 62, "BMI": 28.4, "SystolicBloodPressure": 145, "Glucose": 6.7},
    "Ava Martinez, 39": {"Age": 39, "BMI": 26.3, "SystolicBloodPressure": 122, "Glucose": 5.4},
    "Ethan Davis, 51": {"Age": 51, "BMI": 33.5, "SystolicBloodPressure": 138, "Glucose": 6.1},
    "Emma Thompson, 45": {"Age": 45, "BMI": 25.9, "SystolicBloodPressure": 128, "Glucose": 5.7},
    "Noah Garcia, 58": {"Age": 58, "BMI": 30.8, "SystolicBloodPressure": 150, "Glucose": 7.0}
}

# Sidebar for patient selection
st.sidebar.title("Patient Selection")
selected_patient = st.sidebar.selectbox(
    "Choose a patient",
    list(fictitious_patients.keys())
)

# Get selected patient data
patient_data = fictitious_patients[selected_patient]

# Main content
st.title("🛡️ GlucoGuard: AI-Powered Diabetes-II Prediction")
st.markdown("""
GlucoGuard is an advanced AI-powered diabetes prediction tool. This application 
utilizes comprehensive health data and illustrative patient profiles to showcase the 
sophisticated capabilities of our predictive model. Explore the intricate interplay of 
various factors contributing to diabetes risk assessment in this cutting-edge demonstration.
""")

# Display patient data and risk assessment cards side by side
col1, col2 = st.columns(2)

with col1:
    # Display patient data in a card
    st.markdown(f"<div class='patient-card'>", unsafe_allow_html=True)
    st.subheader(f"Patient Information: {selected_patient}")
    col1a, col1b, col1c, col1d = st.columns(4)
    col1a.metric("Age", patient_data['Age'])
    col1b.metric("BMI", f"{patient_data['BMI']:.1f}")
    col1c.metric("Systolic BP", patient_data['SystolicBloodPressure'])
    col1d.metric("HbA1c (%)", patient_data['Glucose'])
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # Calculate risk score and factor contributions
    risk_factors = ['Age', 'BMI', 'SystolicBloodPressure', 'Glucose']
    factor_weights = [0.2, 0.3, 0.2, 0.3]  # Example weights, adjust as needed

    risk_contributions = {}
    total_risk_score = 0

    for factor, weight in zip(risk_factors, factor_weights):
        factor_percentile = np.percentile(data[factor], [25, 50, 75])
        factor_score = np.searchsorted(factor_percentile, patient_data[factor]) / 3  # Normalize to 0-1
        weighted_score = factor_score * weight
        risk_contributions[factor] = weighted_score
        total_risk_score += weighted_score

    # Display risk assessment in a card
    st.markdown(f"<div class='risk-card'>", unsafe_allow_html=True)
    st.subheader("Diabetes Risk Assessment")
    st.markdown(f"**Overall Risk Score: {total_risk_score:.2%}**")
    st.progress(total_risk_score)

    if total_risk_score > 0.75:
        st.warning("High risk of diabetes. Recommend further testing and lifestyle changes.")
    elif total_risk_score > 0.5:
        st.info("Moderate risk of diabetes. Consider lifestyle modifications and regular check-ups.")
    else:
        st.success("Low risk of diabetes. Maintain a healthy lifestyle.")
    st.markdown("</div>", unsafe_allow_html=True)

# Display factor contributions and distributions side by side
col1, col2 = st.columns(2)

with col1:
    st.subheader("Risk Factor Contributions")
    fig, ax = plt.subplots(figsize=(6, 6))
    factor_names = list(risk_contributions.keys())
    factor_scores = list(risk_contributions.values())
    colors = plt.cm.RdYlGn_r(np.array(factor_scores) / max(factor_scores))
    bars = ax.barh(factor_names, factor_scores, color=colors)
    ax.set_xlim(0, max(factor_scores) * 1.1)
    ax.set_xlabel('Risk Contribution')
    ax.set_title('Contribution of Each Factor to Overall Risk')

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{factor_scores[i]:.2f}', 
                ha='left', va='center', fontweight='bold', color='black', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("Risk Factor Distributions")
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    axes = axes.ravel()

    for i, factor in enumerate(risk_factors):
        sns.histplot(data[factor], kde=True, ax=axes[i])
        axes[i].axvline(patient_data[factor], color='r', linestyle='--')
        axes[i].set_title(f"{factor}")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        axes[i].tick_params(labelsize=8)

    plt.tight_layout()
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer**: This demo uses real NHANES data and fictitious patient profiles for illustrative purposes only. 
The risk assessment provided is based on a simplified model and should not be considered as medical advice. 
For actual medical advice, please consult with healthcare professionals.
""")