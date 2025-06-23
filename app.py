import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🧬 Ultimate Stroke Risk Dashboard")

# Load and clean data
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df.columns = df.columns.str.lower().str.strip()
df = df[df['gender'].isin(['Male','Female'])]  # ensure only valid genders
df = df[(df['bmi'] > 10) & (df['bmi'] < 60)]

# Sidebar controls
st.sidebar.title("Build Your Patient")
age = st.sidebar.slider("Age", 0, 82, 45)
gender = st.sidebar.radio("Gender", ['Male','Female'])
bmi = st.sidebar.slider("BMI", 10, 59, 25)
smoking_status = st.sidebar.radio("Smoking Status", ['never smoked','smokes','formerly smoked'])
hypertension = st.sidebar.checkbox("Hypertension?")
heart_disease = st.sidebar.checkbox("Heart Disease?")

# Filter data for this “patient”
mask = (
    (df['age'].between(age-2, age+2)) &
    (df['gender']==gender) &
    (df['bmi'].between(bmi-1, bmi+1)) &
    (df['smoking_status']==smoking_status) &
    (df['hypertension']==int(hypertension)) &
    (df['heart_disease']==int(heart_disease))
)
df_patient = df[mask]
risk = df_patient['stroke'].mean()*100 if len(df_patient)>0 else np.nan

# --- Build-a-Patient Avatar ---
def draw_avatar(ax, age, bmi, smoke, htn, hd):
    # Base figure
    ax.plot([0,0], [0,1.1], lw=15, color="#e0e0e0")  # body
    ax.plot([-0.22,0.22], [0.65,0.65], lw=10, color="#e0e0e0")  # arms
    ax.plot([-0.14,0.14], [0,0.5], lw=6, color="#e0e0e0")  # legs
    circ = plt.Circle((0,1.2), 0.18, color="#e0e0e0")
    ax.add_patch(circ)
    # Age = more wrinkles (add lines)
    if age>60:
        ax.plot([-0.1,0.1],[1.28,1.28],color="gray", lw=2)
    # BMI = fatter body
    fat = (bmi-20)/30
    ax.plot([0-fat,0+fat],[0,1.1], lw=15+10*fat, color="#8bc34a")
    # Smoking = puff
    if smoke!="never smoked":
        ax.plot([0.2,0.3], [1.1,1.22], color="brown", lw=3)
        ax.text(0.32,1.25,"💨",fontsize=18)
    # Hypertension = red face
    if htn:
        circ = plt.Circle((0,1.2), 0.18, color="#e57373", alpha=0.7)
        ax.add_patch(circ)
    # Heart disease = heart icon
    if hd:
        ax.text(0,-0.1,"❤️",fontsize=24,ha="center")
    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(-0.25,1.45)
    ax.axis('off')

# Display avatar + risk %
col1, col2 = st.columns([1,2])
with col1:
    fig, ax = plt.subplots(figsize=(2,4))
    draw_avatar(ax, age, bmi, smoking_status, hypertension, heart_disease)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("<center><i>Patient avatar: fills and changes as you add risks!</i></center>", unsafe_allow_html=True)

with col2:
    st.markdown(f"### Estimated Stroke Risk: <span style='color:#d32f2f'>{risk:.1f}%</span>" if not np.isnan(risk) else "No similar cases in data.", unsafe_allow_html=True)
    st.write("_Based on real people with these risk factors in the data._")

