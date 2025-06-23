import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(layout="wide")

# --- LOAD AND PREPARE DATA ---
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df.columns = df.columns.str.strip().str.lower()
df['gender'] = df['gender'].str.lower().str.strip()
df = df[df['gender'].isin(['male', 'female'])]
df = df[(df['bmi'] > 10) & (df['bmi'] < 60)]
df = df[(df['avg_glucose_level'] > 40) & (df['avg_glucose_level'] < 250)]
df['stroke'] = df['stroke'].astype(int)
numeric = ['age', 'bmi', 'avg_glucose_level', 'hypertension', 'heart_disease']

# --- LAYOUT: 2 rows, each with 2 or 3 graphs, all fit on single screen ---
st.title("🩺 Stroke Risk Factors: Dashboard (Advanced EDA)")

col1, col2, col3 = st.columns([1.2,1,1])

# --- 1. Correlation Heatmap ---
with col1:
    st.subheader("Numeric Feature Correlation to Stroke")
    corr = df[numeric+['stroke']].corr()
    plt.figure(figsize=(3.5,2.5))
    sns.heatmap(corr[['stroke']][:-1], annot=True, cmap="Blues", vmin=-0.1, vmax=0.2, cbar=False)
    plt.title("Correlation with Stroke")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

# --- 2. Risk by Categorical (Sorted Bar) ---
with col2:
    st.subheader("Highest-Risk Groups (Mean Stroke %)")
    cat_vars = ['ever_married','smoking_status','work_type','residence_type','gender']
    risks = []
    for c in cat_vars:
        means = df.groupby(c)['stroke'].mean()*100
        for k,v in means.items():
            risks.append((f"{c}: {k}", v))
    risks = sorted(risks, key=lambda x: -x[1])[:7]
    fig, ax = plt.subplots(figsize=(3.3,2.5))
    names, vals = zip(*risks)
    ax.barh(names, vals, color=plt.cm.viridis(np.linspace(0,1,len(vals))))
    ax.set_xlabel("Stroke Rate (%)")
    ax.set_xlim(0,max(vals)+2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

# --- 3. Joint Effect: Hypertension × Smoking Stacked Bar ---
with col3:
    st.subheader("Joint Effect: Hypertension & Smoking")
    temp = df[df['smoking_status']!='Unknown'].copy()
    joint = temp.groupby(['hypertension','smoking_status'])['stroke'].mean().unstack()*100
    joint = joint.loc[[0,1]]  # Ensure 0,1 order
    joint.index = ['No Hyper','Hyper']
    fig, ax = plt.subplots(figsize=(3.3,2.5))
    bottom = np.zeros(len(joint))
    for s in joint.columns:
        ax.bar(joint.index, joint[s], bottom=bottom, label=s)
        bottom += joint[s].values
    ax.legend(fontsize=8)
    ax.set_ylabel("Stroke Rate (%)")
    ax.set_ylim(0, max(bottom)+4)
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

# --- 4. Age Curve (full-width below) ---
st.subheader("Age vs. Stroke Risk (%)", divider="rainbow")
curve_col1, curve_col2 = st.columns([2,1])

with curve_col1:
    curve = df.groupby('age')['stroke'].mean()*100
    fig, ax = plt.subplots(figsize=(6.8,2.1))
    ax.plot(curve.index, curve.values, color="#d84315", lw=2)
    ax.set_xlabel("Age")
    ax.set_ylabel("Stroke %")
    ax.set_title("Stroke Probability by Age")
    ax.grid(alpha=0.15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()
with curve_col2:
    st.write("**How to read this dashboard:**")
    st.markdown("""
    - **Left:** Numeric features most associated with stroke
    - **Center:** Most at-risk subgroups (by mean %)
    - **Right:** Combined impact (e.g. smokers with hypertension)
    - **Below:** How risk rises with age

    _Sample sizes for rare subgroups are small—interpret with care!_
    """)

st.markdown("---")
st.markdown("<center><i>Dashboard by ChatGPT | All visuals show group-level risk, not personal prediction. Data is illustrative.</i></center>", unsafe_allow_html=True)
