import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("🩺 Stroke Risk Factors: Advanced Dashboard (Grid EDA)")

# --- LOAD AND PREPARE DATA ---
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df.columns = df.columns.str.strip().str.lower()
df['gender'] = df['gender'].str.lower().str.strip()
df = df[df['gender'].isin(['male', 'female'])]
df = df[(df['bmi'] > 10) & (df['bmi'] < 60)]
df = df[(df['avg_glucose_level'] > 40) & (df['avg_glucose_level'] < 250)]
df['stroke'] = df['stroke'].astype(int)
numeric = ['age', 'bmi', 'avg_glucose_level', 'hypertension', 'heart_disease']

# --- HEADER METRIC ---
stroke_rate = df['stroke'].mean() * 100
st.metric("Dataset Stroke Rate", f"{stroke_rate:.2f}% ({df['stroke'].sum()}/{len(df)})")

# --- FIRST ROW: 3 COMPACT GRAPHS ---
col1, col2, col3 = st.columns([1,1,1])

# 1. COMPACT CORRELATION HEATMAP
with col1:
    st.caption("Correlation with Stroke")
    corr = df[numeric+['stroke']].corr()
    fig = plt.figure(figsize=(2.8,2.4))
    sns.heatmap(corr[['stroke']][:-1], annot=True, cmap="Blues", vmin=-0.1, vmax=0.2, cbar=False, annot_kws={"size":10})
    plt.title("")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# 2. CATEGORICAL RISK BAR (Top 5 groups)
with col2:
    st.caption("Top 5 Highest-Risk Groups")
    cat_vars = ['ever_married','smoking_status','work_type','residence_type','gender']
    risks = []
    for c in cat_vars:
        means = df.groupby(c)['stroke'].mean()*100
        for k,v in means.items():
            risks.append((f"{c}: {k}", v))
    risks = sorted(risks, key=lambda x: -x[1])[:5]
    fig2, ax2 = plt.subplots(figsize=(2.6,2.4))
    names, vals = zip(*risks)
    ax2.barh(names, vals, color=plt.cm.viridis(np.linspace(0,1,len(vals))))
    ax2.set_xlabel("Stroke %")
    ax2.set_xlim(0,max(vals)+2)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# 3. COMPACT HYPERTENSION & HEART DISEASE MOSAIC (interpretable)
with col3:
    st.caption("Risk by Hypertension + Heart Disease")
    grid = df.groupby(['hypertension','heart_disease'])['stroke'].mean().unstack(fill_value=0)*100
    fig3, ax3 = plt.subplots(figsize=(2.3,2.4))
    sns.heatmap(grid, annot=True, fmt=".1f", cmap="RdPu", cbar=False, 
                xticklabels=['No Heart\nDisease','Heart\nDisease'], yticklabels=['No Hyper','Hyper'],
                annot_kws={"size":11})
    ax3.set_xlabel("Heart Disease")
    ax3.set_ylabel("Hypertension")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

# --- SECOND ROW: 3 MORE VISUALS ---
col4, col5, col6 = st.columns([1,1,1])

# 4. STROKE % BY AGE (DENSE CURVE)
with col4:
    st.caption("Stroke Probability by Age")
    curve = df.groupby('age')['stroke'].mean()*100
    fig4, ax4 = plt.subplots(figsize=(2.8,2.1))
    ax4.plot(curve.index, curve.values, color="#d84315", lw=2)
    ax4.fill_between(curve.index, curve.values, color="#ce93d8", alpha=0.3)
    ax4.set_xlabel("Age")
    ax4.set_ylabel("%")
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

# 5. VIOLIN/BOX PLOT: BMI by Stroke
with col5:
    st.caption("BMI by Stroke Status")
    fig5, ax5 = plt.subplots(figsize=(2.6,2.1))
    sns.violinplot(data=df, x="stroke", y="bmi", inner="quartile", palette="muted", ax=ax5)
    ax5.set_xlabel("Stroke (0=No, 1=Yes)")
    ax5.set_ylabel("BMI")
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)

# 6. STACKED BAR: SMOKING & STROKE
with col6:
    st.caption("Stroke % by Smoking Status")
    temp = df[df['smoking_status']!='Unknown'].copy()
    smoke = temp.groupby(['smoking_status','stroke']).size().unstack(fill_value=0)
    perc = smoke.div(smoke.sum(axis=1), axis=0) * 100
    fig6, ax6 = plt.subplots(figsize=(2.4,2.1))
    perc[[0,1]].plot(kind='bar', stacked=True, color=['#90caf9','#c62828'], ax=ax6, width=0.85)
    ax6.legend(['No Stroke','Stroke'], fontsize=8, loc='upper right')
    ax6.set_ylabel("% of Group")
    ax6.set_xlabel("Smoking Status")
    ax6.set_xticklabels(perc.index, rotation=25)
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close(fig6)

    _Sample sizes for rare subgroups are small—interpret with care!_
    """)

st.markdown("---")
st.markdown("<center><i>Dashboard by ChatGPT | All visuals show group-level risk, not personal prediction. Data is illustrative.</i></center>", unsafe_allow_html=True)
