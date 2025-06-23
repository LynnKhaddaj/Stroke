import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🧬 Creative Stroke Risk Data Story")

# --- DATA PREP ---
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df.columns = df.columns.str.lower().str.strip()
df['gender'] = df['gender'].str.lower().str.strip()
df = df[df['gender'].isin(['male', 'female'])]
df = df[(df['bmi'] > 10) & (df['bmi'] < 60)]
df = df[(df['avg_glucose_level'] > 40) & (df['avg_glucose_level'] < 250)]

# --- INTERACTIVE PROFILE (slider/filter controls) ---
st.markdown("#### 🔎 Select a Profile to Visualize Stroke Risk")
col1, col2, col3, col4 = st.columns(4)
with col1:
    age = st.slider("Age", int(df['age'].min()), int(df['age'].max()), 45)
with col2:
    bmi = st.slider("BMI", int(df['bmi'].min()), int(df['bmi'].max()), 25)
with col3:
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "smokes", "formerly smoked"])
with col4:
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])

# --- Calculate personalized risk from dataset ---
sel = df[
    (df['age']>=age-2) & (df['age']<=age+2) &
    (df['bmi']>=bmi-1) & (df['bmi']<=bmi+1) &
    (df['smoking_status']==smoking_status) &
    (df['hypertension']==(1 if hypertension=="Yes" else 0))
]
personal_risk = sel['stroke'].mean()*100 if len(sel)>0 else np.nan

# --- Big visual: Animated Circular Gauge for Personalized Risk ---
st.markdown("### 🎯 Your Profile's Stroke Risk Gauge")
gauge_col, spider_col = st.columns([2,2])

with gauge_col:
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = personal_risk if not np.isnan(personal_risk) else 0,
        delta = {'reference': df['stroke'].mean()*100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 20]},
            'bar': {'color': "crimson"},
            'steps' : [
                {'range': [0, 5], 'color': "#d1e7dd"},
                {'range': [5, 10], 'color': "#ffe699"},
                {'range': [10, 20], 'color': "#f5c2c7"}
            ],
            'threshold' : {
                'line': {'color': "purple", 'width': 4},
                'thickness': 0.8,
                'value': personal_risk if not np.isnan(personal_risk) else 0
            }
        },
        number = {'suffix': "%"},
        title = {'text': "Stroke Risk"}
    ))
    fig.update_layout(margin=dict(l=20,r=20,t=40,b=20), height=350)
    st.plotly_chart(fig, use_container_width=True)
    if np.isnan(personal_risk):
        st.write("_No matching profiles in dataset—try relaxing your filters._")
    else:
        st.write(f"_This means {personal_risk:.2f}% of people like you had a stroke in this dataset._")

# --- Spider/Radar Plot for Main Risk Factors ---
with spider_col:
    st.markdown("#### 🕸️ Multi-Factor Risk (Radar Chart)")
    factors = ['age','bmi','avg_glucose_level','hypertension','heart_disease']
    means = df.groupby('stroke')[factors].mean()
    radar_vals = list(means.loc[1])  # average for stroke cases
    radar_base = list(means.loc[0])  # average for non-stroke
    radar_labels = ['Age','BMI','Glucose','Hypertension','Heart Disease']

    fig2 = go.Figure()
    fig2.add_trace(go.Scatterpolar(
        r = radar_vals,
        theta = radar_labels,
        fill='toself',
        name='Stroke'
    ))
    fig2.add_trace(go.Scatterpolar(
        r = radar_base,
        theta = radar_labels,
        fill='toself',
        name='No Stroke'
    ))
    fig2.update_layout(
      polar=dict(
        radialaxis=dict(visible=True, range=[0, max(df['age'].max(), df['bmi'].max(), df['avg_glucose_level'].max(),1.2)])
      ),
      showlegend=True,
      height=350,
      margin=dict(l=20,r=20,t=20,b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.write("_See how risk factors cluster for stroke vs no-stroke profiles._")

st.markdown("---")

# --- Heart Plot: Hypertension and Stroke Risk (iconic, small) ---
h1, h2 = st.columns([1,2])
with h1:
    st.markdown("#### ❤️ Hypertension Visual")
    h_risk = df.groupby('hypertension')['stroke'].mean()*100
    fig, ax = plt.subplots(figsize=(1.3,1.2))
    for val, color in zip([0,1], ['#abd699', '#ff6f61']):
        pct = h_risk[val] if val in h_risk else 0
        theta = np.linspace(0, np.pi, 100)
        x = np.sin(theta)
        y = np.cos(theta)
        ax.fill_between(x*0.5+val, 0, y*0.8, color=color, alpha=0.8)
        ax.text(val, 0.7, f"{pct:.1f}%", color="black", ha="center", va="center", fontsize=10, fontweight='bold')
        ax.text(val, -0.05, "No Htn" if val==0 else "Htn", ha="center", va="center", fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with h2:
    st.markdown("#### 🏥 Heart Disease & Smoking Effects (Horizontal Bar Art)")
    hd_smoke = df.groupby(['heart_disease','smoking_status'])['stroke'].mean().unstack()*100
    fig3, ax3 = plt.subplots(figsize=(4.2,1.2))
    hd_smoke.T.plot(kind="barh", ax=ax3, width=0.82, color=["#7bc043","#fdc900"])
    ax3.legend(title="Heart Disease", labels=["No","Yes"], fontsize=8, title_fontsize=8)
    ax3.set_xlabel("Stroke Rate (%)")
    ax3.set_ylabel("Smoking Status")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

# --- Compact Age + BMI Joint Plot (Heatmap) ---
st.markdown("#### 📊 Age x BMI: Stroke Risk Heatmap")
age_bmi_pivot = df.copy()
age_bmi_pivot['age_bin'] = pd.cut(age_bmi_pivot['age'], bins=np.linspace(df['age'].min(), df['age'].max(), 8))
age_bmi_pivot['bmi_bin'] = pd.cut(age_bmi_pivot['bmi'], bins=np.linspace(df['bmi'].min(), df['bmi'].max(), 7))
pivot = age_bmi_pivot.pivot_table(index='age_bin', columns='bmi_bin', values='stroke', aggfunc='mean')*100
fig4, ax4 = plt.subplots(figsize=(5.2,1.3))
im = ax4.imshow(pivot, cmap="Reds", aspect='auto')
ax4.set_xlabel("BMI Bin")
ax4.set_ylabel("Age Bin")
plt.colorbar(im, ax=ax4, fraction=0.045)
ax4.set_title("")
plt.tight_layout()
st.pyplot(fig4)
plt.close(fig4)

st.markdown("---")
st.markdown("<center><i>Visuals show group risk—not individual predictions. Try changing the sliders to see real data patterns. Dashboard by ChatGPT.</i></center>", unsafe_allow_html=True)
