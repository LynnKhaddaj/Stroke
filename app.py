import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🧬 Creative Stroke Risk Explorer")

# ---- Load and Preprocess Data ----
@st.cache_data
def load_data():
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    df.columns = df.columns.str.lower().str.strip()
    df = df[df['gender'].isin(['Male','Female'])]  # only realistic
    df = df[(df['bmi'] > 10) & (df['bmi'] < 60)]   # remove BMI outliers
    return df

df = load_data()

# --- Binning for Smoother Risk Estimates ---
df['age_group'] = pd.cut(df['age'], [0,40,60,82], labels=['<40','40-60','60+'], right=True)
df['bmi_group'] = pd.cut(df['bmi'], [10,18.5,25,30,60], labels=['Underweight','Normal','Overweight','Obese'], right=False)
df['glucose_group'] = pd.cut(df['avg_glucose_level'], [0,100,125,150,300], labels=['Normal','Prediabetes','High','Very High'], right=False)

# --- Sidebar Controls ---
with st.sidebar:
    st.header("🔧 Build a Patient Profile")
    age = st.slider("Age", 0, 82, 50)
    gender = st.radio("Gender", ['Male','Female'])
    bmi = st.slider("BMI", 10, 59, 25)
    hypertension = st.radio("Hypertension?", ['No', 'Yes'])
    heart_disease = st.radio("Heart Disease?", ['No', 'Yes'])
    smoking_status = st.radio("Smoking Status", ['never smoked', 'smokes', 'formerly smoked'])
    avg_glucose_level = st.slider("Avg. Glucose Level", 50, 300, 100)

# --- Assign Profile to Bins ---
age_group = pd.cut([age], [0,40,60,82], labels=['<40','40-60','60+'])[0]
bmi_group = pd.cut([bmi], [10,18.5,25,30,60], labels=['Underweight','Normal','Overweight','Obese'])[0]
glucose_group = pd.cut([avg_glucose_level], [0,100,125,150,300], labels=['Normal','Prediabetes','High','Very High'])[0]

htn_bin = 1 if hypertension=="Yes" else 0
hd_bin = 1 if heart_disease=="Yes" else 0

# --- Main Calculation: Multi-Factor Bin ---
main_factors = ['age_group','bmi_group','smoking_status','hypertension','heart_disease']
query = (
    (df['age_group']==age_group) &
    (df['bmi_group']==bmi_group) &
    (df['smoking_status']==smoking_status) &
    (df['hypertension']==htn_bin) &
    (df['heart_disease']==hd_bin)
)
filtered = df[query]
N = len(filtered)
stroke_rate = filtered['stroke'].mean() if N>=5 else np.nan

# Fallback: broader group if too few cases
fallback_query = (
    (df['age_group']==age_group) &
    (df['hypertension']==htn_bin) &
    (df['smoking_status']==smoking_status)
)
fallback = df[fallback_query]
fallback_N = len(fallback)
fallback_stroke_rate = fallback['stroke'].mean() if fallback_N>=5 else df['stroke'].mean()

# --- Avatar Drawing ---
def draw_avatar(ax, age, bmi, smoke, htn, hd):
    # Body: size scales with BMI
    base_width = 15 + 10*(bmi-18.5)/41.5
    ax.plot([0,0], [0,1.1], lw=base_width, color="#b3cde0")  # body
    # Arms
    ax.plot([-0.23,0.23], [0.7,0.7], lw=8, color="#b3cde0")
    # Legs
    ax.plot([-0.1,0.1], [0,0.55], lw=6, color="#b3cde0")
    # Head
    circ = plt.Circle((0,1.25), 0.18, color="#b3cde0")
    ax.add_patch(circ)
    # Smoking
    if smoke!="never smoked":
        ax.plot([0.2,0.3], [1.12,1.3], color="brown", lw=3)
        ax.text(0.33,1.3,"💨",fontsize=20)
    # Hypertension: face turns reddish
    if htn:
        circ2 = plt.Circle((0,1.25), 0.18, color="#e57373", alpha=0.6)
        ax.add_patch(circ2)
    # Heart Disease: add heart
    if hd:
        ax.text(0,-0.18,"❤️",fontsize=30,ha="center")
    # Age: add cane for older patients
    if age>65:
        ax.plot([0.27,0.27],[0.55,0.9],color="gray", lw=6)
        ax.plot([0.27,0.33],[0.55,0.5],color="gray", lw=3)
    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(-0.25,1.5)
    ax.axis('off')

col1, col2 = st.columns([1,2])
with col1:
    fig, ax = plt.subplots(figsize=(2,4))
    draw_avatar(ax, age, bmi, smoking_status, htn_bin, hd_bin)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("<center><i>Avatar responds to risk factors!</i></center>", unsafe_allow_html=True)

with col2:
    if not np.isnan(stroke_rate):
        st.markdown(f"### Estimated Stroke Risk: <span style='color:#d32f2f'>{stroke_rate*100:.1f}%</span> <sub style='color:gray'>(n={N})</sub>", unsafe_allow_html=True)
    elif not np.isnan(fallback_stroke_rate):
        st.markdown(f"### Estimated Stroke Risk for similar profile: <span style='color:#d32f2f'>{fallback_stroke_rate*100:.1f}%</span> <sub style='color:gray'>(n={fallback_N})</sub>", unsafe_allow_html=True)
        st.info("Not enough people with your exact profile, showing a broader estimate.")
    else:
        st.markdown("### Estimated Stroke Risk: <span style='color:#d32f2f'>Data too sparse</span>", unsafe_allow_html=True)
    st.caption("_Based on grouped real data, not just your unique combo._")

# --- Creative Multi-Factor Visuals ---
st.subheader("🧪 Multi-Risk Factor Explorer")

# 1. Risk Ladder (how risk increases as you add risk factors)
ladder_cols = ['All','+ Hypertension','+ Heart Disease','+ Smoking']
rates = []
nvals = []
f = df[df['age_group']==age_group]
rates.append(f['stroke'].mean())
nvals.append(len(f))
f1 = f[f['hypertension']==htn_bin]
rates.append(f1['stroke'].mean())
nvals.append(len(f1))
f2 = f1[f1['heart_disease']==hd_bin]
rates.append(f2['stroke'].mean())
nvals.append(len(f2))
f3 = f2[f2['smoking_status']==smoking_status]
rates.append(f3['stroke'].mean())
nvals.append(len(f3))

fig2, ax2 = plt.subplots(figsize=(6,2))
bars = ax2.barh(ladder_cols, [r*100 if r==r else 0 for r in rates], color=['#b3cde0','#74a9cf','#0570b0','#d32f2f'])
for i,v in enumerate(rates):
    ax2.text((v*100 if v==v else 0)+0.2, i, f"{v*100:.1f}%" if v==v else "n/a", va='center', fontweight='bold')
ax2.set_xlabel("Stroke Risk (%)")
ax2.set_xlim(0, max(20, np.nanmax([r*100 for r in rates if r==r])+2))
ax2.invert_yaxis()
ax2.grid(axis='x', ls=':')
st.pyplot(fig2)
st.caption("_See how stroke risk rises as more risk factors are added._")

# 2. Age/BMI Heatmap of Stroke Risk
st.subheader("📊 Age vs. BMI: Stroke Risk Heatmap")
heat_data = df.groupby(['age_group','bmi_group'])['stroke'].mean().unstack()
fig3, ax3 = plt.subplots(figsize=(6,2))
im = ax3.imshow(heat_data, cmap='Reds', aspect='auto')
ax3.set_xticks(np.arange(len(heat_data.columns)), labels=heat_data.columns)
ax3.set_yticks(np.arange(len(heat_data.index)), labels=heat_data.index)
plt.colorbar(im, ax=ax3, label='Stroke Risk')
for i in range(len(heat_data.index)):
    for j in range(len(heat_data.columns)):
        val = heat_data.iloc[i,j]
        ax3.text(j,i,f"{val*100:.1f}%" if val==val else "", ha='center', va='center', color='black', fontsize=8)
st.pyplot(fig3)

# 3. Stacked Bar: Stroke Rate by Smoking Status & Hypertension
st.subheader("🚬 Smoking & Hypertension: Combined Stroke Risk")
smoke_htn = df.groupby(['smoking_status','hypertension'])['stroke'].mean().unstack()
fig4, ax4 = plt.subplots(figsize=(4,2))
smoke_htn.plot(kind='bar', ax=ax4, color=['#e0e0e0','#d32f2f'])
ax4.set_ylabel("Stroke Rate")
ax4.set_ylim(0,0.2)
ax4.legend(['No Hypertension','Hypertension'],title='Hypertension')
st.pyplot(fig4)

# 4. Population Comparison: Who gets stroke in the data?
st.subheader("🌍 Dataset Population by Risk Group")
pop = df.groupby('age_group')['stroke'].agg(['mean','count','sum'])
fig5, ax5 = plt.subplots(figsize=(4,2))
ax5.bar(pop.index, pop['sum'], color="#d32f2f", alpha=0.7, label='Stroke Cases')
ax5.bar(pop.index, pop['count']-pop['sum'], bottom=pop['sum'], color="#b3cde0", alpha=0.6, label='No Stroke')
ax5.set_ylabel("Number of People")
ax5.legend()
st.pyplot(fig5)
st.caption("_Most people do not have strokes, but risk increases with age group._")
