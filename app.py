import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Creative Stroke Risk Dashboard", layout="wide")
st.title("🚀 Stroke Risk: A Data Story with Visuals")

# --- Load and clean data ---
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df.columns = df.columns.str.strip().str.lower()
df = df[df['gender'].isin(['male', 'female'])]
df = df[(df['bmi'] > 10) & (df['bmi'] < 60)]
df = df[(df['avg_glucose_level'] > 40) & (df['avg_glucose_level'] < 250)]
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
st.write("Loaded:", df.shape)
df.columns = df.columns.str.strip().str.lower()
df = df[df['gender'].isin(['male', 'female'])]
st.write("After gender filter:", df.shape)
df = df[(df['bmi'] > 10) & (df['bmi'] < 60)]
st.write("After BMI filter:", df.shape)
df = df[(df['avg_glucose_level'] > 40) & (df['avg_glucose_level'] < 250)]
st.write("After glucose filter:", df.shape)
# --- Dataset stats ---
stroke_rate = df['stroke'].mean() * 100
stroke_count = df['stroke'].sum()
st.metric("Dataset Stroke Rate", f"{stroke_rate:.2f}%")
st.caption(f"({stroke_count} stroke cases out of {len(df)}) — Class imbalance is expected in real health data.")

st.markdown("---")

# --- BMI Visual: Custom Body Shape/Water Fill ---
st.header("1. BMI & Stroke Risk: Visualized Silhouettes")
# Quantile-based bins so every bin has people
df['bmi_bin'] = pd.qcut(df['bmi'], 3, labels=['Low', 'Medium', 'High'])
binned = df.groupby('bmi_bin')['stroke'].agg(['count', 'sum', 'mean'])
bmi_value = st.slider("Choose a BMI for Silhouette", float(df['bmi'].min()), float(df['bmi'].max()), float(df['bmi'].median()))
bmi_category = pd.qcut([bmi_value], 3, labels=['Low', 'Medium', 'High'])[0]
cat_idx = ['Low', 'Medium', 'High'].index(bmi_category)
risk = binned['mean'].iloc[cat_idx]
risk_percent = float(risk) * 100
sample_size = binned['count'].iloc[cat_idx]
stroke_count_bin = binned['sum'].iloc[cat_idx]

# Draw custom silhouette with fill (matplotlib, no PNG needed)
fig, ax = plt.subplots(figsize=(2.2, 3))
width = [0.8, 1.1, 1.45][cat_idx]
height = 2.8
x = np.array([-width, width, width, -width, -width]) / 2
y = np.array([0, 0, height, height, 0])
ax.fill(x, y, color="#d2c7f5", edgecolor="black")
fill_height = height * (risk_percent/100)
ax.fill_between(x, 0, fill_height, color="#41b6e6", alpha=0.5)
ax.set_xlim(-1, 1)
ax.set_ylim(0, 3)
ax.axis("off")
ax.set_title(f"{bmi_category} BMI", fontsize=11)
st.pyplot(fig)
st.metric("Stroke Rate", f"{risk_percent:.2f}%")
st.caption(f"{stroke_count_bin} stroke cases / {sample_size} in this BMI group.")

st.markdown("---")

# --- Smoking Visual: Emoji Bars + Metrics ---
st.header("2. Smoking Status & Stroke Risk: Emoji Visuals")
smoking_emojis = {"never smoked": "🚭", "smokes": "🚬", "formerly smoked": "🚬❌"}
smoking_risk = df.groupby('smoking_status')['stroke'].agg(['mean', 'count', 'sum']).reindex(['never smoked', 'smokes', 'formerly smoked'])
sm_cols = st.columns(3)
for i, status in enumerate(['never smoked', 'smokes', 'formerly smoked']):
    with sm_cols[i]:
        emoji = smoking_emojis.get(status, "❓")
        mean_risk = smoking_risk.loc[status, 'mean'] if status in smoking_risk.index else 0
        n = int(smoking_risk.loc[status, 'count']) if status in smoking_risk.index else 0
        n_strokes = int(smoking_risk.loc[status, 'sum']) if status in smoking_risk.index else 0
        st.markdown(f"<h1 style='text-align: center;'>{emoji}</h1>", unsafe_allow_html=True)
        st.metric(status.replace('_', ' ').title(), f"{mean_risk*100:.2f}%")
        st.caption(f"{n_strokes} strokes / {n} people")
        bar = "█" * int(mean_risk * 50)
        st.markdown(f"<span style='color:#0d47a1;font-size:1.5em'>{bar}</span>", unsafe_allow_html=True)
        if n < 10:
            st.warning("Few samples — risk unreliable.")

st.markdown("---")

# --- Hypertension: Heart Visual (ASCII) ---
st.header("3. Hypertension & Stroke Risk: Heart Icon Visual")
hyper_risk = df.groupby('hypertension')['stroke'].agg(['mean', 'count', 'sum'])
htn_cols = st.columns(2)
for htn, label, heart in zip([0, 1], ['No Hypertension', 'Hypertension'], ["💚", "❤️"]):
    with htn_cols[htn]:
        group = hyper_risk.loc[htn] if htn in hyper_risk.index else pd.Series({'mean': 0, 'count': 0, 'sum': 0})
        risk_pct = float(group['mean']) * 100
        n = int(group['count'])
        n_stroke = int(group['sum'])
        st.markdown(f"<h1 style='text-align:center'>{heart}</h1>", unsafe_allow_html=True)
        st.metric(label, f"{risk_pct:.2f}%")
        st.caption(f"{n_stroke} strokes / {n} people")
        bar = "█" * int(risk_pct * 2)
        st.markdown(f"<span style='color:#e57373;font-size:1.5em'>{bar}</span>", unsafe_allow_html=True)
        if n < 10:
            st.warning("Few samples — risk unreliable.")

st.markdown("---")

# --- Age Group: Risk Curve ---
st.header("4. Age & Stroke Risk: Curve")
age_curve = df.groupby('age')['stroke'].agg(['mean', 'count', 'sum'])
fig2, ax2 = plt.subplots(figsize=(6,2.5))
ax2.plot(age_curve.index, age_curve['mean']*100, color="#6a1b9a", linewidth=2)
ax2.fill_between(age_curve.index, age_curve['mean']*100, color="#ce93d8", alpha=0.3)
ax2.set_xlabel("Age")
ax2.set_ylabel("Stroke Risk (%)")
ax2.set_title("Stroke Risk Rises Sharply With Age")
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
st.pyplot(fig2)

st.markdown("---")

# --- Gender Visual: Emoji + Metric ---
st.header("5. Gender & Stroke Risk")
gen_risk = df.groupby('gender')['stroke'].agg(['mean', 'count', 'sum'])
gender_cols = st.columns(2)
for i, (g, emoji) in enumerate(zip(['male', 'female'], ["♂️", "♀️"])):
    with gender_cols[i]:
        group = gen_risk.loc[g] if g in gen_risk.index else pd.Series({'mean': 0, 'count': 0, 'sum': 0})
        risk_pct = float(group['mean']) * 100
        n = int(group['count'])
        n_stroke = int(group['sum'])
        st.markdown(f"<h1 style='text-align:center'>{emoji}</h1>", unsafe_allow_html=True)
        st.metric(g.title(), f"{risk_pct:.2f}%")
        st.caption(f"{n_stroke} strokes / {n} people")

st.markdown("---")
st.markdown("<center><i>Dashboard by ChatGPT | All visuals show group risk, not personal prediction. Data is illustrative.</i></center>", unsafe_allow_html=True)
