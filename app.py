import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stroke Dashboard", layout="wide")
st.title("Interactive Stroke Risk Dashboard")

# --- LOAD & FIX DATA ---
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df.columns = df.columns.str.strip().str.lower()
df['gender'] = df['gender'].str.strip().str.lower()  # Fix: force lower case!
df = df[df['gender'].isin(['male', 'female'])]

# You can tune/relax these filters:
df = df[(df['bmi'] > 10) & (df['bmi'] < 60)]
df = df[(df['avg_glucose_level'] > 40) & (df['avg_glucose_level'] < 250)]

# --- OVERALL DATASET METRIC ---
stroke_rate = df['stroke'].mean() * 100
stroke_count = df['stroke'].sum()
st.metric("Dataset Stroke Rate", f"{stroke_rate:.2f}%")
st.caption(f"({stroke_count} stroke cases out of {len(df)}) — Note: Medical data is often imbalanced.")

st.markdown("---")

# --- BMI WITH IMAGES ---
st.header("1. BMI & Stroke Risk (Image Silhouette)")
bmi_bins = [0, 18.5, 24.9, 29.9, 34.9, 100]
bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II+']
fig_paths = [f"bmi_fig_{i+1}.png" for i in range(5)]
df['bmi_bin'] = pd.cut(df['bmi'], bins=bmi_bins, labels=False, include_lowest=True)
binned = df.groupby('bmi_bin')['stroke'].agg(['count', 'sum', 'mean']).reindex(range(5), fill_value=0)
bmi_value = st.slider("Select BMI", 15.0, 45.0, 24.0, step=0.1)
bmi_bin = np.digitize([bmi_value], bmi_bins)[0] - 1
bmi_bin = min(max(bmi_bin, 0), 4)
img_path = fig_paths[bmi_bin]
category = bmi_labels[bmi_bin]
risk = binned['mean'].iloc[bmi_bin]
risk_percent = float(risk) * 100
sample_size = binned['count'].iloc[bmi_bin]
stroke_count_bin = binned['sum'].iloc[bmi_bin]

def fill_image_with_water(img, fill_percent):
    img = img.copy()
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    water_height = int(h * (fill_percent / 100))
    draw = ImageDraw.Draw(mask)
    draw.rectangle([0, h - water_height, w, h], fill=120)
    blue = Image.new("RGBA", (w, h), (60, 120, 255, 128))
    img.paste(blue, mask=mask)
    return img

if os.path.exists(img_path):
    img = Image.open(img_path).convert("RGBA")
    filled_img = fill_image_with_water(img, risk_percent)
    st.image(
        filled_img,
        caption=f"{category} (BMI {bmi_value:.1f}), Stroke Risk: {risk_percent:.2f}%, n={sample_size}, strokes={stroke_count_bin}",
        use_column_width=False
    )
else:
    st.warning(f"Image {img_path} not found.")

if sample_size < 10:
    st.warning("Low sample size in this BMI group! Interpret risk cautiously.")

st.markdown("---")

# --- SMOKING STATUS & STROKE RISK WITH ICONS ---
st.header("2. Stroke Risk by Smoking Status (with Icons)")
smoking_icons = {
    "never smoked": "cigarette_never.png",
    "smokes": "cigarette_smokes.png",
    "formerly smoked": "cigarette_former.png"
}
smoking_risk = df.groupby('smoking_status')['stroke'].agg(['mean', 'count', 'sum']).reindex(['never smoked', 'smokes', 'formerly smoked'])
cols = st.columns(3)
for i, status in enumerate(['never smoked', 'smokes', 'formerly smoked']):
    with cols[i]:
        iconfile = smoking_icons[status]
        if os.path.exists(iconfile):
            st.image(iconfile, width=60)
        else:
            st.write("🚬")
        mean_risk = smoking_risk.loc[status, 'mean'] if status in smoking_risk.index else 0
        n = int(smoking_risk.loc[status, 'count']) if status in smoking_risk.index else 0
        n_strokes = int(smoking_risk.loc[status, 'sum']) if status in smoking_risk.index else 0
        st.metric(status.replace('_', ' ').title(), f"{mean_risk*100:.2f}%")
        st.caption(f"{n_strokes} strokes / {n} people")
        if n < 10:
            st.warning("Few cases, risk may not be reliable.")

st.markdown("---")

# --- HYPERTENSION HEART ---
st.header("3. Hypertension & Stroke Risk (Heart Visual)")
hyper_risk = df.groupby('hypertension')['stroke'].agg(['mean', 'count', 'sum'])
htn_cols = st.columns(2)
for htn, label, color in zip([0, 1], ['No Hypertension', 'Hypertension'], ['#81c784', '#e57373']):
    with htn_cols[htn]:
        group = hyper_risk.loc[htn] if htn in hyper_risk.index else pd.Series({'mean': 0, 'count': 0, 'sum': 0})
        risk_pct = float(group['mean']) * 100
        n = int(group['count'])
        n_stroke = int(group['sum'])
        # Heart shape using matplotlib
        fig, ax = plt.subplots(figsize=(1.4,1.4))
        t = np.linspace(0, 2*np.pi, 100)
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
        y_min = y.min()
        fill_cut = y_min + (y.max() - y_min) * (1 - risk_pct/100)
        ax.fill_between(x, y, fill_cut, color=color, alpha=0.8)
        ax.plot(x, y, color="#333")
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"{label}\n{risk_pct:.2f}% stroke risk\n({n_stroke}/{n})", fontsize=10)
        st.pyplot(fig)

st.markdown("---")

# --- AGE RISK CURVE ---
st.header("4. Age & Stroke Risk: Curve")
age_curve = df.groupby('age')['stroke'].agg(['mean', 'count', 'sum'])
fig2, ax2 = plt.subplots(figsize=(6,2.5))
ax2.plot(age_curve.index, age_curve['mean']*100, color="#6a1b9a", linewidth=2)
ax2.fill_between(age_curve.index, age_curve['mean']*100, color="#ce93d8", alpha=0.3)
ax2.set_xlabel("Age")
ax2.set_ylabel("Stroke Risk (%)")
ax2.set_title("Stroke Risk Rises With Age")
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
st.pyplot(fig2)

st.markdown("---")
st.info("Stroke risk is calculated as % of people with stroke in each group. If a group is small, its percentage may be unreliable. Always check sample sizes!")

