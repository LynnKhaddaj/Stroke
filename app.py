import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import os

st.set_page_config(page_title="Stroke Dashboard", layout="wide")
st.title("Interactive Stroke Risk Dashboard")

# --- LOAD DATA ---
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df.columns = df.columns.str.strip().str.lower()
df = df[df['gender'].isin(['male', 'female'])]
df = df[(df['bmi'] > 10) & (df['bmi'] < 60)]  # relaxed filtering
df = df[(df['avg_glucose_level'] > 40) & (df['avg_glucose_level'] < 250)]

bmi_bins = [0, 25, 30, 100]
bmi_labels = ['Normal or Underweight', 'Overweight', 'Obese']
fig_paths = ["bmi_fig_1.png", "bmi_fig_3.png", "bmi_fig_5.png"]

df['bmi_bin'] = pd.cut(df['bmi'], bins=bmi_bins, labels=False, include_lowest=True)

binned = df.groupby('bmi_bin')['stroke'].agg(['count', 'sum', 'mean']).reindex(range(3), fill_value=0)
bmi_value = st.slider("Select BMI", 15.0, 45.0, 24.0, step=0.1)
bmi_bin = np.digitize([bmi_value], bmi_bins)[0] - 1
bmi_bin = min(max(bmi_bin, 0), 2)
img_path = fig_paths[bmi_bin]
category = bmi_labels[bmi_bin]
risk = binned['mean'].iloc[bmi_bin]
risk_percent = float(risk) * 100
sample_size = binned['count'].iloc[bmi_bin]
stroke_count = binned['sum'].iloc[bmi_bin]

st.write(f"Sample size in this bin: {sample_size}, Stroke cases: {stroke_count}")

# (rest of image/water code unchanged)


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

try:
    img = Image.open(img_path).convert("RGBA")
    filled_img = fill_image_with_water(img, risk_percent)
    st.image(
        filled_img,
        caption=f"{category} (BMI {bmi_value:.1f}), Stroke Risk: {risk_percent:.2f}%, n={sample_size}",
        use_column_width=False
    )
except Exception as e:
    st.error(f"Image {img_path} not found. Error: {e}")

if sample_size < 10:
    st.warning("Low sample size in this BMI group! Interpret risk cautiously.")
import streamlit as st
import os

smoking_icons = {
    "never smoked": "cigarette_never.png",
    "smokes": "cigarette_smokes.png",
    "formerly smoked": "cigarette_former.png"
}

# Calculate stroke risk and counts per group
smoking_risk = df.groupby('smoking_status')['stroke'].agg(['mean', 'count', 'sum']).reindex(['never smoked', 'smokes', 'formerly smoked'])

st.header("Stroke Risk by Smoking Status")
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
        st.write(f"Strokes: {n_strokes} / {n}")
        if n < 10:
            st.warning("Few cases, risk may not be reliable.")

# --- HYPERTENSION & STROKE RISK ---
st.header("3. Hypertension and Stroke Risk (Heart Visual)")
hyper_risk = df.groupby('hypertension')['stroke'].mean()
col1, col2 = st.columns(2)
for htn, label, color in zip([0, 1], ['No Hypertension', 'Hypertension'], ['#81c784', '#e57373']):
    with [col1, col2][htn]:
        risk_pct = float(hyper_risk.get(htn, 0)) * 100
        # Draw heart shape filled up to risk_pct
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
        ax.set_title(f"{label}\n{risk_pct:.2f}% stroke risk", fontsize=10)
        st.pyplot(fig)

# --- AGE GROUP CURVE ---
st.header("4. Stroke Risk by Age")
age_curve = df.groupby('age')['stroke'].mean()
fig2, ax2 = plt.subplots(figsize=(5,2))
ax2.plot(age_curve.index, age_curve.values*100, color="#673ab7", linewidth=3)
ax2.fill_between(age_curve.index, age_curve.values*100, color="#b39ddb", alpha=0.3)
ax2.set_xlabel("Age")
ax2.set_ylabel("Stroke Risk %")
ax2.set_title("Stroke Risk by Age")
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
st.pyplot(fig2)

# --- DATA IMBALANCE REMINDER ---
st.info("Stroke risk is calculated as % of people with stroke in each group. If a group is small, its percentage may be unreliable. Always check sample sizes!")

