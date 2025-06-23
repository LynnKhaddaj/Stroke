import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

st.set_page_config(page_title="BMI & Stroke Visual", layout="centered")
st.title("BMI Silhouette: Stroke Risk Visualizer")

# ---- LOAD DATA ----
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df.columns = df.columns.str.strip().str.lower()
df = df[df['gender'].isin(['male', 'female'])]
df = df[(df['bmi'] < 60) & (df['bmi'] > 10)]
df = df[(df['avg_glucose_level'] < 250) & (df['avg_glucose_level'] > 40)]

# ---- BMI BINS & IMAGE PATHS ----
bmi_bins = [0, 18.5, 24.9, 29.9, 34.9, 100]
bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II+']
fig_paths = [f"bmi_fig_{i+1}.png" for i in range(5)]

df['bmi_bin'] = pd.cut(df['bmi'], bins=bmi_bins, labels=False, include_lowest=True)

# --- Always create 5 stroke risk values, 0 if missing ---
bmi_stroke = df.groupby('bmi_bin')['stroke'].mean().reindex(range(5), fill_value=0)

# ---- BMI SLIDER ----
bmi_value = st.slider("Select BMI", 15.0, 45.0, 24.0, step=0.1)
bmi_bin = np.digitize([bmi_value], bmi_bins)[0] - 1
bmi_bin = min(max(bmi_bin, 0), 4)

img_path = fig_paths[bmi_bin]
try:
    img = Image.open(img_path).convert("RGBA")
except Exception as e:
    st.error(f"Image {img_path} not found. Error: {e}")
    st.stop()

risk = bmi_stroke.iloc[bmi_bin]
risk_percent = float(risk) * 100

# ---- "WATER LEVEL" FILL FUNCTION ----
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

filled_img = fill_image_with_water(img, risk_percent)
category = bmi_labels[bmi_bin]

st.image(
    filled_img,
    caption=f"{category} (BMI {bmi_value:.1f}) — Stroke Risk: {risk_percent:.2f}%",
    use_column_width=False
)
st.caption("Silhouette matches BMI category. Blue fill = stroke probability for this BMI group.")

# ---- Optionally, show all risks in a table ----
with st.expander("Show stroke risk by BMI category"):
    result = pd.DataFrame({
        "BMI Category": bmi_labels,
        "Stroke Risk (%)": [f"{x*100:.2f}" for x in bmi_stroke]
    })
    st.table(result)

# ---- 3. SMOKING STATUS SECTION ----
with st.container():
    st.header("2. Stroke Risk by Smoking Status and Age Group")
    smoking_icons = {
        "never smoked": "cigarette_never.png",
        "smokes": "cigarette_smokes.png",
        "formerly smoked": "cigarette_former.png"
    }
    age_bins = [0, 18, 40, 60, 82]
    age_labels = ['Child', 'Young Adult', 'Adult', 'Senior']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, include_lowest=True)
    stroke_by_group = df.groupby(['smoking_status', 'age_group'])['stroke'].mean().unstack().fillna(0)
    st.write("Visuals: Full cigarette (never), lit/short (smokes), burnt-out (formerly smoked).")
    sm_cols = st.columns(3)
    for idx, status in enumerate(['never smoked', 'smokes', 'formerly smoked']):
        with sm_cols[idx]:
            st.image(smoking_icons[status], width=80)
            st.markdown(f"<center><b>{status.title()}</b></center>", unsafe_allow_html=True)
            for age_group in stroke_by_group.columns:
                percent = stroke_by_group.loc[status, age_group]*100
                bar = "█" * int(percent/3)  # Fun text bar visual
                st.write(f"{age_group}: {bar} {percent:.2f}%")

# ---- 4. HYPERTENSION "HEART BATTERY" ----
with st.container():
    st.header("3. Hypertension: 'Heart Battery' and Stroke Risk")
    # Create a heart "battery" image in code (or upload heart images per risk level)
    # Calculate stroke risk by hypertension group
    hyper_risk = df.groupby('hypertension')['stroke'].mean()
    colh1, colh2 = st.columns(2)
    with colh1:
        # Draw heart icon with fill according to risk %
        for htn in [0, 1]:
            risk_pct = hyper_risk[htn] * 100
            fig, ax = plt.subplots(figsize=(1.4,1.4))
            t = np.linspace(0, 2*np.pi, 100)
            x = 16 * np.sin(t) ** 3
            y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
            # Fill heart by stroke risk
            y_min = y.min()
            fill_cut = y_min + (y.max() - y_min) * (1 - risk_pct/100)
            ax.fill_between(x, y, fill_cut, color="#f06292" if htn else "#81c784", alpha=0.8)
            ax.plot(x, y, color="#333")
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(f"{'No' if htn==0 else 'Yes'} Hypertension\n{risk_pct:.2f}% stroke risk", fontsize=9)
            st.pyplot(fig)
    with colh2:
        st.markdown("**Green heart:** No hypertension (lower risk)<br>**Pink heart:** Has hypertension (higher risk)", unsafe_allow_html=True)
        st.caption("Heart fills up to the group stroke risk.")

# ---- 5. AGE GROUP: RISK CURVE ----
with st.container():
    st.header("4. Age and Stroke Risk Curve")
    # Smooth risk curve
    age_curve = df.groupby('age')['stroke'].mean()
    fig, ax = plt.subplots(figsize=(5,2))
    ax.plot(age_curve.index, age_curve.values*100, color="#673ab7", linewidth=3)
    ax.fill_between(age_curve.index, age_curve.values*100, color="#b39ddb", alpha=0.3)
    ax.set_xlabel("Age")
    ax.set_ylabel("Stroke Risk %")
    ax.set_title("Stroke Risk Increases Sharply With Age")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    st.pyplot(fig)

# ---- 6. WORK TYPE: ICON + COLOR BAR ----
with st.container():
    st.header("5. Work Type and Stroke Risk")
    work_icons = {
        'private': "work_private.png",
        'self-employed': "work_selfemployed.png",
        'govt_job': "work_govt.png",
        'children': "work_children.png",
        'never_worked': "work_nowork.png"
    }
    work_stroke = df.groupby('work_type')['stroke'].mean().sort_values(ascending=False)
    st.write("Each icon represents a work type. Bars are colored by stroke risk.")
    wt_cols = st.columns(len(work_stroke))
    for i, (wt, risk) in enumerate(work_stroke.items()):
        with wt_cols[i]:
            iconfile = work_icons.get(wt, "")
            if iconfile:
                st.image(iconfile, width=60)
            col = "#d32f2f" if risk > 0.05 else "#fbc02d" if risk > 0.03 else "#388e3c"
            st.markdown(f'<div style="width: 40px; height: {int(risk*400)}px; background:{col}; border-radius: 7px; margin-bottom:7px;"></div>', unsafe_allow_html=True)
            st.caption(f"{wt.title()}<br>{risk*100:.2f}%", unsafe_allow_html=True)

# ---- 7. GENDER SPLIT ----
with st.container():
    st.header("6. Gender and Stroke Risk")
    gen_risk = df.groupby('gender')['stroke'].mean()
    gc = st.columns(2)
    with gc[0]:
        st.image("gender_male.png", width=70)
        st.metric("Male Stroke %", f"{gen_risk['male']*100:.2f}")
    with gc[1]:
        st.image("gender_female.png", width=70)
        st.metric("Female Stroke %", f"{gen_risk['female']*100:.2f}")

st.markdown("---")
st.markdown("<center><i>Dashboard by [Your Name] | All visuals use group stroke probability, not individual prediction.</i></center>", unsafe_allow_html=True)
