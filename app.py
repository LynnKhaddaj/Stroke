import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="WAW Stroke Dashboard", layout="wide")

# -------- LOAD AND CLEAN DATA --------
@st.cache_data
def load_data():
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    df = df[df['gender'].isin(['Male', 'Female'])]      # Remove 'Other'
    df = df[df['bmi'] < 60]                             # Remove impossible BMI
    df = df[df['avg_glucose_level'] < 250]              # Remove impossible glucose
    df['age_group'] = pd.cut(df['age'], [0, 18, 40, 60, 82],
                            labels=['Child', 'Young Adult', 'Adult', 'Senior'])
    return df

df = load_data()

# -------- SIDEBAR FILTERS --------
st.sidebar.header("Filters")
gender = st.sidebar.multiselect("Gender", options=df["gender"].unique(), default=list(df["gender"].unique()))
age_group_labels = df['age_group'].cat.categories.tolist()
age_group = st.sidebar.multiselect("Age Group", options=age_group_labels, default=age_group_labels)
smoking = st.sidebar.multiselect(
    "Smoking Status",
    options=[x for x in df["smoking_status"].unique() if x != "Unknown"],
    default=[x for x in df["smoking_status"].unique() if x != "Unknown"]
)

df_filtered = df[
    df["gender"].isin(gender) &
    df["age_group"].isin(age_group) &
    df["smoking_status"].isin(smoking)
]

# -------- KPI CARDS --------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Patients", len(df_filtered))
col2.metric("Stroke Rate (%)", f"{df_filtered['stroke'].mean()*100:.2f}")
col3.metric("Avg Age", f"{df_filtered['age'].mean():.1f}")
col4.metric("Hypertension Rate (%)", f"{df_filtered['hypertension'].mean()*100:.2f}")

# -------- 1. GAUGE: STROKE RATE --------
stroke_pct = df_filtered['stroke'].mean() * 100
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=stroke_pct,
    title={'text': "Stroke Rate (%)"},
    gauge={'axis': {'range': [0, 10]},  # Make gauge sensitive for your rare event!
           'bar': {'color': "purple"}}
))
st.plotly_chart(fig_gauge, use_container_width=True)

# -------- 2. POPULATION PYRAMID (AGE & GENDER) --------
age_bins = pd.cut(df_filtered['age'], bins=range(0, 90, 10), right=False)
age_gender = df_filtered.groupby([age_bins, 'gender']).size().unstack(fill_value=0)

fig_pyramid = go.Figure()
fig_pyramid.add_trace(go.Bar(y=age_gender.index.astype(str), x=age_gender['Male'], name='Male', orientation='h'))
fig_pyramid.add_trace(go.Bar(y=age_gender.index.astype(str), x=-age_gender['Female'], name='Female', orientation='h'))
fig_pyramid.update_layout(title='Population Pyramid: Age and Gender', barmode='overlay',
                         xaxis=dict(tickvals=[-300, -200, -100, 0, 100, 200, 300],
                                    ticktext=[300, 200, 100, 0, 100, 200, 300]))
st.plotly_chart(fig_pyramid, use_container_width=True)

# -------- 3. BMI ZONES (COLOR BAND BAR) --------
st.markdown("### BMI Zones")
fig, ax = plt.subplots(figsize=(8, 2))
colors = ['#ADD8E6', '#90EE90', '#FFD700', '#FF6347']  # blue, green, yellow, red
ranges = [10, 18.5, 25, 30, 60]
labels = ['Underweight', 'Healthy', 'Overweight', 'Obese']
for i in range(len(colors)):
    ax.barh(0, ranges[i+1]-ranges[i], left=ranges[i], color=colors[i], edgecolor='black', height=0.5)
mean_bmi = df_filtered['bmi'].mean()
ax.scatter(mean_bmi, 0, color='black', zorder=5, s=100)
ax.text(mean_bmi+0.5, 0.05, f'Mean BMI: {mean_bmi:.1f}', color='black')
ax.set_xlim(10, 50)
ax.set_yticks([])
ax.set_title("BMI Zones with Mean BMI")
for i in range(len(labels)):
    ax.text((ranges[i]+ranges[i+1])/2, 0.2, labels[i], ha='center')
st.pyplot(fig)

# -------- 4. SANKEY: HYPERTENSION → STROKE --------
label = ["All", "Hypertensive", "Stroke"]
source = [0, 1]
target = [1, 2]
value = [
    df_filtered["hypertension"].sum(),
    df_filtered[(df_filtered["hypertension"] == 1) & (df_filtered["stroke"] == 1)].shape[0]
]
fig_sankey = go.Figure(go.Sankey(
    node=dict(label=label),
    link=dict(source=source, target=target, value=value)
))
fig_sankey.update_layout(title="Risk Funnel: Hypertension and Stroke")
st.plotly_chart(fig_sankey, use_container_width=True)

# -------- 5. TREEMAP: AGE GROUP, SMOKING, STROKE --------
fig_tree = px.treemap(df_filtered,
                      path=['age_group', 'smoking_status', 'stroke'],
                      title="Treemap: Age Group, Smoking Status, and Stroke")
st.plotly_chart(fig_tree, use_container_width=True)

# -------- 6. PATIENT PROFILE CARD --------
st.markdown("### Random Patient Profile")
random_patient = df_filtered.sample(1).iloc[0]
st.info(
    f"""
    **Gender:** {random_patient['gender']}  
    **Age:** {random_patient['age']}  
    **BMI:** {random_patient['bmi']:.1f}  
    **Hypertension:** {'Yes' if random_patient['hypertension'] else 'No'}  
    **Smoking Status:** {random_patient['smoking_status']}  
    **Stroke:** {'Yes' if random_patient['stroke'] else 'No'}
    """
)

# -------- 7. CORRELATION HEATMAP (PLOTLY) --------
st.markdown("### Correlation Matrix")
import plotly.figure_factory as ff
numeric_df = df_filtered.select_dtypes(include='number')
corr = numeric_df.corr()
fig_heat = ff.create_annotated_heatmap(
    z=np.array(corr),
    x=list(corr.columns),
    y=list(corr.index),
    annotation_text=corr.round(2).astype(str).values,
    colorscale='Viridis'
)
st.plotly_chart(fig_heat, use_container_width=True)

# -------- 8. STROKE RATE BY AGE GROUP & SMOKING (HEATMAP TABLE) --------
st.markdown("### Stroke Rate by Age Group and Smoking Status")
stroke_rate_table = df_filtered.groupby(['age_group', 'smoking_status'])['stroke'].mean().unstack().style.format("{:.2%}")
st.dataframe(stroke_rate_table)

# --- (Optional) Add more custom visuals below! ---
import streamlit as st
import pandas as pd

# --- Load your filtered DataFrame ---
# (Assume df is your already-filtered main DataFrame from earlier in app.py)

# For this demo, reload cleaned data (or reuse df_filtered if you have it)
@st.cache_data
def load_data():
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    df = df[df['gender'].isin(['Male', 'Female'])]
    df = df[df['bmi'] < 60]
    df = df[df['avg_glucose_level'] < 250]
    df['age_group'] = pd.cut(df['age'], [0, 18, 40, 60, 82], labels=['Child', 'Young Adult', 'Adult', 'Senior'])
    return df
df = load_data()

import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np

# -- Load your cleaned data, use df as in earlier code
@st.cache_data
def load_data():
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    df = df[df['gender'].isin(['Male', 'Female'])]
    df = df[df['bmi'] < 60]
    df = df[df['avg_glucose_level'] < 250]
    return df
df = load_data()

# -- Bin BMI and calculate stroke risk by BMI bin
bmi_bins = np.arange(15, 45.1, 2)
df['bmi_bin'] = pd.cut(df['bmi'], bins=bmi_bins)
bmi_stroke = df.groupby('bmi_bin')['stroke'].mean()
bmi_bin_centers = [interval.left + 1 for interval in bmi_stroke.index]

# -- User controls BMI and gender
st.title("BMI Silhouette: Stroke Risk Visualizer")

col1, col2 = st.columns([2, 3])
with col1:
    gender_select = st.radio("Gender", ['Both', 'Male', 'Female'], horizontal=True)
    bmi_value = st.slider("Select BMI", 15.0, 45.0, 24.0, step=0.5)
with col2:
    # -- Get stroke probability at selected BMI (optionally by gender)
    bin_index = np.digitize(bmi_value, bmi_bins) - 1
    if gender_select == "Both":
        risk = bmi_stroke.iloc[bin_index] if bin_index < len(bmi_stroke) else 0
    else:
        subset = df[df['gender'] == gender_select]
        bmi_stroke_gender = subset.groupby('bmi_bin')['stroke'].mean()
        risk = bmi_stroke_gender.iloc[bin_index] if bin_index < len(bmi_stroke_gender) else 0

    risk_percent = 0 if pd.isna(risk) else risk * 100

    # -- Morph width, but keep head/arms proportional so figure isn't deformed
    width_factor = (bmi_value - 15) / (45 - 15)
    base_width = 90
    morph_width = int(base_width + width_factor * 70)  # Head/body width
    fill_height = int(160 * (risk_percent / 100))  # Fill proportionally to stroke rate

    # Gender color
    color = "#90caf9" if gender_select == "Male" else "#F48FB1" if gender_select == "Female" else "#ba99ff"

    # Stroke fill color: blue <20%, purple 20-50%, red >50%
    fill_color = "#90caf9" if risk_percent < 20 else "#bb55ff" if risk_percent < 50 else "#b71c1c"

    # SVG for body, with risk “water level” fill
    svg = f"""
    <svg width="{morph_width}" height="240" viewBox="0 0 160 240" xmlns="http://www.w3.org/2000/svg">
      <!-- Fill (risk “water” tank) -->
      <rect x="{morph_width//2-27}" y="{220-fill_height}" width="{54}" height="{fill_height}" rx="22"
        fill="{fill_color}" opacity="0.5"/>
      <!-- Head -->
      <ellipse cx="{morph_width//2}" cy="48" rx="{morph_width//5}" ry="28" fill="{color}" />
      <!-- Body -->
      <rect x="{morph_width//2-27}" y="70" width="54" height="110" rx="27" fill="{color}"/>
      <!-- Arms -->
      <rect x="{morph_width//2-56}" y="85" width="27" height="75" rx="16" fill="{color}"/>
      <rect x="{morph_width//2+29}" y="85" width="27" height="75" rx="16" fill="{color}"/>
      <!-- Legs -->
      <rect x="{morph_width//2-21}" y="175" width="18" height="50" rx="8" fill="{color}"/>
      <rect x="{morph_width//2+3}" y="175" width="18" height="50" rx="8" fill="{color}"/>
      <!-- Outline for clarity -->
      <rect x="{morph_width//2-27}" y="70" width="54" height="110" rx="27" fill="none" stroke="#333" stroke-width="2"/>
    </svg>
    """

    st.markdown(f"<center>{svg}</center>", unsafe_allow_html=True)
    st.markdown(f"<center><h2>BMI: {bmi_value:.1f}</h2>"
                f"<h3>Stroke Risk: <span style='color:{fill_color}'>{risk_percent:.2f}%</span></h3></center>",
                unsafe_allow_html=True)
    st.caption("The silhouette width matches BMI. The colored fill inside represents the group stroke risk at that BMI.")

---

## **2. “Smoking” Visual: Cigarette Emoji as Status**

Here’s the creative, immediately understandable visual:

```python
st.title("Smoking Status & Stroke Risk")

def cigarette_svg(lit=False, smoked=False):
    # Base body of cigarette
    body = f'<rect x="20" y="30" width="120" height="20" rx="8" fill="#F5DEB3" stroke="#A0522D" stroke-width="3"/>'
    # Lit tip or not
    tip = ('<circle cx="140" cy="40" r="10" fill="#B22222"/>' if lit
           else '<circle cx="140" cy="40" r="10" fill="#F5DEB3" stroke="#A0522D" stroke-width="3"/>')
    # Ash (burnt) effect if smoked
    ash = ('<rect x="80" y="33" width="60" height="14" fill="#666" opacity="0.6"/>' if smoked else '')
    return f'<svg width="180" height="70" xmlns="http://www.w3.org/2000/svg">{body}{tip}{ash}</svg>'

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<center><b>Never Smoked</b></center>", unsafe_allow_html=True)
    st.markdown(f"<center>{cigarette_svg(lit=False, smoked=False)}</center>", unsafe_allow_html=True)
with col2:
    st.markdown("<center><b>Smokes</b></center>", unsafe_allow_html=True)
    st.markdown(f"<center>{cigarette_svg(lit=True, smoked=True)}</center>", unsafe_allow_html=True)
with col3:
    st.markdown("<center><b>Formerly Smoked</b></center>", unsafe_allow_html=True)
    st.markdown(f"<center>{cigarette_svg(lit=False, smoked=True)}</center>", unsafe_allow_html=True)
