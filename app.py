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
