import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Waw Stroke Dashboard", layout="wide")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    df = df[df['gender'].isin(['Male', 'Female'])]  # remove 'Other'
    df = df[df['bmi'] < 60]                         # remove outliers
    df = df[df['avg_glucose_level'] < 250]
    return df

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Filters")
gender = st.sidebar.multiselect("Gender", options=df["gender"].unique(), default=list(df["gender"].unique()))
age_group_bins = [0, 18, 40, 60, 82]
age_group_labels = ["Child", "Young Adult", "Adult", "Senior"]
df["age_group"] = pd.cut(df["age"], bins=age_group_bins, labels=age_group_labels)
age_group = st.sidebar.multiselect("Age Group", options=age_group_labels, default=age_group_labels)
smoking = st.sidebar.multiselect("Smoking Status", options=[x for x in df["smoking_status"].unique() if x != "Unknown"], default=[x for x in df["smoking_status"].unique() if x != "Unknown"])

df_filtered = df[
    df["gender"].isin(gender) &
    df["age_group"].isin(age_group) &
    df["smoking_status"].isin(smoking)
]

# --- KPI Cards ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Patients", len(df_filtered))
col2.metric("Stroke Rate (%)", f"{df_filtered['stroke'].mean()*100:.2f}")
col3.metric("Avg Age", f"{df_filtered['age'].mean():.1f}")
col4.metric("Hypertension Rate (%)", f"{df_filtered['hypertension'].mean()*100:.2f}")

# --- Visual 1: Donut Chart for Stroke Distribution ---
stroke_count = df_filtered['stroke'].value_counts().rename({0: "No Stroke", 1: "Stroke"})
fig1 = px.pie(stroke_count, names=stroke_count.index, values=stroke_count.values, hole=0.5,
              title="Stroke Outcome Distribution")
st.plotly_chart(fig1, use_container_width=True)

# --- Visual 2: Sunburst for Age Group + Smoking Status + Stroke ---
fig2 = px.sunburst(df_filtered, path=['age_group', 'smoking_status', 'stroke'],
                   values=None, title="Sunburst: Age Group > Smoking Status > Stroke")
st.plotly_chart(fig2, use_container_width=True)

# --- Visual 3: Violin Plot of Age by Stroke Status ---
fig3 = px.violin(df_filtered, y="age", x="stroke", color="stroke", box=True, points="all",
                 labels={"stroke": "Stroke"}, title="Age Distribution by Stroke Status")
st.plotly_chart(fig3, use_container_width=True)

# --- Visual 4: Funnel Chart for Risk Path ---
st.subheader("Risk Funnel: Population Breakdown")
funnel_stages = ["All", "Hypertensive", "Heart Disease", "Stroke"]
funnel_values = [
    len(df_filtered),
    df_filtered["hypertension"].sum(),
    df_filtered["heart_disease"].sum(),
    df_filtered["stroke"].sum()
]
fig4 = px.funnel(x=funnel_stages, y=funnel_values)
st.plotly_chart(fig4, use_container_width=True)

# --- Visual 5: Correlation Heatmap ---
import seaborn as sns
import matplotlib.pyplot as plt
numeric_df = df_filtered.select_dtypes(include='number')
fig5, ax = plt.subplots()
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", ax=ax)
st.pyplot(fig5)

# --- Visual 6: Table of Stroke Rates by Group ---
st.subheader("Stroke Rate by Age Group and Smoking")
stroke_rate_table = df_filtered.groupby(['age_group', 'smoking_status'])['stroke'].mean().unstack().style.format("{:.2%}")
st.dataframe(stroke_rate_table)
