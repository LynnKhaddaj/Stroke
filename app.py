import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib_venn import venn3
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(layout="wide", page_title="Creative Stroke Risk Dashboard")

# --- LOAD DATA ---
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# --- CLEAN DATA ---
df = df[df['gender'].isin(['Male', 'Female'])]
df = df.dropna(subset=["bmi"])
df["age_group"] = pd.cut(df["age"], bins=[0,18,40,60,82], labels=["Child","Young Adult","Adult","Senior"])

# --- SIDEBAR GLOBAL FILTERS ---
gender_filter = st.sidebar.selectbox("Gender", options=["All"]+list(df["gender"].unique()))
age_group_filter = st.sidebar.selectbox("Age Group", options=["All"]+list(df["age_group"].unique()))

filtered = df.copy()
if gender_filter != "All":
    filtered = filtered[filtered["gender"] == gender_filter]
if age_group_filter != "All":
    filtered = filtered[filtered["age_group"] == age_group_filter]

st.title("🧬 **Creative Stroke Risk Dashboard**")
st.markdown("##### Each visual tells a unique data story about stroke risk factors. *Hover for details!*")

# --- 1. SUNBURST: Gender > Age Group > Stroke ---
fig1 = px.sunburst(
    filtered, path=['gender', 'age_group', 'stroke'],
    values=None,
    color="stroke",
    color_continuous_scale="RdBu",
    title="Stroke Risk Sunburst: Gender → Age → Stroke"
)
fig1.update_layout(margin=dict(t=40, l=0, r=0, b=0), height=400)

# --- 2. BMI: Human icons or bar ---
bmi_bins = [0,18.5,25,30,35,100]
bmi_labels = ['Underweight','Normal','Overweight','Obese I','Obese II+']
filtered["bmi_bin"] = pd.cut(filtered["bmi"], bins=bmi_bins, labels=bmi_labels)
bmi_risk = filtered.groupby("bmi_bin")["stroke"].mean().reset_index()
bmi_risk["stroke_pct"] = 100 * bmi_risk["stroke"]

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=bmi_labels,
    y=bmi_risk["stroke_pct"],
    marker_color=['#96d38c','#78c4a0','#f6cd61','#f79d84','#e94f37'],
    text=[f"{p:.1f}%" for p in bmi_risk["stroke_pct"]],
    textposition="outside"
))
fig2.update_layout(
    title="BMI Group Stroke Risk (%)",
    yaxis_title="Stroke Rate (%)",
    height=400
)

# --- 3. Smoking: Icon-based bar ---
smoke_order = ['never smoked', 'formerly smoked', 'smokes']
smoke_map = {"never smoked":"🚭", "formerly smoked":"🟤", "smokes":"🚬"}
smoke_risk = filtered.groupby("smoking_status")["stroke"].mean().reindex(smoke_order)
fig3 = go.Figure(go.Bar(
    x=[smoke_map.get(s,s) for s in smoke_order],
    y=[100*x if pd.notna(x) else 0 for x in smoke_risk],
    marker_color=['#b6e3e9','#f3b391','#e94f37'],
    text=[f"{100*x:.1f}%" if pd.notna(x) else "N/A" for x in smoke_risk],
    textposition="auto"
))
fig3.update_layout(
    title="Stroke Risk by Smoking Status",
    yaxis_title="Stroke Rate (%)",
    height=400
)

# --- 4. Glucose vs Age by Stroke: Scatter + density ---
fig4 = px.scatter(
    filtered, x="age", y="avg_glucose_level", color="stroke",
    labels={"stroke":"Stroke"}, opacity=0.5,
    color_discrete_map={0: "steelblue", 1: "crimson"},
    title="Glucose vs Age: Stroke Overlay"
)
fig4.update_traces(marker=dict(size=6))
fig4.update_layout(height=400)

# --- 5. Hypertension/Heart Disease Diverging Bar ---
htn_hd = filtered.groupby(['hypertension', 'heart_disease'])["stroke"].mean().reset_index()
htn_hd["label"] = ["Htn+HD","Htn+NoHD","NoHtn+HD","NoHtn+NoHD"]
fig5 = go.Figure()
fig5.add_trace(go.Bar(
    y=htn_hd["label"],
    x=100*htn_hd["stroke"],
    orientation="h",
    marker_color=["#e94f37" if x>0.07 else "#96d38c" for x in htn_hd["stroke"]],
    text=[f"{100*x:.1f}%" for x in htn_hd["stroke"]],
    textposition="auto"
))
fig5.update_layout(
    title="Stroke Rate: Hypertension & Heart Disease Combo",
    xaxis_title="Stroke Rate (%)",
    height=400
)

# --- 6. Venn Diagram: Overlapping Risks ---
venn_data = filtered.copy()
venn_data["HighBMI"] = venn_data["bmi"] > 30
venn_data["HighGlucose"] = venn_data["avg_glucose_level"] > 150
venn_data["Smoker"] = venn_data["smoking_status"] == "smokes"
groupA = venn_data[venn_data["HighBMI"]]
groupB = venn_data[venn_data["HighGlucose"]]
groupC = venn_data[venn_data["Smoker"]]
plt.figure(figsize=(4,4))
venn3([
    set(groupA.index),
    set(groupB.index),
    set(groupC.index)
], set_labels=('High BMI','High Glucose','Smoker'))
plt.title("Overlap of Major Stroke Risk Factors")
buf = BytesIO()
plt.savefig(buf, format="png", bbox_inches="tight", transparent=True)
plt.close()
buf.seek(0)

# ---- LAYOUT: Arrange the charts in a grid ----
col1, col2 = st.columns([2,2])
with col1:
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
with col2:
    st.plotly_chart(fig3, use_container_width=True)
    st.plotly_chart(fig4, use_container_width=True)

# New row
col3, col4 = st.columns([2,2])
with col3:
    st.plotly_chart(fig5, use_container_width=True)
with col4:
    st.image(buf, caption="Venn: Overlap of High BMI, Glucose, Smoking", use_column_width=True)

st.markdown("**Try using the sidebar filters to explore subgroups!**")

