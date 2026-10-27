"""Interactive Streamlit dashboard for K-means customer segmentation."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Mall_Customers.csv"
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "customer_segmentation_model.joblib"
CLUSTERED_PATH = ARTIFACTS / "clustered_customers.csv"
PROFILE_PATH = ARTIFACTS / "cluster_profiles.csv"
METRICS_PATH = ARTIFACTS / "metrics.json"

st.set_page_config(page_title="Customer Segmentation Studio", page_icon="🧩", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: radial-gradient(circle at 8% 0%, rgba(37,76,139,.26), transparent 27%), #090f1e; }
[data-testid="stHeader"] { background: rgba(9,15,30,.72); }
.block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 2rem; }
.hero { padding: 2rem 2.2rem; border: 1px solid rgba(124,156,220,.28); border-radius: 24px; background: linear-gradient(120deg, rgba(30,43,79,.96), rgba(13,63,77,.88)); margin-bottom: 1.1rem; }
.eyebrow { color: #67eadb; font-size: .72rem; letter-spacing: .16em; font-weight: 800; text-transform: uppercase; }
.hero h1 { margin: .55rem 0 .35rem; font-size: clamp(2rem,4vw,3.45rem); line-height: 1.1; color: #f4f7ff; }
.hero p { color: #c6d5f2; max-width: 760px; font-size: 1.02rem; }
.hero-note { padding: 1rem 1.15rem; border: 1px solid rgba(135,176,226,.24); border-radius: 16px; background: rgba(8,17,35,.34); }
.section-title { color: #70e5da; text-transform: uppercase; letter-spacing: .13em; font-size: .73rem; font-weight: 800; margin: .7rem 0 .25rem; }
div[data-testid="stMetric"] { background: linear-gradient(145deg, rgba(20,31,54,.9), rgba(16,28,47,.8)); border: 1px solid rgba(124,156,220,.18); padding: 1rem 1.1rem; border-radius: 16px; }
div[data-testid="stMetricLabel"] { color: #a9c3eb; }
div[data-testid="stMetricValue"] { color: #f5f8ff; }
div[data-testid="stForm"] { border: 1px solid rgba(124,156,220,.22); border-radius: 18px; padding: 1.1rem; background: rgba(13,21,39,.64); }
div.stButton > button, div.stFormSubmitButton > button { border-radius: 11px; border: 1px solid rgba(102,228,216,.35); min-height: 2.7rem; }
.footer { text-align:center; padding: 1.8rem 0 .4rem; color:#8098c1; font-size:.76rem; }
@media(max-width: 700px) { .block-container { padding-left: 1rem; padding-right: 1rem; } .hero { padding: 1.35rem; } }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, tuple[pd.DataFrame, dict[str, Any]]]:
    if not all(p.exists() for p in [DATA_PATH, CLUSTERED_PATH, PROFILE_PATH, METRICS_PATH]):
        raise FileNotFoundError("Project artifacts are missing. Run `python train_model.py` from the project folder first.")
    raw = pd.read_csv(DATA_PATH)
    clustered = pd.read_csv(CLUSTERED_PATH)
    profiles = pd.read_csv(PROFILE_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return raw, clustered, (profiles, metrics)

@st.cache_resource
def load_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run `python train_model.py` first.")
    return joblib.load(MODEL_PATH)

try:
    raw, clustered, payload = load_data()
    profiles, metrics = payload
    bundle = load_model()
except Exception as exc:
    st.error(str(exc))
    st.code("pip install -r requirements.txt\npython train_model.py\nstreamlit run app.py", language="bash")
    st.stop()

st.markdown(f"""
<div class="hero"><div class="eyebrow">Unsupervised learning · Retail analytics</div>
<h1>Customer Segmentation Studio</h1>
<p>Discover customer groups by annual income and spending score. Explore the clusters, compare their profiles, and classify a sample customer with K-means.</p>
<div class="hero-note"><b>Model ready</b><br><span style="color:#b9cee9">{len(raw):,} customer records · K-means · {metrics['selected_k']} segments · Standardized features</span></div></div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Customers", f"{len(raw):,}", "Rows in uploaded dataset")
m2.metric("Segments", str(metrics["selected_k"]), "Selected by silhouette score")
m3.metric("Silhouette score", f"{metrics['silhouette_score']:.3f}", "Higher is better, within context")
m4.metric("Input features", "2", "Annual income + spending score")

st.markdown('<div class="section-title">Explore the results</div>', unsafe_allow_html=True)
tab_overview, tab_segments, tab_predict, tab_method = st.tabs(["Overview", "Segment profiles", "Classify a customer", "Methodology"])

with tab_overview:
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        fig = px.scatter(clustered, x="Annual Income (k$)", y="Spending Score (1-100)", color="Segment",
                         hover_data={"CustomerID": True, "Age": True, "Gender": True, "Cluster": True},
                         title="Customer map · Income vs spending", template="plotly_dark",
                         color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_traces(marker=dict(size=10, opacity=.85, line=dict(width=.5, color="#dce8ff")))
        fig.update_layout(height=530, margin=dict(l=8,r=8,t=55,b=8), legend_title_text="Customer segment",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(12,20,37,.6)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        counts = clustered["Segment"].value_counts().rename_axis("Segment").reset_index(name="Customers")
        fig2 = px.bar(counts, x="Customers", y="Segment", orientation="h", text="Customers", title="Customers per segment",
                      template="plotly_dark", color="Segment", color_discrete_sequence=px.colors.qualitative.Bold)
        fig2.update_layout(height=530, showlegend=False, margin=dict(l=8,r=8,t=55,b=8),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(12,20,37,.6)")
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)
    st.caption("Gender and age are shown for hover-level exploration only; they are not inputs to the K-means model.")

with tab_segments:
    st.subheader("Segment summary")
    st.dataframe(profiles.rename(columns={"Customer_Count":"Customers", "Average_Age":"Avg. age", "Female_Percent":"Female (%)", "Average_Income_kUSD":"Avg. income (k$)", "Average_Spending_Score":"Avg. spending score"}), use_container_width=True, hide_index=True)
    fig = px.scatter(profiles, x="Average_Income_kUSD", y="Average_Spending_Score", size="Customer_Count", color="Segment",
                     text="Segment", title="Cluster centers and segment sizes", template="plotly_dark",
                     labels={"Average_Income_kUSD":"Average annual income (k$)", "Average_Spending_Score":"Average spending score"},
                     color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_traces(textposition="top center")
    fig.update_layout(height=480, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(12,20,37,.6)")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("View customers with assigned segments"):
        st.dataframe(clustered.sort_values(["Segment", "CustomerID"]), use_container_width=True, hide_index=True)
    st.download_button("Download segmented customer CSV", data=clustered.to_csv(index=False).encode("utf-8"), file_name="clustered_customers.csv", mime="text/csv")

with tab_predict:
    st.subheader("Classify a sample customer")
    st.write("Enter annual income and spending score to find the closest learned customer segment. Income is entered in thousands of USD (k$).")
    presets = {
        "Lower income · low spending": (25.0, 20),
        "Lower income · high spending": (25.0, 80),
        "Mid-range · moderate spending": (55.0, 50),
        "High income · low spending": (90.0, 20),
        "High income · high spending": (90.0, 80),
    }
    def apply_customer_preset() -> None:
        selected = st.session_state.get("customer_profile", "Custom")
        if selected in presets:
            st.session_state["income_input"] = float(presets[selected][0])
            st.session_state["spending_input"] = int(presets[selected][1])

    st.session_state.setdefault("income_input", 60.0)
    st.session_state.setdefault("spending_input", 50)
    st.selectbox("Quick profile", ["Custom", *presets.keys()], key="customer_profile", on_change=apply_customer_preset)
    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        c1.number_input("Annual income (thousand USD)", min_value=0.1, max_value=250.0, step=1.0, key="income_input")
        c2.slider("Spending score", min_value=1, max_value=100, key="spending_input")
        submitted = st.form_submit_button("Classify customer", type="primary", use_container_width=True)
    income = float(st.session_state["income_input"])
    score = int(st.session_state["spending_input"])
    if submitted:
        feature_row = pd.DataFrame([[income, score]], columns=bundle["feature_columns"])
        label = int(bundle["model"].predict(bundle["scaler"].transform(feature_row))[0])
        segment = bundle["cluster_to_segment"][label]
        center = bundle["scaler"].inverse_transform(bundle["model"].cluster_centers_)[label]
        st.markdown("---")
        st.success(f"Assigned segment: **{segment}**")
        r1, r2, r3 = st.columns(3)
        r1.metric("Cluster ID", str(label))
        r2.metric("Income input", f"${income:.1f}k")
        r3.metric("Spending score", f"{score}/100")
        st.info(f"This cluster's center is approximately ${center[0]:.1f}k income and a spending score of {center[1]:.1f}. Segment names summarize cluster centers; they are not guarantees about an individual customer's behavior.")
    st.caption("This is a segment assignment, not a prediction of future purchases or customer value.")

with tab_method:
    st.subheader("How the model works")
    st.markdown("""
    1. **Load and validate** the supplied `Mall_Customers.csv` data.
    2. **Select features:** annual income and spending score. `CustomerID`, gender, and age are not used to form the clusters.
    3. **Standardize** both features so their scales do not dominate Euclidean distance.
    4. **Compare k = 2–10** with inertia (elbow method) and silhouette score.
    5. **Fit K-means** with the selected number of clusters and a fixed random seed.
    6. **Profile the segments** by their centers and summarize customer counts and demographics after clustering.
    """)
    st.markdown("**Cluster selection**")
    score_df = pd.DataFrame(metrics["k_search"])
    left, right = st.columns(2)
    with left:
        fig = px.line(score_df, x="k", y="inertia", markers=True, title="Elbow / inertia", template="plotly_dark")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(12,20,37,.6)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.line(score_df, x="k", y="silhouette_score", markers=True, title="Silhouette score", template="plotly_dark")
        fig.add_vline(x=metrics["selected_k"], line_dash="dash", line_color="#67eadb")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(12,20,37,.6)")
        st.plotly_chart(fig, use_container_width=True)
    st.warning("Dataset limitation: this CSV contains a spending score, not detailed transaction history. The project therefore demonstrates segmentation using income and score as proxies, rather than clustering actual receipts or item-level purchases.")

st.markdown(
    '<div class="footer">'
    'CUSTOMER SEGMENTATION STUDIO · K-MEANS CLUSTERING<br>'
    '<div style="text-align:center;color:#d1d5db;font-size:.78rem;'
    'padding:1.25rem .25rem .1rem;">'
    'Developed by '
    '<a href="https://www.linkedin.com/in/saratbanik" '
    'target="_blank" rel="noopener noreferrer" '
    'style="color:inherit;font-weight:700;text-decoration:none;">'
    'SARAT BANIK</a>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)