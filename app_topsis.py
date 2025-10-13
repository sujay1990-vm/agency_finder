# app_topsis_from_kpi.py
# Streamlit app: Run TOPSIS on the core business KPIs only.
# KPIs used (if present in kpi.csv):
#   - LossRatio (lower is better)
#   - RetentionRate (higher is better)
#   - EarnedPremium (higher is better)
#   - SubmissionQuality (higher is better)

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# TOPSIS core
# -----------------------------
def run_topsis(df, weights, benefit_flags):
    metrics = [m for m in df.columns if m in weights]
    if not metrics:
        raise ValueError("No metrics to score. Check your selections/weights.")

    X = df[metrics].replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if X.empty:
        raise ValueError("No rows left after dropping missing values for selected metrics.")

    # Normalize weights
    w = np.array([weights[m] for m in metrics], dtype=float)
    w = w / (w.sum() if w.sum() > 0 else 1.0)

    # Vector normalization
    M = X.values.astype(float)
    norms = np.linalg.norm(M, axis=0)
    norms[norms == 0] = 1.0
    R = M / norms

    # Apply weights
    V = R * w

    # Ideal / Negative-ideal
    ideals, nadirs = [], []
    for j, m in enumerate(metrics):
        if benefit_flags.get(m, True):
            ideals.append(V[:, j].max());  nadirs.append(V[:, j].min())
        else:
            ideals.append(V[:, j].min());  nadirs.append(V[:, j].max())
    ideals = np.array(ideals); nadirs = np.array(nadirs)

    # Distances
    d_pos = np.linalg.norm(V - ideals, axis=1)
    d_neg = np.linalg.norm(V - nadirs, axis=1)

    # Closeness Coefficient
    cc = d_neg / (d_pos + d_neg + 1e-12)

    out = X.copy()
    out["CC"] = cc
    out["Rank"] = out["CC"].rank(ascending=False, method="dense").astype(int)
    out = out.sort_values(["Rank", "CC"], ascending=[True, False])
    return out

# -----------------------------
# App config + state
# -----------------------------
st.set_page_config(page_title="TOPSIS — Core KPIs", page_icon="📊", layout="wide")
st.title("📊 Agency TOPSIS — Core KPIs")

# Initialize state
for k, v in {
    "kpi_df": None,
    "active_metrics": [],
    "weights_used": {},
    "benefit_used": {},
    "ranking_df": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------
# Sidebar: upload + mode + preset
# -----------------------------
with st.sidebar:
    st.header("1) Upload KPI file")
    kpi_file = st.file_uploader("kpi.csv (with AgentCode)", type=["csv"])

    st.header("2) Run mode")
    auto_run = st.toggle("Auto-run on change", value=True, help="If off, use the form’s Run TOPSIS button.")

    st.header("3) Executive preset")
    preset = st.selectbox(
        "Choose a focus:",
        [
            "Balanced",
            "Loss Ratio First",
            "Growth (Earned Premium)",
            "Loyalty (Retention)",
            "Sales Efficiency (SubmissionQuality)",
        ],
        help="Sets default weights so you can answer executive questions quickly.",
    )

# -----------------------------
# Load KPI once (cache)
# -----------------------------
def load_kpi(file):
    df = pd.read_csv(file)
    for dcol in ["PeriodStart", "PeriodEnd"]:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    return df

if kpi_file is not None:
    st.session_state.kpi_df = load_kpi(kpi_file)

# Guard: need data
if st.session_state.kpi_df is None:
    st.info("Upload your **kpi.csv** aggregated at the **AgentCode** level (one row per AgentCode for a period).")
    st.stop()

kpi = st.session_state.kpi_df.copy()
if "AgentCode" not in kpi.columns:
    st.error("kpi.csv must have an 'AgentCode' column (one row per AgentCode).")
    st.stop()

st.subheader("KPI preview")
st.dataframe(kpi.head(20), use_container_width=True)

# -----------------------------
# Core metrics only
# -----------------------------
CORE_METRICS = ["LossRatio", "RetentionRate", "EarnedPremium", "SubmissionQuality"]

# which are present & numeric?
numeric_cols = [c for c in kpi.columns if pd.api.types.is_numeric_dtype(kpi[c])]
present_metrics = [m for m in CORE_METRICS if m in numeric_cols]

if not present_metrics:
    st.error("No usable core KPIs found. Need at least one of: LossRatio, RetentionRate, EarnedPremium, SubmissionQuality.")
    st.stop()

# Default directions (benefit/cost)
default_benefit_map = {
    "LossRatio": False,         # lower is better
    "RetentionRate": True,      # higher is better
    "EarnedPremium": True,      # higher is better
    "SubmissionQuality": True,  # higher is better
}

# Preset weights (before availability pruning)
preset_weights = {
    "Balanced":                    {"LossRatio": 30, "RetentionRate": 30, "EarnedPremium": 20, "SubmissionQuality": 20},
    "Loss Ratio First":            {"LossRatio": 60, "RetentionRate": 20, "EarnedPremium": 10, "SubmissionQuality": 10},
    "Growth (Earned Premium)":     {"LossRatio": 15, "RetentionRate": 15, "EarnedPremium": 55, "SubmissionQuality": 15},
    "Loyalty (Retention)":         {"LossRatio": 15, "RetentionRate": 55, "EarnedPremium": 15, "SubmissionQuality": 15},
    "Sales Efficiency (SubmissionQuality)": {"LossRatio": 15, "RetentionRate": 15, "EarnedPremium": 15, "SubmissionQuality": 55},
}

# Build working weight map limited to present metrics
base_w = preset_weights[preset]
weights_default = {m: base_w.get(m, 0) for m in present_metrics}

# -----------------------------
# Controls (simple: sliders only for present metrics)
# -----------------------------
def controls_ui():
    st.subheader("Tune weights (optional)")
    cols = st.columns(4)
    weights_used = {}
    benefit_used = {}
    active_metrics = []

    for i, m in enumerate(present_metrics):
        with cols[i % 4]:
            default_w = int(weights_default.get(m, 0))
            w = st.slider(f"{m}", 0, 100, default_w, step=5, help=("Higher weight = more important"))
            weights_used[m] = w
            benefit_used[m] = default_benefit_map[m]
            if w > 0:
                active_metrics.append(m)

    # show directions (fixed)
    with st.expander("Metric directions (fixed)"):
        st.markdown(
            "- **LossRatio**: lower is better  \n"
            "- **RetentionRate**: higher is better  \n"
            "- **EarnedPremium**: higher is better  \n"
            "- **SubmissionQuality**: higher is better"
        )

    return active_metrics, weights_used, benefit_used

if auto_run:
    active_metrics, weights_used, benefit_used = controls_ui()
else:
    with st.form("controls_form", clear_on_submit=False):
        active_metrics, weights_used, benefit_used = controls_ui()
        submitted = st.form_submit_button("Run TOPSIS")
        if not submitted:
            if st.session_state.ranking_df is not None:
                st.subheader("TOPSIS Ranking (last run)")
                st.dataframe(st.session_state.ranking_df, use_container_width=True)
            st.stop()

# -----------------------------
# Compute ranking
# -----------------------------
if not active_metrics:
    st.error("All selected weights are 0. Give at least one metric a positive weight (or choose another preset).")
    st.stop()

df_metrics = kpi[["AgentCode"] + active_metrics].copy()
for m in active_metrics:
    df_metrics[m] = pd.to_numeric(df_metrics[m], errors="coerce")

df_metrics = df_metrics.set_index("AgentCode").replace([np.inf, -np.inf], np.nan).dropna(how="any")
if df_metrics.empty:
    st.warning("All rows dropped due to missing values across selected metrics. Fill data or reduce metrics.")
    st.stop()

try:
    ranking = run_topsis(
        df=df_metrics,
        weights={m: float(weights_used[m]) for m in active_metrics},
        benefit_flags={m: bool(benefit_used[m]) for m in active_metrics},
    )
except Exception as e:
    st.exception(e)
    st.stop()

# Persist for stability between reruns
st.session_state.active_metrics = active_metrics
st.session_state.weights_used = weights_used
st.session_state.benefit_used = benefit_used
st.session_state.ranking_df = ranking.copy()

# Show normalized weights
st.subheader("Weights actually used (normalized)")
ws = np.array([weights_used[m] for m in active_metrics], dtype=float)
ws = ws / (ws.sum() if ws.sum() > 0 else 1.0)
st.write({m: round(float(w), 4) for m, w in zip(active_metrics, ws)})

# Results table + chart
st.subheader("TOPSIS Ranking (higher CC = better)")
st.dataframe(ranking, use_container_width=True)

st.subheader("Closeness Coefficient by AgentCode")
chart_df = ranking[["CC"]].copy()
chart_df["AgentCode"] = ranking.index
st.bar_chart(chart_df.set_index("AgentCode"))

# Download
st.download_button(
    "Download ranking as CSV",
    data=ranking.reset_index().to_csv(index=False).encode("utf-8"),
    file_name="topsis_ranking.csv",
    mime="text/csv",
)

# Executive explainer
with st.expander("How to answer executive questions with presets"):
    st.markdown("""
**Pick a preset** in the sidebar:

- **Loss Ratio First** → “Show me agencies that keep losses low, even if it costs growth.”
- **Growth (Earned Premium)** → “Who drives the most premium, accepting some risk.”
- **Loyalty (Retention)** → “Who keeps the book renewing.”
- **Sales Efficiency (SubmissionQuality)** → “Who converts pipeline to bound business best.”
- **Balanced** → Blended view across all four KPIs.

You can fine-tune the sliders if needed.  
Weights are normalized automatically over the metrics you’ve enabled.
""")
