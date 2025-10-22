# app_topsis_from_kpi.py
# Streamlit app: Run TOPSIS on the core business KPIs only.

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

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
    s = w.sum()
    w = w / (s if s > 0 else 1.0)

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
st.title("📊 Agency Finder — Core KPIs")

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
                "Expense Discipline",              # <— add this if you want it visible
                "Growth (Earned Premium)",
                "Loyalty (Retention)",
                "Sales Efficiency (SubmissionQuality)",
            ],
            help="Sets default weights so you can answer executive questions quickly.",
        )


# -----------------------------
# Load KPI once (cache)
# -----------------------------
@st.cache_data
def load_kpi(file):
    df = pd.read_csv(file)
    for dcol in ["PeriodStart", "PeriodEnd"]:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    return df

if kpi_file is not None:
    st.session_state.kpi_df = load_kpi(kpi_file)

if st.session_state.kpi_df is None:
    st.info("Upload your **kpi.csv** aggregated at the **AgentCode** level (one row per AgentCode for a period).")
    st.stop()

kpi = st.session_state.kpi_df.copy()
if "AgentCode" not in kpi.columns:
    st.error("kpi.csv must have an 'AgentCode' column (one row per AgentCode).")
    st.stop()

st.subheader("KPI preview")
st.dataframe(kpi.head(20), width=1200) 

# -----------------------------
# Core metrics only
# -----------------------------

CORE_METRICS = ["RetentionRate", "SubmissionQuality", "LossRatio", "ExpenseRatio", "EarnedPremium"]

numeric_cols = [c for c in kpi.columns if pd.api.types.is_numeric_dtype(kpi[c])]
present_metrics = [m for m in CORE_METRICS if m in numeric_cols]

if not present_metrics:
    st.error("No usable core KPIs found. Need at least one of: LossRatio, ExpenseRatio, RetentionRate, EarnedPremium, SubmissionQuality.")
    st.stop()


# Human-friendly labels & help for sliders
# Human-friendly labels & help for sliders
LABELS = {
    "RetentionRate":  "Retention Rate",
    "SubmissionQuality": "Submission Quality",
    "LossRatio":      "Loss Ratio",
    "ExpenseRatio":   "Expense Ratio",
    "EarnedPremium":  "Earned Premium",
}

HELP = {
    "RetentionRate":     "Higher is better.",
    "SubmissionQuality": "Higher is better.",
    "LossRatio":         "Lower is better.",
    "ExpenseRatio":      "Lower is better (typ. 24–40%).",
    "EarnedPremium":     "Use with care — bigger agencies tend to score higher.",
}



default_benefit_map = {
    "LossRatio": False,        # lower is better
    "ExpenseRatio": False,     # lower is better
    "RetentionRate": True,
    "EarnedPremium": True,
    "SubmissionQuality": True,
}

preset_weights = {
    "Balanced": {
        "LossRatio": 20, "ExpenseRatio": 20, "RetentionRate": 20, "EarnedPremium": 20, "SubmissionQuality": 20
    },
    "Loss Ratio First": {
        "LossRatio": 50, "ExpenseRatio": 15, "RetentionRate": 20, "EarnedPremium": 5, "SubmissionQuality": 10
    },
    "Expense Discipline": {
        "LossRatio": 20, "ExpenseRatio": 50, "RetentionRate": 15, "EarnedPremium": 5, "SubmissionQuality": 10
    },
    "Growth (Earned Premium)": {
        "LossRatio": 15, "ExpenseRatio": 10, "RetentionRate": 15, "EarnedPremium": 50, "SubmissionQuality": 10
    },
    "Loyalty (Retention)": {
        "LossRatio": 10, "ExpenseRatio": 10, "RetentionRate": 55, "EarnedPremium": 10, "SubmissionQuality": 15
    },
    "Sales Efficiency (SubmissionQuality)": {
        "LossRatio": 10, "ExpenseRatio": 10, "RetentionRate": 15, "EarnedPremium": 10, "SubmissionQuality": 55
    },
}


base_w = preset_weights[preset]
weights_default = {m: base_w.get(m, 0) for m in present_metrics}

# -----------------------------
# Controls
# -----------------------------
def controls_ui():
    st.subheader("Tune weights (optional)")
    cols = st.columns(4)
    weights_used, benefit_used, active_metrics = {}, {}, []

    for i, m in enumerate(present_metrics):
        with cols[i % 4]:
            default_w = int(weights_default.get(m, 0))
            label = LABELS.get(m, m)
            help_txt = HELP.get(m, "Higher weight = more important")
            w = st.slider(label, 0, 100, default_w, step=5, help=help_txt)
            weights_used[m] = w
            benefit_used[m] = default_benefit_map.get(m, True)
            if w > 0:
                active_metrics.append(m)

    with st.expander("Metric directions (fixed)"):
        lines = []
        for m in present_metrics:
            name = LABELS.get(m, m)
            dir_txt = "higher is better" if default_benefit_map.get(m, True) else "lower is better"
            lines.append(f"- **{name}**: {dir_txt}")
        st.markdown("\n".join(lines))

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
                st.dataframe(st.session_state.ranking_df, width=1200)
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

st.session_state.active_metrics = active_metrics
st.session_state.weights_used = weights_used
st.session_state.benefit_used = benefit_used
st.session_state.ranking_df = ranking.copy()

# ---- Display controls ----
st.subheader("Display options")
c1, c2, c3 = st.columns([1,1,2])
max_top = max(1, min(50, len(ranking)))          # robust slider bounds
default_top = min(15, max_top)
with c1:
    top_n = st.slider("Show top N", 1, max_top, default_top, step=1)
with c2:
    search = st.text_input("Filter AgentCode (contains)", "")
with c3:
    show_contrib = st.toggle("Show per-metric contributions", value=True)

# Prep data
rank_df = ranking.reset_index().rename(columns={"index": "AgentCode"})
if search.strip():
    rank_df = rank_df[rank_df["AgentCode"].astype(str).str.contains(search.strip(), case=False, na=False)]

rank_df = rank_df.sort_values("CC", ascending=False).head(top_n)

# ---- Closeness Coefficient bar ----
st.subheader("Closeness Coefficient (higher = better)")
bar = (
    alt.Chart(rank_df)
    .mark_bar()
    .encode(
        x=alt.X("CC:Q", title="Closeness Coefficient"),
        y=alt.Y("AgentCode:N", sort="-x", title="AgentCode"),
        tooltip=[alt.Tooltip("AgentCode:N"),
                 alt.Tooltip("Rank:Q"),
                 alt.Tooltip("CC:Q", format=".3f"),
                 *[alt.Tooltip(m + ":Q", format=".3f") for m in rank_df.columns if m in active_metrics]],
        color=alt.Color("Rank:O", legend=None),
    )
    .properties(height=28 * len(rank_df), width=800)
)
labels = (
    alt.Chart(rank_df)
    .mark_text(align="left", dx=4)
    .encode(x=alt.X("CC:Q"), y=alt.Y("AgentCode:N", sort="-x"), text=alt.Text("CC:Q", format=".3f"))
)
st.altair_chart((bar + labels).properties(width=900))


# ---- Per-metric contributions ----
if show_contrib and len(active_metrics) > 1 and len(rank_df) > 0:
    X = rank_df[active_metrics].values.astype(float)
    norms = np.linalg.norm(X, axis=0)
    norms[norms == 0] = 1.0
    R = X / norms
    w_vec = np.array([weights_used[m] for m in active_metrics], dtype=float)
    w_vec = w_vec / (w_vec.sum() if w_vec.sum() > 0 else 1.0)
    V = R * w_vec
    row_sums = V.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    contrib = V / row_sums

    contrib_df = (
        pd.DataFrame(contrib, columns=active_metrics)
        .assign(AgentCode=rank_df["AgentCode"].values)
        .melt(id_vars="AgentCode", var_name="Metric", value_name="Share")
    )
    st.subheader("What’s driving each score? (share of weighted normalized value)")

    # Make a display label for the metric
    _contrib = contrib_df.copy()
    _contrib["MetricLabel"] = _contrib["Metric"].map(LABELS).fillna(_contrib["Metric"])

    # Keep consistent legend order to match your UI order
    metric_order = [LABELS.get(m, m) for m in present_metrics if m in _contrib["Metric"].unique()]

    stacked = (
        alt.Chart(_contrib)
        .mark_bar()
        .encode(
            x=alt.X("Share:Q", axis=alt.Axis(format="%"), title=None),
            y=alt.Y("AgentCode:N",
                    sort=list(rank_df.sort_values("CC", ascending=False)["AgentCode"])),
            color=alt.Color(
                "MetricLabel:N",
                title="Metric",
                scale=alt.Scale(domain=metric_order)  # preserves friendly order
            ),
            tooltip=[
                alt.Tooltip("AgentCode:N", title="Agent"),
                alt.Tooltip("MetricLabel:N", title="Metric"),
                alt.Tooltip("Share:Q", format=".1%", title="Share"),
            ],
        )
        .properties(height=24 * len(rank_df), width=800)
    )
    st.altair_chart(stacked.properties(width=900))



# ---- Ranking table ----
st.subheader("Ranking table")
tbl = rank_df[["AgentCode", "Rank", "CC"] + [m for m in active_metrics if m in rank_df.columns]].copy()
st.dataframe(tbl.style.format({**{m: "{:.3f}" for m in active_metrics}, "CC": "{:.3f}"}), width=1200 )

# ---- Download ----
st.download_button(
    "Download ranking as CSV",
    data=rank_df.to_csv(index=False).encode("utf-8"),
    file_name="topsis_ranking_topN.csv",
    mime="text/csv",
)

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
