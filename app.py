"""Step 3 - compliance dashboard.  Run with:  streamlit run app.py"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

OUT = Path(__file__).parent / "outputs"
LIVE_DATE = "1 October 2026"

st.set_page_config(page_title="Coal Mine Compliance Monitor", layout="wide")


@st.cache_data
def load():
    ranked = pd.read_csv(OUT / "ranked_sites.csv")
    items = pd.read_csv(OUT / "open_items.csv", parse_dates=["due_date"])
    metrics = json.loads((OUT / "metrics.json").read_text())
    importance = pd.read_csv(OUT / "feature_importance.csv")
    return ranked, items, metrics, importance


if not (OUT / "ranked_sites.csv").exists():
    st.error("No outputs found. Run `python src/generate_data.py` and "
             "`python src/train_model.py` first.")
    st.stop()

ranked, items, metrics, importance = load()

st.title("Coal Mine Compliance Monitor")
st.caption(f"Position as of {LIVE_DATE}. Demonstration on SYNTHETIC data: "
           "sites and operators are fictional.")

# ---------------- filters ----------------
with st.sidebar:
    st.header("Filters")
    states = st.multiselect("State", sorted(ranked.state.unique()))
    operators = st.multiselect("Operator", sorted(ranked.operator.unique()))
    types = st.multiselect("Mine type", sorted(ranked.mine_type.unique()))
    top_k = st.slider("Inspections available this quarter (K)", 5, 50, 20, step=5)

view = ranked.copy()
if states:
    view = view[view.state.isin(states)]
if operators:
    view = view[view.operator.isin(operators)]
if types:
    view = view[view.mine_type.isin(types)]

overdue = items[items.status == "Overdue"]
view_overdue = overdue[overdue.site_id.isin(view.site_id)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sites monitored", len(view))
c2.metric("High-risk sites", int((view.risk_band == "High").sum()))
c3.metric("Overdue obligations", len(view_overdue))
c4.metric("Overdue, high severity", int((view_overdue.severity >= 4).sum()))

tab1, tab2, tab3, tab4 = st.tabs(
    ["Inspection priority list", "Site detail", "Upcoming deadlines", "Model performance"])

# ---------------- tab 1: ranked list ----------------
with tab1:
    st.subheader(f"Top {top_k} sites to inspect")
    st.write("Sites are ranked by predicted risk that an inspection would find a lapse. "
             "The driving obligation is the specific item most likely to be the problem.")
    table = view.sort_values("risk_score", ascending=False).head(top_k)[[
        "rank", "site_name", "state", "operator", "mine_type", "risk_score", "risk_band",
        "n_overdue", "driving_obligation", "driving_days_overdue", "driving_basis",
        "baseline_rank"]]
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "rank": "Rank",
            "site_name": "Site",
            "state": "State",
            "operator": "Operator",
            "mine_type": "Type",
            "risk_score": st.column_config.ProgressColumn(
                "Risk score", min_value=0, max_value=1, format="%.2f"),
            "risk_band": "Band",
            "n_overdue": "Overdue items",
            "driving_obligation": "Driving obligation",
            "driving_days_overdue": "Days overdue",
            "driving_basis": "Why flagged",
            "baseline_rank": st.column_config.NumberColumn(
                "Baseline rank", help="Rank if sites were ordered only by overdue count"),
        })
    st.download_button("Download full ranked list (CSV)",
                       view.sort_values("rank").to_csv(index=False),
                       "ranked_sites.csv", "text/csv")

# ---------------- tab 2: site detail ----------------
with tab2:
    options = view.sort_values("rank")
    if options.empty:
        st.info("No sites match the filters.")
    else:
        choice = st.selectbox(
            "Site", options.site_id,
            format_func=lambda s: f"{options.set_index('site_id').site_name[s]}  "
                                  f"(rank {options.set_index('site_id')['rank'][s]})")
        row = options.set_index("site_id").loc[choice]
        a, b, c, d = st.columns(4)
        a.metric("Risk score", f"{row.risk_score:.2f}", row.risk_band, delta_color="off")
        b.metric("Rank", int(row["rank"]))
        c.metric("Overdue obligations", int(row.n_overdue))
        d.metric("Past inspections with a lapse", f"{row.past_lapse_rate:.0%}")
        st.write(f"**{row.site_name}** | {row.mine_type} | {row.state} | {row.operator} | "
                 f"{row.capacity_mtpa} MTPA | workforce {int(row.workforce)}")
        st.warning(f"Driving obligation: **{row.driving_obligation}** "
                   f"({row.driving_basis.lower()}"
                   + (f", {int(row.driving_days_overdue)} days" if row.driving_days_overdue else "")
                   + ")")
        site_items = items[items.site_id == choice].sort_values(
            ["status", "severity", "days_overdue"], ascending=[False, False, False])
        st.write("Open items: overdue now, or due within 60 days")
        st.dataframe(
            site_items[["obligation_name", "category", "authority", "severity",
                        "due_date", "status", "days_overdue"]],
            hide_index=True, width="stretch",
            column_config={
                "obligation_name": "Obligation", "category": "Category",
                "authority": "Authority", "severity": "Severity (1-5)",
                "due_date": st.column_config.DateColumn("Due date", format="DD MMM YYYY"),
                "status": "Status",
                "days_overdue": st.column_config.NumberColumn(
                    "Days overdue", help="Negative = days remaining"),
            })

# ---------------- tab 3: prevention ----------------
with tab3:
    st.subheader("Deadlines in the next 60 days")
    st.write("These are not lapses yet. Acting on them is how lapses are prevented "
             "rather than found at inspection.")
    soon = items[(items.status == "Due soon") & items.site_id.isin(view.site_id)].merge(
        ranked[["site_id", "site_name", "risk_score"]], on="site_id")
    soon["days_left"] = -soon.days_overdue
    soon = soon.sort_values(["days_left", "severity"], ascending=[True, False])
    st.dataframe(
        soon[["site_name", "obligation_name", "authority", "severity", "due_date",
              "days_left", "risk_score"]],
        hide_index=True, width="stretch",
        column_config={
            "site_name": "Site", "obligation_name": "Obligation", "authority": "Authority",
            "severity": "Severity (1-5)",
            "due_date": st.column_config.DateColumn("Due date", format="DD MMM YYYY"),
            "days_left": "Days left",
            "risk_score": st.column_config.NumberColumn("Site risk score", format="%.2f"),
        })
    st.subheader("Overdue obligations by type")
    by_type = (view_overdue.groupby("obligation_name").size()
               .sort_values(ascending=False).rename("Overdue sites"))
    st.bar_chart(by_type, horizontal=True)

# ---------------- tab 4: model performance ----------------
with tab4:
    st.subheader("Does the model beat the baseline?")
    st.write(f"Tested on the {len(metrics['test_quarters'])} most recent quarters "
             f"({metrics['test_rows']} inspections the model never saw during training). "
             f"{metrics['test_lapse_rate']:.0%} of those inspections found a lapse, so that "
             "is the precision you would get by picking sites at random.")
    perf = pd.DataFrame(metrics["ranking"]).rename(columns={
        "method": "Method", "roc_auc": "ROC-AUC",
        "precision_at_10": "Precision at top 10", "precision_at_20": "Precision at top 20"})
    st.dataframe(perf, hide_index=True, width="stretch")

    st.subheader("Obligation detection accuracy")
    st.write(f"Of {metrics['obligation_accuracy_n']} test inspections that found a lapse, "
             "how often did the system name the same obligation the inspector recorded?")
    acc = pd.DataFrame(
        [{"Method": k, "Accuracy": f"{v:.1%}"}
         for k, v in metrics["obligation_detection_accuracy"].items()])
    st.dataframe(acc, hide_index=True, width="stretch")

    st.subheader("What drives the risk score")
    st.write("Permutation importance: how much ranking quality drops when each input is shuffled.")
    st.bar_chart(importance.set_index("label").importance.rename("Importance"), horizontal=True)
    st.info("All figures come from simulated data. They show that the pipeline works "
            "end to end, not how well it would predict lapses at real mines.")
