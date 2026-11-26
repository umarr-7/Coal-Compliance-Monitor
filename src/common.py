"""Shared helpers: paths, dates, data loading and feature building.

Everything here works "as of" a date T. We only use information that would
have been known on that date, so the model never peeks into the future.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"

# "Today" for the dashboard. The live ranking is produced for this date.
LIVE_DATE = pd.Timestamp("2026-10-01")

# Past quarter-start dates used to train and test the model.
# For each one we build features as of that date, and the label is what the
# inspection in the following 3 months actually found.
SNAPSHOTS = pd.date_range("2024-04-01", "2026-07-01", freq="QS")
N_TEST_SNAPSHOTS = 3  # the most recent 3 quarters are held out for testing

FEATURES = [
    "n_overdue",
    "sev_weighted_overdue",
    "max_days_overdue",
    "n_overdue_high_sev",
    "n_due_next_30d",
    "late_rate_12m",
    "avg_delay_days_12m",
    "past_inspections",
    "past_lapse_rate",
    "last_inspection_lapse",
    "days_since_last_inspection",
    "is_underground",
    "capacity_mtpa",
    "workforce",
    "age_years",
]

FEATURE_LABELS = {
    "n_overdue": "Number of overdue obligations (the baseline)",
    "sev_weighted_overdue": "Overdue obligations weighted by severity and delay",
    "max_days_overdue": "Longest overdue period (days)",
    "n_overdue_high_sev": "Overdue high-severity obligations",
    "n_due_next_30d": "Obligations falling due in next 30 days",
    "late_rate_12m": "Share of obligations completed late (last 12 months)",
    "avg_delay_days_12m": "Average delay in days (last 12 months)",
    "past_inspections": "Number of past inspections",
    "past_lapse_rate": "Share of past inspections that found a lapse",
    "last_inspection_lapse": "Last inspection found a lapse",
    "days_since_last_inspection": "Days since last inspection",
    "is_underground": "Underground workings",
    "capacity_mtpa": "Production capacity (MTPA)",
    "workforce": "Workforce",
    "age_years": "Age of mine (years)",
}


def load_data():
    sites = pd.read_csv(DATA / "sites.csv")
    obligations = pd.read_csv(DATA / "obligations.csv")
    inst = pd.read_csv(
        DATA / "obligation_instances.csv", parse_dates=["due_date", "completed_date"]
    )
    insp = pd.read_csv(DATA / "inspections.csv", parse_dates=["inspection_date"])
    return sites, obligations, inst, insp


def open_overdue(inst, T):
    """Obligations that were past their due date and still not done on date T."""
    mask = (inst.due_date < T) & (inst.completed_date.isna() | (inst.completed_date > T))
    od = inst.loc[mask].copy()
    od["days_overdue"] = (T - od.due_date).dt.days
    # Score used to pick the "driving" obligation: severe and long-overdue first.
    od["drive_score"] = od.severity * (1 + od.days_overdue.clip(upper=365) / 90)
    return od


def build_features(sites, inst, insp, T):
    """One row of features per site, using only what was known on date T."""
    f = sites.set_index("site_id")[["capacity_mtpa", "workforce", "age_years"]].copy()
    f["is_underground"] = (sites.set_index("site_id").mine_type != "Opencast").astype(int)

    # --- current overdue position ---
    od = open_overdue(inst, T)
    od["sev_x_delay"] = od.severity * od.days_overdue.clip(upper=180) / 180
    g = od.groupby("site_id")
    f["n_overdue"] = g.size()
    f["sev_weighted_overdue"] = g.sev_x_delay.sum()
    f["max_days_overdue"] = g.days_overdue.max()
    f["n_overdue_high_sev"] = od[od.severity >= 4].groupby("site_id").size()

    soon = inst[
        (inst.due_date >= T)
        & (inst.due_date < T + pd.Timedelta(days=30))
        & (inst.completed_date.isna() | (inst.completed_date > T))
    ]
    f["n_due_next_30d"] = soon.groupby("site_id").size()

    # --- 12-month track record ---
    last = inst[(inst.due_date < T) & (inst.due_date >= T - pd.Timedelta(days=365))].copy()
    last["late"] = (last.completed_date.isna() | (last.completed_date > last.due_date)).astype(int)
    # If it was still open on T we only know it was at least (T - due) days late.
    done_by_T = last.completed_date.where(last.completed_date <= T, T)
    last["delay_days"] = (done_by_T - last.due_date).dt.days.clip(lower=0)
    f["late_rate_12m"] = last.groupby("site_id").late.mean()
    f["avg_delay_days_12m"] = last.groupby("site_id").delay_days.mean()

    # --- inspection history ---
    past = insp[insp.inspection_date < T]
    gp = past.groupby("site_id")
    f["past_inspections"] = gp.size()
    f["past_lapse_rate"] = gp.lapse_found.mean()
    last_insp = past.sort_values("inspection_date").groupby("site_id").tail(1).set_index("site_id")
    f["last_inspection_lapse"] = last_insp.lapse_found
    f["days_since_last_inspection"] = (T - last_insp.inspection_date).dt.days

    # Sites with no history get neutral defaults.
    f["past_lapse_rate"] = f["past_lapse_rate"].fillna(0.3)
    f["days_since_last_inspection"] = f["days_since_last_inspection"].fillna(730)
    f = f.fillna(0)
    return f[FEATURES]


def driving_obligation(inst, T):
    """For each site, the single obligation most likely to be the problem.

    Rule: among overdue obligations pick the highest severity x delay score.
    If nothing is overdue, fall back to the obligation the site has most often
    been late on in the past (weighted by severity).
    Also returns the simple alternative "longest overdue" for comparison.
    """
    od = open_overdue(inst, T)
    top = (
        od.sort_values("drive_score", ascending=False)
        .groupby("site_id").head(1).set_index("site_id")
    )
    longest = (
        od.sort_values("days_overdue", ascending=False)
        .groupby("site_id").head(1).set_index("site_id")
    )

    hist = inst[inst.due_date < T].copy()
    hist["late"] = (hist.completed_date.isna() | (hist.completed_date > hist.due_date)).astype(int)
    tr = (
        hist.groupby(["site_id", "obligation_code"])
        .agg(late_rate=("late", "mean"), severity=("severity", "first"))
        .reset_index()
    )
    tr["fb_score"] = tr.severity * (0.2 + tr.late_rate)
    fb = tr.sort_values("fb_score", ascending=False).groupby("site_id").head(1).set_index("site_id")

    res = pd.DataFrame(index=sorted(inst.site_id.unique()))
    res.index.name = "site_id"
    res["driving_code"] = top.obligation_code
    res["driving_days_overdue"] = top.days_overdue
    res["driving_basis"] = "Overdue now"
    missing = res.driving_code.isna()
    res.loc[missing, "driving_code"] = fb.obligation_code.reindex(res.index)[missing]
    res.loc[missing, "driving_basis"] = "Nothing overdue; repeatedly late in the past"
    res["driving_days_overdue"] = res.driving_days_overdue.fillna(0).astype(int)
    res["longest_overdue_code"] = longest.obligation_code
    # Top 3 candidates (used for the "top-3" accuracy figure).
    top3 = (
        od.sort_values("drive_score", ascending=False)
        .groupby("site_id").obligation_code.apply(lambda c: "|".join(c.drop_duplicates().head(3)))
    )
    res["top3_codes"] = top3.reindex(res.index).fillna(res.driving_code)
    return res
