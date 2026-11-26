"""Step 2 - baseline, risk model, evaluation, and the live ranking.

Baseline : rank sites by the number of overdue obligations.
Model    : gradient-boosted trees on the features in common.py.
           (A logistic regression is also reported as a simpler reference.)

Metrics (as asked in the problem statement):
  * precision at top-K inspections
  * obligation detection accuracy

Split is by TIME: train on older quarters, test on the 3 most recent ones.
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (FEATURES, FEATURE_LABELS, LIVE_DATE, N_TEST_SNAPSHOTS, OUT,
                    SNAPSHOTS, build_features, driving_obligation, load_data,
                    open_overdue)

K_VALUES = [10, 20]


def make_model():
    return GradientBoostingClassifier(
        n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0
    )


def precision_at_k(df, score_col, k):
    """Average over test quarters: of the top-K ranked sites, how many had a lapse?"""
    rng = np.random.default_rng(0)
    vals = []
    for _, d in df.groupby("snapshot"):
        d = d.assign(_tie=rng.random(len(d)))  # fair random tie-break
        top = d.sort_values([score_col, "_tie"], ascending=False).head(k)
        vals.append(top.lapse_found.mean())
    return float(np.mean(vals))


def main():
    OUT.mkdir(exist_ok=True)
    sites, obligations, inst, insp = load_data()

    # ---------- build the training table ----------
    parts = []
    for T in SNAPSHOTS:
        X = build_features(sites, inst, insp, T)
        window = insp[(insp.inspection_date >= T)
                      & (insp.inspection_date < T + pd.DateOffset(months=3))]
        labels = window.set_index("site_id")[["lapse_found", "primary_lapse_obligation"]]
        drv = driving_obligation(inst, T)[["driving_code", "longest_overdue_code", "top3_codes"]]
        df = X.join(labels, how="inner").join(drv)
        df["snapshot"] = T
        parts.append(df.reset_index())
    panel = pd.concat(parts, ignore_index=True)

    test_dates = SNAPSHOTS[-N_TEST_SNAPSHOTS:]
    train = panel[~panel.snapshot.isin(test_dates)]
    test = panel[panel.snapshot.isin(test_dates)].copy()

    # ---------- fit ----------
    gbm = make_model().fit(train[FEATURES], train.lapse_found)
    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    logit.fit(train[FEATURES], train.lapse_found)

    test["score_baseline"] = test.n_overdue
    test["score_model"] = gbm.predict_proba(test[FEATURES])[:, 1]
    test["score_logit"] = logit.predict_proba(test[FEATURES])[:, 1]

    # ---------- evaluate ----------
    methods = {"Baseline: overdue count": "score_baseline",
               "Logistic regression": "score_logit",
               "Risk model (gradient boosting)": "score_model"}
    ranking = []
    for name, col in methods.items():
        row = {"method": name, "roc_auc": round(roc_auc_score(test.lapse_found, test[col]), 3)}
        for k in K_VALUES:
            row[f"precision_at_{k}"] = round(precision_at_k(test, col, k), 3)
        ranking.append(row)

    lapses = test[(test.lapse_found == 1) & test.primary_lapse_obligation.notna()
                  & (test.primary_lapse_obligation != "")]
    obligation_acc = {
        "Longest-overdue obligation (simple)":
            round(float((lapses.longest_overdue_code == lapses.primary_lapse_obligation).mean()), 3),
        "Severity x delay rule (this system)":
            round(float((lapses.driving_code == lapses.primary_lapse_obligation).mean()), 3),
        "Severity x delay rule, correct within top 3":
            round(float(np.mean([t in str(c).split("|") for t, c in
                                 zip(lapses.primary_lapse_obligation, lapses.top3_codes)])), 3),
    }

    metrics = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_quarters": [d.strftime("%Y-%m-%d") for d in test_dates],
        "test_lapse_rate": round(float(test.lapse_found.mean()), 3),
        "ranking": ranking,
        "obligation_detection_accuracy": obligation_acc,
        "obligation_accuracy_n": int(len(lapses)),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    imp = permutation_importance(gbm, test[FEATURES], test.lapse_found,
                                 n_repeats=20, random_state=0, scoring="roc_auc")
    imp_df = pd.DataFrame({"feature": FEATURES,
                           "label": [FEATURE_LABELS[f] for f in FEATURES],
                           "importance": imp.importances_mean.round(4)})
    imp_df.sort_values("importance", ascending=False).to_csv(OUT / "feature_importance.csv", index=False)

    # ---------- live ranking for "today" ----------
    final = make_model().fit(panel[FEATURES], panel.lapse_found)
    live = build_features(sites, inst, insp, LIVE_DATE)
    live["risk_score"] = final.predict_proba(live[FEATURES])[:, 1].round(3)
    live = live.join(driving_obligation(inst, LIVE_DATE))
    names = obligations.set_index("obligation_code").obligation_name
    live["driving_obligation"] = live.driving_code.map(names)
    live["risk_band"] = pd.cut(live.risk_score, [-1, 0.35, 0.6, 2], labels=["Low", "Medium", "High"])
    live["rank"] = live.risk_score.rank(ascending=False, method="first").astype(int)
    live["baseline_rank"] = live.n_overdue.rank(ascending=False, method="min").astype(int)
    live = live.drop(columns=["capacity_mtpa", "workforce", "age_years"])
    for col in ["n_overdue", "max_days_overdue", "n_overdue_high_sev", "n_due_next_30d",
                "past_inspections", "last_inspection_lapse", "days_since_last_inspection"]:
        live[col] = live[col].astype(int)
    live = sites.set_index("site_id").join(live).sort_values("rank")
    live.reset_index().to_csv(OUT / "ranked_sites.csv", index=False)

    # Open items per site: overdue now, or due within 60 days.
    od = open_overdue(inst, LIVE_DATE).assign(status="Overdue")
    soon = inst[(inst.due_date >= LIVE_DATE)
                & (inst.due_date < LIVE_DATE + pd.Timedelta(days=60))
                & inst.completed_date.isna()].assign(status="Due soon")
    soon["days_overdue"] = (LIVE_DATE - soon.due_date).dt.days  # negative = days left
    items = pd.concat([od, soon], ignore_index=True).merge(
        obligations[["obligation_code", "obligation_name", "category", "authority"]],
        on="obligation_code")
    items[["site_id", "obligation_code", "obligation_name", "category", "authority",
           "severity", "due_date", "status", "days_overdue"]].to_csv(
        OUT / "open_items.csv", index=False)

    # ---------- print a summary ----------
    print(f"train rows: {len(train)}   test rows: {len(test)}   "
          f"lapse rate in test: {metrics['test_lapse_rate']:.1%}")
    print(pd.DataFrame(ranking).to_string(index=False))
    print(f"obligation detection accuracy (n={len(lapses)}):")
    for k, v in obligation_acc.items():
        print(f"  {k}: {v:.1%}")


if __name__ == "__main__":
    main()
