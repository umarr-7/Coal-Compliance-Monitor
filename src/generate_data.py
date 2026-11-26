"""Step 1 - create a SYNTHETIC dataset.

No clean public dataset links mine-level obligations to inspection outcomes,
so we simulate one. Site names and operators are fictional on purpose: the
data must never look like real findings about a real mine.

The obligation TYPES are modelled on real Indian coal-mine requirements
(see README), but cycles and severities are simplified - verify them against
the actual law before quoting them.

Output (in data/):
  sites.csv                 one row per mine site
  obligations.csv           the obligations register (master list)
  obligation_instances.csv  every due date for every site, and when it was done
  inspections.csv           inspection results (the "ground truth" labels)
"""
import numpy as np
import pandas as pd

from common import DATA, LIVE_DATE

rng = np.random.default_rng(42)

N_SITES = 150
SIM_START = pd.Timestamp("2023-01-01")

# code, name, category, authority, cycle_days, severity(1-5), difficulty, applies_to
OBLIGATIONS = [
    ("SMP", "Safety Management Plan review", "Safety", "DGMS", 365, 5, 0.2, "all"),
    ("VENT", "Ventilation survey and report", "Safety", "DGMS", 90, 5, 0.0, "ug"),
    ("SLOPE", "Slope and dump stability monitoring study", "Safety", "DGMS", 180, 5, 0.3, "oc"),
    ("EQPT", "Statutory examination of machinery and equipment", "Safety", "DGMS", 180, 4, 0.1, "all"),
    ("PLAN", "Mine plans and sections update", "Safety", "DGMS", 90, 3, 0.0, "all"),
    ("DRILL", "Emergency plan mock rehearsal", "Safety", "DGMS", 180, 4, 0.1, "all"),
    ("VT", "Vocational training coverage return", "Labour", "DGMS", 365, 3, 0.0, "all"),
    ("PME", "Periodic medical examination coverage", "Health", "DGMS", 365, 4, 0.3, "all"),
    ("RET", "Annual return to DGMS", "Reporting", "DGMS", 365, 2, -0.3, "all"),
    ("EXPL", "Explosives licence renewal", "Licence", "PESO", 365, 5, -0.4, "all"),
    ("EC", "Environmental clearance half-yearly compliance report", "Environment", "MoEFCC", 182, 3, 0.0, "all"),
    ("CTO", "Consent to Operate renewal", "Environment", "State PCB", 365, 5, 0.1, "all"),
    ("ENVS", "Annual environmental statement", "Environment", "State PCB", 365, 2, -0.2, "all"),
    ("HW", "Hazardous waste annual return", "Environment", "State PCB", 365, 2, 0.0, "all"),
    ("GW", "Groundwater abstraction NOC renewal", "Environment", "CGWA", 365, 3, 0.2, "all"),
    ("FC", "Forest clearance compliance report", "Environment", "MoEFCC", 365, 3, 0.1, "forest"),
    ("MCP", "Mine closure plan escrow deposit", "Finance", "Coal Controller", 365, 4, 0.2, "all"),
]
OB_COLS = ["obligation_code", "obligation_name", "category", "authority",
           "cycle_days", "severity", "difficulty", "applies_to"]

STATES = {"Jharkhand": "JH", "Odisha": "OD", "Chhattisgarh": "CG", "West Bengal": "WB",
          "Madhya Pradesh": "MP", "Telangana": "TS", "Maharashtra": "MH"}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_sites():
    operators = [f"Operator {c}" for c in "ABCDEF"]
    op_effect = dict(zip(operators, rng.normal(0, 0.5, len(operators))))
    rows = []
    for i in range(1, N_SITES + 1):
        state = rng.choice(list(STATES))
        mine_type = rng.choice(["Opencast", "Underground", "Mixed"], p=[0.55, 0.35, 0.10])
        operator = rng.choice(operators)
        short = {"Opencast": "OC", "Underground": "UG", "Mixed": "MX"}[mine_type]
        rows.append({
            "site_id": f"CM-{i:03d}",
            "site_name": f"{STATES[state]}-{short}-{i:03d}",
            "state": state,
            "operator": operator,
            "mine_type": mine_type,
            "capacity_mtpa": round(float(rng.lognormal(0.6, 0.8)), 2),
            "workforce": int(rng.lognormal(6.6, 0.6)),
            "age_years": int(rng.integers(3, 60)),
            "has_forest_land": int(rng.random() < 0.55),
            # Hidden "management quality". Used only to simulate behaviour;
            # it is NOT saved, so the model cannot see it.
            "hidden_quality": op_effect[operator] + rng.normal(0, 0.85),
        })
    return pd.DataFrame(rows)


def applies(ob_applies, site):
    if ob_applies == "all":
        return True
    if ob_applies == "ug":
        return site.mine_type in ("Underground", "Mixed")
    if ob_applies == "oc":
        return site.mine_type in ("Opencast", "Mixed")
    if ob_applies == "forest":
        return site.has_forest_land == 1
    return False


def main():
    DATA.mkdir(exist_ok=True)
    sites = make_sites()
    obligations = pd.DataFrame(OBLIGATIONS, columns=OB_COLS)

    # Each site goes through good and bad patches over time (per quarter).
    n_q = 20
    slump = {}
    for s in sites.itertuples():
        x, arr = 0.0, []
        for _ in range(n_q):
            x = 0.6 * x + rng.normal(0, 0.6)
            arr.append(x)
        slump[s.site_id] = arr

    def quality_at(site, date):
        q_idx = min(max((date - SIM_START).days // 91, 0), n_q - 1)
        return site.hidden_quality - 0.6 * slump[site.site_id][q_idx]

    # ---------- obligation history ----------
    rows = []
    for s in sites.itertuples():
        for ob in obligations.itertuples():
            if not applies(ob.applies_to, s):
                continue
            due = SIM_START + pd.Timedelta(days=int(rng.integers(0, ob.cycle_days)))
            while due < LIVE_DATE + pd.Timedelta(days=120):
                q = quality_at(s, due)
                p_late = sigmoid(-1.5 - 0.9 * q + ob.difficulty)
                if rng.random() < p_late:
                    delay = 1 + int(rng.exponential(40 * (1 + 0.5 * max(-q, 0))))
                else:
                    delay = -int(rng.integers(0, 21))
                completed = due + pd.Timedelta(days=delay)
                if completed >= LIVE_DATE:  # not done yet as of "today"
                    completed = pd.NaT
                rows.append({
                    "site_id": s.site_id, "obligation_code": ob.obligation_code,
                    "severity": ob.severity, "due_date": due, "completed_date": completed,
                })
                due += pd.Timedelta(days=int(ob.cycle_days))
    inst = pd.DataFrame(rows)
    inst["completed_date"] = pd.to_datetime(inst["completed_date"])

    # ---------- inspections ----------
    insp_rows = []
    by_site = {sid: d for sid, d in inst.groupby("site_id")}
    n = 0
    for q_start in pd.date_range("2023-07-01", "2026-07-01", freq="QS"):
        for s in sites.itertuples():
            if rng.random() > 0.6:  # about 60% of sites are inspected each quarter
                continue
            D = q_start + pd.Timedelta(days=int(rng.integers(0, 90)))
            d = by_site[s.site_id]
            od = d[(d.due_date < D) & (d.completed_date.isna() | (d.completed_date > D))].copy()
            od["days"] = (D - od.due_date).dt.days
            sw = float((od.severity * od.days.clip(upper=180) / 180).sum())
            q = quality_at(s, D)
            logit = (-1.9 + 0.5 * sw - 0.7 * q
                     + 0.5 * (s.mine_type != "Opencast")
                     + 0.2 * np.log(s.workforce / 800))
            lapse = int(rng.random() < sigmoid(logit))
            primary, n_viol = "", 0
            if lapse:
                n_viol = 1 + int(rng.poisson(1.2))
                if len(od) and rng.random() < 0.8:
                    w = (od.severity ** 1.5 * np.sqrt(od.days + 1)).to_numpy()
                    primary = rng.choice(od.obligation_code.to_numpy(), p=w / w.sum())
                else:
                    # Something the records did not show (e.g. done on paper only)
                    cand = d.drop_duplicates("obligation_code")
                    w = cand.severity.to_numpy().astype(float)
                    primary = rng.choice(cand.obligation_code.to_numpy(), p=w / w.sum())
            n += 1
            insp_rows.append({
                "inspection_id": f"INS-{n:05d}", "site_id": s.site_id,
                "inspection_date": D, "lapse_found": lapse,
                "violations_count": n_viol, "primary_lapse_obligation": primary,
            })
    insp = pd.DataFrame(insp_rows)

    sites.drop(columns=["hidden_quality"]).to_csv(DATA / "sites.csv", index=False)
    obligations.drop(columns=["difficulty"]).to_csv(DATA / "obligations.csv", index=False)
    inst.to_csv(DATA / "obligation_instances.csv", index=False)
    insp.to_csv(DATA / "inspections.csv", index=False)

    print(f"sites: {len(sites)}  obligation records: {len(inst)}  inspections: {len(insp)}")
    print(f"share of inspections that found a lapse: {insp.lapse_found.mean():.1%}")


if __name__ == "__main__":
    main()
