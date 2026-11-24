# Coal Mine Compliance Monitor

Prototype for Smart India Hackathon 2026, problem statement SIH26024:
"AI-Based Smart Governance and Compliance Monitoring System for Coal Mines" (Ministry of Coal).

**Problem.** Statutory compliance across many mine sites is tracked on spreadsheets, so lapses
are found at inspection rather than prevented.

**What this does.** It scores every site for compliance risk, ranks sites for inspection, and
names the specific obligation driving the risk. It also lists deadlines coming up so they can be
acted on before they become lapses.

## How to run

```
pip install -r requirements.txt
python src/generate_data.py     # 1. create the synthetic dataset  -> data/
python src/train_model.py       # 2. baseline, model, metrics      -> outputs/
streamlit run app.py            # 3. open the dashboard in a browser
```

Needs Python 3.10 or newer. Steps 1 and 2 take a few seconds.

## What is in the project

| File | What it does |
|---|---|
| `src/generate_data.py` | Simulates 150 mine sites, 17 obligation types, about 3.5 years of due dates and completions, and inspection results |
| `src/common.py` | Loads data and builds the features "as of" a date |
| `src/train_model.py` | Baseline, risk model, evaluation, and today's ranked list |
| `app.py` | Dashboard: priority list, site detail, upcoming deadlines, model performance |

## How it works

1. **Obligations register.** Each site has a list of recurring obligations (for example Consent to
   Operate renewal, ventilation survey, Safety Management Plan review). Each has a cycle, an
   authority and a severity from 1 to 5.
2. **Features.** For a given date we compute, per site: how many obligations are overdue, how
   severe and how long overdue, how often the site was late in the last 12 months, what past
   inspections found, and basic site facts (underground or opencast, size, workforce).
3. **Label.** Did the inspection in the following 3 months find a lapse (yes/no)?
4. **Baseline.** Rank sites by number of overdue obligations.
5. **Model.** Gradient-boosted trees (scikit-learn) predict the probability of a lapse. A logistic
   regression is reported as a simpler reference.
6. **Driving obligation.** A transparent rule, not ML: among a site's overdue items pick the highest
   `severity x (1 + days overdue / 90)`. If nothing is overdue, pick the obligation the site has most
   often been late on.
7. **Time-based split.** Train on older quarters, test on the 3 most recent. Features use only what
   was known on the snapshot date, so there is no leakage from the future.

## Results on the synthetic data

Test set: 276 inspections in the last 3 quarters; 26% found a lapse (that is what random picking gives).

| Method | Precision at top 10 | Precision at top 20 | ROC-AUC |
|---|---|---|---|
| Baseline: overdue count | 0.67 | 0.52 | 0.72 |
| Logistic regression | 0.77 | 0.58 | 0.76 |
| Risk model (gradient boosting) | 0.80 | 0.62 | 0.72 |

Obligation detection accuracy (73 lapses): longest-overdue rule 25%, severity x delay rule 26%,
correct obligation within the top 3 candidates 40%.

Read these honestly:

- The model beats the baseline at the top of the list, which is what matters for inspections.
  On overall ranking (ROC-AUC) the gradient boosting model is no better than the baseline, and
  the simple logistic regression is slightly ahead. With only about 600 training rows a simple
  model is a fair choice.
- The test set is small. Precision at top 10 rests on 30 sites, so a few sites change it a lot.
- Obligation detection is weak. The system names the exact obligation about one time in four.
  This is the part with most room for improvement.

## Limitations you must state to judges

- **The data is synthetic.** Sites and operators are fictional. The results show the pipeline works
  end to end. They do not show how well it would predict lapses at real mines.
- The simulation was written by the same people who wrote the model, so the model is finding
  patterns that were put there. Real data will be messier.
- In real life only inspected sites have labels, and inspectors choose where to go. That selection
  bias would need handling.
- Obligation cycles and severities are simplified. Check each one against the actual rule before
  quoting it: Mines Act 1952, Coal Mines Regulations 2017, Mines Rules 1955, Mines Vocational
  Training Rules 1966, Explosives Rules 2008, EIA Notification 2006, the Air and Water Acts,
  Hazardous Waste Rules 2016, the Forest (Conservation) Act and mine closure plan guidelines.

## Moving towards real data

Replace the four CSV files in `data/` with real ones in the same column layout; nothing else needs
to change. Possible public sources: Ministry of Coal and Coal Controller mine lists, DGMS annual
reports and circulars, the PARIVESH portal for environmental and forest clearances and compliance
reports, and State Pollution Control Board consent records.

## Ideas to extend

- Predict lapses per obligation, not just per site, to improve obligation detection.
- Email or SMS alerts 30, 15 and 7 days before each deadline.
- Upload of compliance documents with automatic date extraction.
- A map view by coalfield.
