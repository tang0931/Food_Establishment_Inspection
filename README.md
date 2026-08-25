# King County Food Safety: Analysis & Violation Prediction

Using historical inspection data to help health inspectors prioritize what to check next.

Inspection records tell you what a restaurant did wrong last time. 
This project asks a different question: what's it likely to do wrong next time — combining exploratory analysis with a machine learning model that predicts the most likely violation types for a restaurant's next inspection, giving inspectors a targeted checklist.

## Data Sources

- [Food Establishment Inspection Data](https://data.kingcounty.gov/Health-Wellness/Food-Establishment-Inspection-Data/r878-4sxa/about_data) — the primary dataset: inspections, violations, scores, and grades.
- [Businesses (Food) — King County](https://data.kingcounty.gov/Health/Businesses-Food-King-County/sgr5-u9rw/about_data) — merged in to recover each restaurant's latitude/longitude. Addresses still missing after the merge were filled in using [GeoPy](https://geopy.readthedocs.io/).
- [Zip code population data](https://www.zip-codes.com/city/wa-seattle.asp#zipcodes) — used to compute restaurant density per zip code.

After merging, the dataset covers **12,261 unique restaurants**.

![Overall map of restaurants](map_from_tableau/overall_map.png)

## Exploratory Findings

**Higher-risk restaurants earn fewer "Excellent" grades.** Of the three risk categories, Risk Level 3 has the lowest share of Excellent grades (58%, vs. 77% for Risk Levels 1 and 2) — risk category is a real, visible signal in the grading outcome, not just a label.

![Grade distribution by risk category](image.png)

**The most common violations, overall:**

![Top 10 violation types](image-2.png)

**41.24% of restaurants have at least one repeat violation** — the same issue cited again at a later inspection. Repeat violations skew toward the lower-severity **blue** category rather than the higher-severity **red** category.

![Repeat violation type breakdown](image-3.png)

**Restaurants with repeat violations score worse per inspection**, on average, than restaurants without repeat violations.

![Inspection score vs. repeat violation status](image-4.png)

**Restaurant density varies sharply by zip code.** Combining the inspection data with zip code population figures shows which areas are under-served relative to population — the regions below the dashed line have lower restaurant density than expected. ([Reference: King County rating system](https://kingcounty.gov/en/dept/dph/health-safety/food-safety/inspection-rating-system/rating-system))

![Restaurant density by zip code](image-5.png)

**Business growth has slowed since its 2022 peak.** Most food business categories are still expanding, but at a lower rate than in 2022. (2026 is excluded from this trend since the year isn't complete yet.)

![Growth rate by business category](image-6.png)

**47 restaurants** currently hold a "Needs To Improve" grade. Relative to category size, **Mobile Food Units have the highest rate** of receiving this grade among all classifications.

![Needs To Improve rate by classification](image-7.png)

Their violation profile also looks different from the citywide pattern: **7 of their top 10 violation types are "red"** (higher severity), compared to a much more blue-leaning mix citywide.

![Violation type breakdown for Needs To Improve restaurants](image-8.png)

They're also far less likely to resolve those violations: only **12.8%** of "Needs To Improve" restaurants have fixed their violations, compared to **58.8%** fixed rate across all restaurants — a large gap, not a marginal one.

![Fix rate: Needs To Improve restaurants](image-9.png)
![Fix rate: all restaurants](image-10.png)

## Machine Learning: Predicting Violation Types

**Goal**: for each restaurant's next Routine Inspection, predict which of the 53 known violation types are most likely to occur, and surface the top 3 as a checklist for the inspector — not a replacement for a full inspection, but a way to prioritize attention.

**Approach**: a single shared model, trained on one row per (inspection, candidate violation type) pair, using only information available *before* that inspection (a business's own violation history, how common each violation type is citywide, days since last inspection, and its Risk Category, Classification, and Seating Range). The train/test split is chronological, so the model is always evaluated on inspections that happen after the ones it trained on.

**Evaluation**: Precision@3, Recall@3, and F1@3 — of the 3 violation types suggested, how many were actually cited (precision); of the violations that did occur, how many were caught in the top 3 (recall); and F1@3 balances the two into a single score, since a model can't win just by maximizing one at the other's expense.

### Model comparison

| Model | Precision@3 | Recall@3 | F1@3 |
|---|---|---|---|
| Random guess (3 of 53) | 0.017 | 0.056 | 0.027 |
| Always suggest the 3 most common violations | 0.067 | 0.215 | 0.102 |
| Logistic Regression | 0.145 | 0.197 | 0.167 |
| Random Forest | 0.145 | 0.234 | **0.179** |
| **XGBoost** | **0.158** | 0.208 | **0.179** |

Random Forest and XGBoost are essentially tied on F1@3, each roughly **6.6x better than random guessing** and **~75% better than the naive "always suggest the most common violations"** baseline — without any hyperparameter tuning.

**What actually drives the prediction** (Random Forest feature importance):
1. Citywide base rate of the violation type — **32%**
2. Whether that violation happened at the restaurant's *previous* inspection — **22%**
3. How often that violation has happened at this restaurant historically — **21%**

Together, a restaurant's own violation history accounts for **43%** of the model's decision-making — more than twice the weight of the citywide baseline alone.

**A second decision, before the top-3 list**: not every inspection should get a checklist — a restaurant with a clean history usually doesn't need one. Before ranking violation types, the model first estimates the probability that *any* violation will occur at all, and only issues a top-3 list when that probability crosses a threshold — otherwise inspectors see nothing extra for that visit. This gate has an AUC of ~0.75, meaning it meaningfully separates at-risk from clean inspections, though it isn't perfect — some false alarms and some misses are inevitable trade-offs, tuned via the threshold.

![alt text](image-11.png)

## Future Works

- Try gradient-boosting hyperparameter tuning beyond the current defaults.
- Revisit the "how many suggestions to show" trade-off (currently top 3) with input from an actual health inspector.
