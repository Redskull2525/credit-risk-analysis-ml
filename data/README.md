# Data: German Credit (Statlog)

`german_credit.csv`: 1,000 rows x 21 columns (about 140 KB). It is committed to the repository so
that `python src/train.py` is reproducible without any network access or account.

## Source and licence

* Downloaded from **OpenML dataset 31, "credit-g", version 1** (https://www.openml.org/d/31) with
  `sklearn.datasets.fetch_openml("credit-g", version=1)`. No login is needed. Re-download with
  `python src/data.py`.
* The OpenML copy is the UCI **Statlog (German Credit Data)** dataset, donated by Prof. Hans Hofmann
  (University of Hamburg) in 1994:
  https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
* **Licence: Creative Commons Attribution 4.0 International (CC BY 4.0)**, which allows
  redistribution with attribution.
* Citation: Hofmann, H. (1994). *Statlog (German Credit Data)* [Dataset]. UCI Machine Learning
  Repository. https://doi.org/10.24432/C5NC77
* OpenML replaces the original A11, A12, ... codes with readable labels such as `<0` or `no checking`.
  Money amounts are in Deutsche Mark (DM), the German currency before the euro.
* The dataset ships with a **cost matrix** from its authors: *"It is worse to class a customer as good
  when they are bad (5), than it is to class a customer as bad when they are good (1)."* This project
  uses that 5:1 ratio as its (explicit) cost assumption in `src/scoring.py`.

## Why not the Kaggle credit-risk data from the first version of this repo?

Version 1 of this repository used an 11-column Kaggle dataset (`person_age`, `person_income`,
`loan_grade`, ...). Kaggle downloads require a login and that dataset's licence does not clearly
permit redistribution, so the training data could not be committed and nobody could reproduce the
model. The German Credit data is public, clearly licensed and small enough to commit, so the schema
was switched.

## Columns

| column | type | meaning | model input? |
|---|---|---|---|
| `checking_status` | categorical | balance of the existing checking account: `<0`, `0<=X<200`, `>=200` (DM), `no checking` | yes |
| `duration` | integer | loan duration in months (4 to 72) | yes |
| `credit_history` | categorical | `no credits/all paid`, `all paid`, `existing paid`, `delayed previously`, `critical/other existing credit` | yes |
| `purpose` | categorical | `radio/tv`, `education`, `furniture/equipment`, `new car`, `used car`, `business`, `domestic appliance`, `repairs`, `other`, `retraining` | yes |
| `credit_amount` | integer | loan amount in DM (250 to 18,424) | yes |
| `savings_status` | categorical | savings/bonds: `<100`, `100<=X<500`, `500<=X<1000`, `>=1000` (DM), `no known savings` | yes |
| `employment` | categorical | years in present job: `unemployed`, `<1`, `1<=X<4`, `4<=X<7`, `>=7` | yes |
| `installment_commitment` | integer | installment rate as a share of disposable income, ordinal code 1 to 4 | yes |
| `personal_status` | categorical | sex and marital status (`male single`, `female div/dep/mar`, `male div/sep`, `male mar/wid`) | **no** (protected) |
| `other_parties` | categorical | other debtors or guarantors: `none`, `co applicant`, `guarantor` | yes |
| `residence_since` | integer | years at the present residence, code 1 to 4 | yes |
| `property_magnitude` | categorical | most valuable property: `real estate`, `life insurance`, `car`, `no known property` | yes |
| `age` | integer | age in years (19 to 75) | **no** (protected) |
| `other_payment_plans` | categorical | other installment plans: `bank`, `stores`, `none` | yes |
| `housing` | categorical | `rent`, `own`, `for free` | yes |
| `existing_credits` | integer | number of existing credits at this bank (1 to 4) | yes |
| `job` | categorical | `unemp/unskilled non res`, `unskilled resident`, `skilled`, `high qualif/self emp/mgmt` | yes |
| `num_dependents` | integer | number of people financially dependent on the applicant (1 or 2) | yes |
| `own_telephone` | categorical | telephone registered under the applicant's name: `none`, `yes` | yes |
| `foreign_worker` | categorical | `yes`, `no` | **no** (protected) |
| `class` | target | `good` (700 rows) = repaid, `bad` (300 rows) = did not repay | target |

The three protected columns stay in the file so that the fairness ablation in `src/train.py` can
measure what excluding them costs (see `reports/model_card.md`).
