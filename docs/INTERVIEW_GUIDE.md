# Interview guide: Credit risk analysis (German Credit)

Use this to explain the project confidently and **honestly**. Read it next to the code: you should be able
to explain every line you show. All numbers come from one run of `python src/train.py` (2026-10-19) and
are also in `reports/metrics.json`.

> Golden rules. The data is a 1994 sample from one German bank. Never say "my model saves the bank X" or
> "this would work in production". Say what you measured, how you measured it, and what you would need
> before anyone trusted it with real money.

---

## 1. Pitches

**30 seconds**

> "I built a credit-risk classifier on the public German Credit dataset: 1,000 loan applications, 30%
> of them bad. It is a scikit-learn pipeline with scaling and one-hot encoding, comparing logistic regression
> and a random forest with stratified cross-validation. The random forest won by a small margin and scores
> ROC-AUC 0.79 on a held-out test set. The interesting part is the decision threshold: using the dataset's
> own cost matrix, where a missed bad loan costs five times a wrongly rejected good one, the best threshold is
> 0.32, not 0.5. That lifts recall of bad loans from 73% to 93% at the price of rejecting many more
> applicants. The model is served in a Streamlit app whose form is generated from the saved feature list,
> and 20 tests run in GitHub Actions."

**60 seconds**

> "The business problem is a bank deciding whom to lend to. Each applicant has 17 attributes: checking and
> savings balances, loan amount and duration, purpose, employment, housing and so on. The data also has age,
> sex and foreign-worker status; I deliberately left those out because using them for credit decisions is
> discriminatory, and I measured the cost of leaving them out: 0.003 ROC-AUC, essentially nothing.
>
> I split 80/20 with stratification, built a ColumnTransformer with StandardScaler and OneHotEncoder, and
> compared logistic regression and a random forest, both class-weighted, using 5-fold stratified CV on the
> training set only. Random forest got 0.793 versus 0.781, which is inside the fold-to-fold spread, so I
> would not claim it is truly better; the script picks by the rule it declares. I chose the threshold by
> minimising expected cost on out-of-fold training predictions, so the test set was touched exactly once:
> ROC-AUC 0.788, PR-AUC 0.640, recall of bad loans 0.93 at threshold 0.32.
>
> The first version of this repo had a pickled model with no training code, unscaled features that never
> converged, and an app that sent 5 of the 11 inputs. The rebuild fixes each of those with a test: the form
> is built from the saved feature list, stored metrics are re-checked against the saved model, and the
> scikit-learn version is pinned and asserted."

---

## 2. Plain-English walkthrough of `src/train.py`

1. **Load** `data/german_credit.csv` (1,000 rows). The target is `class`: `good` or `bad`. We predict
   `bad`, so `y = 1` means "defaulted".
2. **Split** 800 train / 200 test with `stratify=y`, so both parts keep 30% bad. `random_state=42` makes
   the split identical every run; the tests rely on that.
3. **Preprocess inside a Pipeline.** `StandardScaler` puts the 6 numeric columns on the same scale (needed
   for logistic regression to converge and for coefficients to be comparable); `OneHotEncoder` turns the 11
   text columns into 0/1 columns (`handle_unknown="ignore"` so an unseen category becomes all zeros instead of
   crashing the app). Because it is inside the pipeline, the scaler and encoder are fitted on training folds
   only. Fitting them on all data first would leak information from the test set.
4. **Compare two models with 5-fold stratified CV on the training set.** Each model is trained 5 times on
   640 rows and scored on the other 160; we average ROC-AUC. Both use `class_weight="balanced"` so the 300
   bad examples count as much as the 700 good ones.
5. **Pick the winner** by mean CV ROC-AUC. In this run: random forest.
6. **Choose the threshold** with `cross_val_predict`: every training row gets a prediction from a model that
   did not see it ("out-of-fold"). On those 800 predictions we try thresholds 0.05 to 0.95 and keep the one
   with the lowest cost (5 x false negatives + 1 x false positives). Ties go to the higher threshold (fewer
   rejections for the same cost).
7. **Fit the winner on all 800 training rows and evaluate once on the 200 test rows** at both 0.50 and the
   chosen threshold. We also report what "approve everyone" and "reject everyone" would cost.
8. **Save** one joblib file containing the fitted pipeline *and* metadata: feature names, per-feature ranges
   and categories (for the app's form), threshold, costs, metrics, scikit-learn version. Write
   `reports/metrics.json`, `reports/model_card.md` and the logistic-regression coefficients.

The notebook imports `run_training` from the same file, so it cannot show different numbers from the script.

---

## 3. Metrics, explained the way an interviewer wants to hear them

**Why accuracy is not enough.** 70% of applicants are good. A "model" that approves everyone is 70% accurate
and catches zero bad loans. Our model at threshold 0.32 has *lower* accuracy (0.53) and is far more useful,
because accuracy weighs a missed default and a rejected good customer equally, and the bank does not.

**Confusion matrix** (positive = bad). On the test set at 0.32: TN 50, FP 90, FN 4, TP 56.
* TP: bad loans we rejected (good). FN: bad loans we approved (expensive). FP: good applicants we rejected
  (lost business). TN: good applicants we approved.

**Precision (bad)** = TP / (TP + FP) = 56 / 146 = 0.38: of the people we reject, 38% really were bad.
**Recall (bad)** = TP / (TP + FN) = 56 / 60 = 0.93: of the bad loans, we caught 93%.
**F1** is their harmonic mean; it treats precision and recall as equally important, which here they are not,
so F1 is reported but not optimised.

**ROC-AUC (0.788)**: take a random bad applicant and a random good one; the model gives the bad one the
higher score 78.8% of the time. It measures *ranking* and does not depend on the threshold. 0.5 is chance,
1.0 is perfect. Caveat: with imbalanced data ROC-AUC can look flattering because true negatives are plentiful.

**PR-AUC / average precision (0.640)**: the area under the precision-recall curve. It ignores true negatives,
so it is the harsher, more informative number for the rare class. Chance level equals the bad rate, 0.30, so
0.64 is a real improvement.

**ROC vs PR in one line**: ROC asks "can the model separate the classes?", PR asks "when the model flags
someone, how often is it right, and how many does it find?" Report both.

**What a threshold does.** The model outputs a score in [0, 1]. Threshold 0.5 is an arbitrary default.
Lowering it flags more people: recall goes up, precision and accuracy go down. Moving the threshold slides
along the ROC and PR curves; it does not change either AUC. Our cost-based threshold is 0.32.

**Expected cost.** Cost = 5 x FN + 1 x FP. Test set: 120 at 0.50, 110 at 0.32, 300 for approving everyone,
140 for rejecting everyone. The 5:1 ratio is the dataset authors' suggestion and an assumption; the real
numbers depend on loan size, interest margin and recoveries.

**Calibration.** `class_weight="balanced"` pushes scores upward: the mean predicted score on the test set is
0.45 while the real bad rate is 0.30. So a score of 0.45 is not "a 45% chance of default". The app says this
explicitly. Fixing it would mean `CalibratedClassifierCV` or re-tuning the threshold after calibration.

---

## 4. Why logistic regression is the baseline in credit scoring

* **Interpretability and reason codes.** Regulators and customers expect a lender to explain a rejection.
  Logistic regression gives one coefficient per input; `exp(coefficient)` is an odds ratio you can read out:
  "an overdrawn checking account doubles the odds of default". Tree ensembles need extra tooling (SHAP,
  permutation importance) to say anything similar.
* **Stability and monotonic behaviour.** A linear model behaves predictably outside the training range and is
  easy to monitor. Banks have used scorecards built on logistic regression for decades.
* **It is hard to beat on small tabular data.** Here it was 0.012 ROC-AUC behind the random forest, within
  the fold-to-fold noise. If a complex model cannot clearly beat it, the simple model usually wins in practice.

Coefficients from this run (standardised numeric columns; one-hot levels compared within their own column).
Positive raises the odds of default.

| Feature | Coefficient | Odds ratio | Plain English |
|---|---|---|---|
| purpose = used car | -1.49 | 0.22 | used-car loans default far less than the average purpose |
| checking_status = no checking | -1.00 | 0.37 | no checking account at this bank: lower risk (a known quirk of this dataset) |
| other_parties = guarantor | -0.92 | 0.40 | a guarantor cuts the odds by 60% |
| purpose = education | +0.84 | 2.32 | education loans default more often |
| credit_history = critical/other existing credit | -0.84 | 0.43 | counter-intuitive but consistent across studies of this dataset |
| checking_status = <0 | +0.69 | 2.00 | overdrawn checking account doubles the odds |
| savings_status = >=1000 | -0.67 | 0.51 | large savings halve the odds |
| savings_status = <100 | +0.58 | 1.78 | little savings: 78% higher odds |
| duration (per +1 SD, about 12 months) | +0.35 | 1.42 | longer loans are riskier |
| credit_amount (per +1 SD, about 2,800 DM) | +0.32 | 1.38 | larger loans are riskier |
| installment_commitment (per +1 SD) | +0.30 | 1.35 | higher installment burden is riskier |

How to defend "counter-intuitive" rows: they are associations in one bank's 1994 sample, after conditioning
on the other inputs; they are not causal claims, and you would investigate them with the business before
using them as reason codes.

---

## 5. Fifteen questions you should expect

**Q1. What does the model predict, exactly?**
The probability-like score that an applicant turns out "bad" (does not repay), from 17 attributes known at
application time. Then a decision: HIGH risk if the score is at or above 0.32, else LOW.
*Follow-up: why a score and not just a yes/no?* Because the bank sets the cut-off, and may use several
(approve, manual review, decline). The score also lets you rank and price.

**Q2. Why this dataset and not the Kaggle one from version 1?**
Kaggle needs a login and the licence did not clearly allow redistribution, so nobody could rerun the
training. German Credit is CC BY 4.0, downloads from OpenML with one scikit-learn call and is small enough to
commit. *Follow-up: downsides?* 1,000 rows, 1994, one bank, Deutsche Mark. Fine for learning, useless for
a real lender today.

**Q3. Walk me through the preprocessing.**
`ColumnTransformer`: `StandardScaler` on 6 numeric columns, `OneHotEncoder(handle_unknown="ignore")` on 11
categorical ones, wrapped in a `Pipeline` with the model. *Follow-up: why inside the pipeline?* So the scaler
is fitted on training folds only (no leakage) and the app can call `predict_proba` on raw inputs with no
hand-written preprocessing that could drift.

**Q4. Why a stratified split and why random_state=42?**
Stratification keeps 30% bad in both train and test; with 200 test rows a random split could easily give 25%
or 35% and change every metric. The seed makes the split reproducible; the test
`test_saved_metrics_are_reproducible` rebuilds it and checks the stored ROC-AUC.

**Q5. Why cross-validation if you have a test set?**
Model selection must not use the test set, otherwise the test score is optimistic. CV on the training data
picks the model and the threshold; the test set is used once, at the end, to report. *Follow-up: what does the
+/- 0.065 tell you?* That ROC-AUC varies by about 0.065 between folds of 160 rows, so the 0.012 gap between
the models is noise-level.

**Q6. Random forest won. Is it really better?**
Not demonstrably. 0.793 vs 0.781 with std 0.065. I kept the declared rule (highest mean CV ROC-AUC) rather
than override it, and I keep the logistic regression for interpretation. *Follow-up: how would you find out?*
Repeated stratified CV (e.g. 10 x 5 folds) and a paired comparison of fold scores, or simply more data.

**Q7. Why `class_weight="balanced"`?**
With 30% positives an unweighted model leans towards "good". Balanced weights make each bad example count
7/3 times as much, which raises recall of the bad class. *Follow-up: side effect?* Scores shift upward and are
no longer calibrated; mean score 0.45 vs bad rate 0.30 on the test set. The threshold compensates, and the
app warns about it.

**Q8. What does ROC-AUC 0.788 mean in words?**
If you pick one bad and one good applicant at random, the model scores the bad one higher 79% of the time.
*Follow-up: why also report PR-AUC?* Because ROC-AUC is inflated by the many easy true negatives; PR-AUC
(0.64 against a 0.30 chance level) focuses on how well we find and identify the rare bad class.

**Q9. Why is the threshold 0.32?**
Under the 5:1 cost matrix, that threshold minimised cost on out-of-fold training predictions (433 vs 511 at
0.50). On the test set it gave cost 110 vs 120 and recall 0.93 vs 0.73. *Follow-up: who decides the cost
ratio, and what if it were 10:1?* The business, from loan size, margin and recovery data. A bigger ratio pushes
the threshold lower and rejects even more people; the exercise in section 7 asks you to try it.

**Q10. Accuracy dropped from 0.72 to 0.53 at your threshold. Did you make the model worse?**
No; the model (and its AUCs) is identical, only the cut-off moved. Accuracy is the wrong objective here:
approving everyone scores 0.70. What changed is the trade: 4 missed defaults instead of 16, at the cost of
rejecting 90 good applicants instead of 40. *Follow-up: would a bank accept rejecting 73% of applicants?*
Probably not; that is the honest limitation of a pure cost-minimising threshold. A real lender would add a
volume constraint or a manual-review band.

**Q11. Where could data leakage hide in your pipeline?**
Fitting the scaler/encoder on all data before splitting; tuning the threshold on the test set; picking the
model after looking at test scores. The pipeline, out-of-fold threshold and single test evaluation address
each. *Follow-up: anything left?* EDA used the full dataset, including the test rows. No modelling decision was
based on it, but a strict setup would do EDA on the training split only.

**Q12. Why exclude age, sex and foreign-worker status? They are predictive.**
Because deciding credit on them is discriminatory and illegal in many jurisdictions (US ECOA, EU
equal-treatment law). I measured the cost: CV ROC-AUC 0.793 without them vs 0.796 with them. *Follow-up: is
the model now fair?* Not necessarily. Other columns can act as proxies (job, housing, purpose). A fairness
audit would compare recall and false-positive rates across groups on the test set; I list it as an exercise.

**Q13. Version 1 of this repo was broken. What exactly went wrong and how do the tests prevent it?**
Three things: (1) the model was trained on 11 raw unscaled columns and never converged (coefficients around
1e-7), (2) the app sent 5 columns to a model expecting 11, (3) requirements were unpinned, so the server's
scikit-learn could differ from the one that made the pickle. Now: the scaler is in the pipeline and the
pipeline must beat chance on a fixture; the form is generated from the saved feature list and a test asserts
the form widgets equal the pipeline's `feature_names_in_`; requirements are pinned and
`test_sklearn_version_matches_artifact` fails on a mismatch.

**Q14. How would you explain a rejection to a customer?**
With the logistic-regression odds ratios: for example "overdrawn checking account (odds x2.0), loan for
education (x2.3), savings under 100 DM (x1.8)". For the random forest I would compute per-applicant
contributions with SHAP or fall back to the logistic model as the explanation layer. *Follow-up: what is an
odds ratio?* `exp(coefficient)`: the multiplicative change in odds of default for a one-unit (one standard
deviation, for scaled numeric columns) increase in that input, holding the others fixed.

**Q15. What would you do next with more time?**
Repeated CV to settle the model choice; hyperparameter tuning with nested CV; probability calibration and a
reliability plot; a fairness audit by group; a two-threshold policy (approve / review / decline) with a volume
constraint; monitoring for drift; and, above all, newer and larger data.

---

## 6. Common mistakes (some of which version 1 of this repo made)

* Shipping a pickle with no training code or data. Nobody, including you, can reproduce or defend it.
* Serving a model with a different set of columns than it was trained on. Generate the form from the model.
* Unscaled features in logistic regression: the solver does not converge and coefficients are meaningless.
* Unpinned `requirements.txt`: the server installs a different scikit-learn and the pickle fails or misbehaves.
* Reporting accuracy on imbalanced data as if it meant something.
* Choosing the threshold, or the model, on the test set.
* Fitting the scaler or encoder before the train/test split.
* Reading class-weighted scores as calibrated probabilities.
* Claiming a business impact ("saves the bank 20%") that was never measured.
* Using protected attributes because "they improve the score".

---

## 7. Six exercises

1. **Change the cost ratio.** Set `COST_FALSE_NEGATIVE` to 10, then to 2, re-run `python src/train.py`, and
   record the threshold, recall of bad loans and rejected share each time. Explain the pattern in two sentences.
2. **Add a third candidate.** Put `HistGradientBoostingClassifier` in `candidate_models()`. Does it win the
   CV comparison? Add a test that every candidate produces probabilities in [0, 1].
3. **Tune honestly.** Use `GridSearchCV` for `C` (logistic regression) and `min_samples_leaf` (random forest)
   *inside* the cross-validation (nested CV). Compare the nested score with the un-tuned 0.793 and explain why a
   non-nested grid search would be optimistic.
4. **Calibrate.** Wrap the winner in `CalibratedClassifierCV(method="isotonic")`, plot a reliability diagram
   on the test set, and re-run the threshold search. Does the chosen threshold move towards 0.5? Why?
5. **Fairness audit.** Using the protected columns only for evaluation, compute recall and false-positive rate
   of the saved model separately for each `personal_status` group and each age band on the test set. Are the
   error rates equal? What would you say to the business?
6. **Reason codes in the app.** For the logistic regression, show the three inputs that push the current
   applicant's score up the most (`coefficient x standardised value`). Add an AppTest that checks they appear.
