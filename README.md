# H1N1 Vaccine Allocation Challenge — Dataset B

**University of Hertfordshire · MSc Data Science · Research Methods Group Challenge**  
**Hackathon:** 11 November 2026 · **Presentation:** 17 November 2026

> **Research question:** Can we predict H1N1 vaccine uptake and use those predictions to allocate vaccines at a lower estimated total cost to the town?

## Project overview

This project builds, evaluates, and compares machine-learning models for predicting the binary target `h1n1_vaccine`. The practical objective is **cost-effective vaccine allocation**, not accuracy alone. Our workflow covers exploratory data analysis (EDA), leakage-aware preprocessing, model comparison, cross-validation, hyperparameter tuning, a cost-sensitive allocation rule, and generation of five distinct test-set submissions.

**Submission status:** The uploaded notebook contains saved output showing that five distinct submission CSVs were generated. Actual upload to Canvas and official hidden-test scores must be verified separately.

## Data

| File | Rows | Columns | Purpose |
|---|---:|---:|---|
| `dataset_B_training.csv` | 4,756 | 31 | Features and labelled target |
| `dataset_B_testing.csv` | 4,749 | 30 | Features only; generate predictions |
| `dataset_B_example_submission.csv` | 4,749 | 2 | Required output schema |
| `Dataset B Information.pdf` | — | — | Feature definitions and challenge costs |

- **Target:** `h1n1_vaccine` (`1` = vaccinated; `0` = not vaccinated).
- **ID:** `respondent_id` is used to match outputs to respondents, not as a model feature.
- **Class balance:** 1,874 vaccinated (**39.4%**), 2,882 not vaccinated (**60.6%**).
- **Examples of inputs:** H1N1 concern and knowledge, vaccine opinions, doctor recommendation, age, healthcare-worker status, household circumstances, and chronic medical conditions.

### Selected EDA findings

- A doctor's H1N1 vaccine recommendation was associated with **72.75%** vaccination uptake, compared with **27.91%** without a recorded recommendation.
- Respondents who rated the vaccine *very effective* had **62.54%** uptake, versus **10.79%** among those rating it *not at all effective*.
- Missingness was high for `employment_sector` (**49.71%**) and `health_insurance` (**40.48%**).

These are descriptive associations, **not causal effects**.

## Methods

### 1. Exploratory analysis

Examined the target distribution, missingness, age-group vaccination rates, vaccine attitudes, doctor recommendations, chronic conditions, and Spearman correlations. Charts and calculations are documented in the notebook.

### 2. Preprocessing

The notebook uses a `ColumnTransformer` inside scikit-learn `Pipeline` objects:

| Feature group | Treatment |
|---|---|
| Numeric | Median imputation, missingness indicators, standard scaling |
| Binary | Most-frequent imputation, missingness indicators |
| Categorical | Explicit missing category, one-hot encoding |
| Vaccine-opinion responses | Categorical encoding to avoid treating “Don't know” as a numeric midpoint |

The respondent ID and target are excluded from predictive features. Preprocessing is fitted **within each training fold**, rather than on validation data.

### 3. Model development

Compared:

- Logistic Regression (interpretable baseline)
- Random Forest (bagged nonlinear trees)
- Gradient Boosting (sequentially boosted nonlinear trees)

The labelled dataset was split **80% development / 20% holdout validation**, stratified by target (3,804 / 952 respondents). Five-fold stratified cross-validation was used on the development subset. Hyperparameters were tested using `RandomizedSearchCV` with the estimated financial-cost objective.

### 4. Evaluation

Reported classification accuracy, ROC-AUC, Brier score, confusion matrices and **estimated total financial cost**. Additional comparisons may include log loss and average precision. The best model for allocation need not be the one with the best accuracy.

## Financial decision model

The challenge specifies:

- Vaccine delivery: **$500** for every dose sent.
- Compensation: **$500** if an individual who would accept vaccination is not sent a vaccine.
- Hospitalisation: **$25,000** per hospitalisation.
- Infection is assumed for all respondents as stated in the challenge.

For individual *i*, let `s_i` indicate a vaccine **sent**, `y_i` indicate observed vaccine uptake (used here as a proxy for acceptance), and `h_u,i` / `h_v,i` indicate estimated hospitalisation risk without / with vaccination. The notebook's financial rule uses:

```text
Total estimated cost =
    500 × number of vaccines sent
  + 500 × number of actual vaccinators not sent a vaccine
  + 25,000 × sum(expected hospitalisation probabilities)
```

Hospitalisation protection is assigned only when a vaccine is **both sent and accepted**. For predicted acceptance probability `p_i`, the expected-cost allocation threshold is:

```text
send vaccine if:
    p_i × [500 + 25,000 × (h_u,i - h_v,i)] > 500
```

**Crucial limitation:** The dataset does not include hospitalisation outcomes. The uploaded notebook uses **illustrative** assumptions (1% baseline hospitalisation risk, 50% relative risk reduction from vaccination, plus age/chronic-condition multipliers). These are **not validated epidemiological estimates**. The financial values below are therefore scenario-dependent, not official costs. The team should document an externally supported risk model and use the **same audit set, costs, individual risk arrays, decision thresholds, and formula** when comparing members' models.

## Saved model results

The following values are from the **952-person holdout validation set in the uploaded notebook**, not from the team's separately reserved audit set or the hidden competition test set.

| Tuned model | Accuracy | ROC-AUC | Brier score | Estimated holdout cost |
|---|---:|---:|---:|---:|
| **Gradient Boosting** | 0.7763 | **0.8413** | **0.1552** | **$560,117.50** |
| Logistic Regression | **0.7784** | 0.8362 | 0.1570 | $561,735.00 |
| Random Forest | 0.7710 | 0.8297 | 0.1636 | $566,425.00 |

A useful critical finding is that tuning did **not** help every algorithm: compared with the baseline, estimated holdout costs decreased for Gradient Boosting but increased for Logistic Regression and Random Forest. Observed differences between leading strategies were small relative to the uncertainty in the cost assumptions.

## Five ranked challenge submissions

The uploaded notebook's saved Step 10 output reported the following **distinct test-set allocation strategies**, ranked using its provisional holdout financial evaluation:

| Rank | Strategy | Decision-factor | Estimated holdout cost | Vaccines sent on hidden test |
|---|---|---:|---:|---:|
| 1 | Tuned Gradient Boosting | 1.00 | $560,117.50 | 904 |
| 2 | Tuned Logistic Regression | 0.95 | $560,130.00 | 1,026 |
| 3 | Baseline Random Forest | 0.85 | $560,167.50 | 1,064 |
| 4 | Baseline Gradient Boosting | 0.95 | $560,227.50 | 1,067 |
| 5 | Baseline Logistic Regression | 1.05 | $560,370.00 | 908 |

The final predictions are binary `0` / `1`, preserve `respondent_id` ordering, and use the required file format:

```csv
respondent_id,h1n1_vaccine
4757,1
4758,0
```

*The rows above illustrate the schema, not verified predictions for those IDs.*

Expected filenames generated by the saved notebook:

```text
challenge_submission_group_G_order_1.csv
challenge_submission_group_G_order_2.csv
challenge_submission_group_G_order_3.csv
challenge_submission_group_G_order_4.csv
challenge_submission_group_G_order_5.csv
```

**Note:** “G” was the group identifier entered in the saved run. If your official identifier is different, regenerate filenames appropriately. The five test allocation counts are **not** vaccine counts for the 952-person validation split.

## Explainability and critical evaluation

The analysis should consider both **global feature reliance** and **local explanations**:

- **Permutation importance:** Quantifies the change in validation performance when an input feature is shuffled.
- **SHAP:** Describes feature contributions to model predictions; interpret output units and one-hot-encoded features carefully.

The **uploaded notebook version reviewed for this README does not yet contain saved SHAP or permutation-importance code/output**. If these analyses have since been added in Colab, save and commit the updated `.ipynb` file before claiming them as completed work in your presentation.

Neither technique proves causation or directly validates the hospitalisation-risk assumptions.

## How to run in Google Colab

1. Open the team's shared Colab notebook (link below) or upload `H1N1_Vaccine_Challenge.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Add `dataset_B_training.csv`, `dataset_B_testing.csv` and `dataset_B_example_submission.csv` to the Colab working directory.
3. Run cells in notebook order: data loading → EDA → preprocessing → model comparison → financial evaluation → tuning → final submission generation.
4. Enter the official group identifier when Step 10 requests it.
5. Review the model results and risk assumptions; download the five generated CSVs and preserve the exact submission schema.
6. Save the latest notebook to the organisation's **public** GitHub repository. Ensure shared Colab permissions meet the assignment requirements.

**Core Python packages:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.  
**For optional explainability:** `shap`.  
Running in Colab requires the dataset files to be available in the active runtime.

## Reproducibility, teamwork and presentation

- Most model steps use `random_state=42`; the notebook documents model choices, preprocessing and figures.
- Each team member should contribute through their own GitHub account and record meaningful commits.
- Keep evidence of discussions, model decisions, testing and shared Colab editing.
- Presentation: **10 minutes + 5 minutes of questions**, including EDA, models, validation, costs, XAI, limitations, teamwork, and a **non-technical stakeholder summary**.
- The presentation's first slide must link to the GitHub organisation and shared Colab notebook.

**Links to complete before submission:**

- GitHub organisation: **[ADD PUBLIC ORGANISATION URL]**
- Project repository: **[ADD PUBLIC REPOSITORY URL]**
- Shared Colab notebook: **[ADD COLAB URL]**
- Team members and contributions: **[ADD VERIFIED NAMES / ROLES]**

## Limitations

1. The outcome records **historical vaccination**, not a directly measured willingness-to-accept outcome.
2. Financial estimates depend on **unverified clinical risk assumptions** in the saved notebook.
3. Choosing among models and thresholds on one holdout set can cause **selection optimism**.
4. Small validation-cost differences are not sufficient evidence that the ranking will generalise.
5. Cross-team model rankings are valid only with the **same development/audit split and financial scoring convention**.
6. The **hidden test labels and official leaderboard results** are not available in the uploaded notebook.

## Deliverables

- [x] EDA, preprocessing, three model families, cross-validation and tuning in the uploaded notebook.
- [x] Saved notebook output reporting five different prediction strategies.
- [ ] Verify submission of five individual CSVs on Canvas.
- [ ] Verify public GitHub repository, individual contribution history and shared Colab permissions.
- [ ] Add and document XAI results in the final committed notebook if completed.
- [ ] Replace provisional financial assumptions with justified estimates, or clearly present a sensitivity-limited interpretation.
- [ ] Complete group presentation and peer feedback.

---

*Academic group project. Numerical results reflect the saved notebook's internal validation and assumptions; they are not clinical recommendations or official challenge scores.*
