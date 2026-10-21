# Hotel Booking ML: Cancellation (Classification) and Price (Regression)

In this project I used one hotel booking dataset to solve two problems:

1. **Classification:** will a booking be cancelled? (yes / no)
2. **Regression:** what will be the price per night (ADR) of a booking?

The main focus is **the process**: how I started with a simple baseline, compared models fairly, and improved the results step by step using proper metrics. I did not just pick one model and report accuracy.

## Dataset

Hotel Booking Demand dataset (City Hotel and Resort Hotel in Portugal, 2015-2017), 119,390 rows and 32 columns. I downloaded the CSV from the TidyTuesday GitHub repo (it is the same data that is on Kaggle), and it is saved in `data/hotels.csv`.

## What I did (step by step)

### 1. Data checking and cleaning (`src/01_data_audit.py`)
- Found **31,994 duplicate rows** and removed them. After this the data has 87,213 rows. I did this **before** splitting, otherwise the same booking can be in both train and test.
- Missing values: `children` (4) and `country` (488) were filled. `agent` and `company` are mostly empty (company is 94% empty), so I changed them to simple yes/no columns (`has_agent`, `has_company`).
- Removed 180 bookings that have 0 guests.
- After removing duplicates the cancellation rate changed from 37% to **27.5%**, so the classes are imbalanced.
- For the price problem I also removed rows with price <= 0 and price > 500 (1,647 rows).

### 2. Leakage guard (`src/02_leakage_demo.py`)
The column `reservation_status` tells directly if the booking was cancelled, but we only know it after the outcome. When I left it in, a simple decision tree got **100% accuracy**, which is fake. After removing it (and also `reservation_status_date`, `assigned_room_type`, `booking_changes`) the same model gets **80.5% accuracy and 0.86 ROC-AUC**. This honest score is the starting point for everything else.

| Setup | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| With leakage | 1.000 | 1.000 | 1.000 |
| Leakage guard ON | 0.805 | 0.605 | 0.861 |

### 3. Simple feature engineering
- From the arrival date: weekday, weekend arrival
- From counts: total nights, total guests, has children
- For price only: `hotel x month` (city and resort hotels have different price patterns across the year)

### 4. Splits and cross-validation
- 60% train / 20% validation / 20% test (stratified for classification).
- All scaling, imputing and encoding are inside a sklearn `Pipeline`, so they learn only from training data.
- Models are compared with 5-fold cross-validation on the training set. The validation set is used for choosing the threshold and the alpha values. The **test set is used only at the end**.

### 5. Choosing the final model: metric + simplicity
I did not just take the highest score. Rule: take the **simplest model whose CV score is within 1 standard deviation of the best**. In both problems only Random Forest was inside this range, so it was selected, but this rule would have picked a simpler model if it was close.

## Part A: Classification results

Models: Dummy baseline, Logistic Regression, Decision Tree, Random Forest. Metrics: accuracy, precision, recall, F1, ROC-AUC.

**Why accuracy alone is not enough:** the dummy model that always says "not cancelled" already gets 72.5% accuracy, but it catches 0 cancellations (F1 = 0).

Improvement journey on the **test set**:

| Step | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| 1. Dummy baseline | 0.725 | 0.000 | 0.000 | 0.000 | 0.500 |
| 2. Logistic Regression (default) | 0.790 | 0.676 | 0.454 | 0.543 | 0.834 |
| 3. Logistic Regression (tuned, class_weight balanced) | 0.746 | 0.526 | 0.776 | 0.627 | 0.835 |
| 4. Decision Tree (tuned) | 0.814 | 0.684 | 0.604 | 0.641 | 0.867 |
| 5. Random Forest (tuned, balanced) | 0.814 | 0.630 | 0.787 | 0.700 | 0.894 |
| 6. Random Forest + tuned threshold (0.54) | 0.823 | 0.658 | 0.747 | 0.699 | 0.894 |

What I learnt from this:
- Class weights (step 2 to 3) increased recall a lot (0.45 to 0.78) because the model stopped ignoring the minority class.
- Going from a linear model to a Random Forest gave the biggest improvement (F1 0.54 to 0.70, ROC-AUC 0.83 to 0.89).
- **Threshold tuning:** I picked the threshold (0.54) on the validation set, not on test. On this data it did **not** improve F1 (0.700 vs 0.699). It only moved the balance towards higher precision and accuracy with a little lower recall. I kept this result as it is, because it shows threshold tuning is a trade-off and not always a gain.
- Cross-validation scores (mean of 5 folds) for the tuned models: Logistic Regression ROC-AUC 0.832, Decision Tree 0.869, Random Forest 0.895.

Error analysis (test set, 17,443 bookings): 14,361 correct, 1,868 false alarms, 1,214 missed cancellations. The model is weakest for Online TA bookings (accuracy 77%) and City Hotel (80%). It misses many cancellations in the Corporate, Direct and Aviation segments, where cancellations are rare. The most important features are lead time, country (Portugal), parking spaces, special requests and the Online TA market segment.

## Part B: Regression results

Models: Dummy (mean), Linear Regression, Ridge, Lasso, Decision Tree, Random Forest. Metrics: MAE, RMSE, R2.

**Regularization:** I tested many alpha values on the validation set (`results/figures/11_regularization_sweep.png`). Best Ridge alpha = 43.3 and best Lasso alpha = 0.015. Lasso removed 13 of 86 features. Ridge and Lasso gave almost the same score as plain Linear Regression, because there are 51,000 training rows and few features, so overfitting is not the main problem here. The real problem is that price is not a linear function of the features.

Improvement journey on the **test set**:

| Step | MAE | RMSE | R2 |
|---|---|---|---|
| 1. Dummy (predict mean) | 38.83 | 50.08 | 0.000 |
| 2. Linear Regression | 21.93 | 29.41 | 0.655 |
| 3. Ridge (tuned alpha) | 21.93 | 29.42 | 0.655 |
| 4. Ridge + log of target | 21.74 | 30.58 | 0.627 |
| 5. Ridge + log target + hotel x month feature | 15.76 | 23.10 | 0.787 |
| 6. Random Forest + hotel x month | 11.07 | 17.63 | 0.876 |
| 7. Random Forest + hotel x month + log target | 11.32 | 18.02 | 0.871 |

What I learnt from this:
- A simple feature (`hotel x month`) reduced the Ridge MAE from 21.7 to 15.8, which is more than the regularization or any model tuning did.
- The log transform of the target did not help (RMSE went up in step 4 and step 7), so I did not keep it. A negative result is still useful.
- The final Random Forest (step 6) has MAE 11.07 compared to 38.83 for the dummy baseline, about **71% lower error**, and R2 = 0.876.

Error analysis: errors are biggest in August (MAE about 14) when prices are high and change a lot, and smallest in January/February (MAE about 7). In the residual plot the errors are spread wider for higher predicted prices, so the model is less sure for expensive bookings. There are also a few bookings with a very low real price (close to 0) that the model predicts as normal prices. The 15 worst predictions are saved in `results/04_worst_predictions_top15.csv`.

## Part C: Extensions (Gradient Boosting, PR-AUC, cost-based threshold, time split)

I added these after the first version to test my results more strictly. Code: `src/05_extensions.py`. Note: in this part I removed the `arrival_date_year` column, because a time split cannot use a year (2017) that is not in the training data. So the numbers here are a little different from Part A and Part B.

### 1. Gradient Boosting (HistGradientBoosting)
Gradient Boosting builds many small trees one after another, and each new tree tries to fix the mistakes of the previous trees. A Random Forest builds its trees independently. I tuned it with a small random search.

| Model (random split, test) | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| Logistic Regression | 0.833 | 0.657 | 0.624 |
| Random Forest | 0.887 | 0.753 | 0.695 |
| Gradient Boosting | 0.888 | 0.757 | 0.686 |

Result: Gradient Boosting is **only about the same as Random Forest** (the difference is smaller than the CV standard deviation). So it did not give a real improvement on this data. For price, Gradient Boosting (RMSE 19.89, R2 0.842) is also close to Random Forest (RMSE 20.20, R2 0.837).

### 2. PR-AUC (average precision)
PR-AUC measures how well the model ranks cancelled bookings above the others, focusing on the cancelled class (precision vs recall). ROC-AUC can look good when the classes are imbalanced, because the many "not cancelled" bookings make it easy. Only 27.5% of bookings are cancelled, so PR-AUC is a stricter metric. A random guess has PR-AUC of about 0.275, and our models get 0.66 to 0.76. Plot: `results/figures/16_pr_curves.png`.

### 3. Time-based split
Instead of mixing all bookings randomly, I trained on the past (Jul 2015 to Aug 2016), used Sep to Dec 2016 for validation, and tested on 2017. This is how a real hotel would use the model (predict the future from the past).

| Test ROC-AUC | Random split | Time split |
|---|---|---|
| Logistic Regression | 0.833 | 0.796 |
| Random Forest | 0.887 | 0.784 |
| Gradient Boosting | 0.888 | 0.795 |

What I found:
- All models are worse on the time split. The random split was **too optimistic**.
- The cancellation rate changed over time: 23.9% in training, 31.9% in 2017. The data drifts.
- Random Forest dropped the most (0.887 to 0.784) and the simple Logistic Regression dropped the least (0.833 to 0.796). On the future data the simple model was almost as good as the complex ones.
- For price, the models under-predict in 2017 by about 17 per night on average (prices went up over time), and MAE is 21.7 to 22.1 instead of 13.3 to 13.5 on the random split. This is why the year column matters for price but cannot be used for forecasting.

### 4. Cost-based threshold
F1 treats a missed cancellation and a false alarm as equally bad. In a real hotel they cost different amounts. I assumed (this is my assumption, not real hotel data) that a missed cancellation costs 1 unit (empty room) and a false alarm costs 2 units (guest arrives but the room was resold). I picked the threshold on the validation set that gives the lowest total cost.

Random split, test set (Gradient Boosting):

| Threshold rule | Threshold | Missed cancellations | False alarms | Cost per 1000 bookings |
|---|---|---|---|---|
| Default | 0.50 | 849 | 2762 | 365 |
| Best F1 | 0.58 | 1184 | 2019 | 299 |
| Lowest cost | 0.82 | 2735 | 514 | 216 |

The cost-based threshold gives about 41% lower cost than 0.50, even though its F1 is lower (0.56). This shows the "best" threshold depends on the business cost, not only on F1. The best threshold also moves with the assumption (from 0.56 to 0.91 when false alarms get more costly), see `results/05_cost_threshold_sensitivity.csv`.

Warning: on the time split the thresholds chosen on validation did not work well on 2017 (the F1 threshold of 0.06 gave 14,264 false alarms). This is because the data drifted between validation and test. In a real project the threshold would need to be re-checked regularly.

## Part D: Hyperparameter search (all trials are saved)

Every search used cross-validation on the **training set only** (the test set was never used). `src/07_search_report.py` collects all trials into tables and plots, so the search can be checked: `results/07_search_summary.csv`, `results/03_search_trials_*.csv`, `results/05_search_trials_*.csv`, and figures `26_search_trials_classification.png` and `27_tuning_gain.png`.

| Model | Method | Space tried | Trials | Best CV score | Worst CV score | Best setting |
|---|---|---|---|---|---|---|
| Logistic Regression | Grid search | C in 0.01, 0.1, 1, 10 | 4 | 0.8322 (ROC-AUC) | 0.8311 | C=10 |
| Decision Tree | Grid search | depth (5 values) x min leaf (3 values) | 15 | 0.8638 (ROC-AUC) | 0.8080 | depth 12, leaf 50 |
| Random Forest | Random search | depth x min leaf x max features | 6 | 0.8919 (ROC-AUC) | 0.8722 | depth 20, leaf 5, max features 0.3 |
| Gradient Boosting (cancel) | Random search | learning rate, leaves, min leaf, L2, iterations | 8 | 0.7481 (PR-AUC) | 0.7223 | lr 0.1, 31 leaves, min leaf 100, L2 10, 400 iterations |
| Gradient Boosting (price) | Random search | same space | 8 | RMSE 20.17 | RMSE 22.59 | lr 0.1, 63 leaves, min leaf 20, L2 0, 600 iterations |
| Ridge (price) | Alpha sweep (validation set) | 12 alphas, log scale | 12 | RMSE 29.69 | RMSE 30.53 | alpha 43.3 |
| Lasso (price) | Alpha sweep (validation set) | 12 alphas, log scale | 12 | RMSE 29.69 | RMSE 34.41 | alpha 0.015 |

How much did the search help? I compared the tuned model with the hand-picked setting from round 1 (same 5-fold CV):

| Model | ROC-AUC before | ROC-AUC after | F1 before | F1 after |
|---|---|---|---|---|
| Logistic Regression | 0.8323 | 0.8325 | 0.623 | 0.623 |
| Decision Tree (depth 6 to depth 12) | 0.8425 | 0.8688 | 0.556 | 0.634 |
| Random Forest | 0.8935 | 0.8947 | 0.700 | 0.699 |

What I learnt: the search helped a lot for the Decision Tree (the depth 6 tree was too simple), and it helped choose between Gradient Boosting settings (price RMSE 22.59 to 20.17). For Logistic Regression and Random Forest the gain was almost zero, so the starting settings were already good. A search does not always give a big improvement.

## Part E: Explainability with SHAP

SHAP gives each feature a value for every prediction, which shows how much that feature pushed the predicted cancellation probability up or down. I used it on the final Random Forest (1,500 test bookings sampled). Code: `src/06_explainability.py`.

**Which features matter most** (grouped, average absolute SHAP value):

| Feature | SHAP | Rank by RF impurity | Rank by permutation |
|---|---|---|---|
| country | 0.128 | 2 | 1 |
| market_segment | 0.095 | 3 | 2 |
| total_of_special_requests | 0.081 | 5 | 5 |
| lead_time | 0.078 | 1 | 3 |
| required_car_parking_spaces | 0.048 | 4 | 4 |

All three methods (SHAP, Random Forest impurity importance, permutation importance) give the same top 5 features, but the order is a little different. The impurity method puts `lead_time` first, while SHAP and permutation put `country` first. This is expected, because impurity importance can favor numeric features with many values. Plots: `20_shap_importance_bar.png`, `25_importance_methods_compared.png`.

**Direction of the effect** (beeswarm plot `21_shap_beeswarm.png`, table `06_shap_direction_summary.csv`):
- Longer lead time, higher price (ADR), longer stay and more previous cancellations push towards **cancellation**.
- More special requests, a required parking space and repeat guests push towards **staying**.
- Guests from Portugal and bookings from Online travel agents push towards cancellation.

**Single bookings** (waterfall plots `24_shap_waterfall_*.png`, table `06_shap_example_bookings.csv`):
- High-risk and correct: lead time 134 days, non-refundable deposit, Portuguese guest. Predicted 0.999 and it was cancelled.
- Low-risk and correct: Online TA booking with a parking space. Predicted 0.00 and the guest stayed.
- Missed cancellation: lead time of only 24 days, guest from the UK, offline agent. Predicted 0.006 but it was cancelled. The waterfall shows that country and market segment pushed the score down, so this is a case where the model is wrong, and these are the cases I would study next.

Things to be careful about:
- The base value in the SHAP plots is 0.5, not 0.275, because the Random Forest was trained with balanced class weights. So the predicted numbers are scores, not exact probabilities.
- `required_car_parking_spaces` and `total_of_special_requests` are in the top 5. They may be filled in after booking, so SHAP makes the leakage question in the Limitations more important, not less.
- `country` is the top feature. In a real hotel, using nationality to treat guests differently would need careful thought, so I only use it here to understand the model.

## Limitations (what I would improve)
- Part A and B use a random split, which turned out to be optimistic. Part C shows the time-split results, which are the more realistic ones.
- The costs in the cost-based threshold are my own assumptions, not real hotel numbers.
- `required_car_parking_spaces` and `total_of_special_requests` can sometimes be added after booking, so they might leak a little information. I kept them and removed only the clear leakage columns, but I would check this with the hotel in a real project.
- The hyper-parameter searches are small (4 to 15 trials each) to keep the run time short on a normal laptop. A bigger search might find slightly better settings.
- Only scikit-learn's HistGradientBoosting was used (no XGBoost), because the goal was to compare models and focus on evaluation.

## How to run
```bash
pip install -r requirements.txt
python run_all.py          # runs all 7 scripts (about 45 minutes)
```
Or run each script in `src/` separately (01 to 07, in this order). All tables are saved in `results/` and all plots in `results/figures/`.

## Folder structure
```
hotel_booking_ml/
  data/hotels.csv
  src/
    config.py            settings and column lists
    prep.py              cleaning, features, split, preprocessing pipeline
    utils.py             plot helper
    01_data_audit.py     duplicates, missing values, class balance
    02_leakage_demo.py   fake score vs honest score
    03_classification.py baselines, CV, tuning, selection, threshold, error analysis
    04_regression.py     Linear/Ridge/Lasso/Forest, alpha sweep, improvements, error analysis
    05_extensions.py     Gradient Boosting, PR-AUC, time split, cost-based threshold
    06_explainability.py SHAP, permutation importance, single-booking explanations
    07_search_report.py  tables and plots of all hyper-parameter trials
  results/               csv tables (metrics, selection, errors)
  results/figures/       all plots
  run_all.py
```

## Tools
Python, pandas, NumPy, scikit-learn, SHAP, matplotlib, seaborn
