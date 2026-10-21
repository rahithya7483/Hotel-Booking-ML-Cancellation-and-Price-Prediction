"""Step 6 - Explainability with SHAP (final cancellation model: tuned Random Forest).

Questions answered
  1. Which features matter most overall?           (mean |SHAP|)
  2. In which direction do they push the prediction? (beeswarm plot, dependence plot)
  3. Why did the model give ONE booking a high / low cancellation risk? (waterfall plots)
  4. Do SHAP, permutation importance and Random Forest importance agree?
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

import config as C
from prep import drop_leaky, get_clean_data, make_splits
from utils import savefig

# ---------------------------------------------------------------- same data as 03_classification.py
df = get_clean_data()
y = df["is_canceled"]
X = drop_leaky(df).drop(columns=["is_canceled"])
X = X[C.CLF_NUM_COLS + C.BIN_COLS + C.CAT_COLS]
X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y, stratify=y)

saved = joblib.load(C.MODELS / "best_classifier.joblib")
pipe, THRESH = saved["model"], saved["threshold"]
pre, rf = pipe.named_steps["pre"], pipe.named_steps["clf"]
names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
print("model:", saved["name"], "| threshold:", THRESH, "| features after encoding:", len(names), flush=True)

# SHAP on a random sample of the test set (SHAP on trees is slow for big data)
rng = np.random.RandomState(C.SEED)
idx = rng.choice(len(X_test), size=1500, replace=False)
X_s = X_test.iloc[idx]
y_s = y_test.iloc[idx]
Z = pre.transform(X_s)

explainer = shap.TreeExplainer(rf)
sv = explainer.shap_values(Z, check_additivity=False)
if isinstance(sv, list):                 # older shap: list [class0, class1]
    sv = sv[1]
elif sv.ndim == 3:                       # newer shap: (rows, features, classes)
    sv = sv[:, :, 1]
base = explainer.expected_value
base = float(base[1]) if np.ndim(base) > 0 else float(base)
print("SHAP values computed:", sv.shape, "| base value (avg predicted cancel prob):", round(base, 3), flush=True)

# ---------------------------------------------------------------- 1. global importance
mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=names).sort_values(ascending=False)
plt.figure(figsize=(8, 6.5))
shap.summary_plot(sv, Z, feature_names=names, plot_type="bar", max_display=15, show=False)
plt.title("SHAP: average impact on cancellation probability")
savefig("20_shap_importance_bar.png")

# ---------------------------------------------------------------- 2. direction (beeswarm)
plt.figure(figsize=(9, 7))
shap.summary_plot(sv, Z, feature_names=names, max_display=15, show=False)
plt.title("SHAP beeswarm (red = high feature value)")
savefig("21_shap_beeswarm.png")

for feat, fname in [("lead_time", "22_shap_dependence_lead_time.png"),
                    ("total_of_special_requests", "23_shap_dependence_special_requests.png")]:
    plt.figure(figsize=(6.5, 4.8))
    shap.dependence_plot(feat, sv, Z, feature_names=names, show=False, interaction_index=None)
    plt.title(f"SHAP dependence: {feat}")
    savefig(fname)

# ---------------------------------------------------------------- 3. single bookings
proba = rf.predict_proba(Z)[:, 1]
pred = (proba >= THRESH).astype(int)
hit_cancel = np.where((y_s.values == 1) & (pred == 1))[0]
hit_stay = np.where((y_s.values == 0) & (pred == 0))[0]
miss_cancel = np.where((y_s.values == 1) & (pred == 0))[0]
cases = {
    "high_risk_correct": hit_cancel[np.argmax(proba[hit_cancel])],
    "low_risk_correct": hit_stay[np.argmin(proba[hit_stay])],
    "missed_cancellation": miss_cancel[np.argmin(proba[miss_cancel])],
}
case_rows = []
for k, (label, i) in enumerate(cases.items()):
    exp = shap.Explanation(values=sv[i], base_values=base, data=Z[i], feature_names=names)
    plt.figure(figsize=(8, 5.5))
    shap.plots.waterfall(exp, max_display=10, show=False)
    savefig(f"24_shap_waterfall_{label}.png", tight=False)
    row = X_s.iloc[i]
    case_rows.append({"case": label, "predicted_cancel_prob": round(float(proba[i]), 3),
                      "actually_cancelled": int(y_s.iloc[i]),
                      "lead_time": row["lead_time"], "deposit_type": row["deposit_type"],
                      "market_segment": row["market_segment"], "country": row["country"],
                      "total_of_special_requests": row["total_of_special_requests"],
                      "required_car_parking_spaces": row["required_car_parking_spaces"]})
pd.DataFrame(case_rows).to_csv(C.RESULTS / "06_shap_example_bookings.csv", index=False)

# ---------------------------------------------------------------- 4. compare three importance methods
def group_of(col):
    """map an encoded column back to its original feature (e.g. country_PRT -> country)"""
    for c in sorted(C.CAT_COLS, key=len, reverse=True):
        if col.startswith(c + "_"):
            return c
    return col


shap_grp = mean_abs.groupby(group_of).sum()
rf_imp = pd.Series(rf.feature_importances_, index=names).groupby(group_of).sum()

# permutation importance works on the ORIGINAL columns (the whole pipeline is shuffled)
sub = X_val.sample(4000, random_state=C.SEED)
pi = permutation_importance(pipe, sub, y_val.loc[sub.index], scoring="roc_auc",
                            n_repeats=3, random_state=C.SEED, n_jobs=1)
perm = pd.Series(pi.importances_mean, index=X_val.columns)

comp = pd.DataFrame({"shap_mean_abs": shap_grp, "rf_impurity": rf_imp, "permutation_auc_drop": perm})
for c in list(comp.columns):
    comp[c + "_rank"] = comp[c].rank(ascending=False).astype(int)
comp = comp.sort_values("shap_mean_abs", ascending=False)
comp.round(4).to_csv(C.RESULTS / "06_importance_comparison.csv")
print(comp.head(12).round(4).to_string(), flush=True)

top = comp.head(10)[::-1]
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, col, title, color in zip(axes, ["shap_mean_abs", "rf_impurity", "permutation_auc_drop"],
                                 ["SHAP (mean |value|)", "RF impurity importance", "Permutation (AUC drop)"],
                                 ["#4c72b0", "#55a868", "#c44e52"]):
    ax.barh(top.index, top[col], color=color)
    ax.set_title(title)
savefig("25_importance_methods_compared.png")

# direction summary for the main numeric features (simple table, easy to quote)
dir_rows = []
for feat in ["lead_time", "total_of_special_requests", "required_car_parking_spaces",
             "previous_cancellations", "adr", "total_nights", "is_repeated_guest"]:
    j = names.index(feat)
    corr = np.corrcoef(Z[:, j], sv[:, j])[0, 1]
    dir_rows.append({"feature": feat, "corr_feature_vs_shap": round(float(corr), 3),
                     "meaning": "higher value -> more cancellation risk" if corr > 0.1
                     else ("higher value -> less cancellation risk" if corr < -0.1 else "weak / non-linear")})
pd.DataFrame(dir_rows).to_csv(C.RESULTS / "06_shap_direction_summary.csv", index=False)
print(pd.DataFrame(dir_rows).to_string(index=False), flush=True)
print("Done.")
