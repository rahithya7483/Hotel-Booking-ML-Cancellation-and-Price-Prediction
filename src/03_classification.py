"""Step 3 - Classification: will a booking be cancelled?

Workflow
  A. split 60/20/20 (stratified)            F. model selection (metric + simplicity)
  B. dummy baseline + simple baselines      G. threshold tuning on VALIDATION set
  C. 5-fold CV comparison (round 1)         H. one final look at the TEST set
  D. imbalance check (class_weight)         I. error analysis + feature importance
  E. hyper-parameter tuning (round 2)
"""
import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import config as C
from prep import build_preprocessor, drop_leaky, get_clean_data, make_splits
from utils import savefig

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)




def save_trials(search, filename):
    """Save every hyper-parameter combination that was tried, with its CV score."""
    t = pd.DataFrame(search.cv_results_)
    keep = [c for c in t.columns if c.startswith("param_")] + [
        "mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time"]
    t = t[keep].rename(columns=lambda c: c.replace("param_clf__", "").replace("param_reg__", ""))
    t.sort_values("rank_test_score").round(4).to_csv(C.RESULTS / filename, index=False)

# ---------------------------------------------------------------- A. data/split
df = get_clean_data()
y = df["is_canceled"]
X = drop_leaky(df).drop(columns=["is_canceled"])
NUM = C.CLF_NUM_COLS
X = X[NUM + C.BIN_COLS + C.CAT_COLS]          # keep only the planned features

X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y, stratify=y)
log(f"rows: train={len(X_train)} val={len(X_val)} test={len(X_test)}  "
    f"cancel rate train={y_train.mean():.3f} val={y_val.mean():.3f} test={y_test.mean():.3f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.SEED)
SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
           "f1": "f1", "roc_auc": "roc_auc"}


def make_pipe(clf, scale):
    return Pipeline([("pre", build_preprocessor(NUM, scale=scale)), ("clf", clf)])


def cv_table(models):
    rows = []
    for name, pipe in models.items():
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1)
        row = {"model": name}
        for m in SCORING:
            row[f"{m}_mean"] = res[f"test_{m}"].mean()
            row[f"{m}_std"] = res[f"test_{m}"].std()
        rows.append(row)
        log(f"CV done: {name:32s} F1={row['f1_mean']:.3f}  AUC={row['roc_auc_mean']:.3f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- B + C + D. round 1
round1 = {
    "Dummy (most frequent)":        make_pipe(DummyClassifier(strategy="most_frequent"), True),
    "Logistic Regression":          make_pipe(LogisticRegression(max_iter=1000), True),
    "Logistic Reg. (balanced)":     make_pipe(LogisticRegression(max_iter=1000, class_weight="balanced"), True),
    "Decision Tree (depth 6)":      make_pipe(DecisionTreeClassifier(max_depth=6, random_state=C.SEED), False),
    "Random Forest (default-ish)":  make_pipe(RandomForestClassifier(
        n_estimators=100, min_samples_leaf=3, max_features="sqrt",
        random_state=C.SEED, n_jobs=1), False),
    "Random Forest (balanced)":     make_pipe(RandomForestClassifier(
        n_estimators=100, min_samples_leaf=3, max_features="sqrt",
        class_weight="balanced_subsample", random_state=C.SEED, n_jobs=1), False),
}
log("Round 1: 5-fold CV on the training set ...")
r1 = cv_table(round1)
r1.round(4).to_csv(C.RESULTS / "03_cv_round1_baselines.csv", index=False)

# ---------------------------------------------------------------- E. tuning
log("Round 2: hyper-parameter tuning (CV on training set only) ...")
best_params = {}

lr_search = GridSearchCV(
    make_pipe(LogisticRegression(max_iter=1000, class_weight="balanced"), True),
    {"clf__C": [0.01, 0.1, 1, 10]}, scoring="roc_auc", cv=3, n_jobs=1)
lr_search.fit(X_train, y_train)
best_params["Logistic Regression"] = lr_search.best_params_
save_trials(lr_search, "03_search_trials_logreg.csv")
log(f"LR best: {lr_search.best_params_}  AUC={lr_search.best_score_:.4f}")

dt_search = GridSearchCV(
    make_pipe(DecisionTreeClassifier(random_state=C.SEED), False),
    {"clf__max_depth": [4, 6, 8, 12, 16], "clf__min_samples_leaf": [5, 20, 50]},
    scoring="roc_auc", cv=3, n_jobs=1)
dt_search.fit(X_train, y_train)
best_params["Decision Tree"] = dt_search.best_params_
save_trials(dt_search, "03_search_trials_tree.csv")
log(f"DT best: {dt_search.best_params_}  AUC={dt_search.best_score_:.4f}")

rf_search = RandomizedSearchCV(
    make_pipe(RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample",
                                     random_state=C.SEED, n_jobs=1), False),
    {"clf__max_depth": [12, 16, 20, None],
     "clf__min_samples_leaf": [2, 5, 10],
     "clf__max_features": ["sqrt", 0.3]},
    n_iter=6, scoring="roc_auc", cv=3, random_state=C.SEED, n_jobs=1)
rf_search.fit(X_train, y_train)
best_params["Random Forest"] = rf_search.best_params_
save_trials(rf_search, "03_search_trials_forest.csv")
log(f"RF best: {rf_search.best_params_}  AUC={rf_search.best_score_:.4f}")

with open(C.RESULTS / "03_best_params.json", "w") as f:
    json.dump({k: {a: (None if b is None else b) for a, b in v.items()}
               for k, v in best_params.items()}, f, indent=2)

tuned = {
    "Logistic Regression (tuned)": lr_search.best_estimator_,
    "Decision Tree (tuned)": dt_search.best_estimator_,
    "Random Forest (tuned)": rf_search.best_estimator_,
}
log("Round 2: 5-fold CV of the tuned models ...")
r2 = cv_table(tuned)
r2.round(4).to_csv(C.RESULTS / "03_cv_round2_tuned.csv", index=False)

# ---------------------------------------------------------------- F. selection
# Rule: pick the SIMPLEST model whose CV ROC-AUC is within one standard
# deviation of the best model (a "one standard error"-style rule).
complexity = {"Logistic Regression (tuned)": 1, "Decision Tree (tuned)": 2,
              "Random Forest (tuned)": 3}
best_row = r2.loc[r2["roc_auc_mean"].idxmax()]
cutoff = best_row["roc_auc_mean"] - best_row["roc_auc_std"]
r2["complexity_rank"] = r2["model"].map(complexity)
r2["within_1_std_of_best"] = r2["roc_auc_mean"] >= cutoff
eligible = r2[r2["within_1_std_of_best"]].sort_values("complexity_rank")
chosen_name = eligible.iloc[0]["model"]
r2["chosen"] = r2["model"] == chosen_name
r2.round(4).to_csv(C.RESULTS / "03_model_selection.csv", index=False)
log(f"Selected model: {chosen_name}  (AUC cutoff = {cutoff:.4f})")

# plot CV comparison (round 1 + 2)
allcv = pd.concat([r1, r2[r1.columns]], ignore_index=True)
plot_df = allcv[allcv["model"] != "Dummy (most frequent)"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, metric in zip(axes, ["f1", "roc_auc"]):
    ax.barh(plot_df["model"], plot_df[f"{metric}_mean"], xerr=plot_df[f"{metric}_std"],
            color="#4c72b0", capsize=3)
    ax.set_title(f"5-fold CV {metric.upper().replace('_', '-')} (mean +/- std)")
    ax.set_xlim(plot_df[f"{metric}_mean"].min() - 0.08, 1.0)
savefig("04_cv_model_comparison.png")

# ---------------------------------------------------------------- G. threshold
final_models = dict(tuned)
final_models["Dummy (most frequent)"] = round1["Dummy (most frequent)"].fit(X_train, y_train)
final_models["Logistic Regression (default, no balancing)"] = round1["Logistic Regression"].fit(X_train, y_train)

best_pipe = tuned[chosen_name]
val_proba = best_pipe.predict_proba(X_val)[:, 1]
grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
rows = []
for t in grid:
    pred = (val_proba >= t).astype(int)
    rows.append({"threshold": t,
                 "precision": precision_score(y_val, pred, zero_division=0),
                 "recall": recall_score(y_val, pred),
                 "f1": f1_score(y_val, pred)})
thr = pd.DataFrame(rows)
best_t = float(thr.loc[thr["f1"].idxmax(), "threshold"])
thr.round(4).to_csv(C.RESULTS / "03_threshold_table_validation.csv", index=False)
log(f"Best threshold on validation (max F1): {best_t}")

plt.figure(figsize=(7, 4.5))
plt.plot(thr["threshold"], thr["precision"], label="Precision")
plt.plot(thr["threshold"], thr["recall"], label="Recall")
plt.plot(thr["threshold"], thr["f1"], label="F1", lw=2.5)
plt.axvline(0.5, color="grey", ls=":", label="default 0.5")
plt.axvline(best_t, color="red", ls="--", label=f"tuned {best_t:.2f}")
plt.xlabel("Decision threshold (validation set)")
plt.title(f"Threshold tuning - {chosen_name}")
plt.legend()
savefig("05_threshold_tuning.png")

# ---------------------------------------------------------------- H. final test
def metrics_at(model, X_, y_, t=0.5):
    p = model.predict_proba(X_)[:, 1]
    pred = (p >= t).astype(int)
    return {"accuracy": accuracy_score(y_, pred),
            "precision": precision_score(y_, pred, zero_division=0),
            "recall": recall_score(y_, pred),
            "f1": f1_score(y_, pred),
            "roc_auc": roc_auc_score(y_, p)}


journey_rows = []
steps = [
    ("1. Dummy baseline (always 'not cancelled')", final_models["Dummy (most frequent)"], 0.5),
    ("2. Logistic Regression (default)", final_models["Logistic Regression (default, no balancing)"], 0.5),
    ("3. Logistic Regression (tuned, balanced)", tuned["Logistic Regression (tuned)"], 0.5),
    ("4. Decision Tree (tuned)", tuned["Decision Tree (tuned)"], 0.5),
    ("5. Random Forest (tuned, balanced)", tuned["Random Forest (tuned)"], 0.5),
    (f"6. {chosen_name} + tuned threshold ({best_t:.2f})", best_pipe, best_t),
]
for label, model, t in steps:
    r = {"step": label, "threshold": t}
    r.update({f"val_{k}": v for k, v in metrics_at(model, X_val, y_val, t).items()})
    r.update({f"test_{k}": v for k, v in metrics_at(model, X_test, y_test, t).items()})
    journey_rows.append(r)
journey = pd.DataFrame(journey_rows)
journey.round(4).to_csv(C.RESULTS / "03_improvement_journey.csv", index=False)
log("Final test results:")
print(journey[["step", "threshold", "test_accuracy", "test_precision", "test_recall",
               "test_f1", "test_roc_auc"]].round(3).to_string(index=False), flush=True)

# improvement bar chart on test F1 / recall / precision
fig, ax = plt.subplots(figsize=(10, 4.8))
x = np.arange(len(journey))
w = 0.27
ax.bar(x - w, journey["test_precision"], w, label="Precision")
ax.bar(x, journey["test_recall"], w, label="Recall")
ax.bar(x + w, journey["test_f1"], w, label="F1")
ax.set_xticks(x)
ax.set_xticklabels([f"step {i + 1}" for i in range(len(journey))])
ax.set_ylim(0, 1)
ax.set_title("Improvement journey (test set) - see README for step names")
ax.legend()
savefig("06_improvement_journey.png")

# ROC curves on test
plt.figure(figsize=(6, 5.5))
for name in ["Logistic Regression (tuned)", "Decision Tree (tuned)", "Random Forest (tuned)"]:
    p = tuned[name].predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, p):.3f})")
plt.plot([0, 1], [0, 1], "k:", label="Random guess")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curves - test set")
plt.legend(loc="lower right", fontsize=8)
savefig("07_roc_curves_test.png")

# confusion matrices: threshold 0.5 vs tuned
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
test_proba = best_pipe.predict_proba(X_test)[:, 1]
for ax, t in zip(axes, [0.5, best_t]):
    cm = confusion_matrix(y_test, (test_proba >= t).astype(int))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: stay", "Pred: cancel"])
    ax.set_yticklabels(["Actual: stay", "Actual: cancel"])
    ax.set_title(f"threshold = {t:.2f}")
    ax.grid(False)
savefig("08_confusion_matrix_threshold.png")

# ---------------------------------------------------------------- I. error analysis
test_df = X_test.copy()
test_df["actual"] = y_test.values
test_df["proba"] = test_proba
test_df["pred"] = (test_proba >= best_t).astype(int)
test_df["error"] = np.where((test_df["actual"] == 1) & (test_df["pred"] == 0), "missed cancellation (FN)",
                    np.where((test_df["actual"] == 0) & (test_df["pred"] == 1), "false alarm (FP)", "correct"))
seg_rows = []
for col in ["deposit_type", "market_segment", "hotel", "customer_type"]:
    g = test_df.groupby(col).apply(
        lambda d: pd.Series({"n": len(d), "accuracy": (d["actual"] == d["pred"]).mean(),
                             "fn_rate_of_cancels": ((d["error"] == "missed cancellation (FN)").sum()
                                                    / max((d["actual"] == 1).sum(), 1)),
                             "cancel_rate": d["actual"].mean()}),
        include_groups=False).reset_index().rename(columns={col: "group"})
    g.insert(0, "feature", col)
    seg_rows.append(g)
seg = pd.concat(seg_rows, ignore_index=True)
seg.round(4).to_csv(C.RESULTS / "03_error_analysis_by_segment.csv", index=False)
test_df.groupby("error").size().rename("count").to_csv(C.RESULTS / "03_error_counts.csv")

# feature importance (Random Forest) and LR coefficients
pre = tuned["Random Forest (tuned)"].named_steps["pre"]
names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
imp = pd.Series(tuned["Random Forest (tuned)"].named_steps["clf"].feature_importances_,
                index=names).sort_values(ascending=False)
imp.head(20).round(4).to_csv(C.RESULTS / "03_rf_feature_importance_top20.csv", header=["importance"])
plt.figure(figsize=(7, 6))
imp.head(15)[::-1].plot.barh(color="#55a868")
plt.title("Random Forest - top 15 features")
savefig("09_feature_importance_rf.png")

lr_pre = tuned["Logistic Regression (tuned)"].named_steps["pre"]
lr_names = [n.split("__", 1)[1] for n in lr_pre.get_feature_names_out()]
coef = pd.Series(tuned["Logistic Regression (tuned)"].named_steps["clf"].coef_[0], index=lr_names)
top = pd.concat([coef.sort_values().head(8), coef.sort_values().tail(8)])
plt.figure(figsize=(7, 6))
top.plot.barh(color=["#c44e52" if v > 0 else "#4c72b0" for v in top.values])
plt.title("Logistic Regression - strongest coefficients\n(+ = more likely to cancel)")
savefig("10_logreg_coefficients.png")

# save the chosen model
joblib.dump({"model": best_pipe, "threshold": best_t, "name": chosen_name,
             "features": list(X.columns)}, C.MODELS / "best_classifier.joblib", compress=3)
log("Saved model + all result files. Done.")
