"""Step 5 - Extensions: Gradient Boosting, PR-AUC, cost-based threshold, time-based split.

A. Gradient Boosting (HistGradientBoosting) added to the model comparison
B. PR-AUC (average precision) added next to ROC-AUC, because only 27.5 % of bookings are cancelled
C. Time-based split: train on the past, test on the future (2017) - compared with the random split
D. Cost-based threshold: choose the threshold by money-like cost, not by F1
E. Same Gradient Boosting vs Random Forest comparison for price (regression)

Note: the year column is removed here, because a time split cannot use a year
(2017) that never appears in the training data.
"""
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_recall_curve, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (KFold, RandomizedSearchCV, StratifiedKFold,
                                     cross_validate)
from sklearn.pipeline import Pipeline

import config as C
from prep import (MONTHS, build_preprocessor, drop_leaky, get_clean_data, make_splits)
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

# ---------------------------------------------------------------- data
df = get_clean_data()
month_num = df["arrival_date_month"].map({m: i + 1 for i, m in enumerate(MONTHS)})
arrival = pd.to_datetime(dict(year=df["arrival_date_year"], month=month_num,
                              day=df["arrival_date_day_of_month"]))

NUM_CLF = [c for c in C.CLF_NUM_COLS if c != "arrival_date_year"]
y = df["is_canceled"]
X = drop_leaky(df).drop(columns=["is_canceled"])[NUM_CLF + C.BIN_COLS + C.CAT_COLS]

# random split (same recipe as before)
rs = make_splits(X, y, stratify=y)
# time split: past -> train, Sep-Dec 2016 -> validation, 2017 -> test
m_train = arrival < "2016-09-01"
m_val = (arrival >= "2016-09-01") & (arrival < "2017-01-01")
m_test = arrival >= "2017-01-01"
ts = (X[m_train], X[m_val], X[m_test], y[m_train], y[m_val], y[m_test])
log(f"time split sizes: train={m_train.sum()} val={m_val.sum()} test={m_test.sum()}; "
    f"cancel rate train={y[m_train].mean():.3f} val={y[m_val].mean():.3f} test={y[m_test].mean():.3f}")
SPLITS = {"random": rs, "time": ts}


def make_pipe(clf, scale=False):
    return Pipeline([("pre", build_preprocessor(NUM_CLF, scale=scale)), ("clf", clf)])


# ---------------------------------------------------------------- A. tune Gradient Boosting
log("A. tuning HistGradientBoosting (CV on random-split training set only) ...")
X_train, X_val, X_test, y_train, y_val, y_test = rs
hgb_search = RandomizedSearchCV(
    make_pipe(HistGradientBoostingClassifier(class_weight="balanced", early_stopping=True,
                                             random_state=C.SEED)),
    {"clf__learning_rate": [0.03, 0.05, 0.1],
     "clf__max_leaf_nodes": [15, 31, 63],
     "clf__min_samples_leaf": [20, 50, 100],
     "clf__l2_regularization": [0.0, 1.0, 10.0],
     "clf__max_iter": [200, 400]},
    n_iter=8, scoring="average_precision", cv=3, random_state=C.SEED, n_jobs=1)
hgb_search.fit(X_train, y_train)
hgb_params = {k.replace("clf__", ""): v for k, v in hgb_search.best_params_.items()}
save_trials(hgb_search, "05_search_trials_gradient_boosting_clf.csv")
log(f"HGB best: {hgb_params}  CV PR-AUC={hgb_search.best_score_:.4f}")
with open(C.RESULTS / "05_hgb_best_params.json", "w") as f:
    json.dump({k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
               for k, v in hgb_params.items()}, f, indent=2)


def model_factory(name):
    if name == "Logistic Regression":
        return make_pipe(LogisticRegression(C=10, class_weight="balanced", max_iter=1000), scale=True)
    if name == "Random Forest":
        return make_pipe(RandomForestClassifier(
            n_estimators=100, max_depth=20, min_samples_leaf=5, max_features=0.3,
            class_weight="balanced_subsample", random_state=C.SEED, n_jobs=1))
    if name == "Gradient Boosting":
        return make_pipe(HistGradientBoostingClassifier(
            class_weight="balanced", early_stopping=True, random_state=C.SEED, **hgb_params))
    raise ValueError(name)


MODELS = ["Logistic Regression", "Random Forest", "Gradient Boosting"]

# ---------------------------------------------------------------- A+B. cross-validation with PR-AUC
log("A+B. 5-fold CV on random-split training set (ROC-AUC, PR-AUC, F1) ...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.SEED)
cv_rows = []
for name in MODELS:
    res = cross_validate(model_factory(name), X_train, y_train, cv=cv, n_jobs=1,
                         scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1"})
    row = {"model": name}
    for m in ["roc_auc", "pr_auc", "f1"]:
        row[f"{m}_mean"] = res[f"test_{m}"].mean()
        row[f"{m}_std"] = res[f"test_{m}"].std()
    cv_rows.append(row)
    log(f"CV {name:20s} ROC-AUC={row['roc_auc_mean']:.3f}  PR-AUC={row['pr_auc_mean']:.3f}  F1={row['f1_mean']:.3f}")
cvdf = pd.DataFrame(cv_rows)
complexity = {"Logistic Regression": 1, "Random Forest": 3, "Gradient Boosting": 3}
best = cvdf.loc[cvdf["pr_auc_mean"].idxmax()]
cvdf["within_1_std_of_best"] = cvdf["pr_auc_mean"] >= best["pr_auc_mean"] - best["pr_auc_std"]
cvdf["complexity_rank"] = cvdf["model"].map(complexity)
elig = cvdf[cvdf["within_1_std_of_best"]].sort_values(["complexity_rank", "pr_auc_mean"],
                                                      ascending=[True, False])
chosen = elig.iloc[0]["model"]
cvdf["chosen"] = cvdf["model"] == chosen
cvdf.round(4).to_csv(C.RESULTS / "05_cv_with_prauc.csv", index=False)
log(f"Chosen by PR-AUC + simplicity rule: {chosen}")

# ---------------------------------------------------------------- B + C. random vs time split
def metrics(y_true, p, t=0.5):
    pred = (p >= t).astype(int)
    return {"accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred), "f1": f1_score(y_true, pred),
            "roc_auc": roc_auc_score(y_true, p), "pr_auc": average_precision_score(y_true, p)}


fitted, proba = {}, {}
rows = []
for split_name, (Xtr, Xva, Xte, ytr, yva, yte) in SPLITS.items():
    for name in MODELS:
        m = model_factory(name).fit(Xtr, ytr)
        fitted[(split_name, name)] = m
        pv, pt = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
        proba[(split_name, name)] = (pv, pt)
        r = {"split": split_name, "model": name, "test_prevalence": yte.mean()}
        r.update({f"test_{k}": v for k, v in metrics(yte, pt).items()})
        rows.append(r)
        log(f"{split_name:6s} split | {name:20s} test ROC-AUC={r['test_roc_auc']:.3f} "
            f"PR-AUC={r['test_pr_auc']:.3f} F1={r['test_f1']:.3f}")
split_df = pd.DataFrame(rows)
split_df.round(4).to_csv(C.RESULTS / "05_split_comparison_classification.csv", index=False)

# PR curves (random split, test)
plt.figure(figsize=(6.5, 5.5))
for name in MODELS:
    p = proba[("random", name)][1]
    pr, rc, _ = precision_recall_curve(y_test, p)
    plt.plot(rc, pr, label=f"{name} (PR-AUC={average_precision_score(y_test, p):.3f})")
plt.axhline(y_test.mean(), color="k", ls=":", label=f"Random guess ({y_test.mean():.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall curves - random split, test set")
plt.legend(fontsize=8)
savefig("16_pr_curves.png")

# random vs time split bars
fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=False)
for ax, metric in zip(axes, ["test_roc_auc", "test_pr_auc", "test_f1"]):
    piv = split_df.pivot(index="model", columns="split", values=metric).loc[MODELS]
    piv.plot.bar(ax=ax, color=["#4c72b0", "#c44e52"], rot=15)
    ax.set_title(metric.replace("test_", "").upper().replace("_", "-") + " (test)")
    ax.set_xlabel("")
    ax.set_ylim(piv.values.min() - 0.1, min(1.0, piv.values.max() + 0.05))
savefig("17_random_vs_time_split.png")

# ---------------------------------------------------------------- D. cost-based threshold
COST_FN = 1.0          # missed cancellation = room stays empty (lost revenue) -> cost 1
COST_FP_MAIN = 2.0     # false alarm = guest comes but room was resold -> relocation, 2x worse (assumption)
grid = np.round(np.arange(0.05, 0.96, 0.01), 2)


def cost_per_1000(y_true, p, t, c_fp):
    pred = (p >= t).astype(int)
    fn = ((y_true == 1) & (pred == 0)).sum()
    fp = ((y_true == 0) & (pred == 1)).sum()
    return (COST_FN * fn + c_fp * fp) / len(y_true) * 1000, int(fn), int(fp)


sens_rows, cmp_rows = [], []
for split_name in ["random", "time"]:
    _, _, _, _, yva, yte = SPLITS[split_name]
    pv, pt = proba[(split_name, chosen)]
    # F1-optimal threshold on validation
    f1s = [f1_score(yva, (pv >= t).astype(int)) for t in grid]
    t_f1 = float(grid[int(np.argmax(f1s))])
    for c_fp in [0.5, 1.0, 2.0, 4.0]:
        costs = [cost_per_1000(yva, pv, t, c_fp)[0] for t in grid]
        t_cost = float(grid[int(np.argmin(costs))])
        sens_rows.append({"split": split_name, "cost_false_alarm": c_fp, "cost_missed_cancel": COST_FN,
                          "best_threshold_on_validation": t_cost})
        if c_fp == COST_FP_MAIN:
            t_main = t_cost
    for label, t in [("default 0.50", 0.5), (f"F1-optimal ({t_f1:.2f})", t_f1),
                     (f"cost-optimal ({t_main:.2f})", t_main)]:
        c, fn, fp = cost_per_1000(yte, pt, t, COST_FP_MAIN)
        mm = metrics(yte, pt, t)
        cmp_rows.append({"split": split_name, "threshold_rule": label, "threshold": t,
                         "missed_cancellations_FN": fn, "false_alarms_FP": fp,
                         "cost_per_1000_bookings": round(c, 1),
                         "precision": round(mm["precision"], 3), "recall": round(mm["recall"], 3),
                         "f1": round(mm["f1"], 3)})
    if split_name == "random":
        plt.figure(figsize=(7, 4.5))
        for c_fp in [0.5, 1.0, 2.0, 4.0]:
            plt.plot(grid, [cost_per_1000(yva, pv, t, c_fp)[0] for t in grid], label=f"false alarm costs {c_fp}x")
        plt.axvline(0.5, color="grey", ls=":")
        plt.xlabel("Decision threshold (validation set)")
        plt.ylabel("Cost per 1000 bookings")
        plt.title(f"Cost vs threshold - {chosen}")
        plt.legend()
        savefig("18_cost_vs_threshold.png")
pd.DataFrame(sens_rows).to_csv(C.RESULTS / "05_cost_threshold_sensitivity.csv", index=False)
cmp_df = pd.DataFrame(cmp_rows)
cmp_df.to_csv(C.RESULTS / "05_cost_threshold_test_comparison.csv", index=False)
log("D. cost-based threshold results:")
print(cmp_df.to_string(index=False), flush=True)

# ---------------------------------------------------------------- E. regression: GB vs RF, both splits
log("E. regression (price) with Gradient Boosting ...")
dfr = df[(df["adr"] > 0) & (df["adr"] <= 500)].copy()
arr_r = arrival[dfr.index]
dfr["hotel_month"] = dfr["hotel"] + "_" + dfr["arrival_date_month"]
NUM_R = [c for c in C.NUM_COLS if c != "arrival_date_year"]
CAT_R = C.CAT_COLS + ["hotel_month"]
yr = dfr["adr"]
Xr = drop_leaky(dfr).drop(columns=["is_canceled", "adr"])[NUM_R + C.BIN_COLS + CAT_R]
rsr = make_splits(Xr, yr)
tr_m, va_m, te_m = arr_r < "2016-09-01", (arr_r >= "2016-09-01") & (arr_r < "2017-01-01"), arr_r >= "2017-01-01"
tsr = (Xr[tr_m], Xr[va_m], Xr[te_m], yr[tr_m], yr[va_m], yr[te_m])


def rpipe(reg, scale=False):
    return Pipeline([("pre", build_preprocessor(NUM_R, scale=scale, cat_cols=CAT_R)), ("reg", reg)])


hgbr_search = RandomizedSearchCV(
    rpipe(HistGradientBoostingRegressor(early_stopping=True, random_state=C.SEED)),
    {"reg__learning_rate": [0.03, 0.05, 0.1], "reg__max_leaf_nodes": [15, 31, 63],
     "reg__min_samples_leaf": [20, 50, 100], "reg__l2_regularization": [0.0, 1.0, 10.0],
     "reg__max_iter": [300, 600]},
    n_iter=8, scoring="neg_root_mean_squared_error",
    cv=KFold(3, shuffle=True, random_state=C.SEED), random_state=C.SEED, n_jobs=1)
hgbr_search.fit(rsr[0], rsr[3])
hgbr_params = {k.replace("reg__", ""): v for k, v in hgbr_search.best_params_.items()}
save_trials(hgbr_search, "05_search_trials_gradient_boosting_reg.csv")
log(f"HGB regressor best: {hgbr_params}  CV RMSE={-hgbr_search.best_score_:.2f}")


def reg_factory(name):
    if name == "Ridge (hotel x month)":
        return rpipe(Ridge(alpha=43.3), scale=True)
    if name == "Random Forest":
        return rpipe(RandomForestRegressor(n_estimators=100, min_samples_leaf=3, max_features=0.4,
                                           random_state=C.SEED, n_jobs=1))
    return rpipe(HistGradientBoostingRegressor(early_stopping=True, random_state=C.SEED, **hgbr_params))


reg_rows = []
for split_name, (Xtr, Xva, Xte, ytr, yva, yte) in {"random": rsr, "time": tsr}.items():
    for name in ["Ridge (hotel x month)", "Random Forest", "Gradient Boosting"]:
        m = reg_factory(name).fit(Xtr, ytr)
        p = m.predict(Xte)
        reg_rows.append({"split": split_name, "model": name,
                         "test_mae": mean_absolute_error(yte, p),
                         "test_rmse": mean_squared_error(yte, p) ** 0.5,
                         "test_r2": r2_score(yte, p),
                         "bias_mean_residual": float((yte - p).mean())})
        log(f"{split_name:6s} | {name:22s} MAE={reg_rows[-1]['test_mae']:.2f} "
            f"RMSE={reg_rows[-1]['test_rmse']:.2f} R2={reg_rows[-1]['test_r2']:.3f} "
            f"bias={reg_rows[-1]['bias_mean_residual']:.2f}")
regdf = pd.DataFrame(reg_rows)
regdf.round(4).to_csv(C.RESULTS / "05_regression_split_comparison.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
for ax, metric in zip(axes, ["test_mae", "test_r2"]):
    piv = regdf.pivot(index="model", columns="split", values=metric).loc[
        ["Ridge (hotel x month)", "Random Forest", "Gradient Boosting"]]
    piv.plot.bar(ax=ax, color=["#4c72b0", "#c44e52"], rot=15)
    ax.set_title(metric.replace("test_", "").upper() + " (test)")
    ax.set_xlabel("")
savefig("19_regression_random_vs_time.png")
log("Done.")
