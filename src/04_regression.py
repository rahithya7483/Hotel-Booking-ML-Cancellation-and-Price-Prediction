"""Step 4 - Regression: predict the price per night (ADR) of a booking.

Workflow
  A. clean target (0 < adr <= 500) and split 60/20/20
  B. dummy baseline + Linear / Ridge / Lasso / Tree / Forest, 5-fold CV
  C. regularization: alpha sweep on the validation set (Ridge + Lasso)
  D. improvements: log-target, simple feature (hotel x month)
  E. model selection (metric + simplicity), one final look at TEST
  F. error analysis (residuals, worst predictions, error by segment)
"""
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

import config as C
from prep import build_preprocessor, drop_leaky, get_clean_data, make_splits
from utils import savefig

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------- A. data
df = get_clean_data()
n_before = len(df)
df = df[(df["adr"] > 0) & (df["adr"] <= 500)].reset_index(drop=True)
log(f"kept {len(df)} of {n_before} rows after removing adr<=0 (free/negative) and adr>500 (outliers)")

# simple extra feature: price level depends on hotel type AND season
df["hotel_month"] = df["hotel"] + "_" + df["arrival_date_month"]

y = df["adr"]
base_feats = drop_leaky(df).drop(columns=["is_canceled", "adr"])
NUM = C.NUM_COLS                       # adr is the target, so not in the features
CAT_BASE = C.CAT_COLS
CAT_PLUS = C.CAT_COLS + ["hotel_month"]
X = base_feats[NUM + C.BIN_COLS + CAT_PLUS]

X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)
log(f"rows: train={len(X_train)} val={len(X_val)} test={len(X_test)}")

cv = KFold(n_splits=5, shuffle=True, random_state=C.SEED)
SCORING = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}


def make_pipe(reg, scale=True, cats=CAT_BASE, log_target=False):
    pipe = Pipeline([("pre", build_preprocessor(NUM, scale=scale, cat_cols=cats)), ("reg", reg)])
    if log_target:
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    return pipe


def cv_table(models):
    rows = []
    for name, pipe in models.items():
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1)
        row = {"model": name,
               "mae_mean": -res["test_mae"].mean(), "mae_std": res["test_mae"].std(),
               "rmse_mean": -res["test_rmse"].mean(), "rmse_std": res["test_rmse"].std(),
               "r2_mean": res["test_r2"].mean(), "r2_std": res["test_r2"].std()}
        rows.append(row)
        log(f"CV done: {name:34s} MAE={row['mae_mean']:.2f} RMSE={row['rmse_mean']:.2f} R2={row['r2_mean']:.3f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- B. round 1
round1 = {
    "Dummy (predict mean)":      make_pipe(DummyRegressor(strategy="mean")),
    "Linear Regression":         make_pipe(LinearRegression()),
    "Ridge (alpha=10)":          make_pipe(Ridge(alpha=10)),
    "Lasso (alpha=0.01)":        make_pipe(Lasso(alpha=0.01, max_iter=5000)),
    "Decision Tree (depth 10)":  make_pipe(DecisionTreeRegressor(max_depth=10, min_samples_leaf=10,
                                                                 random_state=C.SEED), scale=False),
    "Random Forest":             make_pipe(RandomForestRegressor(n_estimators=100, min_samples_leaf=3,
                                                                 max_features=0.4, random_state=C.SEED,
                                                                 n_jobs=1), scale=False),
}
log("Round 1: 5-fold CV on the training set ...")
r1 = cv_table(round1)
r1.round(4).to_csv(C.RESULTS / "04_cv_round1_baselines.csv", index=False)

# ---------------------------------------------------------------- C. regularization sweep
log("Regularization: alpha sweep on the validation set ...")
pre = build_preprocessor(NUM, scale=True, cat_cols=CAT_BASE).fit(X_train)
Xtr, Xva = pre.transform(X_train), pre.transform(X_val)
ridge_alphas = np.logspace(-2, 3, 12)
lasso_alphas = np.logspace(-4, 0, 12)
sweep = []
for a in ridge_alphas:
    m = Ridge(alpha=a).fit(Xtr, y_train)
    sweep.append({"model": "Ridge", "alpha": a,
                  "val_rmse": mean_squared_error(y_val, m.predict(Xva)) ** 0.5,
                  "nonzero_coefs": int((np.abs(m.coef_) > 1e-8).sum())})
for a in lasso_alphas:
    m = Lasso(alpha=a, max_iter=10000).fit(Xtr, y_train)
    sweep.append({"model": "Lasso", "alpha": a,
                  "val_rmse": mean_squared_error(y_val, m.predict(Xva)) ** 0.5,
                  "nonzero_coefs": int((np.abs(m.coef_) > 1e-8).sum())})
sweep = pd.DataFrame(sweep)
sweep.round(5).to_csv(C.RESULTS / "04_regularization_sweep.csv", index=False)
best_ridge_alpha = float(sweep[sweep.model == "Ridge"].sort_values("val_rmse").iloc[0]["alpha"])
best_lasso_alpha = float(sweep[sweep.model == "Lasso"].sort_values("val_rmse").iloc[0]["alpha"])
log(f"best Ridge alpha={best_ridge_alpha:.4g}  best Lasso alpha={best_lasso_alpha:.4g}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for name, color in [("Ridge", "#4c72b0"), ("Lasso", "#c44e52")]:
    s = sweep[sweep.model == name]
    axes[0].plot(s["alpha"], s["val_rmse"], "o-", label=name, color=color)
axes[0].set_xscale("log")
axes[0].set_xlabel("alpha (regularization strength)")
axes[0].set_ylabel("Validation RMSE")
axes[0].set_title("Regularization sweep")
axes[0].legend()
s = sweep[sweep.model == "Lasso"]
axes[1].plot(s["alpha"], s["nonzero_coefs"], "o-", color="#c44e52")
axes[1].set_xscale("log")
axes[1].set_xlabel("alpha")
axes[1].set_ylabel("Number of non-zero coefficients")
axes[1].set_title("Lasso removes features as alpha grows")
savefig("11_regularization_sweep.png")

# ---------------------------------------------------------------- D. improvements
round2 = {
    "Linear Regression":                      make_pipe(LinearRegression()),
    "Ridge (tuned alpha)":                    make_pipe(Ridge(alpha=best_ridge_alpha)),
    "Lasso (tuned alpha)":                    make_pipe(Lasso(alpha=best_lasso_alpha, max_iter=10000)),
    "Ridge + log target":                     make_pipe(Ridge(alpha=best_ridge_alpha), log_target=True),
    "Ridge + log target + hotel x month":     make_pipe(Ridge(alpha=best_ridge_alpha), cats=CAT_PLUS,
                                                        log_target=True),
    "Random Forest + hotel x month":          make_pipe(RandomForestRegressor(
        n_estimators=100, min_samples_leaf=3, max_features=0.4, random_state=C.SEED, n_jobs=1),
        scale=False, cats=CAT_PLUS),
    "Random Forest + hotel x month + log target": make_pipe(RandomForestRegressor(
        n_estimators=100, min_samples_leaf=3, max_features=0.4, random_state=C.SEED, n_jobs=1),
        scale=False, cats=CAT_PLUS, log_target=True),
}
log("Round 2: improvements, 5-fold CV ...")
r2 = cv_table(round2)

# ---------------------------------------------------------------- E. selection
complexity = {"Linear Regression": 1, "Ridge (tuned alpha)": 1, "Lasso (tuned alpha)": 1,
              "Ridge + log target": 2, "Ridge + log target + hotel x month": 2,
              "Random Forest + hotel x month": 3, "Random Forest + hotel x month + log target": 3}
best_row = r2.loc[r2["rmse_mean"].idxmin()]
cutoff = best_row["rmse_mean"] + best_row["rmse_std"]
r2["complexity_rank"] = r2["model"].map(complexity)
r2["within_1_std_of_best"] = r2["rmse_mean"] <= cutoff
eligible = r2[r2["within_1_std_of_best"]].sort_values(["complexity_rank", "rmse_mean"])
chosen_name = eligible.iloc[0]["model"]
r2["chosen"] = r2["model"] == chosen_name
r2.round(4).to_csv(C.RESULTS / "04_cv_round2_improvements.csv", index=False)
log(f"Selected model: {chosen_name} (RMSE cutoff {cutoff:.2f})")

allcv = pd.concat([r1[r1.model.isin(["Dummy (predict mean)", "Decision Tree (depth 10)", "Random Forest"])],
                   r2[r1.columns]], ignore_index=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, metric, col in [(axes[0], "mae", "#4c72b0"), (axes[1], "rmse", "#dd8452")]:
    ax.barh(allcv["model"], allcv[f"{metric}_mean"], xerr=allcv[f"{metric}_std"], color=col, capsize=3)
    ax.set_title(f"5-fold CV {metric.upper()} (lower is better)")
savefig("12_regression_cv_comparison.png")

# final fit on TRAIN only, evaluate on val + test
def evaluate(model):
    out = {}
    for tag, Xs, ys in [("val", X_val, y_val), ("test", X_test, y_test)]:
        p = model.predict(Xs)
        out[f"{tag}_mae"] = mean_absolute_error(ys, p)
        out[f"{tag}_rmse"] = mean_squared_error(ys, p) ** 0.5
        out[f"{tag}_r2"] = r2_score(ys, p)
    return out


journey_steps = [
    ("1. Dummy baseline (predict mean)", round1["Dummy (predict mean)"]),
    ("2. Linear Regression", round2["Linear Regression"]),
    ("3. Ridge (tuned alpha)", round2["Ridge (tuned alpha)"]),
    ("4. Ridge + log target", round2["Ridge + log target"]),
    ("5. Ridge + log target + hotel x month", round2["Ridge + log target + hotel x month"]),
    ("6. Random Forest + hotel x month", round2["Random Forest + hotel x month"]),
    ("7. Random Forest + hotel x month + log target", round2["Random Forest + hotel x month + log target"]),
]
journey, fitted = [], {}
for label, model in journey_steps:
    model.fit(X_train, y_train)
    fitted[label] = model
    row = {"step": label}
    row.update(evaluate(model))
    journey.append(row)
    log(f"fitted + evaluated: {label}")
journey = pd.DataFrame(journey)
journey.round(4).to_csv(C.RESULTS / "04_improvement_journey.csv", index=False)
print(journey[["step", "test_mae", "test_rmse", "test_r2"]].round(3).to_string(index=False), flush=True)

lbl = [f"step {i + 1}" for i in range(len(journey))]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
axes[0].bar(lbl, journey["test_mae"], color="#4c72b0")
axes[0].set_title("Test MAE per step (lower is better)")
axes[1].bar(lbl, journey["test_r2"], color="#55a868")
axes[1].set_title("Test R2 per step (higher is better)")
for ax in axes:
    ax.tick_params(axis="x", rotation=45)
savefig("13_regression_journey.png")

# Lasso: which features survive?
lasso_pre = build_preprocessor(NUM, scale=True, cat_cols=CAT_BASE).fit(X_train)
lasso = Lasso(alpha=best_lasso_alpha, max_iter=10000).fit(lasso_pre.transform(X_train), y_train)
names = [n.split("__", 1)[1] for n in lasso_pre.get_feature_names_out()]
coef = pd.Series(lasso.coef_, index=names)
coef[coef.abs() > 1e-8].sort_values(key=abs, ascending=False).round(3).head(25).to_csv(
    C.RESULTS / "04_lasso_top_coefficients.csv", header=["coef"])
log(f"Lasso keeps {(coef.abs() > 1e-8).sum()} of {len(coef)} features")

# ---------------------------------------------------------------- F. error analysis
chosen_label = [l for l, _ in journey_steps if chosen_name in l][0]
best_model = fitted[chosen_label]
pred = best_model.predict(X_test)
res = pd.DataFrame({"actual": y_test.values, "pred": pred})
res["residual"] = res["actual"] - res["pred"]
res["abs_err"] = res["residual"].abs()
res = pd.concat([res, X_test.reset_index(drop=True)[["hotel", "arrival_date_month", "market_segment",
                                                      "total_nights", "total_guests", "reserved_room_type"]]], axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
axes[0].scatter(res["pred"], res["actual"], s=4, alpha=0.25)
lim = [0, 400]
axes[0].plot(lim, lim, "r--")
axes[0].set_xlim(lim)
axes[0].set_ylim(lim)
axes[0].set_xlabel("Predicted ADR")
axes[0].set_ylabel("Actual ADR")
axes[0].set_title(f"Predicted vs actual - {chosen_label[3:]}", fontsize=10)
axes[1].scatter(res["pred"], res["residual"], s=4, alpha=0.25)
axes[1].axhline(0, color="r", ls="--")
axes[1].set_xlabel("Predicted ADR")
axes[1].set_ylabel("Residual (actual - predicted)")
axes[1].set_title("Residual plot")
savefig("14_regression_residuals.png")

order = ["January", "February", "March", "April", "May", "June", "July", "August",
         "September", "October", "November", "December"]
by_month = res.groupby("arrival_date_month")["abs_err"].mean().reindex(order)
by_hotel = res.groupby("hotel")["abs_err"].mean()
fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [3, 1]})
axes[0].bar([m[:3] for m in order], by_month.values, color="#8172b2")
axes[0].set_title("Test MAE by arrival month")
axes[1].bar(by_hotel.index, by_hotel.values, color="#937860")
axes[1].set_title("Test MAE by hotel")
savefig("15_regression_error_by_segment.png")

seg_rows = []
for col in ["hotel", "arrival_date_month", "market_segment", "reserved_room_type"]:
    g = res.groupby(col).agg(n=("abs_err", "size"), mae=("abs_err", "mean"),
                             mean_actual=("actual", "mean")).reset_index().rename(columns={col: "group"})
    g.insert(0, "feature", col)
    seg_rows.append(g)
pd.concat(seg_rows).round(3).to_csv(C.RESULTS / "04_error_analysis_by_segment.csv", index=False)
res.sort_values("abs_err", ascending=False).head(15).round(2).to_csv(
    C.RESULTS / "04_worst_predictions_top15.csv", index=False)

joblib.dump({"model": best_model, "name": chosen_label, "features": list(X.columns)},
            C.MODELS / "best_regressor.joblib", compress=3)
log("Saved model + all result files. Done.")
