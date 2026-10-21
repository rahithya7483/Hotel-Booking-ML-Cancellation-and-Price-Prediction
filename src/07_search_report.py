"""Step 7 - Show the hyper-parameter searches clearly.

Reads the trial tables saved by 03_classification.py and 05_extensions.py
(every combination that was tried, with its cross-validation score) and makes
  - results/07_search_summary.csv   : search type, space, #trials, best vs worst, gain over default
  - figures 26 and 27               : all trials as plots
"""
import matplotlib.pyplot as plt
import pandas as pd

import config as C
from utils import savefig

R = C.RESULTS


def load(name):
    return pd.read_csv(R / name)


def param_cols(t):
    return [c for c in t.columns if c not in
            ("mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time")]


def best_str(t):
    row = t.sort_values("rank_test_score").iloc[0]
    def fmt(v):
        return int(v) if isinstance(v, float) and v.is_integer() else v
    return ", ".join(f"{c}={fmt(row[c])}" for c in param_cols(t))


lr = load("03_search_trials_logreg.csv")
dt = load("03_search_trials_tree.csv")
rf = load("03_search_trials_forest.csv")
gbc = load("05_search_trials_gradient_boosting_clf.csv")
gbr = load("05_search_trials_gradient_boosting_reg.csv")
sweep = load("04_regularization_sweep.csv")
r1 = load("03_cv_round1_baselines.csv").set_index("model")

rows = [
    {"model": "Logistic Regression", "method": "GridSearchCV (3-fold)", "metric": "ROC-AUC",
     "search_space": "C in [0.01, 0.1, 1, 10]", "trials": len(lr),
     "best_score": lr["mean_test_score"].max(), "worst_score": lr["mean_test_score"].min(),
     "best_params": best_str(lr)},
    {"model": "Decision Tree", "method": "GridSearchCV (3-fold)", "metric": "ROC-AUC",
     "search_space": "max_depth in [4,6,8,12,16] x min_samples_leaf in [5,20,50]", "trials": len(dt),
     "best_score": dt["mean_test_score"].max(), "worst_score": dt["mean_test_score"].min(),
     "best_params": best_str(dt)},
    {"model": "Random Forest", "method": "RandomizedSearchCV (3-fold)", "metric": "ROC-AUC",
     "search_space": "max_depth [12,16,20,None] x min_samples_leaf [2,5,10] x max_features [sqrt,0.3]; 6 random trials",
     "trials": len(rf), "best_score": rf["mean_test_score"].max(), "worst_score": rf["mean_test_score"].min(),
     "best_params": best_str(rf)},
    {"model": "Gradient Boosting (classification)", "method": "RandomizedSearchCV (3-fold)", "metric": "PR-AUC",
     "search_space": "learning_rate x max_leaf_nodes x min_samples_leaf x l2_regularization x max_iter; 8 random trials",
     "trials": len(gbc), "best_score": gbc["mean_test_score"].max(), "worst_score": gbc["mean_test_score"].min(),
     "best_params": best_str(gbc)},
    {"model": "Gradient Boosting (price)", "method": "RandomizedSearchCV (3-fold)", "metric": "RMSE (lower is better)",
     "search_space": "same space as above, 8 random trials", "trials": len(gbr),
     "best_score": -gbr["mean_test_score"].max(), "worst_score": -gbr["mean_test_score"].min(),
     "best_params": best_str(gbr)},
]
for name in ["Ridge", "Lasso"]:
    s = sweep[sweep["model"] == name]
    best = s.sort_values("val_rmse").iloc[0]
    rows.append({"model": f"{name} (price)", "method": "alpha sweep on validation set", "metric": "RMSE (lower is better)",
                 "search_space": f"{len(s)} alphas, {s['alpha'].min():.3g} to {s['alpha'].max():.3g} (log scale)",
                 "trials": len(s), "best_score": best["val_rmse"], "worst_score": s["val_rmse"].max(),
                 "best_params": f"alpha={best['alpha']:.4g}"})
summary = pd.DataFrame(rows)
summary.round(4).to_csv(R / "07_search_summary.csv", index=False)
print(summary[["model", "method", "trials", "best_score", "worst_score", "best_params"]].round(4).to_string(index=False))

# gain of tuned model over a hand-picked setting (CV ROC-AUC, same 5-fold CV)
r2 = load("03_cv_round2_tuned.csv").set_index("model")
gain = pd.DataFrame({
    "hand_picked_setting": ["Logistic Reg. (balanced)", "Decision Tree (depth 6)", "Random Forest (balanced)"],
    "tuned_model": ["Logistic Regression (tuned)", "Decision Tree (tuned)", "Random Forest (tuned)"]})
gain["roc_auc_before"] = [r1.loc[m, "roc_auc_mean"] for m in gain["hand_picked_setting"]]
gain["roc_auc_after"] = [r2.loc[m, "roc_auc_mean"] for m in gain["tuned_model"]]
gain["f1_before"] = [r1.loc[m, "f1_mean"] for m in gain["hand_picked_setting"]]
gain["f1_after"] = [r2.loc[m, "f1_mean"] for m in gain["tuned_model"]]
gain.round(4).to_csv(R / "07_tuning_gain.csv", index=False)
print(gain.round(4).to_string(index=False))

# ---------------------------------------------------------------- figure 26: tree grid + forest + boosting trials
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
piv = dt.pivot_table(index="max_depth", columns="min_samples_leaf", values="mean_test_score")
im = axes[0].imshow(piv.values, cmap="YlGn", aspect="auto")
axes[0].set_xticks(range(len(piv.columns)))
axes[0].set_xticklabels(piv.columns)
axes[0].set_yticks(range(len(piv.index)))
axes[0].set_yticklabels(piv.index)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        dark = v > piv.values.min() + 0.6 * (piv.values.max() - piv.values.min())
        axes[0].text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                     color="white" if dark else "black")
axes[0].set_xlabel("min_samples_leaf")
axes[0].set_ylabel("max_depth")
axes[0].set_title("Decision Tree grid (CV ROC-AUC)")
axes[0].grid(False)


def trial_bars(ax, t, title, label_cols):
    t = t.sort_values("mean_test_score")
    def nice(v):
        if pd.isna(v):
            return "None"
        return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)
    labels = [" | ".join(nice(r[c]) for c in label_cols) for _, r in t.iterrows()]
    ax.barh(range(len(t)), t["mean_test_score"], xerr=t["std_test_score"], color="#4c72b0", capsize=2)
    ax.barh(len(t) - 1, t["mean_test_score"].iloc[-1], color="#55a868")   # best trial in green
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(t["mean_test_score"].min() - 0.01, t["mean_test_score"].max() + 0.01)
    ax.set_title(title)


trial_bars(axes[1], rf, "Random Forest trials (CV ROC-AUC)\nlabel: leaf | max_features | depth",
           ["min_samples_leaf", "max_features", "max_depth"])
trial_bars(axes[2], gbc, "Gradient Boosting trials (CV PR-AUC)\nlabel: leaf | l2 | lr | max_iter | leaves",
           ["min_samples_leaf", "l2_regularization", "learning_rate", "max_iter", "max_leaf_nodes"])
savefig("26_search_trials_classification.png")

# ---------------------------------------------------------------- figure 27: tuning gain
fig, ax = plt.subplots(figsize=(7.5, 4.5))
x = range(len(gain))
ax.bar([i - 0.2 for i in x], gain["roc_auc_before"], 0.4, label="hand-picked setting", color="#9aa5b1")
ax.bar([i + 0.2 for i in x], gain["roc_auc_after"], 0.4, label="after search", color="#4c72b0")
ax.set_xticks(list(x))
ax.set_xticklabels(["Logistic Reg.", "Decision Tree", "Random Forest"])
ax.set_ylim(0.8, 0.9)
ax.set_ylabel("5-fold CV ROC-AUC")
ax.set_title("Gain from hyper-parameter search")
ax.legend()
savefig("27_tuning_gain.png")
print("Done.")
