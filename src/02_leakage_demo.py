"""Step 2 - Leakage demo.

`reservation_status` says Canceled / Check-Out / No-Show. It is only known AFTER
the outcome, so a model that sees it is cheating. This script shows how big the
fake score is, and what the honest score looks like after the leakage guard.
"""
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import config as C
from prep import get_clean_data, make_splits, build_preprocessor, drop_leaky

df = get_clean_data()
y = df["is_canceled"]


def run(feature_df, num_cols, cat_cols_extra=None, label=""):
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(feature_df, y, stratify=y)
    pre = build_preprocessor(num_cols, scale=False)
    if cat_cols_extra:
        # add the extra categorical column(s) to the one-hot step
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        pre = ColumnTransformer(pre.transformers + [
            ("leak", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols_extra)])
    model = Pipeline([("pre", pre), ("clf", DecisionTreeClassifier(max_depth=8, random_state=C.SEED))])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {"setup": label,
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "f1": round(f1_score(y_test, pred), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4)}


feats_honest = drop_leaky(df).drop(columns=["is_canceled", "arrival_date_month_num"], errors="ignore")
feats_leaky = feats_honest.copy()
feats_leaky["reservation_status"] = df["reservation_status"]

rows = [
    run(feats_leaky, C.CLF_NUM_COLS, ["reservation_status"],
        "WITH leakage (reservation_status included)"),
    run(feats_honest, C.CLF_NUM_COLS, None,
        "Leakage guard ON (only info known at booking time)"),
]
out = pd.DataFrame(rows)
out.to_csv(C.RESULTS / "02_leakage_demo.csv", index=False)
print(out.to_string(index=False))
print("\nThe first row is a fake score. The second row is the honest one.")
