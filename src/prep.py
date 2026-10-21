"""Data loading, cleaning, feature engineering, splitting and preprocessing.

Everything that learns from data (imputing, scaling, encoding) lives inside a
sklearn Pipeline so it is fitted on the TRAIN part only -> no leakage.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (DATA_PATH, SEED, HARD_LEAK_COLS, AFTER_BOOKING_COLS,
                    BIN_COLS, CAT_COLS)

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def load_raw():
    return pd.read_csv(DATA_PATH)


def clean(df, drop_duplicates=True):
    """Basic cleaning. Returns a new dataframe."""
    df = df.copy()

    # 1. exact duplicate rows (removed BEFORE splitting, otherwise the same
    #    booking can land in both train and test)
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)

    # 2. missing values
    df["children"] = df["children"].fillna(0)            # only 4 missing
    df["country"] = df["country"].fillna("Unknown")      # ~0.4 % missing
    # agent / company are ID columns with lots of NaN (NaN = "no agent").
    # The ID itself is not useful, but "has an agent?" is.
    df["has_agent"] = df["agent"].notna().astype(int)
    df["has_company"] = df["company"].notna().astype(int)
    df = df.drop(columns=["agent", "company"])
    if drop_duplicates:
        # rows that only differed by the agent/company ID are now identical
        df = df.drop_duplicates().reset_index(drop=True)

    # 3. impossible rows: bookings with zero guests
    guests = df["adults"] + df["children"] + df["babies"]
    df = df[guests > 0].reset_index(drop=True)
    return df


def add_features(df):
    """Simple feature engineering from dates and counts."""
    df = df.copy()
    month_num = df["arrival_date_month"].map({m: i + 1 for i, m in enumerate(MONTHS)})
    arrival = pd.to_datetime(dict(year=df["arrival_date_year"], month=month_num,
                                  day=df["arrival_date_day_of_month"]))
    df["arrival_weekday"] = arrival.dt.day_name()
    df["is_weekend_arrival"] = arrival.dt.dayofweek.isin([4, 5]).astype(int)
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["has_children"] = ((df["children"] + df["babies"]) > 0).astype(int)
    return df


def drop_leaky(df):
    return df.drop(columns=HARD_LEAK_COLS + AFTER_BOOKING_COLS, errors="ignore")


def get_clean_data():
    return add_features(clean(load_raw()))


def make_splits(X, y, stratify=None):
    """60 % train / 20 % validation / 20 % test."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=stratify)
    strat_tmp = y_tmp if stratify is not None else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, random_state=SEED, stratify=strat_tmp)  # 0.25 * 0.8 = 0.2
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(num_cols, scale=True, cat_cols=None):
    """Impute + (scale) numbers, one-hot categories. Rare categories are
    grouped into 'infrequent' (learned on train only)."""
    cat_cols = CAT_COLS if cat_cols is None else cat_cols
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num_cols),
        ("bin", "passthrough", BIN_COLS),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              min_frequency=0.01, sparse_output=False), cat_cols),
    ])
