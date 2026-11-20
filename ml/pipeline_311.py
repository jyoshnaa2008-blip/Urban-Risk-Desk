# ============================================================
# NYC 311 PUBLIC SERVICE COMPLAINTS
# PREDICTION PIPELINE
#
# Predicts which complaints stay unclosed after 48 hours.
#
# Steps:
#   1. Load the cleaned dataset
#   2. Build the 48-hour target and filing-time features
#   3. Compute recent slow-rate features (no leakage)
#   4. Chronological train / validation / test split
#   5. Compare model variants on unseen test weeks
#   6. Daily recalibration on the final model
#   7. Write the priority table and saved model
#
# IMPORTANT:
# Every feature uses only information available at the moment
# a complaint is filed. No closed_date, status or resolution text
# is ever used as a feature.
# ============================================================

# 1. IMPORT LIBRARIES

import os
import json
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import brentq

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss


# 2. PROJECT PATHS AND SETTINGS

DATA_PATH = "data/nyc311_clean.csv"
OUTPUT_DIR = "ml"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HOURS = 48                      # "slow" = not closed within 48 hours
TRAIN_END = "2026-10-25"        # train: before this date
VAL_END = "2026-10-28"          # validation: up to here; test: after this
NS = 10**9

# Filing-time features (all known when a complaint is submitted)
CATEGORICAL_FEATURES = ["complaint_type", "agency", "borough", "channel", "incident_zip"]
NUMERIC_FEATURES = ["hour", "dow", "pt_rr", "ag_rr"]


# 3. LOAD DATASET

def load_data():
    print("\n========== LOAD DATASET ==========")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["created_date"] = pd.to_datetime(df["created_date"])

    # Outcome is known only for complaints filed at least 48h before the end of data
    end = df["created_date"].max()
    df = df[df["created_date"] <= end - pd.Timedelta(hours=HOURS)].copy()

    df["slow"] = df["slow_48h"].astype(int)
    df["hour"] = df["created_date"].dt.hour
    df["dow"] = df["created_date"].dt.day_of_week
    df["day"] = df["created_date"].dt.floor("D")

    print("Rows with known outcome:", f"{len(df):,}")
    print("Created:", df["created_date"].min(), "to", df["created_date"].max())
    print("Slow share:", round(df["slow"].mean(), 3))
    return df


# 4. RECENT SLOW-RATE FEATURES (NO LEAKAGE)

def compute_recent_slow_rate(df, key):
    """Slow rate of the same type/agency over the trailing 7 days, ending 3 days ago.

    Outcomes in that window are already known at filing time, so this is safe.
    """
    daily = df.groupby([key, "day"]).slow.agg(["sum", "count"]).reset_index()
    out = []
    for k, g in daily.groupby(key):
        days = pd.date_range(g["day"].min() - pd.Timedelta(days=10), g["day"].max(), freq="D")
        s = g.set_index("day")["sum"].reindex(days, fill_value=0)
        n = g.set_index("day")["count"].reindex(days, fill_value=0)
        roll_s = s.rolling(7).sum().shift(3)   # window ends 3 days before the complaint
        roll_n = n.rolling(7).sum().shift(3)
        out.append(pd.DataFrame({key: k, "day": days, key + "_rr": (roll_s / roll_n).where(roll_n >= 20)}))
    return df.merge(pd.concat(out), on=[key, "day"], how="left")


def prepare_data(df):
    print("\n========== RECENT SLOW-RATE FEATURES ==========")

    df = compute_recent_slow_rate(df, "complaint_type")
    df = df.rename(columns={"complaint_type_rr": "pt_rr"})
    df = compute_recent_slow_rate(df, "agency")
    df = df.rename(columns={"agency_rr": "ag_rr"})

    print("pt_rr missing:", round(df["pt_rr"].isna().mean(), 3))
    print("ag_rr missing:", round(df["ag_rr"].isna().mean(), 3))
    return df


# 5. PREPROCESSING PIPELINE

def make_model(num_features):
    """One-hot encode categories, pass numbers through, then gradient boosting."""
    pre = ColumnTransformer([
        ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), CATEGORICAL_FEATURES),
        ("n", "passthrough", num_features),
    ])
    return make_pipeline(pre, HistGradientBoostingClassifier(
        max_depth=5, learning_rate=0.06, max_iter=150, min_samples_leaf=40, random_state=0))


# 6. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT

def chronological_split(df):
    """Split by date: train before TRAIN_END, validation up to VAL_END, test after."""
    train = df[df["created_date"] < pd.Timestamp(TRAIN_END)].copy()
    valid = df[(df["created_date"] >= pd.Timestamp(TRAIN_END)) & (df["created_date"] < pd.Timestamp(VAL_END))].copy()
    test = df[df["created_date"] >= pd.Timestamp(VAL_END)].copy()

    if train.empty or valid.empty or test.empty:
        raise ValueError("Chronological split produced an empty group. Check the dates.")

    print("\nTraining rows:", f"{len(train):,}", "(slow", round(train["slow"].mean(), 3), ")")
    print("Validation rows:", f"{len(valid):,}", "(slow", round(valid["slow"].mean(), 3), ")")
    print("Testing rows:", f"{len(test):,}", "(slow", round(test["slow"].mean(), 3), ")")
    return train, valid, test


# 7. METRICS

def evaluate(y_true, y_pred):
    """Ranking metrics for an imbalanced yes/no problem."""
    y_true = np.asarray(y_true)
    k = int(0.2 * len(y_pred))
    top = np.argsort(-y_pred)[:k]
    return {
        "PR_AUC": round(average_precision_score(y_true, y_pred), 3),
        "ROC_AUC": round(roc_auc_score(y_true, y_pred), 3),
        "Brier": round(brier_score_loss(y_true, y_pred), 3),
        "mean_pred": round(float(y_pred.mean()), 3),
        "actual": round(float(y_true.mean()), 3),
        "top20_precision": round(float(y_true[top].mean()), 3),
        "top20_lift": round(float(y_true[top].mean() / y_true.mean()), 2),
    }


# 8. DAILY RECALIBRATION

def recalibrate_daily(df, preds, test_mask, window_days=7):
    """Shift predictions so the average matches outcomes already known each day."""
    lo = np.log(np.clip(preds, 1e-6, 1 - 1e-6) / (1 - np.clip(preds, 1e-6, 1 - 1e-6)))
    out = preds.copy()
    y = df["slow"].values
    t = df["created_date"]
    pool = ~(df["created_date"] < pd.Timestamp(TRAIN_END)).values   # validation + test rows only

    for day in sorted(t[test_mask].dt.floor("D").unique()):
        known = pool & (t < day - pd.Timedelta(hours=HOURS)).values & (t >= day - pd.Timedelta(days=window_days + 2)).values
        rows = test_mask & (t.dt.floor("D") == day).values
        if known.sum() < 300 or y[known].min() == y[known].max():
            continue
        f = lambda s: (1 / (1 + np.exp(-(lo[known] + s)))).mean() - y[known].mean()
        try:
            s = brentq(f, -5, 5)
        except ValueError:
            continue
        out[rows] = 1 / (1 + np.exp(-(lo[rows] + s)))
    return out


# 9. MODEL TRAINING AND COMPARISON

def run_model_comparison(train, valid, test, y_train, y_valid, y_test):
    """Compare model variants on the unseen test weeks. Chosen on validation, not test."""
    print("\n========== MODEL TRAINING AND COMPARISON ==========")

    sets = {
        "B  boosting, basic features": ["hour", "dow"],
        "R  B + recent slow rate (type, agency)": ["hour", "dow", "pt_rr", "ag_rr"],
        "C  R + backlog features": ["hour", "dow", "pt_rr", "ag_rr"],
    }
    results, val_score, models = {}, {}, {}
    rate = train.groupby("complaint_type").slow.mean()

    # Baselines
    results["A0 baseline: type slow-rate from training"] = evaluate(
        y_test, test["complaint_type"].map(rate).fillna(y_train.mean()).values)
    results["A1 baseline: type slow-rate of last week"] = evaluate(
        y_test, test["pt_rr"].fillna(y_train.mean()).values)

    for name, num in sets.items():
        m = make_model(num).fit(train[CATEGORICAL_FEATURES + num], y_train)
        models[name] = (m, CATEGORICAL_FEATURES + num)
        val_score[name] = average_precision_score(y_valid, m.predict_proba(valid[CATEGORICAL_FEATURES + num])[:, 1])
        results[name] = evaluate(y_test, m.predict_proba(test[CATEGORICAL_FEATURES + num])[:, 1])

    best = max(val_score, key=val_score.get)          # chosen on VALIDATION, not test
    print("validation PR-AUC by variant:", {k: round(v, 3) for k, v in val_score.items()}, "-> chosen:", best)
    return results, val_score, models, best


# 10. PRIORITY TABLE

def build_priority_table(df, test, preds, model_name):
    """Rank (ZIP, complaint type) groups by expected slow complaints."""
    print("\n========== PRIORITY TABLE ==========")

    df["p"] = preds
    t = df[df["created_date"] >= pd.Timestamp(VAL_END)]

    pri = t[t["incident_zip"] != "NA"].groupby(["incident_zip", "complaint_type"]).agg(
        complaints=("slow", "size"), expected_slow=("p", "sum"), actual_slow=("slow", "sum")).reset_index()
    pri["priority_score"] = pri["expected_slow"].round(1)
    pri = pri.sort_values("priority_score", ascending=False)

    top = pri.head(max(1, int(0.1 * len(pri))))
    cap_model = top["actual_slow"].sum() / pri["actual_slow"].sum()
    cap_vol = pri.sort_values("complaints", ascending=False).head(len(top))["actual_slow"].sum() / pri["actual_slow"].sum()
    print(f"Priority check: top 10% of (ZIP, type) groups hold {cap_model:.1%} of "
          f"actually-slow complaints by model score vs {cap_vol:.1%} by raw complaint volume")

    pri.to_csv(os.path.join(OUTPUT_DIR, "priority_by_zip_type.csv"), index=False)
    return pri, cap_model, cap_vol


# 11. SAVE RESULTS

def save_results(results, val_score, chosen, mixed_types, cap_model, cap_vol, model, features):
    """Save the model, results and priority table to disk."""
    print("\n========== SAVE RESULTS ==========")

    joblib.dump({"model": model, "features": features, "mixed_types": mixed_types, "threshold_hours": HOURS},
                os.path.join(OUTPUT_DIR, "model_311.joblib"))
    json.dump({"variants": results, "validation_pr_auc": val_score, "chosen": chosen,
               "priority_capture_model": round(float(cap_model), 3),
               "priority_capture_volume": round(float(cap_vol), 3),
               "mixed_types": mixed_types},
              open(os.path.join(OUTPUT_DIR, "results.json"), "w"), indent=2)
    print("saved: model_311.joblib, results.json, priority_by_zip_type.csv in", OUTPUT_DIR)


# 12. MAIN PROGRAM

def main():
    print("\n==============================================")
    print(" NYC 311 - DELAY PREDICTION PIPELINE")
    print(" Predict 48-hour resolution risk from filing-time features")
    print("==============================================")

    df = load_data()
    df = prepare_data(df)

    # Choose complaint types from the training period only (no peeking at test)
    train_mask = df["created_date"] < pd.Timestamp(TRAIN_END)
    g = df[train_mask].groupby("complaint_type").slow.agg(["size", "mean"])
    mixed = list(g[(g["size"] >= 1500) & (g["mean"].between(0.15, 0.85))].index)
    print(f"\n{len(mixed)} mixed-outcome complaint types (chosen on training period only): {mixed}")

    d = df[df["complaint_type"].isin(mixed)].reset_index(drop=True)
    train, valid, test = chronological_split(d)
    y_train, y_valid, y_test = train["slow"], valid["slow"], test["slow"]

    results, val_score, models, best = run_model_comparison(
        train, valid, test, y_train, y_valid, y_test)

    # Final model: best variant + daily recalibration
    bm, bcols = models[best]
    preds = bm.predict_proba(d[bcols])[:, 1]
    test_mask = d["created_date"] >= pd.Timestamp(VAL_END)
    preds_cal = recalibrate_daily(d, preds, test_mask)
    results[f"E  {best} + daily recalibration (final)"] = evaluate(y_test, preds_cal[test_mask.values])

    print("\nVARIANT COMPARISON on unseen test weeks (PR/ROC higher is better; Brier lower)")
    print(pd.DataFrame(results).T.to_string())

    pri, cap_model, cap_vol = build_priority_table(d, test, preds_cal, best)
    save_results(results, val_score, best, mixed, cap_model, cap_vol, bm, bcols)

    print("\n==============================================")
    print(" PIPELINE COMPLETE")
    print("==============================================")


if __name__ == "__main__":
    main()