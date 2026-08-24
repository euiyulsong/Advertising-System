# criteo_autobidding_experiment.py

import argparse
import warnings

import numpy as np
import pandas as pd

from datasets import load_dataset
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, log_loss

warnings.filterwarnings("ignore")

SEED = 42


# ============================================================
# 1. Load data
# ============================================================

def load_criteo(num_rows=1_000_000):
    ds = load_dataset(
        "criteo/criteo-attribution-dataset",
        split="train",
        streaming=True,
    )

    rows = []

    for i, row in enumerate(ds):
        if i >= num_rows:
            break

        rows.append(row)

        if (i + 1) % 100_000 == 0:
            print(f"Loaded {i + 1:,} rows")

    df = pd.DataFrame(rows)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nShape:", df.shape)

    return df


# ============================================================
# 2. Preprocess
# ============================================================

def preprocess(df):
    df = df.copy()

    # chronological ordering
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(
            df["timestamp"],
            errors="coerce",
        ).fillna(0)

        df = df.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        df["hour"] = (
            (df["timestamp"] // 3600) % 24
        ).astype(int)

        df["day"] = (
            df["timestamp"] // (3600 * 24)
        ).astype(int)

    # numeric labels / auction fields
    numeric_cols = [
        "click",
        "conversion",
        "cost",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            ).fillna(0)

    # categorical features
    categorical_cols = [
        "campaign",
        "cat1",
        "cat2",
        "cat3",
        "cat4",
        "cat5",
        "cat6",
        "cat7",
        "cat8",
        "cat9",
    ]

    categorical_cols = [
        c for c in categorical_cols
        if c in df.columns
    ]

    for c in categorical_cols:
        df[c] = df[c].astype("category")

    return df, categorical_cols


# ============================================================
# 3. Time split
# ============================================================

def time_split(df):
    n = len(df)

    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    valid = df.iloc[
        train_end:valid_end
    ].copy()

    test = df.iloc[
        valid_end:
    ].copy()

    print("\nSplit")
    print("Train:", len(train))
    print("Valid:", len(valid))
    print("Test :", len(test))

    return train, valid, test


# ============================================================
# 4. CTR model
# ============================================================

def train_ctr_model(
    train,
    valid,
    feature_cols,
    categorical_cols,
):

    print("\n" + "=" * 70)
    print("TRAIN CTR MODEL")
    print("=" * 70)

    model = LGBMClassifier(
        objective="binary",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(
        train[feature_cols],
        train["click"],
        categorical_feature=[
            c for c in categorical_cols
            if c in feature_cols
        ],
        eval_set=[
            (
                valid[feature_cols],
                valid["click"],
            )
        ],
    )

    pred = model.predict_proba(
        valid[feature_cols]
    )[:, 1]

    auc = roc_auc_score(
        valid["click"],
        pred,
    )

    ll = log_loss(
        valid["click"],
        pred,
    )

    print(f"CTR AUC     : {auc:.6f}")
    print(f"CTR LogLoss : {ll:.6f}")

    return model


# ============================================================
# 5. Add pCTR
# ============================================================

def add_pctr(
    model,
    df,
    feature_cols,
):
    df = df.copy()

    df["pctr"] = model.predict_proba(
        df[feature_cols]
    )[:, 1]

    return df


# ============================================================
# 6. Conversion model
# ============================================================

def train_conversion_model(
    train,
    valid,
    feature_cols,
    categorical_cols,
):

    print("\n" + "=" * 70)
    print("TRAIN CONVERSION MODEL")
    print("=" * 70)

    model = LGBMClassifier(
        objective="binary",
        n_estimators=700,
        learning_rate=0.04,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(
        train[feature_cols],
        train["conversion"],
        categorical_feature=[
            c for c in categorical_cols
            if c in feature_cols
        ],
        eval_set=[
            (
                valid[feature_cols],
                valid["conversion"],
            )
        ],
    )

    pred = model.predict_proba(
        valid[feature_cols]
    )[:, 1]

    auc = roc_auc_score(
        valid["conversion"],
        pred,
    )

    print(f"Conversion AUC: {auc:.6f}")

    return model


# ============================================================
# 7. Auction simulation
# ============================================================

def simulate_auction(
    df,
    bids,
    budget=None,
):
    result = df.copy()

    result["bid"] = np.maximum(
        bids,
        0,
    )

    # historical cost is used as
    # approximate clearing price
    result["win"] = (
        result["bid"]
        >= result["cost"]
    ).astype(int)

    winners = result[
        result["win"] == 1
    ].copy()

    if budget is not None:
        winners["cum_spend"] = (
            winners["cost"].cumsum()
        )

        winners = winners[
            winners["cum_spend"]
            <= budget
        ]

    impressions = len(winners)

    clicks = winners["click"].sum()

    conversions = (
        winners["conversion"].sum()
    )

    spend = winners["cost"].sum()

    ctr = (
        clicks / impressions
        if impressions > 0
        else 0
    )

    cvr = (
        conversions / clicks
        if clicks > 0
        else 0
    )

    cpa = (
        spend / conversions
        if conversions > 0
        else np.inf
    )

    budget_utilization = (
        spend / budget
        if budget is not None
        and budget > 0
        else np.nan
    )

    return {
        "impressions": impressions,
        "clicks": clicks,
        "conversions": conversions,
        "CTR": ctr,
        "CVR": cvr,
        "spend": spend,
        "CPA": cpa,
        "budget_utilization": budget_utilization,
    }


# ============================================================
# 8. Tune multiplier to budget
# ============================================================

def find_multiplier_for_budget(
    df,
    raw_scores,
    target_budget,
):
    multipliers = np.logspace(
        -6,
        5,
        400,
    )

    best_mult = 1.0
    best_diff = np.inf
    best_spend = None

    costs = df["cost"].values

    for mult in multipliers:
        bids = raw_scores * mult

        win = bids >= costs

        spend = df.loc[
            win,
            "cost"
        ].sum()

        diff = abs(
            spend - target_budget
        )

        if diff < best_diff:
            best_diff = diff
            best_mult = mult
            best_spend = spend

    print(
        f"  calibration spend = "
        f"{best_spend:.6f}"
    )

    return best_mult


# ============================================================
# 9. Print comparison
# ============================================================

def print_comparison(
    baseline,
    autobid,
):
    result_df = pd.DataFrame(
        {
            "CTR baseline": baseline,
            "Auto bidding": autobid,
        }
    ).T

    print("\n" + "=" * 80)
    print("BUSINESS OUTCOME")
    print("=" * 80)

    print(
        result_df[
            [
                "impressions",
                "clicks",
                "conversions",
                "CTR",
                "CVR",
                "spend",
                "CPA",
                "budget_utilization",
            ]
        ].to_string()
    )

    print("\n" + "=" * 80)
    print("AUTO-BIDDING LIFT")
    print("=" * 80)

    higher_is_better = [
        "CTR",
        "CVR",
        "conversions",
    ]

    for metric in higher_is_better:
        b = baseline[metric]
        a = autobid[metric]

        if b == 0:
            continue

        lift = (
            (a - b)
            / abs(b)
            * 100
        )

        print(
            f"{metric:20s}: "
            f"{lift:+.2f}%"
        )

    if (
        np.isfinite(baseline["CPA"])
        and baseline["CPA"] != 0
    ):
        cpa_change = (
            (
                autobid["CPA"]
                - baseline["CPA"]
            )
            / baseline["CPA"]
            * 100
        )

        print(
            f"{'CPA':20s}: "
            f"{cpa_change:+.2f}% "
            f"(lower is better)"
        )


# ============================================================
# Main
# ============================================================

def main(args):

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_criteo(
        args.rows
    )

    df, categorical_cols = preprocess(
        df
    )

    print("\nLabel statistics")
    print(
        f"CTR        : "
        f"{df['click'].mean():.6f}"
    )

    print(
        f"Conversion : "
        f"{df['conversion'].mean():.6f}"
    )

    print(
        f"Cost mean  : "
        f"{df['cost'].mean():.8f}"
    )

    print(
        f"Cost sum   : "
        f"{df['cost'].sum():.6f}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train, valid, test = time_split(
        df
    )

    # --------------------------------------------------------
    # Pre-auction features
    #
    # Never use:
    # click
    # conversion
    # cost
    # attribution
    # cpo
    #
    # as input features.
    # --------------------------------------------------------

    feature_cols = [
        c for c in [
            "campaign",
            "cat1",
            "cat2",
            "cat3",
            "cat4",
            "cat5",
            "cat6",
            "cat7",
            "cat8",
            "cat9",
            "hour",
            "day",
        ]
        if c in df.columns
    ]

    print(
        "\nCTR features:",
        feature_cols
    )

    # ========================================================
    # 1. CTR model
    # ========================================================

    ctr_model = train_ctr_model(
        train,
        valid,
        feature_cols,
        categorical_cols,
    )

    train = add_pctr(
        ctr_model,
        train,
        feature_cols,
    )

    valid = add_pctr(
        ctr_model,
        valid,
        feature_cols,
    )

    test = add_pctr(
        ctr_model,
        test,
        feature_cols,
    )

    test_ctr_auc = roc_auc_score(
        test["click"],
        test["pctr"],
    )

    print(
        f"\nTEST CTR AUC: "
        f"{test_ctr_auc:.6f}"
    )

    # ========================================================
    # 2. Conversion model
    # ========================================================

    bid_features = (
        feature_cols
        + ["pctr"]
    )

    print(
        "\nConversion features:",
        bid_features
    )

    conversion_model = (
        train_conversion_model(
            train,
            valid,
            bid_features,
            categorical_cols,
        )
    )

    # --------------------------------------------------------
    # Generate pCVR
    # --------------------------------------------------------

    train["pcvr"] = (
        conversion_model.predict_proba(
            train[bid_features]
        )[:, 1]
    )

    valid["pcvr"] = (
        conversion_model.predict_proba(
            valid[bid_features]
        )[:, 1]
    )

    test["pcvr"] = (
        conversion_model.predict_proba(
            test[bid_features]
        )[:, 1]
    )

    test_conversion_auc = roc_auc_score(
        test["conversion"],
        test["pcvr"],
    )

    print(
        f"TEST Conversion AUC: "
        f"{test_conversion_auc:.6f}"
    )

    # ========================================================
    # 3. Budget
    # ========================================================

    historical_test_spend = (
        test["cost"].sum()
    )

    target_budget = (
        historical_test_spend
        * args.budget_ratio
    )

    validation_budget = (
        valid["cost"].sum()
        * args.budget_ratio
    )

    print(
        f"\nHistorical test spend: "
        f"{historical_test_spend:.6f}"
    )

    print(
        f"Target budget         : "
        f"{target_budget:.6f}"
    )

    print(
        f"Validation budget     : "
        f"{validation_budget:.6f}"
    )

    # ========================================================
    # 4. Tune bidding multipliers on validation
    # ========================================================

    print("\nCalibrating CTR baseline")

    ctr_multiplier = (
        find_multiplier_for_budget(
            valid,
            valid["pctr"].values,
            validation_budget,
        )
    )

    print("\nCalibrating Auto bidder")

    conversion_multiplier = (
        find_multiplier_for_budget(
            valid,
            valid["pcvr"].values,
            validation_budget,
        )
    )

    print(
        f"\nCTR bid multiplier       : "
        f"{ctr_multiplier}"
    )

    print(
        f"Conversion bid multiplier: "
        f"{conversion_multiplier}"
    )

    # ========================================================
    # 5. Test bidding
    # ========================================================

    # Baseline:
    # bid proportional to pCTR
    baseline_bids = (
        test["pctr"].values
        * ctr_multiplier
    )

    # Auto bidder:
    # bid proportional to conversion probability
    autobid_bids = (
        test["pcvr"].values
        * conversion_multiplier
    )

    # ========================================================
    # 6. Offline auction replay
    # ========================================================

    baseline_metrics = simulate_auction(
        test,
        baseline_bids,
        budget=target_budget,
    )

    autobid_metrics = simulate_auction(
        test,
        autobid_bids,
        budget=target_budget,
    )

    print_comparison(
        baseline_metrics,
        autobid_metrics,
    )

    # ========================================================
    # 7. Diagnostics
    # ========================================================

    print("\n" + "=" * 80)
    print("MODEL DIAGNOSTICS")
    print("=" * 80)

    print(
        f"Mean pCTR        : "
        f"{test['pctr'].mean():.6f}"
    )

    print(
        f"Actual CTR       : "
        f"{test['click'].mean():.6f}"
    )

    print(
        f"Mean pCVR        : "
        f"{test['pcvr'].mean():.6f}"
    )

    print(
        f"Actual conversion: "
        f"{test['conversion'].mean():.6f}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rows",
        type=int,
        default=1_000_000,
    )

    parser.add_argument(
        "--budget-ratio",
        type=float,
        default=0.30,
    )

    args = parser.parse_args()

    main(args)
