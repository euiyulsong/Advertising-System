# criteo_autobidding_experiment.py

import argparse
import warnings

import numpy as np
import pandas as pd

from datasets import load_dataset
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import roc_auc_score, log_loss, mean_absolute_error

warnings.filterwarnings("ignore")

SEED = 42


# ============================================================
# 1. Load data
# ============================================================

def load_criteo(num_rows=1_000_000):
    """
    Criteo Attribution Dataset.

    Expected columns roughly:
        timestamp
        uid
        campaign
        conversion
        conversion_timestamp
        conversion_id
        attribution
        click
        click_pos
        click_nb
        cost
        cpo
        cat1 ~ cat9

    Streaming is used because full dataset is large.
    """

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
# 2. Feature engineering
# ============================================================

def preprocess(df):
    df = df.copy()

    # chronological ordering
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(
            df["timestamp"],
            errors="coerce",
        ).fillna(0)

        df = df.sort_values("timestamp").reset_index(drop=True)

        # timestamp is seconds in this dataset
        df["hour"] = (
            (df["timestamp"] // 3600) % 24
        ).astype(int)

        df["day"] = (
            df["timestamp"] // (3600 * 24)
        ).astype(int)

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

    # labels
    for col in [
        "click",
        "conversion",
        "attribution",
        "cost",
        "cpo",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(0)

    # --------------------------------------------------------
    # Business value label
    # --------------------------------------------------------
    #
    # attribution:
    #   conversion attribution weight
    #
    # cpo:
    #   value / cost-per-order related signal
    #
    # realized_value is our offline "revenue" approximation.
    #
    # Depending on dataset version, you can replace this with:
    #
    #   conversion * FIXED_VALUE
    #
    # if desired.
    #

    if "cpo" in df.columns:
        df["realized_value"] = (
            df["conversion"]
            * df["cpo"]
        )
    else:
        # simple fallback
        CONVERSION_VALUE = 10.0
        df["realized_value"] = (
            df["conversion"]
            * CONVERSION_VALUE
        )

    return df, categorical_cols


# ============================================================
# 3. Time split
# ============================================================

def time_split(df):
    n = len(df)

    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    valid = df.iloc[train_end:valid_end].copy()
    test = df.iloc[valid_end:].copy()

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
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
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
                valid["click"]
            )
        ],
        callbacks=[],
    )

    pred = model.predict_proba(
        valid[feature_cols]
    )[:, 1]

    auc = roc_auc_score(
        valid["click"],
        pred
    )

    ll = log_loss(
        valid["click"],
        pred
    )

    print(f"CTR AUC     : {auc:.6f}")
    print(f"CTR LogLoss : {ll:.6f}")

    return model


# ============================================================
# 5. Generate pCTR
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
# 6. Auto bidding model
# ============================================================

def train_bidding_model(
    train,
    valid,
    feature_cols,
    categorical_cols,
):

    print("\n" + "=" * 70)
    print("TRAIN AUTO-BIDDING MODEL")
    print("=" * 70)

    #
    # Most impressions have zero value.
    # Tweedie works reasonably well for
    # zero-heavy positive targets.
    #

    model = LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,

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
    )

    model.fit(
        train[feature_cols],
        train["realized_value"],

        categorical_feature=[
            c for c in categorical_cols
            if c in feature_cols
        ],

        eval_set=[
            (
                valid[feature_cols],
                valid["realized_value"]
            )
        ],
    )

    pred = np.maximum(
        model.predict(
            valid[feature_cols]
        ),
        0,
    )

    mae = mean_absolute_error(
        valid["realized_value"],
        pred,
    )

    print(
        f"Value prediction MAE: "
        f"{mae:.6f}"
    )

    return model


# ============================================================
# 7. Auction simulation
# ============================================================

def simulate_auction(
    df,
    bids,
    budget=None,
):
    """
    Simple offline replay.

    Assumption:
        historical cost ~= market clearing price

    Win:
        bid >= cost

    Important:
        This is NOT a perfect counterfactual auction simulator.
        It is an offline approximation for experimentation.
    """

    result = df.copy()

    result["bid"] = np.maximum(
        bids,
        0
    )

    result["win"] = (
        result["bid"]
        >= result["cost"]
    ).astype(int)

    winners = result[
        result["win"] == 1
    ].copy()

    # --------------------------------------------------------
    # Budget constraint
    # --------------------------------------------------------

    if budget is not None:

        winners["cum_spend"] = (
            winners["cost"]
            .cumsum()
        )

        winners = winners[
            winners["cum_spend"]
            <= budget
        ]

    spend = winners["cost"].sum()

    clicks = winners["click"].sum()

    conversions = (
        winners["conversion"].sum()
    )

    revenue = (
        winners["realized_value"].sum()
    )

    impressions = len(winners)

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

    roas = (
        revenue / spend
        if spend > 0
        else 0
    )

    profit = revenue - spend

    return {
        "impressions": impressions,
        "clicks": clicks,
        "conversions": conversions,
        "CTR": ctr,
        "CVR": cvr,
        "spend": spend,
        "revenue": revenue,
        "CPA": cpa,
        "ROAS": roas,
        "profit": profit,
    }


# ============================================================
# 8. Find bid multiplier
# ============================================================

def find_multiplier_for_budget(
    df,
    raw_scores,
    target_budget,
):
    """
    Search multiplier so predicted scores
    generate approximately the target spend.

        bid = score * multiplier
    """

    multipliers = np.logspace(
        -3,
        4,
        200,
    )

    best_mult = 1.0
    best_diff = np.inf

    for mult in multipliers:

        bids = raw_scores * mult

        win = bids >= df["cost"].values

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

    return best_mult


# ============================================================
# 9. Pretty comparison
# ============================================================

def print_comparison(
    baseline,
    autobid,
):

    df = pd.DataFrame(
        {
            "CTR baseline": baseline,
            "Auto bidding": autobid,
        }
    ).T

    print("\n" + "=" * 80)
    print("BUSINESS OUTCOME")
    print("=" * 80)

    print(
        df[
            [
                "impressions",
                "clicks",
                "conversions",
                "CTR",
                "CVR",
                "spend",
                "revenue",
                "CPA",
                "ROAS",
                "profit",
            ]
        ].to_string()
    )

    print("\n" + "=" * 80)
    print("AUTO-BIDDING LIFT")
    print("=" * 80)

    for metric in [
        "CTR",
        "CVR",
        "conversions",
        "revenue",
        "ROAS",
        "profit",
    ]:

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
            f"{metric:12s}: "
            f"{lift:+.2f}%"
        )

    if np.isfinite(
        baseline["CPA"]
    ) and baseline["CPA"] != 0:

        lift = (
            (
                autobid["CPA"]
                - baseline["CPA"]
            )
            / baseline["CPA"]
            * 100
        )

        print(
            f"{'CPA':12s}: "
            f"{lift:+.2f}% "
            f"(lower is better)"
        )


# ============================================================
# Main
# ============================================================

def main(args):

    df = load_criteo(
        args.rows
    )

    df, categorical_cols = preprocess(
        df
    )

    train, valid, test = time_split(
        df
    )

    # --------------------------------------------------------
    # Pre-auction feature only.
    #
    # DO NOT use:
    #     click
    #     conversion
    #     attribution
    #     cost
    #     cpo
    #
    # because they are future/post-auction signals.
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
    # CTR
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

    test_auc = roc_auc_score(
        test["click"],
        test["pctr"],
    )

    print(
        f"\nTEST CTR AUC: "
        f"{test_auc:.6f}"
    )

    # ========================================================
    # Auto bidding regression
    # ========================================================

    bid_features = (
        feature_cols
        + ["pctr"]
    )

    bidding_model = (
        train_bidding_model(
            train,
            valid,
            bid_features,
            categorical_cols,
        )
    )

    train["pred_value"] = np.maximum(
        bidding_model.predict(
            train[bid_features]
        ),
        0,
    )

    valid["pred_value"] = np.maximum(
        bidding_model.predict(
            valid[bid_features]
        ),
        0,
    )

    test["pred_value"] = np.maximum(
        bidding_model.predict(
            test[bid_features]
        ),
        0,
    )

    # ========================================================
    # Budget
    # ========================================================
    #
    # Example:
    # Spend 30% of historical test spend.
    #

    total_test_cost = (
        test["cost"].sum()
    )

    budget = (
        total_test_cost
        * args.budget_ratio
    )

    print(
        f"\nHistorical test spend: "
        f"{total_test_cost:.4f}"
    )

    print(
        f"Target budget         : "
        f"{budget:.4f}"
    )

    # ========================================================
    # Tune multiplier on validation
    # ========================================================

    validation_budget = (
        valid["cost"].sum()
        * args.budget_ratio
    )

    # --------------------------------------------------------
    # Baseline:
    #
    # bid = alpha * pCTR
    #
    # Equivalent to old/simple bidding logic:
    # high click probability -> higher bid
    # --------------------------------------------------------

    ctr_multiplier = (
        find_multiplier_for_budget(
            valid,
            valid["pctr"].values,
            validation_budget,
        )
    )

    # --------------------------------------------------------
    # Auto bidder:
    #
    # bid = beta * expected conversion value
    # --------------------------------------------------------

    value_multiplier = (
        find_multiplier_for_budget(
            valid,
            valid["pred_value"].values,
            validation_budget,
        )
    )

    print(
        "\nCTR bid multiplier:",
        ctr_multiplier
    )

    print(
        "Value bid multiplier:",
        value_multiplier
    )

    # ========================================================
    # Test auction replay
    # ========================================================

    baseline_bids = (
        test["pctr"].values
        * ctr_multiplier
    )

    autobid_bids = (
        test["pred_value"].values
        * value_multiplier
    )

    baseline_metrics = simulate_auction(
        test,
        baseline_bids,
        budget=budget,
    )

    autobid_metrics = simulate_auction(
        test,
        autobid_bids,
        budget=budget,
    )

    print_comparison(
        baseline_metrics,
        autobid_metrics,
    )

    # ========================================================
    # Additional diagnostics
    # ========================================================

    print("\n" + "=" * 80)
    print("MODEL DIAGNOSTICS")
    print("=" * 80)

    print(
        f"Mean pCTR       : "
        f"{test['pctr'].mean():.6f}"
    )

    print(
        f"Actual CTR      : "
        f"{test['click'].mean():.6f}"
    )

    print(
        f"Mean pred value : "
        f"{test['pred_value'].mean():.6f}"
    )

    print(
        f"Actual value    : "
        f"{test['realized_value'].mean():.6f}"
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
