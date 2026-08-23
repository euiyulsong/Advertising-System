# ============================================================
# train_ads_full.py
#
# Real-data advertising experiment:
#
# 1. Targeting    : Criteo streaming -> pCTR
# 2. Auto-Bidding : AuctionNet -> learned bid policy
# 3. Pacing       : AuctionNet -> learned pacing multiplier
# 4. Evaluation   : offline auction replay
#
# pip install -U datasets pandas numpy scikit-learn requests remotezip
# ============================================================

import io
import numpy as np
import pandas as pd

from datasets import load_dataset

from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
    mean_absolute_error,
)
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from remotezip import RemoteZip


SEED = 42
np.random.seed(SEED)


# ============================================================
# CONFIG
# ============================================================

# ---------- Targeting ----------
N_CTR_TRAIN = 200_000
N_CTR_TEST = 20_000
HASH_DIM = 2 ** 18


# ---------- AuctionNet ----------
N_AUCTION = 200_000

AUCTIONNET_URL = (
    "https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/"
    "share/autoBidding_aigb_track_data_period_7-8.zip"
)


# Bid multiplier actions.
BID_ACTIONS = np.array([
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
    3.0,
])


# ============================================================
# PART 1
# TARGETING / pCTR
# ============================================================

print()
print("=" * 70)
print("PART 1 - TARGETING / pCTR")
print("=" * 70)


DATASET_NAME = "criteo/CriteoClickLogs"


print("Streaming Criteo data...")


ds = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True,
)


ds = ds.shuffle(
    seed=SEED,
    buffer_size=10_000,
)


rows = list(
    ds.take(
        N_CTR_TRAIN
        + N_CTR_TEST
    )
)


ctr_df = pd.DataFrame(rows)


print("Loaded:", len(ctr_df))


# ------------------------------------------------------------
# Label
# ------------------------------------------------------------

LABEL = "label"


feature_cols = [
    c
    for c in ctr_df.columns
    if c != LABEL
]


# ------------------------------------------------------------
# Feature construction
# ------------------------------------------------------------

def criteo_features(row):

    feats = {}

    for col in feature_cols:

        value = row[col]

        if pd.isna(value):
            continue

        if isinstance(
            value,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        ):

            value = float(value)

            # numerical stabilization
            feats[col] = np.log1p(
                max(value, 0.0)
            )

        else:

            feats[
                f"{col}={value}"
            ] = 1.0

    return feats


train_ctr = ctr_df.iloc[
    :N_CTR_TRAIN
].copy()


test_ctr = ctr_df.iloc[
    N_CTR_TRAIN:
].copy()


print(
    "Train:",
    len(train_ctr),
    "Test:",
    len(test_ctr),
)


print(
    "Train CTR:",
    train_ctr[LABEL].mean(),
)


# ------------------------------------------------------------
# Feature Hashing
# ------------------------------------------------------------

hasher = FeatureHasher(
    n_features=HASH_DIM,
    input_type="dict",
)


X_train_ctr = hasher.transform(
    [
        criteo_features(row)
        for _, row in train_ctr.iterrows()
    ]
)


X_test_ctr = hasher.transform(
    [
        criteo_features(row)
        for _, row in test_ctr.iterrows()
    ]
)


y_train_ctr = (
    train_ctr[LABEL]
    .astype(int)
    .values
)


y_test_ctr = (
    test_ctr[LABEL]
    .astype(int)
    .values
)


# ------------------------------------------------------------
# CTR model
# ------------------------------------------------------------

ctr_model = SGDClassifier(
    loss="log_loss",
    alpha=1e-5,
    max_iter=100,
    tol=1e-4,
    random_state=SEED,
)


ctr_model.fit(
    X_train_ctr,
    y_train_ctr,
)


pctr = ctr_model.predict_proba(
    X_test_ctr
)[:, 1]


auc = roc_auc_score(
    y_test_ctr,
    pctr,
)


prauc = average_precision_score(
    y_test_ctr,
    pctr,
)


ll = log_loss(
    y_test_ctr,
    pctr,
)


print()
print("TARGETING RESULTS")
print("-" * 50)

print(f"ROC-AUC       : {auc:.6f}")
print(f"PR-AUC        : {prauc:.6f}")
print(f"LogLoss       : {ll:.6f}")
print(f"Actual CTR    : {y_test_ctr.mean():.6f}")
print(f"Predicted CTR : {pctr.mean():.6f}")


# ------------------------------------------------------------
# CTR Lift
# ------------------------------------------------------------

ctr_result = pd.DataFrame({
    "click": y_test_ctr,
    "pctr": pctr,
})


print()
print("CTR LIFT")
print("-" * 50)


for ratio in [
    0.01,
    0.05,
    0.10,
    0.20,
    0.50,
]:

    n = max(
        1,
        int(
            len(ctr_result)
            * ratio
        ),
    )

    selected = (
        ctr_result
        .nlargest(
            n,
            "pctr",
        )
    )

    selected_ctr = (
        selected["click"]
        .mean()
    )

    lift = (
        selected_ctr
        / ctr_result["click"].mean()
    )

    print(
        f"Top {ratio:>5.0%}"
        f" | CTR={selected_ctr:.5f}"
        f" | Lift={lift:.2f}x"
    )


# ============================================================
# PART 2
# LOAD AUCTIONNET SUBSET
# ============================================================

print()
print("=" * 70)
print("PART 2 - AUCTIONNET")
print("=" * 70)


print(
    "Opening remote AuctionNet ZIP..."
)


rz = RemoteZip(
    AUCTIONNET_URL
)


names = rz.namelist()


csv_names = [
    x
    for x in names
    if x.endswith(".csv")
]


print(
    "CSV files in archive:",
    csv_names[:10],
)


# Prefer period-7
candidate = None

for name in csv_names:

    if "period-7" in name:
        candidate = name
        break


if candidate is None:

    candidate = csv_names[0]


print(
    "Using:",
    candidate,
)


# Only consume first N_AUCTION rows.
# RemoteZip uses HTTP range requests so the full ZIP
# does not need to be downloaded first.

with rz.open(candidate) as f:

    auction = pd.read_csv(
        f,
        nrows=N_AUCTION,
    )


print(
    "Auction rows:",
    len(auction),
)


print(
    "Columns:",
    auction.columns.tolist(),
)


# ============================================================
# Normalize AuctionNet column names
# ============================================================

# Some versions use advertiserIndex,
# others advertiserNumber.

if (
    "advertiserIndex"
    in auction.columns
    and
    "advertiserNumber"
    not in auction.columns
):

    auction = auction.rename(
        columns={
            "advertiserIndex":
            "advertiserNumber"
        }
    )


required = [
    "deliveryPeriodIndex",
    "advertiserNumber",
    "advertiserCategoryIndex",
    "budget",
    "CPAConstraint",
    "timeStepIndex",
    "remainingBudget",
    "pvIndex",
    "pValue",
    "pValueSigma",
    "bid",
    "cost",
    "conversionAction",
    "leastWinningCost",
]


missing = [
    x
    for x in required
    if x not in auction.columns
]


if missing:

    raise RuntimeError(
        "Missing AuctionNet columns: "
        + str(missing)
    )


# ============================================================
# Basic cleanup
# ============================================================

auction = auction[
    required
].copy()


for col in required:

    auction[col] = pd.to_numeric(
        auction[col],
        errors="coerce",
    )


auction = auction.dropna()


auction = auction[
    auction["budget"] > 0
].copy()


auction = auction[
    auction["CPAConstraint"] > 0
].copy()


auction["remaining_budget_ratio"] = (
    auction["remainingBudget"]
    / auction["budget"]
).clip(
    0,
    1,
)


# AuctionNet period = 48 time steps
auction["time_ratio"] = (
    auction["timeStepIndex"]
    / 47.0
).clip(
    0,
    1,
)


auction["spent_ratio"] = (
    1.0
    - auction[
        "remaining_budget_ratio"
    ]
)


print()
print(
    "Auction data:",
    auction.shape,
)


print(
    "Advertisers:",
    auction[
        "advertiserNumber"
    ].nunique(),
)


# ============================================================
# TRAIN / TEST SPLIT
#
# chronological split
# ============================================================

auction = auction.sort_values(
    [
        "deliveryPeriodIndex",
        "timeStepIndex",
        "pvIndex",
    ]
).reset_index(drop=True)


split = int(
    len(auction)
    * 0.7
)


train_auc = (
    auction.iloc[:split]
    .copy()
)


test_auc = (
    auction.iloc[split:]
    .copy()
)


print(
    "Auction Train:",
    len(train_auc),
)


print(
    "Auction Test :",
    len(test_auc),
)


# ============================================================
# PART 3
# LEARNED AUTO-BIDDING
# ============================================================

print()
print("=" * 70)
print("PART 3 - LEARNED AUTO-BIDDING")
print("=" * 70)


# ------------------------------------------------------------
# Core idea
#
# pValue = estimated conversion probability
# CPAConstraint = allowed value/cost per conversion
#
# Basic economic value:
#
# expected_value
#   = pValue * CPAConstraint
#
# But instead of manually saying:
#
# bid = expected_value
#
# we define multiple possible bid multipliers
# and learn WHICH multiplier is appropriate.
# ------------------------------------------------------------


def make_bid_features(df):

    return np.column_stack([
        df["pValue"].values,
        df["pValueSigma"].values,

        df[
            "remaining_budget_ratio"
        ].values,

        df[
            "time_ratio"
        ].values,

        df[
            "CPAConstraint"
        ].values,

        df[
            "advertiserCategoryIndex"
        ].values,
    ])


# ------------------------------------------------------------
# Create training target
#
# For every auction, try several possible bids:
#
# base_value = pValue * CPA
#
# candidate_bid = base_value * multiplier
#
# win if:
# candidate_bid >= leastWinningCost
#
# utility if won:
# expected conversion value - market cost
#
# We select multiplier with highest utility.
# ------------------------------------------------------------


def optimal_bid_action(df):

    pvalue = (
        df["pValue"]
        .values
    )

    cpa = (
        df["CPAConstraint"]
        .values
    )

    market_price = (
        df["leastWinningCost"]
        .values
    )


    economic_value = (
        pvalue
        * cpa
    )


    rewards = []


    for multiplier in BID_ACTIONS:

        bid = (
            economic_value
            * multiplier
        )

        win = (
            bid
            >= market_price
        )


        utility = np.where(
            win,
            economic_value
            - market_price,
            0.0,
        )


        # penalize negative utility
        utility = np.where(
            win
            & (
                economic_value
                < market_price
            ),
            economic_value
            - market_price,
            utility,
        )


        rewards.append(
            utility
        )


    rewards = np.stack(
        rewards,
        axis=1,
    )


    best_action = (
        np.argmax(
            rewards,
            axis=1,
        )
    )


    return best_action


X_bid_train = make_bid_features(
    train_auc
)


X_bid_test = make_bid_features(
    test_auc
)


y_bid_train = optimal_bid_action(
    train_auc
)


y_bid_test = optimal_bid_action(
    test_auc
)


print(
    "Bid action distribution:",
    np.bincount(
        y_bid_train,
        minlength=len(BID_ACTIONS),
    ),
)


# ------------------------------------------------------------
# Learned bidding policy
# ------------------------------------------------------------

bid_model = (
    HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=SEED,
    )
)


bid_model.fit(
    X_bid_train,
    y_bid_train,
)


pred_action = bid_model.predict(
    X_bid_test
)


bid_action_acc = accuracy_score(
    y_bid_test,
    pred_action,
)


print(
    f"Bid-policy action accuracy: "
    f"{bid_action_acc:.4f}"
)


# ------------------------------------------------------------
# Convert action -> actual bid
# ------------------------------------------------------------

def learned_raw_bid(
    model,
    df,
):

    X = make_bid_features(df)

    action_index = (
        model.predict(X)
    )

    multiplier = (
        BID_ACTIONS[
            action_index
        ]
    )

    economic_value = (
        df["pValue"].values
        * df["CPAConstraint"].values
    )

    bid = (
        economic_value
        * multiplier
    )


    return (
        bid,
        multiplier,
    )


train_raw_bid, _ = (
    learned_raw_bid(
        bid_model,
        train_auc,
    )
)


test_raw_bid, test_bid_mult = (
    learned_raw_bid(
        bid_model,
        test_auc,
    )
)


test_auc[
    "learned_raw_bid"
] = test_raw_bid


# ============================================================
# PART 4
# LEARNED PACING
# ============================================================

print()
print("=" * 70)
print("PART 4 - LEARNED PACING")
print("=" * 70)


# ------------------------------------------------------------
# Idea:
#
# Auto-bidding:
#   "How valuable is THIS auction?"
#
# Pacing:
#   "Given current budget/time,
#    how aggressive should we be?"
#
#
# We infer historical pacing behavior:
#
# pacing multiplier
#   ≈ historical bid / learned raw bid
#
# then learn this multiplier from campaign state.
# ------------------------------------------------------------


train_auc = train_auc.copy()


train_auc[
    "learned_raw_bid"
] = train_raw_bid


eps = 1e-6


train_auc[
    "historical_pacing_mult"
] = (
    train_auc["bid"]
    /
    (
        train_auc[
            "learned_raw_bid"
        ]
        + eps
    )
)


# Avoid extreme ratios.
train_auc[
    "historical_pacing_mult"
] = (
    train_auc[
        "historical_pacing_mult"
    ]
    .clip(
        0.25,
        4.0,
    )
)


# ------------------------------------------------------------
# Aggregate impression-level auctions into
# campaign/time-step states.
# ------------------------------------------------------------

state_train = (
    train_auc
    .groupby(
        [
            "deliveryPeriodIndex",
            "advertiserNumber",
            "timeStepIndex",
        ]
    )
    .agg(

        remaining_budget_ratio=(
            "remaining_budget_ratio",
            "first",
        ),

        spent_ratio=(
            "spent_ratio",
            "first",
        ),

        time_ratio=(
            "time_ratio",
            "first",
        ),

        avg_pvalue=(
            "pValue",
            "mean",
        ),

        avg_raw_bid=(
            "learned_raw_bid",
            "mean",
        ),

        avg_market_price=(
            "leastWinningCost",
            "mean",
        ),

        pacing_target=(
            "historical_pacing_mult",
            "median",
        ),

    )
    .reset_index()
)


PACING_FEATURES = [
    "remaining_budget_ratio",
    "spent_ratio",
    "time_ratio",
    "avg_pvalue",
    "avg_raw_bid",
    "avg_market_price",
]


X_pacing = (
    state_train[
        PACING_FEATURES
    ]
    .values
)


y_pacing = (
    state_train[
        "pacing_target"
    ]
    .values
)


# ------------------------------------------------------------
# chronological state split
# ------------------------------------------------------------

p_split = int(
    len(state_train)
    * 0.8
)


X_pace_train = (
    X_pacing[:p_split]
)


X_pace_val = (
    X_pacing[p_split:]
)


y_pace_train = (
    y_pacing[:p_split]
)


y_pace_val = (
    y_pacing[p_split:]
)


pacing_model = (
    HistGradientBoostingRegressor(
        max_iter=100,
        learning_rate=0.05,
        max_leaf_nodes=15,
        random_state=SEED,
    )
)


pacing_model.fit(
    X_pace_train,
    y_pace_train,
)


pace_pred = pacing_model.predict(
    X_pace_val
)


pace_mae = mean_absolute_error(
    y_pace_val,
    pace_pred,
)


print(
    f"Pacing multiplier MAE: "
    f"{pace_mae:.4f}"
)


# ============================================================
# Make pacing state for TEST
# ============================================================

state_test = (
    test_auc
    .groupby(
        [
            "deliveryPeriodIndex",
            "advertiserNumber",
            "timeStepIndex",
        ]
    )
    .agg(

        remaining_budget_ratio=(
            "remaining_budget_ratio",
            "first",
        ),

        spent_ratio=(
            "spent_ratio",
            "first",
        ),

        time_ratio=(
            "time_ratio",
            "first",
        ),

        avg_pvalue=(
            "pValue",
            "mean",
        ),

        avg_raw_bid=(
            "learned_raw_bid",
            "mean",
        ),

        avg_market_price=(
            "leastWinningCost",
            "mean",
        ),

    )
    .reset_index()
)


state_test[
    "learned_pacing_mult"
] = (
    pacing_model.predict(
        state_test[
            PACING_FEATURES
        ].values
    )
)


state_test[
    "learned_pacing_mult"
] = (
    state_test[
        "learned_pacing_mult"
    ]
    .clip(
        0.25,
        4.0,
    )
)


# ------------------------------------------------------------
# Join state-level pacing multiplier
# back to auction rows
# ------------------------------------------------------------

test_auc = test_auc.merge(

    state_test[
        [
            "deliveryPeriodIndex",
            "advertiserNumber",
            "timeStepIndex",
            "learned_pacing_mult",
        ]
    ],

    on=[
        "deliveryPeriodIndex",
        "advertiserNumber",
        "timeStepIndex",
    ],

    how="left",
)


test_auc[
    "final_bid"
] = (

    test_auc[
        "learned_raw_bid"
    ]

    *

    test_auc[
        "learned_pacing_mult"
    ]

)


# ============================================================
# PART 5
# OFFLINE AUCTION REPLAY
# ============================================================

print()
print("=" * 70)
print("PART 5 - OFFLINE AUCTION REPLAY")
print("=" * 70)


def evaluate_policy(
    df,
    bid_column,
    name,
):

    bid = (
        df[bid_column]
        .values
    )


    market_price = (
        df[
            "leastWinningCost"
        ]
        .values
    )


    pvalue = (
        df["pValue"]
        .values
    )


    cpa_constraint = (
        df["CPAConstraint"]
        .values
    )


    win = (
        bid
        >= market_price
    )


    spend = (
        market_price[win]
        .sum()
    )


    expected_conversions = (
        pvalue[win]
        .sum()
    )


    expected_value = (
        (
            pvalue[win]
            *
            cpa_constraint[win]
        )
        .sum()
    )


    utility = (
        expected_value
        - spend
    )


    win_rate = (
        win.mean()
    )


    expected_cpa = (
        spend
        /
        max(
            expected_conversions,
            1e-8,
        )
    )


    return {
        "Policy": name,

        "Win Rate":
            win_rate,

        "Spend":
            spend,

        "Expected Conv":
            expected_conversions,

        "Expected CPA":
            expected_cpa,

        "Expected Value":
            expected_value,

        "Utility":
            utility,
    }


# ------------------------------------------------------------
# Baseline 1:
# simple value-based bid
# ------------------------------------------------------------

test_auc[
    "value_bid"
] = (
    test_auc["pValue"]
    *
    test_auc["CPAConstraint"]
)


# ------------------------------------------------------------
# Baseline 2:
# historical logged bid
# ------------------------------------------------------------

results = []


results.append(
    evaluate_policy(
        test_auc,
        "value_bid",
        "Value Bid",
    )
)


results.append(
    evaluate_policy(
        test_auc,
        "bid",
        "Historical Bid",
    )
)


# ------------------------------------------------------------
# Learned Auto-Bid only
# ------------------------------------------------------------

results.append(
    evaluate_policy(
        test_auc,
        "learned_raw_bid",
        "Learned Auto-Bid",
    )
)


# ------------------------------------------------------------
# Learned Auto-Bid + Learned Pacing
# ------------------------------------------------------------

results.append(
    evaluate_policy(
        test_auc,
        "final_bid",
        "Auto-Bid + Pacing",
    )
)


result_df = pd.DataFrame(
    results
)


print()
print(
    result_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# ============================================================
# PART 6
# PACING DIAGNOSTICS
# ============================================================

print()
print("=" * 70)
print("PART 6 - PACING EXAMPLES")
print("=" * 70)


cols = [
    "timeStepIndex",
    "remaining_budget_ratio",
    "spent_ratio",
    "avg_pvalue",
    "learned_pacing_mult",
]


print(
    state_test[
        cols
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    f"""
Targeting
---------
ROC-AUC : {auc:.4f}
PR-AUC  : {prauc:.4f}
LogLoss : {ll:.4f}

Auto-Bidding
------------
Learned bid action accuracy:
{bid_action_acc:.4f}

Pacing
------
Pacing multiplier MAE:
{pace_mae:.4f}

See auction replay table above for:
- Win Rate
- Spend
- Expected Conversions
- Expected CPA
- Utility
"""
)
