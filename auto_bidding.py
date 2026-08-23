# ads_full_learned.py
#
# pip install -U pandas numpy scikit-learn remotezip
#
# Real AuctionNet subset:
#   1. Targeting  -> learned pCVR
#   2. Auto-bid   -> continuous learned bid
#   3. Pacing     -> Q-learning pacing multiplier
#   4. Evaluation -> chronological auction replay

import numpy as np
import pandas as pd

from remotezip import RemoteZip

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    mean_absolute_error,
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42
rng = np.random.default_rng(SEED)

N_ROWS = 200_000

AUCTIONNET_URL = (
    "https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/"
    "share/autoBidding_aigb_track_data_period_7-8.zip"
)

# Pacing action only.
# Bidding itself is CONTINUOUS.
PACING_ACTIONS = np.array([
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    2.00,
])

N_BUDGET_BINS = 10
N_TIME_BINS = 8
N_SPEND_BINS = 6

EPISODES = 20

ALPHA = 0.15
GAMMA = 0.95
EPSILON_START = 0.30
EPSILON_END = 0.03


# ============================================================
# 1. LOAD SMALL REAL AUCTIONNET SUBSET
# ============================================================

print("=" * 70)
print("LOADING AUCTIONNET")
print("=" * 70)

rz = RemoteZip(AUCTIONNET_URL)

csv_files = [
    x for x in rz.namelist()
    if x.endswith(".csv")
]

print("CSV files:")
for x in csv_files[:10]:
    print(" ", x)


# Prefer period-7
filename = next(
    (
        x for x in csv_files
        if "period-7" in x.lower()
    ),
    csv_files[0],
)

print("\nUsing:", filename)


with rz.open(filename) as f:

    df = pd.read_csv(
        f,
        nrows=N_ROWS,
    )


print("Loaded rows:", len(df))
print("Columns:", df.columns.tolist())


# ============================================================
# 2. NORMALIZE COLUMN NAMES
# ============================================================

if (
    "advertiserIndex" in df.columns
    and
    "advertiserNumber" not in df.columns
):
    df = df.rename(
        columns={
            "advertiserIndex":
            "advertiserNumber"
        }
    )


REQUIRED = [
    "deliveryPeriodIndex",
    "advertiserNumber",
    "advertiserCategoryIndex",
    "budget",
    "CPAConstraint",
    "timeStepIndex",
    "remainingBudget",
    "pvIndex",

    "pValue",
    "bid",
    "cost",

    "isExposed",
    "conversionAction",

    "leastWinningCost",
]


missing = [
    c for c in REQUIRED
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing columns: {missing}"
    )


df = df[REQUIRED].copy()


for c in REQUIRED:

    df[c] = pd.to_numeric(
        df[c],
        errors="coerce",
    )


df = df.dropna()

df = df[
    (df["budget"] > 0)
    &
    (df["CPAConstraint"] > 0)
    &
    (df["leastWinningCost"] >= 0)
].copy()


df["remaining_budget_ratio"] = (
    df["remainingBudget"]
    / df["budget"]
).clip(0, 1)


# AuctionNet uses 48 decision steps.
df["time_ratio"] = (
    df["timeStepIndex"]
    / 47.0
).clip(0, 1)


df = df.sort_values([
    "advertiserNumber",
    "deliveryPeriodIndex",
    "timeStepIndex",
    "pvIndex",
]).reset_index(drop=True)


print("\nClean rows:", len(df))
print(
    "Advertisers:",
    df["advertiserNumber"].nunique(),
)


# ============================================================
# TRAIN / TEST SPLIT BY ADVERTISER
#
# Avoid splitting auctions from same advertiser randomly.
# ============================================================

advertisers = (
    df["advertiserNumber"]
    .unique()
)

rng.shuffle(advertisers)

cut = max(
    1,
    int(len(advertisers) * 0.7)
)

train_advertisers = set(
    advertisers[:cut]
)

test_advertisers = set(
    advertisers[cut:]
)


train_df = df[
    df["advertiserNumber"]
    .isin(train_advertisers)
].copy()


test_df = df[
    df["advertiserNumber"]
    .isin(test_advertisers)
].copy()


print(
    "Train auctions:",
    len(train_df)
)

print(
    "Test auctions :",
    len(test_df)
)


# ============================================================
# 3. TARGETING
#
# Here targeting = conversion probability.
#
# AuctionNet already gives pValue.
#
# We learn a contextual calibration model:
#
# pValue + campaign/context
#            ↓
#       learned pCVR
# ============================================================

print("\n" + "=" * 70)
print("TARGETING / pCVR")
print("=" * 70)


TARGET_FEATURES = [
    "pValue",
    "advertiserCategoryIndex",
    "time_ratio",
    "remaining_budget_ratio",
]


# Conversion is only meaningfully observed for exposed ads.
target_train = train_df[
    train_df["isExposed"] == 1
].copy()

target_test = test_df[
    test_df["isExposed"] == 1
].copy()


X_target_train = target_train[
    TARGET_FEATURES
].values

y_target_train = target_train[
    "conversionAction"
].astype(int).values


X_target_test = target_test[
    TARGET_FEATURES
].values

y_target_test = target_test[
    "conversionAction"
].astype(int).values


print(
    "Target train:",
    len(target_train)
)

print(
    "Conversion rate:",
    y_target_train.mean()
)


target_model = (
    HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=SEED,
    )
)


target_model.fit(
    X_target_train,
    y_target_train,
)


p_test = target_model.predict_proba(
    X_target_test
)[:, 1]


if len(np.unique(y_target_test)) > 1:

    auc = roc_auc_score(
        y_target_test,
        p_test,
    )

    pr_auc = average_precision_score(
        y_target_test,
        p_test,
    )

    ll = log_loss(
        y_target_test,
        p_test,
    )

else:

    auc = np.nan
    pr_auc = np.nan
    ll = np.nan


print(f"ROC-AUC : {auc:.4f}")
print(f"PR-AUC  : {pr_auc:.4f}")
print(f"LogLoss : {ll:.4f}")

print(
    "Actual CVR   :",
    y_target_test.mean(),
)

print(
    "Predicted CVR:",
    p_test.mean(),
)


# ============================================================
# Apply learned pCVR to ALL auctions
# ============================================================

train_df["pcvr"] = (
    target_model.predict_proba(
        train_df[
            TARGET_FEATURES
        ].values
    )[:, 1]
)


test_df["pcvr"] = (
    target_model.predict_proba(
        test_df[
            TARGET_FEATURES
        ].values
    )[:, 1]
)


# ============================================================
# 4. AUTO-BIDDING
#
# No discrete 3x action.
#
# We construct a CONTINUOUS oracle bid.
#
# economic_value:
#
#     pCVR * target CPA
#
# If market price < value:
#     profitable auction
#     oracle bid slightly above winning price
#
# Otherwise:
#     do not buy.
#
# Then train regression:
#
# context -> continuous bid
# ============================================================

print("\n" + "=" * 70)
print("CONTINUOUS AUTO-BIDDING")
print("=" * 70)


def add_oracle_bid(data):

    data = data.copy()

    data["economic_value"] = (
        data["pcvr"]
        * data["CPAConstraint"]
    )


    profitable = (
        data["economic_value"]
        >
        data["leastWinningCost"]
    )


    # Minimum bid necessary to win.
    #
    # 1.001 adds a tiny safety margin.
    oracle_bid = np.where(
        profitable,
        data["leastWinningCost"] * 1.001,
        0.0,
    )


    # Never bid above our predicted value.
    oracle_bid = np.minimum(
        oracle_bid,
        data["economic_value"],
    )


    data["oracle_bid"] = (
        oracle_bid
    )

    return data


train_df = add_oracle_bid(
    train_df
)

test_df = add_oracle_bid(
    test_df
)


BID_FEATURES = [
    "pcvr",
    "pValue",
    "CPAConstraint",
    "advertiserCategoryIndex",
    "time_ratio",
]


X_bid_train = train_df[
    BID_FEATURES
].values


# log transform because bid distribution is skewed
y_bid_train = np.log1p(
    train_df["oracle_bid"].values
)


bid_model = (
    HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=SEED,
    )
)


bid_model.fit(
    X_bid_train,
    y_bid_train,
)


def predict_bid(model, data):

    log_bid = model.predict(
        data[
            BID_FEATURES
        ].values
    )

    bid = np.expm1(
        log_bid
    )

    # Bid can't be negative.
    bid = np.maximum(
        bid,
        0.0,
    )

    # Economic safety cap:
    # never bid above expected value.
    value = (
        data["pcvr"].values
        * data["CPAConstraint"].values
    )

    return np.minimum(
        bid,
        value,
    )


train_df["raw_bid"] = predict_bid(
    bid_model,
    train_df,
)


test_df["raw_bid"] = predict_bid(
    bid_model,
    test_df,
)


bid_mae = mean_absolute_error(
    test_df["oracle_bid"],
    test_df["raw_bid"],
)


print(
    f"Continuous Bid MAE: "
    f"{bid_mae:.4f}"
)


print("\nExamples:")

print(
    test_df[
        [
            "pcvr",
            "CPAConstraint",
            "leastWinningCost",
            "oracle_bid",
            "raw_bid",
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# 5. PACING ENVIRONMENT
#
# This time pacing is NOT trained from:
#
# historical_bid / raw_bid
#
# Instead we run campaigns sequentially and learn from reward.
#
#
# State:
#
# budget remaining
# elapsed time
# spend velocity
#
#
# Action:
#
# pacing multiplier
#
#
# final_bid = raw_bid * multiplier
# ============================================================


def discretize(
    remaining_ratio,
    time_ratio,
    recent_spend_ratio,
):

    b = min(
        int(
            remaining_ratio
            * N_BUDGET_BINS
        ),
        N_BUDGET_BINS - 1,
    )

    t = min(
        int(
            time_ratio
            * N_TIME_BINS
        ),
        N_TIME_BINS - 1,
    )

    s = min(
        int(
            recent_spend_ratio
            * N_SPEND_BINS
        ),
        N_SPEND_BINS - 1,
    )

    return b, t, s


Q = np.zeros(
    (
        N_BUDGET_BINS,
        N_TIME_BINS,
        N_SPEND_BINS,
        len(PACING_ACTIONS),
    ),
    dtype=np.float32,
)


# ============================================================
# 6. BUILD CAMPAIGN EPISODES
# ============================================================


def build_episodes(data):

    episodes = []

    grouped = data.groupby([
        "deliveryPeriodIndex",
        "advertiserNumber",
    ])


    for key, campaign in grouped:

        campaign = (
            campaign
            .sort_values(
                [
                    "timeStepIndex",
                    "pvIndex",
                ]
            )
            .copy()
        )

        if len(campaign) < 20:
            continue

        episodes.append(
            (key, campaign)
        )

    return episodes


train_episodes = build_episodes(
    train_df
)


test_episodes = build_episodes(
    test_df
)


print("\n" + "=" * 70)
print("PACING Q-LEARNING")
print("=" * 70)

print(
    "Train episodes:",
    len(train_episodes)
)

print(
    "Test episodes :",
    len(test_episodes)
)


# ============================================================
# 7. SIMULATE ONE TIME STEP
# ============================================================


def run_time_step(
    step_df,
    remaining_budget,
    pacing_multiplier,
):

    spend = 0.0
    expected_conversions = 0.0

    wins = 0


    for row in step_df.itertuples():

        if remaining_budget <= 0:
            break


        final_bid = (
            row.raw_bid
            * pacing_multiplier
        )


        market_price = (
            row.leastWinningCost
        )


        if final_bid < market_price:
            continue


        # Cannot buy if budget insufficient.
        if market_price > remaining_budget:
            continue


        remaining_budget -= (
            market_price
        )

        spend += market_price

        expected_conversions += (
            row.pcvr
        )

        wins += 1


    return (
        remaining_budget,
        spend,
        expected_conversions,
        wins,
    )


# ============================================================
# 8. TRAIN PACING WITH Q-LEARNING
# ============================================================


def train_pacing_q():

    global Q

    total_steps = (
        EPISODES
        * max(len(train_episodes), 1)
    )

    counter = 0


    for epoch in range(EPISODES):

        rng.shuffle(
            train_episodes
        )


        total_reward = 0.0


        for _, campaign in train_episodes:

            original_budget = float(
                campaign["budget"].iloc[0]
            )

            remaining_budget = (
                original_budget
            )

            previous_step_spend = 0.0


            time_steps = sorted(
                campaign[
                    "timeStepIndex"
                ].unique()
            )


            for i, timestep in enumerate(
                time_steps
            ):

                step_df = campaign[
                    campaign["timeStepIndex"]
                    == timestep
                ]


                remaining_ratio = (
                    remaining_budget
                    / original_budget
                )


                time_ratio = (
                    timestep / 47.0
                )


                recent_spend_ratio = (
                    previous_step_spend
                    / original_budget
                )

                recent_spend_ratio = min(
                    recent_spend_ratio * 48,
                    0.999,
                )


                state = discretize(
                    remaining_ratio,
                    time_ratio,
                    recent_spend_ratio,
                )


                # Linear epsilon decay
                progress = (
                    counter
                    / max(total_steps, 1)
                )

                epsilon = (
                    EPSILON_START
                    +
                    progress
                    * (
                        EPSILON_END
                        - EPSILON_START
                    )
                )


                if rng.random() < epsilon:

                    action_idx = (
                        rng.integers(
                            len(
                                PACING_ACTIONS
                            )
                        )
                    )

                else:

                    action_idx = int(
                        np.argmax(
                            Q[state]
                        )
                    )


                multiplier = (
                    PACING_ACTIONS[
                        action_idx
                    ]
                )


                (
                    new_remaining,
                    spend,
                    expected_conv,
                    wins,
                ) = run_time_step(
                    step_df,
                    remaining_budget,
                    multiplier,
                )


                # --------------------------------------------
                # Reward
                #
                # Primary goal:
                # expected conversions
                #
                # Penalty:
                # spending much faster than elapsed time
                # --------------------------------------------

                spent_total = (
                    original_budget
                    - new_remaining
                )


                spent_ratio = (
                    spent_total
                    / original_budget
                )


                target_ratio = min(
                    (timestep + 1)
                    / 48.0,
                    1.0,
                )


                pacing_error = abs(
                    spent_ratio
                    - target_ratio
                )


                # Main reward = expected conversions.
                #
                # Small penalty keeps spending trajectory sane.
                reward = (
                    expected_conv
                    - 0.05
                    * pacing_error
                )


                done = (
                    i
                    == len(time_steps) - 1
                    or
                    new_remaining <= 0
                )


                next_remaining_ratio = (
                    new_remaining
                    / original_budget
                )


                next_state = discretize(
                    next_remaining_ratio,
                    min(
                        (timestep + 1) / 47.0,
                        1.0,
                    ),
                    min(
                        (
                            spend
                            / original_budget
                        )
                        * 48,
                        0.999,
                    ),
                )


                target = reward

                if not done:

                    target += (
                        GAMMA
                        * np.max(
                            Q[next_state]
                        )
                    )


                old = (
                    Q[
                        state
                        + (action_idx,)
                    ]
                )


                Q[
                    state
                    + (action_idx,)
                ] = (
                    old
                    + ALPHA
                    * (
                        target
                        - old
                    )
                )


                remaining_budget = (
                    new_remaining
                )

                previous_step_spend = (
                    spend
                )

                total_reward += reward

                counter += 1


                if done:
                    break


        print(
            f"Epoch {epoch + 1:02d}"
            f" | reward={total_reward:.4f}"
            f" | epsilon={epsilon:.3f}"
        )


train_pacing_q()


# ============================================================
# 9. POLICY EVALUATION
# ============================================================


def evaluate_campaigns(
    episodes,
    pacing_policy,
    name,
):

    total_budget = 0.0
    total_spend = 0.0

    total_expected_conv = 0.0
    total_wins = 0
    total_auctions = 0

    pacing_errors = []


    for _, campaign in episodes:

        budget = float(
            campaign["budget"].iloc[0]
        )

        remaining_budget = budget

        previous_step_spend = 0.0

        total_budget += budget


        steps = sorted(
            campaign[
                "timeStepIndex"
            ].unique()
        )


        for timestep in steps:

            step_df = campaign[
                campaign[
                    "timeStepIndex"
                ]
                == timestep
            ]


            remaining_ratio = (
                remaining_budget
                / budget
            )


            time_ratio = (
                timestep
                / 47.0
            )


            recent_ratio = min(
                (
                    previous_step_spend
                    / budget
                )
                * 48,
                0.999,
            )


            multiplier = pacing_policy(
                remaining_ratio,
                time_ratio,
                recent_ratio,
            )


            (
                remaining_budget,
                spend,
                expected_conv,
                wins,
            ) = run_time_step(
                step_df,
                remaining_budget,
                multiplier,
            )


            total_spend += spend

            total_expected_conv += (
                expected_conv
            )

            total_wins += wins

            total_auctions += len(
                step_df
            )

            previous_step_spend = spend


            spent_ratio = (
                budget
                - remaining_budget
            ) / budget


            target_ratio = min(
                (timestep + 1)
                / 48.0,
                1.0,
            )


            pacing_errors.append(
                abs(
                    spent_ratio
                    - target_ratio
                )
            )


    utilization = (
        total_spend
        / max(total_budget, 1e-8)
    )


    expected_cpa = (
        total_spend
        / max(
            total_expected_conv,
            1e-8,
        )
    )


    win_rate = (
        total_wins
        / max(total_auctions, 1)
    )


    return {
        "Policy": name,

        "Spend":
            total_spend,

        "Budget Util":
            utilization,

        "Expected Conv":
            total_expected_conv,

        "Expected CPA":
            expected_cpa,

        "Win Rate":
            win_rate,

        "Pacing MAE":
            np.mean(
                pacing_errors
            ),
    }


# ------------------------------------------------------------
# Baseline:
# no pacing
# ------------------------------------------------------------

def no_pacing(
    remaining_ratio,
    time_ratio,
    recent_ratio,
):

    return 1.0


# ------------------------------------------------------------
# Simple hand-written pacing controller
# ------------------------------------------------------------

def rule_pacing(
    remaining_ratio,
    time_ratio,
    recent_ratio,
):

    spent_ratio = (
        1.0
        - remaining_ratio
    )


    target = time_ratio


    error = (
        target
        - spent_ratio
    )


    return float(
        np.clip(
            1.0
            + 2.0
            * error,
            0.5,
            2.0,
        )
    )


# ------------------------------------------------------------
# Learned Q pacing
# ------------------------------------------------------------

def learned_pacing(
    remaining_ratio,
    time_ratio,
    recent_ratio,
):

    state = discretize(
        remaining_ratio,
        time_ratio,
        recent_ratio,
    )


    action = int(
        np.argmax(
            Q[state]
        )
    )


    return float(
        PACING_ACTIONS[action]
    )


results = []

results.append(
    evaluate_campaigns(
        test_episodes,
        no_pacing,
        "Learned Bid Only",
    )
)


results.append(
    evaluate_campaigns(
        test_episodes,
        rule_pacing,
        "Bid + Rule Pacing",
    )
)


results.append(
    evaluate_campaigns(
        test_episodes,
        learned_pacing,
        "Bid + Q-Learning Pacing",
    )
)


result_df = pd.DataFrame(
    results
)


print("\n" + "=" * 70)
print("FINAL AUCTION REPLAY")
print("=" * 70)


print(
    result_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# ============================================================
# 10. SHOW LEARNED PACING POLICY
# ============================================================

print("\n" + "=" * 70)
print("LEARNED PACING EXAMPLES")
print("=" * 70)


examples = [
    (0.90, 0.25, 0.10),
    (0.70, 0.50, 0.10),
    (0.40, 0.50, 0.10),
    (0.40, 0.75, 0.10),
    (0.10, 0.90, 0.10),
]


for (
    remaining,
    time_ratio,
    recent,
) in examples:

    action = learned_pacing(
        remaining,
        time_ratio,
        recent,
    )

    print(
        f"remaining={remaining:.2f}"
        f" time={time_ratio:.2f}"
        f" recent_spend={recent:.2f}"
        f" -> multiplier={action:.2f}"
    )


print("\nDONE")
