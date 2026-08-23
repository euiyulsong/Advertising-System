# ads_targeting_streaming.py
#
# pip install -U datasets scikit-learn pandas numpy

import numpy as np
import pandas as pd

from datasets import load_dataset
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score


# ============================================================
# CONFIG
# ============================================================

N_TRAIN = 200_000
N_TEST = 20_000

HASH_DIM = 2 ** 18
SEED = 42


# ============================================================
# 1. STREAM REAL AD DATA
# ============================================================
#
# Example: public Criteo-style CTR dataset on Hugging Face.
#
# streaming=True:
#   전체 dataset 다운로드 X
#
# take():
#   앞에서 N개만 실제로 읽음
# ============================================================

DATASET_NAME = "criteo/CriteoClickLogs"

print("Loading streaming dataset...")

ds = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True,
)


# shuffle buffer는 전체 데이터를 다운로드하는 게 아니라
# 작은 buffer 안에서만 randomization
ds = ds.shuffle(
    seed=SEED,
    buffer_size=10_000,
)


# 총 60k만 읽음
rows = list(
    ds.take(N_TRAIN + N_TEST)
)

df = pd.DataFrame(rows)

print("Loaded rows:", len(df))
print("Columns:")
print(df.columns.tolist())

print("\nExample:")
print(df.iloc[0])


# ============================================================
# 2. AUTO-DETECT LABEL
# ============================================================

possible_labels = [
    "label",
    "click",
    "clicked",
    "target",
]

LABEL = None

for col in possible_labels:
    if col in df.columns:
        LABEL = col
        break

if LABEL is None:
    raise RuntimeError(
        f"Could not find click label. "
        f"Columns={df.columns.tolist()}"
    )

print("\nLabel column:", LABEL)


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

train_df = df.iloc[:N_TRAIN].copy()
test_df = df.iloc[N_TRAIN:].copy()

print("Train:", train_df.shape)
print("Test :", test_df.shape)

print(
    "Train CTR:",
    train_df[LABEL].astype(float).mean()
)


# ============================================================
# 4. FEATURE ENGINEERING
#
# Criteo류 dataset:
#
# numerical:
#   I1 ... I13
#
# categorical:
#   C1 ... C26
#
# 같은 구조를 자동으로 처리
# ============================================================

ignore_cols = {
    LABEL,
}

feature_cols = [
    c
    for c in df.columns
    if c not in ignore_cols
]

print("\nFeature count:", len(feature_cols))


def row_to_features(row):

    feats = {}

    for col in feature_cols:

        value = row[col]

        if pd.isna(value):
            continue

        # ------------------------------
        # numeric
        # ------------------------------

        if isinstance(
            value,
            (int, float, np.integer, np.floating)
        ):
            feats[col] = np.log1p(max(float(value), 0.0))

        # ------------------------------
        # categorical
        # ------------------------------

        else:
            feats[f"{col}={value}"] = 1.0

    return feats


print("Hashing features...")


train_features = [
    row_to_features(row)
    for _, row in train_df.iterrows()
]

test_features = [
    row_to_features(row)
    for _, row in test_df.iterrows()
]


hasher = FeatureHasher(
    n_features=HASH_DIM,
    input_type="dict",
)


X_train = hasher.transform(
    train_features
)

X_test = hasher.transform(
    test_features
)

y_train = (
    train_df[LABEL]
    .astype(int)
    .to_numpy()
)

y_test = (
    test_df[LABEL]
    .astype(int)
    .to_numpy()
)


print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# ============================================================
# 5. TARGETING MODEL
#
# Online-friendly Logistic Regression
#
# SGDClassifier(loss="log_loss")
# ≈ logistic regression trained with SGD
# ============================================================
model = SGDClassifier(
    loss="log_loss",
    alpha=1e-5,
    max_iter=100,
    tol=1e-4,
    random_state=42,
)

print("\nTraining targeting model...")

model.fit(
    X_train,
    y_train,
)


# ============================================================
# 6. TARGETING EVALUATION
# ============================================================

pctr = model.predict_proba(
    X_test
)[:, 1]


auc = roc_auc_score(
    y_test,
    pctr,
)

pr_auc = average_precision_score(
    y_test,
    pctr,
)

ll = log_loss(
    y_test,
    pctr,
)


print("\n" + "=" * 60)
print("TARGETING / pCTR")
print("=" * 60)

print(f"ROC-AUC : {auc:.6f}")
print(f"PR-AUC  : {pr_auc:.6f}")
print(f"LogLoss : {ll:.6f}")

print(
    f"Actual CTR    : {y_test.mean():.6f}"
)

print(
    f"Predicted CTR : {pctr.mean():.6f}"
)


# ============================================================
# 7. SIMPLE AUTO-BIDDING SIMULATION
#
# 실제 targeting prediction을 사용해서
#
# bid = base_bid * pCTR / avgCTR
#
# 여기서는 auction clearing price가 dataset에 없으면
# "bid score"까지만 계산.
# ============================================================

avg_ctr = y_train.mean()

BASE_BID = 100


bid = (
    BASE_BID
    * pctr
    / max(avg_ctr, 1e-8)
)

bid = np.clip(
    bid,
    0,
    1000,
)


result = pd.DataFrame({
    "click": y_test,
    "pctr": pctr,
    "bid": bid,
})


print("\n" + "=" * 60)
print("AUTO-BIDDING SCORE")
print("=" * 60)

print(result.head(20))


# ============================================================
# 8. TARGETING TOP-K EVALUATION
#
# 광고를 모든 사람에게 보여주는 대신
# predicted CTR 높은 user만 targeting했다고 가정
# ============================================================

for ratio in [
    0.01,
    0.05,
    0.10,
    0.20,
    0.50,
]:

    n = int(
        len(result) * ratio
    )

    selected = (
        result
        .sort_values(
            "pctr",
            ascending=False,
        )
        .head(n)
    )

    ctr = selected[
        "click"
    ].mean()

    lift = (
        ctr
        / max(
            result["click"].mean(),
            1e-8,
        )
    )

    print(
        f"Top {ratio:>5.0%}"
        f" | impressions={n:>6}"
        f" | CTR={ctr:.6f}"
        f" | lift={lift:.2f}x"
    )
