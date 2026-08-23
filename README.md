# Advertising System Experiment

## 1. Overview

This experiment implements three core components of an advertising system:

1. **Targeting** — predict how valuable an impression is.
2. **Auto-Bidding** — determine how much to bid for the impression.
3. **Pacing** — control bidding aggressiveness over time under a limited budget.

The overall goal is not simply to maximize CTR/CVR, but to obtain as many valuable
clicks/conversions as possible while efficiently using the campaign budget.

---

# 2. Architecture

```text
User / Ad / Context
        │
        ▼
┌──────────────────┐
│    Targeting     │
│   CTR/CVR Model  │
└────────┬─────────┘
         │
         │ pCTR / pCVR
         ▼
┌──────────────────┐
│   Auto-Bidding   │
│ Continuous Model │
└────────┬─────────┘
         │
         │ Raw Bid
         ▼
┌──────────────────┐
│      Pacing      │
│    Q-Learning    │
└────────┬─────────┘
         │
         │ Pacing Multiplier
         ▼

Final Bid = Raw Bid × Pacing Multiplier

         │
         ▼
      Auction
         │
    ┌────┴────┐
   Win       Lose
    │
    ▼
 Impression
    │
    ▼
Conversion / Cost
````

In simple terms:

* **Targeting:** "Is this impression valuable?"
* **Auto-Bidding:** "How much should I pay for it?"
* **Pacing:** "Given my remaining budget and time, how aggressively should I bid?"

---

# 3. Data

## Criteo Click Logs

Used for the initial targeting experiment.

Each row represents an ad impression and contains:

* 13 numerical features
* 26 categorical features
* `label`: click = 1, no click = 0

Subset:

```text
Train: 200,000
Test : 20,000
CTR  : 3.29%
```

## AuctionNet

Used for the end-to-end bidding and pacing experiment.

Important fields include:

```text
pValue                 predicted conversion probability
CPAConstraint           campaign CPA constraint
budget                  campaign budget
remainingBudget         remaining budget
timeStepIndex           current campaign time step
bid                     logged bid
leastWinningCost        minimum price required to win
cost                    actual cost
conversionAction        conversion label
```

A subset of 200,000 auctions from `period-7` was used.

---

# 4. Targeting

## Goal

Predict the probability that an impression will generate a click or conversion.

### Input → Output

```text
User / Ad / Context Features
            ↓
       Targeting Model
            ↓
       pCTR / pCVR
```

### Algorithm

The Criteo experiment uses logistic regression with feature hashing:

```text
39 raw features
      ↓
Feature Hashing
      ↓
262,144-dimensional sparse vector
      ↓
Logistic Regression
      ↓
pCTR
```

The AuctionNet experiment instead learns conversion probability using a
Gradient Boosting Classifier.

```text
pValue
Advertiser Category
Time
Remaining Budget
       ↓
Gradient Boosting
       ↓
pCVR
```

## Criteo Results

```text
ROC-AUC : 0.6992
PR-AUC  : 0.0882
LogLoss : 0.1420

Actual CTR    : 0.0334
Predicted CTR : 0.0312
```

CTR ranking performance:

```text
Top  1% → CTR 0.2100 → 6.29x lift
Top  5% → CTR 0.1280 → 3.83x lift
Top 10% → CTR 0.0955 → 2.86x lift
Top 20% → CTR 0.0738 → 2.21x lift
Top 50% → CTR 0.0506 → 1.51x lift
```

The model therefore successfully ranks high-click-probability impressions.

## AuctionNet pCVR Results

```text
ROC-AUC : 0.5311
PR-AUC  : 0.0149
LogLoss : 0.0755

Actual CVR    : 0.0133
Predicted CVR : 0.0144
```

Conversion prediction is substantially harder and much more sparse than the
Criteo click prediction task.

---

# 5. Auto-Bidding

## Goal

Convert the predicted value of an impression into a monetary bid.

```text
pCVR + Campaign Information
            ↓
      Auto-Bidding
            ↓
         Raw Bid
```

A simplified expected economic value is:

Expected Value = pCVR × CPA Constraint

For example:

```text
pCVR = 0.02
CPA  = 10

Expected Value = 0.2
```

The model should avoid paying more than the expected value of the impression.

## Algorithm

The final experiment uses **continuous bid regression**.

For training, `leastWinningCost` is used to determine whether an auction is
economically worthwhile.

```text
if Expected Value > Least Winning Cost:

    Oracle Bid ≈ Least Winning Cost

else:

    Oracle Bid = 0
```

Then:

```text
pCVR
pValue
CPA Constraint
Advertiser Category
Time
       ↓
Gradient Boosting Regressor
       ↓
Continuous Raw Bid
```

Unlike the earlier experiment, bidding is not restricted to actions such as
`0.5x`, `1x`, `2x`, or `3x`.

Example outputs:

```text
raw_bid = 0.1309
raw_bid = 0.0821
raw_bid = 0.0006
```

### Result

```text
Continuous Bid MAE: 0.0176
```

---

# 6. Pacing

## Goal

Auto-bidding decides how much an individual impression is worth.

Pacing solves a different problem:

> How aggressively should the campaign spend its budget at the current time?

For example:

```text
50% of campaign time passed
90% of budget remaining

→ spending too slowly
→ increase bidding aggressiveness
```

Conversely:

```text
50% of campaign time passed
20% of budget remaining

→ spending too quickly
→ decrease bidding aggressiveness
```

## Input → Output

```text
Remaining Budget Ratio
Time Progress
Recent Spending Velocity
          ↓
      Q-Learning
          ↓
 Pacing Multiplier
```

Possible actions:

```text
0.50x
0.75x
1.00x
1.25x
1.50x
2.00x
```

The final bid is:

Final Bid = Raw Bid × Pacing Multiplier

## Q-Learning

Unlike the previous supervised pacing experiment, there is **no pacing label**.

Instead, the agent interacts with chronological auction data:

```text
State
  ↓
Choose Pacing Multiplier
  ↓
Auction Replay
  ↓
Spend + Expected Conversion
  ↓
Reward
  ↓
Next State
```

Reward:

```text
Reward =
    Expected Conversions
    - 0.05 × Pacing Error
```

Therefore the policy learns to obtain conversions while considering the
campaign's spending trajectory.

---

# 7. Final Auction Replay

Three strategies were compared:

| Strategy                |   Spend | Budget Util. | Expected Conv. | Expected CPA |   Win Rate | Pacing MAE |
| ----------------------- | ------: | -----------: | -------------: | -----------: | ---------: | ---------: |
| Learned Bid Only        | 2096.02 |        4.46% |         766.57 |     **2.73** |     27.96% | **0.0627** |
| Bid + Rule Pacing       | 2096.02 |        4.46% |         766.57 |     **2.73** |     27.96% | **0.0627** |
| Bid + Q-Learning Pacing | 6085.84 |   **12.96%** |    **1735.08** |         3.51 | **71.81%** |     0.2078 |

The learned pacing policy increased expected conversions from:

```text
766.57 → 1735.08
```

approximately **2.26x**.

However, this came with higher spending and worse expected CPA:

```text
CPA: 2.73 → 3.51
```

Therefore, the RL policy learned to bid more aggressively and acquire more
conversions, but the reward function still needs stronger CPA/budget/pacing
constraints.

---

# 8. Summary

```text
TARGETING
Input:
    User / Ad / Context

Algorithm:
    Logistic Regression / Gradient Boosting

Output:
    pCTR / pCVR

Purpose:
    Estimate impression value


AUTO-BIDDING
Input:
    pCVR + CPA + Auction Context

Algorithm:
    Continuous Gradient Boosting Regression

Output:
    Raw Bid

Purpose:
    Decide how much the impression is worth paying for


PACING
Input:
    Remaining Budget + Time + Recent Spend

Algorithm:
    Q-Learning

Output:
    Pacing Multiplier

Purpose:
    Control how aggressively the campaign spends over time
```

Final pipeline:

```text
Features
   ↓
Targeting
   ↓
pCTR / pCVR
   ↓
Auto-Bidding
   ↓
Raw Bid
   ↓
Pacing
   ↓
Final Bid
   ↓
Auction
   ↓
Win / Lose
   ↓
Cost + Conversion
```

The experiment demonstrates that advertising optimization consists of three
different problems:

**Targeting predicts value, auto-bidding converts value into a price, and
pacing allocates a limited campaign budget over time.**

```

수치는 실제 실행 결과를 그대로 반영했습니다. 특히 첫 번째 Criteo targeting은 AUC `0.6992`와 Top-1% lift `6.29x`였고 :contentReference[oaicite:0]{index=0}, 최종 AuctionNet 실험에서는 Q-learning pacing이 expected conversion을 `766.57 → 1735.08`로 높이는 대신 CPA와 pacing error가 악화되는 trade-off가 나타났습니다. :contentReference[oaicite:1]{index=1}
```

