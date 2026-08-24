# Auto Bidding — LightGBM Regression Experiment

## 1. Objective

CTR 기반 bidding과 **LightGBM value-based auto bidding**을 비교하여 동일한 수준의 budget에서 business outcome이 어떻게 변하는지 평가했다.

---

## 2. Model

### Input

```text
campaign
cat1 ~ cat9
hour
day
pCTR
```

총 **13개 feature**를 사용했다.

### Output

```text
Predicted Expected Value
```

LightGBM Regression으로 각 impression의 expected business value를 예측했다.

```text
Input Features + pCTR
        ↓
LightGBM Regressor
        ↓
Predicted Value
        ↓
Bid = multiplier × predicted value
```

### Training

```text
Training samples: 700,000
Objective       : Tweedie Regression
Prediction MAE  : 0.007183
```

---

## 3. Bidding Strategies

**CTR Baseline**

```text
bid = 0.001176 × pCTR
```

CTR이 높은 impression에 더 높은 bid를 할당한다.

**Auto Bidding**

```text
bid = 0.151672 × predicted_value
```

예측된 business value가 높은 impression에 더 높은 bid를 할당한다.

두 multiplier는 validation set에서 **비슷한 budget을 사용하도록 calibration**했다.

Test target budget:

```text
Historical spend = 40.8259
Target budget    = 12.2478 (30%)
```

---

## 4. Business Outcome

| Metric      | CTR Baseline | Auto Bidding |        Change |
| ----------- | -----------: | -----------: | ------------: |
| Impressions |      122,672 |       76,270 |    **-37.8%** |
| Clicks      |       38,071 |       24,599 |    **-35.4%** |
| Conversions |        4,107 |        3,876 |    **-5.62%** |
| CTR         |       31.03% |       32.25% |    **+3.92%** |
| CVR         |       10.79% |       15.76% |   **+46.06%** |
| Spend       |       12.248 |       11.919 |    **-2.69%** |
| Revenue     |       375.82 |       395.52 |    **+5.24%** |
| CPA         |     0.002982 |     0.003075 | **+3.11% ⚠️** |
| ROAS        |        30.68 |        33.18 |    **+8.15%** |
| Profit      |       363.57 |       383.60 |    **+5.51%** |

---

## 5. Interpretation

Auto bidding은 **더 적은 impression을 선택하면서 가치가 높은 traffic에 집중**했다.

```text
Impressions ↓ 37.8%
        ↓
Higher-value traffic selection
        ↓
CVR ↑ 46.1%
Revenue ↑ 5.2%
ROAS ↑ 8.2%
Profit ↑ 5.5%
```

따라서 **value/ROAS 관점에서는 Auto Bidding이 CTR baseline보다 우수**했다.

다만 CPA는 `+3.11%`로 소폭 악화되었고 conversions도 `-5.62%` 감소했다. 즉 이 모델은 **CPA/Conversion 최대화 모델이 아니라 value 최적화 모델**이므로 자연스러운 trade-off다.

> **Conclusion:** Value-based LightGBM bidding improved ROAS by **8.15%** and profit by **5.51%**, while sacrificing **5.62% of conversions** and worsening CPA by **3.11%**. The bidder therefore learned to purchase fewer but higher-value impressions.

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


