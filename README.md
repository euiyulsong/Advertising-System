# Advertising-System
# Advertising Targeting & Auto-Bidding Experiment

A small experiment to understand **how an advertising system decides who should see an ad and how much to bid for that impression**.

This experiment uses a subset of a **real public advertising click dataset**, rather than synthetic data.

The main pipeline is:

```text id="tmdv8v"
Advertising Impression
        │
        ▼
User / Ad / Context Features
        │
        ▼
   Targeting Model
    (CTR Prediction)
        │
        ▼
       pCTR
        │
        ├──────────────► Targeting
        │                "Should we prioritize this impression?"
        │
        ▼
   Auto-Bidding
"How much should we bid?"
        │
        ▼
       Bid
```

The basic idea is simple:

> **Targeting estimates how valuable an impression is. Auto-bidding converts that estimated value into a bid.**

---

## 1. Why Do We Need These Models?

Suppose millions of users visit websites and apps every second.

An advertiser cannot simply show an ad to everyone. Some users are much more likely to click or convert than others.

For example:

```text id="60kbmv"
                     Predicted CTR

User A                  10%
User B                   3%
User C                  0.2%
```

The advertising system should generally prioritize **User A**.

This is the role of the **targeting model**.

But online advertising usually involves an auction. Multiple advertisers may compete for the same impression.

Therefore, after estimating the value of an impression, the system must also decide:

> How much am I willing to pay for this impression?

This is the role of **auto-bidding**.

---

# 2. Dataset

The experiment uses real advertising click data loaded through Hugging Face Datasets.

Instead of downloading the complete dataset, it uses:

```python id="tvlr9r"
streaming=True
```

and consumes only:

```text id="pnl9en"
220,000 rows

Train : 200,000
Test  :  20,000
```

This makes the experiment small enough to run locally while still using real advertising data.

## What Does One Row Mean?

Conceptually, each row represents **one advertising impression opportunity**.

For example:

```text id="vxkw10"
label                       0

integer_feature_1         5.0
integer_feature_2         NaN
integer_feature_3         5.0
...
integer_feature_12     1828.0

categorical_feature_1    788a5d5b
categorical_feature_2    98e2c109
categorical_feature_3    3cd7902e
...
categorical_feature_26   b757e957
```

There are:

```text id="62mqud"
13 numerical features
        +
26 categorical features
        =
39 input features
```

plus:

```text id="5a2yma"
label
```

which indicates whether the impression received a click.

```text id="drlb03"
label = 1 → Click
label = 0 → No Click
```

The actual feature meanings are anonymized.

Conceptually, advertising features can represent information such as:

```text id="htjxhw"
User information
Device / browser
Publisher / website
Advertisement
Ad slot
Context
Historical activity
```

Categorical values such as:

```text id="b2n4l6"
788a5d5b
98e2c109
3cd7902e
```

are anonymized category IDs.

Missing numerical values are represented as `NaN`, while some categorical values are `None`.

---

# 3. Data Distribution

The training set contains:

```text id="40mbp6"
Train samples : 200,000
Test samples  : 20,000

Train CTR     : 3.286%
Test CTR      : 3.340%
```

So only about **3 out of every 100 impressions are clicked**.

This is typical of CTR prediction: the dataset is highly imbalanced.

```text id="k04ps5"
No Click █████████████████████████████████  ~97%
Click    █                                  ~3%
```

The goal is not simply to predict `0` for everything.

Instead, we want to **rank impressions according to their probability of being clicked**.

---

# 4. Feature Hashing

Advertising datasets contain many categorical features.

For example, imagine millions of different:

```text id="ihgsp3"
users
websites
campaigns
advertisements
devices
publishers
```

Creating a separate feature dimension for every possible category would produce an enormous vocabulary.

Therefore, this experiment uses **Feature Hashing**.

```text id="52sk1d"
39 Raw Features
      │
      ▼
Feature Hasher
      │
      ▼
262,144-dimensional
Sparse Vector
```

The experiment produced:

```text id="zjdy0n"
X_train : (200000, 262144)
X_test  : (20000, 262144)
```

This does **not** mean every sample stores 262,144 numbers.

The vectors are sparse: only a small number of positions are active.

For example:

```text id="7gy3ec"
categorical_feature_1=788a5d5b
              │
              ▼
        Hash Function
              │
              ▼
       dimension 18423


categorical_feature_2=98e2c109
              │
              ▼
        Hash Function
              │
              ▼
       dimension 92341
```

Feature hashing is useful because it is:

```text id="j8l97k"
Fast
Memory efficient
No vocabulary required
Good for high-cardinality features
Suitable for streaming systems
```

---

# 5. Targeting Model

The first model is a **CTR prediction model**.

Its job is:

> Given this impression, what is the probability that the user will click?

Formally:

[
pCTR=P(click=1|user,ad,context)
]

The architecture is:

```text id="n1o51e"
Numerical Features ──────┐
                         │
Categorical Features ────┤
                         ▼
                  Feature Hashing
                         │
                         ▼
                Sparse Feature Vector
                         │
                         ▼
               Logistic Regression
                         │
                         ▼
                       pCTR
```

For example:

```text id="6ifxzm"
Impression A → pCTR = 0.1307 → 13.07%
Impression B → pCTR = 0.0458 →  4.58%
Impression C → pCTR = 0.0006 →  0.06%
```

Therefore the model considers A much more attractive than C.

---

# 6. Targeting Algorithm

The model is logistic regression trained with SGD.

Conceptually:

[
z=w^Tx+b
]

and:

[
pCTR=\sigma(z)
]

where:

[
\sigma(z)=\frac{1}{1+e^{-z}}
]

The easiest way to think about it is:

```text id="oc03c6"
Features
   │
   ▼
Weighted Sum
   │
   ▼
Sigmoid
   │
   ▼
Number between 0 and 1
   │
   ▼
Probability of Click
```

During training:

```text id="jym5gm"
Features
   │
   ▼
Predict pCTR
   │
   ▼
Compare with actual click
   │
   ▼
Binary Cross-Entropy / Log Loss
   │
   ▼
Update weights using SGD
```

So if:

```text id="0whx13"
Actual label = 1
Predicted    = 0.02
```

the model receives a large loss.

If:

```text id="g1q6lr"
Actual label = 1
Predicted    = 0.90
```

the loss is much smaller.

---

# 7. Targeting Results

The new experiment produced:

| Metric        |     Result |
| ------------- | ---------: |
| **ROC-AUC**   | **0.6992** |
| **PR-AUC**    | **0.0882** |
| **LogLoss**   | **0.1420** |
| Actual CTR    |  **3.34%** |
| Predicted CTR |  **3.12%** |

This is a large improvement over the previous 50K training experiment.

## ROC-AUC = 0.699

ROC-AUC measures ranking ability.

Random ranking:

[
AUC=0.5
]

Current model:

[
AUC=0.699
]

So the model has learned meaningful signals for distinguishing high-CTR impressions from low-CTR impressions.

An intuitive interpretation is:

> If we randomly choose one clicked impression and one non-clicked impression, the model gives the clicked impression a higher score roughly 70% of the time.

---

## PR-AUC = 0.088

PR-AUC is particularly useful because clicks are rare.

The positive rate is only:

```text id="h64pf4"
~3.34%
```

while:

```text id="tbk0mc"
PR-AUC = 8.82%
```

So the model performs meaningfully better than the prevalence-level baseline.

---

## LogLoss = 0.142

LogLoss evaluates the actual predicted probabilities, rather than only their ordering.

Lower is better.

The calibration also looks reasonably aligned at the aggregate level:

```text id="fz2ob3"
Actual CTR    = 3.340%
Predicted CTR = 3.117%
```

The difference is only about:

```text id="fgv6lm"
0.22 percentage points
```

So unlike the first experiment, the model is no longer predicting almost everything as zero.

---

# 8. Does Targeting Actually Work?

AUC can feel abstract, so we also perform a much easier test.

Take the impressions with the **highest predicted CTR** and measure their actual CTR.

Results:

| Selected Traffic | Impressions | Actual CTR |  CTR Lift |
| ---------------- | ----------: | ---------: | --------: |
| **Top 1%**       |         200 | **21.00%** | **6.29×** |
| **Top 5%**       |       1,000 | **12.80%** | **3.83×** |
| **Top 10%**      |       2,000 |  **9.55%** | **2.86×** |
| **Top 20%**      |       4,000 |  **7.38%** | **2.21×** |
| **Top 50%**      |      10,000 |  **5.06%** | **1.51×** |
| All Traffic      |      20,000 |  **3.34%** | **1.00×** |

This is probably the easiest result to understand.

The overall CTR is:

[
3.34%
]

but if we select only the model's **Top 1% impressions**:

[
CTR=21%
]

Therefore:

[
CTR\ Lift =
\frac{21}{3.34}
\approx6.29\times
]

In other words:

> The targeting model successfully identified a small group of impressions whose observed CTR was about **6.3× the overall CTR**.

The trend is also consistent:

```text id="lknjdu"
Top  1% → 21.00% CTR  █████████████████████
Top  5% → 12.80% CTR  █████████████
Top 10% →  9.55% CTR  ██████████
Top 20% →  7.38% CTR  ███████
Top 50% →  5.06% CTR  █████
All     →  3.34% CTR  ███
```

This demonstrates what targeting is doing much more clearly than AUC alone.

---

# 9. From Targeting to Auto-Bidding

Targeting tells us:

> **How attractive is this impression?**

But advertising inventory is usually sold through an auction.

Suppose two users appear:

```text id="8dw2va"
User A → pCTR = 13%
User B → pCTR = 1%
```

We probably want to bid more aggressively for User A.

That is the purpose of **auto-bidding**.

The experiment uses a simple linear bidding rule:

[
Bid =
BaseBid \times \frac{pCTR}{AverageCTR}
]

The average training CTR was:

[
AverageCTR=3.286%
]

So if an impression has average click probability:

```text id="hlq5nq"
pCTR ≈ 3.286%

→ Bid ≈ Base Bid
```

If it has twice the average pCTR:

```text id="94r54v"
pCTR ≈ 6.57%

→ Bid ≈ 2 × Base Bid
```

Therefore:

```text id="8l7bpx"
Low pCTR
   ↓
Low Bid

Average pCTR
   ↓
Base Bid

High pCTR
   ↓
High Bid
```

---

# 10. Actual Auto-Bidding Examples

The experiment produced:

| Click |       pCTR |        Bid |
| ----: | ---------: | ---------: |
|     0 |     13.07% |     397.77 |
|     0 |      4.58% |     139.37 |
|     0 |      9.72% |     295.90 |
| **1** | **18.33%** | **557.89** |
|     0 |      1.04% |      31.75 |
|     0 |      0.88% |      26.86 |
|     0 |      0.06% |       1.76 |

For example:

```text id="w6kt67"
Impression #3

pCTR = 18.33%
        ↓
Very high predicted click probability
        ↓
Bid = 557.89
```

while:

```text id="bffijj"
Impression #8

pCTR = 0.058%
        ↓
Very low predicted click probability
        ↓
Bid = 1.76
```

So the bidding algorithm is essentially saying:

> **Spend more money competing for impressions that the targeting model believes are more valuable.**

One important detail: an individual high-pCTR impression can still have `click=0`.

For example:

```text id="0hn7jg"
pCTR = 13%
actual click = 0
```

This is completely normal.

`pCTR = 13%` does **not** mean the user will definitely click. It means that across many similar impressions, roughly 13% are expected to click if the model is well calibrated.

---

# 11. Targeting vs Auto-Bidding

The difference can be summarized with one example.

Suppose:

```text id="t4avc4"
User A → pCTR = 10%
User B → pCTR = 1%
```

### Targeting

Targeting says:

```text id="tj7fru"
A looks much better than B.

A → high priority
B → low priority
```

### Auto-Bidding

Auto-bidding takes that information and says:

```text id="n14lyl"
A → Bid 300
B → Bid 30
```

So:

```text id="ph21cn"
Targeting
"Who is valuable?"
      ↓
     pCTR
      ↓
Auto-Bidding
"How much should I pay?"
      ↓
     Bid
```

---

# 12. Where Does Pacing Fit?

There is one more problem.

Suppose an advertiser has:

```text id="7qr00g"
Daily Budget = $10,000
```

Even if our bidding model is excellent, it could spend all $10,000 in the morning.

```text id="07f38h"
00:00       10:00                         24:00
████████████████████
$10,000 spent
```

Then we cannot participate in valuable auctions later in the day.

**Pacing controls how quickly the budget is spent.**

For example, at 12:00:

```text id="39y6k5"
Expected spend ≈ 50% of budget

Target = $5,000
```

If actual spend is only:

```text id="e5e9ay"
$3,000
```

we are under-spending.

The pacing system can increase bidding:

```text id="hcc13v"
Raw Bid = 100
Pacing Multiplier = 1.2

Final Bid = 120
```

If we have already spent:

```text id="7uc9j6"
$7,000
```

we are spending too quickly:

```text id="0hm01y"
Raw Bid = 100
Pacing Multiplier = 0.7

Final Bid = 70
```

So:

[
FinalBid =
AutoBid \times PacingMultiplier
]

---

# 13. Full Advertising Architecture

Putting everything together:

```text id="q75ob6"
             User visits website/app
                       │
                       ▼
                  Ad Request
                       │
                       ▼
              User / Ad / Context
                       │
                       ▼
              ┌─────────────────┐
              │ Targeting Model │
              │ CTR Prediction  │
              └────────┬────────┘
                       │
                       ▼
                     pCTR
                       │
                       ▼
              ┌─────────────────┐
              │  Auto-Bidding   │
              │ pCTR → Bid      │
              └────────┬────────┘
                       │
                       ▼
                    Raw Bid
                       │
                       ×
              ┌─────────────────┐
Budget ──────►│     Pacing      │
Spend ───────►│   Controller    │
Time ────────►│                 │
              └────────┬────────┘
                       │
                       ▼
                   Final Bid
                       │
                       ▼
                    Auction
                   /       \
                 Win       Lose
                  │
                  ▼
             Ad Impression
                  │
                  ▼
             Click / No Click
```

The easiest way to remember the three components is:

| Component        | Simple Question                             | Output         |
| ---------------- | ------------------------------------------- | -------------- |
| **Targeting**    | Who is likely to click?                     | pCTR           |
| **Auto-Bidding** | How much should we bid for them?            | Bid            |
| **Pacing**       | How aggressively should we spend right now? | Bid multiplier |

Or even shorter:

```text id="79ws9w"
Targeting    = WHO?
Auto-Bidding = HOW MUCH?
Pacing       = HOW FAST?
```

---

# 14. What Is Actually Implemented in This Experiment?

It is important to distinguish the conceptual architecture from what the current dataset allows us to evaluate.

### Implemented

```text id="00axeb"
Real Advertising Data
        ↓
Streaming Subset
        ↓
Feature Hashing
        ↓
Logistic Regression
        ↓
pCTR
        ↓
Targeting Evaluation
        +
Linear Auto-Bidding Score
```

We successfully evaluate the **targeting model** because the dataset contains actual click labels.

The auto-bidding component converts pCTR into bids, demonstrating how value-based bidding works.

### Not Yet Fully Evaluated

A full bidding/pacing experiment needs additional auction information:

```text id="8vdnpd"
Market / Clearing Price
Timestamp
Campaign Budget
Conversions / Value
```

Then we could replay auctions:

```text id="tycjdg"
Our Bid >= Market Price
        ↓
       Win
        ↓
      Spend
        ↓
Click / Conversion
```

and measure actual bidding metrics such as:

```text id="76xxfz"
Win Rate
Clicks
Cost
CPC
CPA
ROAS
Budget Utilization
Pacing Error
```

Therefore, the current `bid` values should be interpreted as **auto-bidding scores/policy outputs**, not as evidence that the bidding strategy itself improved CPC or ROAS.

---

# 15. Experiment Results

## Targeting

```text id="3ac6vn"
Train samples : 200,000
Test samples  : 20,000

Train CTR     : 3.286%
Test CTR      : 3.340%

ROC-AUC       : 0.6992
PR-AUC        : 0.0882
LogLoss       : 0.1420

Actual CTR    : 3.340%
Predicted CTR : 3.117%
```

## Targeting Lift

```text id="1unqsv"
Top  1% → CTR 21.00% → 6.29x lift
Top  5% → CTR 12.80% → 3.83x lift
Top 10% → CTR  9.55% → 2.86x lift
Top 20% → CTR  7.38% → 2.21x lift
Top 50% → CTR  5.06% → 1.51x lift
```

The model therefore learned meaningful ranking behavior: impressions assigned higher pCTR were substantially more likely to receive actual clicks.

---

# 16. Conclusion

This experiment demonstrates the basic logic behind an advertising optimization system using real click data.

The learned targeting model achieved:

[
ROC\text{-}AUC = 0.699
]

and selecting the model's highest-ranked 1% of impressions increased observed CTR from:

[
3.34% \rightarrow 21.0%
]

corresponding to:

[
6.29\times\ CTR\ Lift
]

The most important conceptual pipeline is:

```text id="97g09i"
User / Ad / Context
        ↓
     Targeting
        ↓
       pCTR
        ↓
   Auto-Bidding
        ↓
       Bid
        ↓
      Pacing
        ↓
    Final Bid
        ↓
      Auction
```

In simple terms:

> **Targeting finds valuable impressions, auto-bidding decides how much they are worth paying for, and pacing controls how quickly the campaign spends its budget.**

The current experiment validates the first part particularly well: the pCTR model successfully identifies higher-click-probability impressions. The next natural extension is to use an RTB dataset containing auction prices and timestamps to evaluate **auto-bidding and pacing with actual cost, CPC, budget utilization, and auction replay**.

