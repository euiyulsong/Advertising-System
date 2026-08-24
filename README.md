# Auto Bidding Experiment — CTR vs. Conversion-Based Bidding

## 1. Dataset

**Criteo Attribution Dataset**

| Item                    |     Value |
| ----------------------- | --------: |
| Samples                 | 1,000,000 |
| Features                |        22 |
| Overall CTR             |    34.49% |
| Overall Conversion Rate |     4.86% |
| Total Cost              |    276.04 |

### Data Split

| Split      | Samples |
| ---------- | ------: |
| Train      | 700,000 |
| Validation | 150,000 |
| Test       | 150,000 |

---

## 2. CTR Model

### Input

```text
campaign, cat1~cat9, hour, day
```

### Output

```text
pCTR = P(click | impression)
```

### Model

```text
LightGBM Binary Classifier
```

### Performance

| Metric  | Validation |       Test |
| ------- | ---------: | ---------: |
| ROC-AUC |     0.6833 | **0.6754** |
| LogLoss |     0.5830 |          - |

---

## 3. Auto Bidding — Conversion Model

### Input

```text
campaign, cat1~cat9, hour, day, pCTR
```

### Output

```text
pCVR = P(conversion | impression)
```

### Model

```text
LightGBM Binary Classifier
```

### Performance

| Metric             | Validation |       Test |
| ------------------ | ---------: | ---------: |
| Conversion ROC-AUC |     0.7947 | **0.7981** |

The predicted conversion probability was also well aligned with the observed rate:

```text
Mean pCVR        : 4.575%
Actual Conversion: 4.612%
```

---

## 4. Bidding Strategy

Two bidding policies were compared.

### CTR Baseline

```text
bid = 0.001149 × pCTR
```

Prioritizes impressions with high click probability.

### Auto Bidding

```text
bid = 0.010594 × pCVR
```

Prioritizes impressions with high conversion probability.

The bid multipliers were calibrated on the validation set to achieve approximately the same spending level.

### Test Budget

```text
Historical spend : 40.8259
Target budget    : 12.2478
```

---

## 5. Business Outcome

| Metric             | CTR Baseline | Auto Bidding |      Change |
| ------------------ | -----------: | -----------: | ----------: |
| Impressions        |      124,304 |       91,965 | **-26.02%** |
| Clicks             |       38,523 |       28,288 | **-26.57%** |
| CTR                |       30.99% |       30.76% |  **-0.75%** |
| CVR                |       10.73% |       17.51% | **+63.20%** |
| Conversions        |        4,133 |        4,953 | **+19.84%** |
| Spend              |      12.2478 |      12.2134 |  **-0.28%** |
| CPA                |     0.002963 |     0.002466 | **-16.79%** |
| Budget Utilization |       100.0% |       99.72% |     -0.28%p |

---

## 6. Result

At approximately the **same budget**, conversion-based auto bidding selected fewer but substantially more conversion-efficient impressions.

```text
CTR-based bidding
        ↓
124K impressions
        ↓
4,133 conversions
        ↓
CPA = 0.002963


Conversion-based Auto Bidding
        ↓
92K impressions
        ↓
4,953 conversions
        ↓
CPA = 0.002466
```

The auto bidder achieved:

* **+19.84% conversions**
* **-16.79% CPA**
* **+63.20% CVR**
* **-0.75% CTR**

The slight CTR decrease is expected because the new policy no longer optimizes primarily for clicks. Instead, it prioritizes impressions with a higher probability of producing the downstream business outcome: **conversion**.

### Conclusion

> Replacing CTR-based bidding with conversion-probability-based auto bidding increased conversions by **19.84%** while reducing CPA by **16.79%** under approximately the same budget. This demonstrates that optimizing bidding toward the downstream business objective can outperform click-oriented bidding even when CTR itself does not improve.

