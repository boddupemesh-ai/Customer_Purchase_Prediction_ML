# Customer Purchase Prediction - EDA Report

## Dataset summary
- Samples: 1515  |  Features: 9  |  Target: `Purchase`
- Storage: 323.2 KB
- Duplicate records: 15

### Missing values
|               |   missing |   missing_% |
|:--------------|----------:|------------:|
| Age           |        60 |        3.96 |
| Annual_Income |        88 |        5.81 |
| Device        |        32 |        2.11 |

### Outliers
|                    |   IQR_outliers |   Z>3_outliers |
|:-------------------|---------------:|---------------:|
| Age                |              0 |              0 |
| Annual_Income      |             20 |             12 |
| Time_On_Site_Min   |             56 |             22 |
| Pages_Viewed       |             33 |              6 |
| Previous_Purchases |             86 |             11 |
| Email_Opened       |              0 |              0 |

## Key EDA findings
- Class balance (%): {1: 64.75, 0: 35.25}
- Significant features (p<0.05): Previous_Purchases, Time_On_Site_Min, Pages_Viewed, Email_Opened, Annual_Income, Age, Device

### Skewness & kurtosis
|                    |   skew |   kurtosis | note                           |
|:-------------------|-------:|-----------:|:-------------------------------|
| Age                | -0.025 |     -1.189 | ok                             |
| Annual_Income      |  7.734 |     73.094 | highly skewed -> log/transform |
| Time_On_Site_Min   |  1.45  |      3.078 | highly skewed -> log/transform |
| Pages_Viewed       |  0.543 |      0.497 | ok                             |
| Previous_Purchases |  0.952 |      1.137 | ok                             |
| Email_Opened       |  0.004 |     -2.003 | ok                             |

### Hypothesis tests
| feature            | test       |   statistic |   p_value | significant(p<0.05)   |
|:-------------------|:-----------|------------:|----------:|:----------------------|
| Previous_Purchases | t-test     |       9.47  |    0      | True                  |
| Time_On_Site_Min   | t-test     |       6.487 |    0      | True                  |
| Pages_Viewed       | t-test     |       5.95  |    0      | True                  |
| Email_Opened       | t-test     |       5.5   |    0      | True                  |
| Annual_Income      | t-test     |       5.028 |    0      | True                  |
| Age                | t-test     |       3.512 |    0.0005 | True                  |
| Device             | chi-square |       8.705 |    0.0129 | True                  |
| Region             | chi-square |       0.91  |    0.8231 | False                 |
| Gender             | chi-square |       0.034 |    0.8527 | False                 |

## Preprocessing recommendations
- Remove duplicate rows and rows without a label
- Normalise inconsistent category spellings (strip + title-case)
- Impute numeric gaps with median, categorical with most frequent
- Scale numeric features; one-hot encode categoricals
- Cap or transform extreme income values and skewed features

## Baseline performance
| model               |   accuracy |   precision |   recall |    f1 |   roc_auc |
|:--------------------|-----------:|------------:|---------:|------:|----------:|
| Logistic Regression |      0.657 |       0.779 |    0.655 | 0.711 |     0.734 |
| Random Forest       |      0.687 |       0.753 |    0.768 | 0.76  |     0.712 |
