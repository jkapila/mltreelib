# Decision tree regression in Python: estimate diamond prices

Predict the listed price of a diamond from its measurements and grading labels. Compare a simple baseline and two tree configurations. Inspect the largest errors before using the score for a pricing decision.

## 1. Get the data and make the split

`Dataset.load("diamonds", target="price")` reads the [OpenML diamonds dataset 46923](https://www.openml.org/d/46923) (53,940 records) and marks `price` as the target. The input includes numeric measurements such as carat and raw labels for cut, color and clarity. Keep those labels as strings; the tree routes categories natively.

Take a seeded 8,000-row subset, then split it into 6,000 training rows and 2,000 held-out test rows. A production pricing study needs a split that respects listing date, seller and repeated items where available.

**Next:** fit mean-leaf and linear-leaf trees at depth 4 on the 6,000 training rows, score the 2,000 test rows with `MetricsReport`, and print `summary()` and `leaf_summary(0)`.

## 2. Fit, score and inspect the tree

This snippet loads the dataset with `Dataset.load`, makes a seeded 6,000 / 2,000 split, fits both a mean-leaf tree and a linear-leaf tree at `max_depth=4` (`min_rows=40`, `prune="none"`), and evaluates both against a training-mean baseline using `MetricsReport`. We use depth 4 because a shallower depth-2 tree leaves too much price variation inside each leaf for local slopes to matter, whereas depth 4 gives linear leaves enough room to lower test RMSE by about 5% without extra tuning. It then prints the full `summary()` of the linear-leaf tree and a single leaf's `leaf_summary(0)`.

<!-- run -->
```python
from statistics import mean

from mltreelib import MetricsReport, ModelTree
from mltreelib.data import Dataset

ds = Dataset.load("diamonds", target="price")
sub, _ = ds.split(test_size=ds.n_rows - 8000, seed=2026)
train, test = sub.split(test_size=2000, seed=2026)

base_rep = MetricsReport(
    "regression", y_true=test.y, y_pred=[mean(train.y)] * test.n_rows
)
print(f"baseline rmse={base_rep['rmse']:.2f}")

trees = {}
for leaf in ("mean", "linear"):
    tree = ModelTree(leaf=leaf, max_depth=4, min_rows=40, prune="none")
    tree.fit(train)
    pred = tree.predict(test)
    rep = MetricsReport("regression", y_true=test.y, y_pred=pred)
    trees[leaf] = tree
    print(f"{leaf} rmse={rep['rmse']:.2f} mae={rep['mae']:.2f} r2={rep['r2']:.4f}")

print()
print("=== Linear tree summary() ===")
print(trees["linear"].summary())
print("=== Leaf 0 leaf_summary(0) ===")
print(trees["linear"].leaf_summary(0))
```

<!-- output -->
```text
baseline rmse=3926.26
mean rmse=1248.40 mae=690.52 r2=0.8988
linear rmse=1182.90 mae=598.75 r2=0.9092

=== Linear tree summary() ===
======================================================================
                        mltreelib ModelTree Summary                   
Depth: 4, Total Nodes: 31, Active Features: 9
======================================================================

======================================================================
Leaf Node 0 (Leaf ID 0) [n=1315]
----------------------------------------------------------------------
Residual standard error: 153.3667 on 1305 degrees of freedom
Multiple R-squared:  0.2775,  Adjusted R-squared:  0.2725
F-statistic: 55.69 on 9 and 1305 DF,  p-value: < 2.2e-16

Coefficients:
                       Estimate   Std. Error    t value     Pr(>|t|) 
(Intercept)           -543.1376    2910.0985      -0.19       0.8519    
carat                   62.4231     986.2875       0.06       0.9495    
cut                    -25.0783       5.1499      -4.87     1.118e-6 ***
color                   -9.0927       1.9872      -4.58     4.750e-6 ***
clarity                  0.1176       2.3059       0.05       0.9593    
depth                  -10.4484      45.0513      -0.23       0.8166    
table                    0.6016       3.2086       0.19       0.8513    
x                     1221.9024     355.1129       3.44     5.798e-4 ***
y                     -874.0673     342.4732      -2.55       0.0107 *  
z                      136.5295    1044.3619       0.13       0.8960    
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
======================================================================


======================================================================
Leaf Node 1 (Leaf ID 1) [n=643]
----------------------------------------------------------------------
Residual standard error: 218.2497 on 633 degrees of freedom
Multiple R-squared:  0.1307,  Adjusted R-squared:  0.1183
F-statistic: 10.57 on 9 and 633 DF,  p-value: 2.4346e-15

Coefficients:
                       Estimate   Std. Error    t value     Pr(>|t|) 
(Intercept)           -488.5683    2460.7921      -0.20       0.8426    
carat                  419.0807    1550.1894       0.27       0.7869    
cut                    -34.5405      10.7114      -3.22       0.0013 ** 
color                   -9.3236       3.8504      -2.42       0.0155 *  
clarity                  6.1755       4.6529       1.33       0.1844    
depth                  -35.1124      43.2504      -0.81       0.4169    
table                    8.6566       6.6799       1.30       0.1950    
x                      251.3335     112.9185       2.23       0.0260 *  
y                      -65.2036     425.7619      -0.15       0.8783    
z                      715.8494    1070.1242       0.67       0.5035    
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
======================================================================


======================================================================
Leaf Node 3 (Leaf ID 2) [n=690]
----------------------------------------------------------------------
Residual standard error: 329.1394 on 680 degrees of freedom
Multiple R-squared:  0.2146,  Adjusted R-squared:  0.2042
F-statistic: 20.64 on 9 and 680 DF,  p-value: < 2.2e-16

Coefficients:
                       Estimate   Std. Error    t value     Pr(>|t|) 
(Intercept)          -3188.1811   12744.2559      -0.25       0.8025    
carat                  188.7778    1815.1780       0.10       0.9172    
cut                    -54.7186      14.8439      -3.69     2.276e-4 ***
color                  -30.8996       5.7045      -5.42     6.069e-8 ***
clarity                -34.9325       7.8159      -4.47     7.844e-6 ***
depth                    9.9225     206.8829       0.05       0.9617    
table                   -8.4653       8.8520      -0.96       0.3389    
x                      549.9681    1243.3184       0.44       0.6582    
y                      195.1551    1213.6176       0.16       0.8722    
z                      287.9943    3997.8488       0.07       0.9426    
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
======================================================================


======================================================================
Leaf Node 4 (Leaf ID 3) [n=67]
----------------------------------------------------------------------
Residual standard error: 621.8211 on 57 degrees of freedom
Multiple R-squared:  0.3364,  Adjusted R-squared:  0.2317
F-statistic: 3.21 on 9 and 57 DF,  p-value: 3.2704e-3

Coefficients:
                       Estimate   Std. Error    t value     Pr(>|t|) 
(Intercept)          -2741.0817   84611.4939      -0.03       0.9742    
carat                   54.7573   14580.0766       0.00       0.9970    
cut                   -292.6692      97.4819      -3.00       0.0027 ** 
color                  -83.5138      33.9780      -2.46       0.0140 *  
clarity                  8.0966      34.2367       0.24       0.8131    
depth                  -26.2534    1303.9149      -0.02       0.9839    
table                   34.7508      69.0776       0.50       0.6149    
x                      146.9274    9150.3352       0.02       0.9872    
y                      689.0188    7060.6056       0.10       0.9223    
z                      207.2262   24695.4152       0.01       0.9933    
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
======================================================================


======================================================================
Leaf Node 7 (Leaf ID 4) [n=668]
----------------------------------------------------------------------
Residual standard error: 518.5575 on 658 degrees of freedom
Multiple R-squared:  0.2526,  Adjusted R-squared:  0.2424
F-statistic: 24.71 on 9 and 658 DF,  p-value: < 2.2e-16

Coefficients:
                       Estimate   Std. Error    t value     Pr(>|t|) 
(Intercept)          -4538.9924    4252.4813      -1.07       0.2858    
carat                 1226.6336    2071.8864       0.59       0.5538    
cut                   -146.9060      22.8722      -6.42    1.337e-10 ***
color                  -34.3446       9.5056      -3.61     3.026e-4 ***
clarity                 38.0345      12.4096       3.06       0.0022 ** 
depth                   -0.0371      54.0574      -0.00       0.9995    
table                   -6.7167      12.9127      -0.52       0.6030    
x                     -123.9632     399.5411      -0.31       0.7564    
y                      961.4225     416.0466       2.31       0.0208 *  
z                      595.0476    1222.7786       0.49       0.6265    
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
======================================================================


======================================================================
Leaf Node 8 (Leaf ID 5) [n=125]
----------------------------------------------------------------------
Residual standard error: 298.1147 on 115 degrees of freedom
Multiple R-squared:  0.5385,  Adjusted R-squared:  0.5024
F-statistic: 14.91 on 9 and 115 DF,  p-value: 8.3499e-16

Coefficients:
                       Estimate   Std. Error    t value     Pr(>|t|) 
(Intercept)          -5782.1486   22235.8868      -0.26       0.7948    
carat                  251.4081    2823.2960       0.09       0.9290    
cut                    -85.1396      28.6227      -2.97       0.0029 ** 
color                 -400.9120      60.4104      -6.64    3.213e-11 ***
clarity                 -2.4840      17.5840      -0.14       0.8877    
depth                   26.8369     359.4466       0.07       0.9405    
table                  -11.7324      15.1856      -0.77       0.4398    
x                      571.2024    2060.6048       0.28       0.7816    
y                      692.0117    1911.9249       0.36       0.7174    
z                      384.7419    6275.7228       0.06       0.9511    
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
======================================================================


======================================================================
Leaf Node 10 (Leaf ID 6) [n=282]
----------------------------------------------------------------------
Residual standard error: 733.3794 on 272 degrees of freedom
Multiple R-squared:  0.1824,  Adjusted R-squared:  0.1553
F-statistic: 6.74 on 9 and 272 DF,  p-value: 9.9657e-9

Coefficients:
                       Estimate   Std. Error    t value     Pr(>|t|) 
(Intercept)          -2387.6665   44948.2990      -0.05       0.9576    
carat                 -195.1986    3316.0264      -0.06       0.9531    
cut                   -166.4033      47.6954      -3.49     4.851e-4 ***
color                  -33.3695      20.3413      -1.64       0.1009    
clarity                 93.5445      27.7434       3.37     7.469e-4 ***
depth                   -0.8933     745.9987      -0.00       0.9990    
table                   -5.2078      22.4831      -0.23       0.8168    
x                    -1018.0624    3638.9145      -0.28       0.7797    
y                     1896.2624    3784.7920       0.50       0.6164    
z                      419.8854   12244.8709       0.03       0.9726    
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
======================================================================


======================================================================
Leaf Node 11 (Leaf ID 7) [n=62]
----------------------------------------------------------------------
Residual standard error: 406.5590 on 52 degrees of freedom
Multiple R-squared:  0.2234,  Adjusted R-squared:  0.0890
F-statistic: 1.66 on 9 and 52 DF,  p-value: 1.2238e-1

Coefficients:
                       Estimate   Std. Error    t value     Pr(>|t|) 
(Intercept)          -2789.9568   63768.1559      -0.04       0.9651    
carat                  137.8419    4858.9702       0.03       0.9774    
cut                    -56.3823      59.0273      -0.96       0.3395    
color                 -235.0063     124.2165      -1.89       0.0585 .  
clarity                 20.5655      41.4862       0.50       0.6201    
depth                   10.9297    1013.4835       0.01       0.9914    
table                   -2.3788      29.0124      -0.08       0.9347    
x                      174.9776    5471.1205       0.03       0.9745    
y                      698.7020    5038.0306       0.14       0.8897    
z                      254.2871   16297.4223       0.02       0.9876    
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
======================================================================


======================================================================
Leaf Node 15 (Leaf ID 8) [n=566]
----------------------------------------------------------------------
Residual standard error: 548.6820 on 556 degrees of freedom
Multiple R-squared:  0.4087,  Adjusted R-squared:  0.3991
F-statistic: 42.70 on 9 and 556 DF,  p-value: < 2.2e-16

Coefficients:
                       Estimate   Std. Error    t value     Pr(>|t|) 
(Intercept)          -9900.2560   21782.3290      -0.45       0.6495    
carat                  144.9288    1390.5282       0.10       0.9170    
cut                    -88.0813      25.3792      -3.47     5.193e-4 ***
color                  -92.4108      12.4651      -7.41    1.230e-13 ***
clarity                214.2194      15.6586      13.68      < 2e-16 ***
depth                   19.7113     344.2617       0.06       0.9543    
table                    1.2800      14.3107       0.09       0.9287    
x                       67.3546    1708.4124       0.04       0.9686    
y                     1535.8307    1691.4313       0.91       0.3639    
z                      631.9341    5328.6172       0.12       0.9056    
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
======================================================================


======================================================================
Leaf Node 16 (Leaf ID 9) [n=186]
----------------------------------------------------------------------
Residual standard error: 798.0004 on 176 degrees of freedom
Multiple R-squared:  0.5742,  Adjusted R-squared:  0.5524
F-statistic: 26.37 on 9 and 176 DF,  p-value: < 2.2e-16

Coefficients:
                       Estimate   Std. Error    t value     Pr(>|t|) 
(Intercept)         -27350.1292   94522.2512      -0.29       0.7723    
carat                 1506.3594    4145.8289       0.36       0.7163    
cut                   -217.9846      74.1487      -2.94       0.0033 ** 
color                 -185.3506      32.9557      -5.62     1.863e-8 ***
clarity                320.0308      40.7667       7.85    3.997e-15 ***
depth                   80.0962    1533.5372       0.05       0.9583    
table                    0.6315      45.8083       0.01       0.9890    
x                      792.6257    6796.2612       0.12       0.9072    
y                     2476.0583    6933.4915       0.36       0.7210    
z                      929.5571   22263.1827       0.04       0.9667    
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
======================================================================


======================================================================
Leaf Node 18 (Leaf ID 10) [n=472]
----------------------------------------------------------------------
Residual standard error: 1339.4506 on 462 degrees of freedom
Multiple R-squared:  0.3607,  Adjusted R-squared:  0.3483
F-statistic: 28.96 on 9 and 462 DF,  p-value: < 2.2e-16

Coefficients:
                       Estimate   Std. Error    t value     Pr(>|t|) 
(Intercept)          -8570.2590   17517.4339      -0.49       0.6247    
carat                 2017.9860    3468.8540       0.58       0.5607    
cut                    -73.9609      69.4037      -1.07       0.2866    
color                 -182.0096      28.5420      -6.38    1.807e-10 ***
clarity                317.1856      64.0979       4.95     7.480e-7 ***
depth                    0.1746     176.2444       0.00       0.9992    
table                  -75.5745      40.1647      -1.88       0.0599 .  
x                     -702.9133    1645.6948      -0.43       0.6693    
y                     2655.4229    1534.4465       1.73       0.0835 .  
z                      996.9328    2259.7662       0.44       0.6591    
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
======================================================================


======================================================================
Leaf Node 19 (Leaf ID 11) [n=213]
----------------------------------------------------------------------
Residual standard error: 2686.5217 on 203 degrees of freedom
Multiple R-squared:  0.2426,  Adjusted R-squared:  0.2090
F-statistic: 7.22 on 9 and 203 DF,  p-value: 4.3385e-9

Coefficients:
                       Estimate   Std. Error    t value     Pr(>|t|) 
(Intercept)           5779.8878   66933.1063       0.09       0.9312    
carat                 -287.6203   12677.7897      -0.02       0.9819    
cut                   -707.0816     213.2710      -3.32     9.151e-4 ***
color                 -155.0811      82.5081      -1.88       0.0602 .  
clarity               -212.6185      68.2609      -3.11       0.0018 ** 
depth                 -136.9514    1095.9461      -0.12       0.9006    
table                  -27.1691     123.1635      -0.22       0.8254    
x                    -2301.9697    6497.3852      -0.35       0.7231    
y                     4393.2513    4667.0881       0.94       0.3465    
z                      257.8596   18486.0148       0.01       0.9889    
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
======================================================================


======================================================================
Leaf Node 22 (Leaf ID 12) [n=328]
----------------------------------------------------------------------
Residual standard error: 2157.5753 on 318 degrees of freedom
Multiple R-squared:  0.2907,  Adjusted R-squared:  0.2706
F-statistic: 14.48 on 9 and 318 DF,  p-value: < 2.2e-16

Coefficients:
                       Estimate   Std. Error    t value     Pr(>|t|) 
(Intercept)         -21101.5050   31325.9583      -0.67       0.5006    
carat                 2126.0826    5114.2604       0.42       0.6776    
cut                   -197.9500     141.0675      -1.40       0.1605    
color                 -366.0276      62.2978      -5.88     4.217e-9 ***
clarity                677.2131      96.5386       7.01    2.300e-12 ***
depth                  131.3057     216.5427       0.61       0.5443    
table                  -90.9920      77.7072      -1.17       0.2416    
x                      733.2343    2504.9832       0.29       0.7697    
y                     2092.3167    2560.6052       0.82       0.4139    
z                      822.4470     709.7859       1.16       0.2466    
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
======================================================================


======================================================================
Leaf Node 23 (Leaf ID 13) [n=130]
----------------------------------------------------------------------
Residual standard error: 3314.5793 on 120 degrees of freedom
Multiple R-squared:  0.1748,  Adjusted R-squared:  0.1129
F-statistic: 2.82 on 9 and 120 DF,  p-value: 4.7871e-3

Coefficients:
                       Estimate   Std. Error    t value     Pr(>|t|) 
(Intercept)          14584.8959  151304.5289       0.10       0.9232    
carat                -2683.0682   13723.7630      -0.20       0.8450    
cut                   -630.9787     359.2355      -1.76       0.0790 .  
color                 -202.9782     157.6754      -1.29       0.1980    
clarity               -307.4564     146.7935      -2.09       0.0362 *  
depth                 -240.2536    2387.6408      -0.10       0.9198    
table                  -77.1346     180.1928      -0.43       0.6686    
x                      965.9156   10521.4733       0.09       0.9269    
y                     2434.9759   10864.4775       0.22       0.8227    
z                     -262.7353   33141.3360      -0.01       0.9937    
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
======================================================================


======================================================================
Leaf Node 25 (Leaf ID 14) [n=147]
----------------------------------------------------------------------
Residual standard error: 2677.1595 on 137 degrees of freedom
Multiple R-squared:  0.1097,  Adjusted R-squared:  0.0512
F-statistic: 1.88 on 9 and 137 DF,  p-value: 6.0435e-2

Coefficients:
                       Estimate   Std. Error    t value     Pr(>|t|) 
(Intercept)           3766.5871  182265.4295       0.02       0.9835    
carat                  -57.8592    8001.7908      -0.01       0.9942    
cut                   -550.0866     227.5920      -2.42       0.0156 *  
color                 -274.6979     126.4573      -2.17       0.0298 *  
clarity                159.6902     143.2517       1.11       0.2650    
depth                  171.5109    2851.6306       0.06       0.9520    
table                  145.9058     124.2037       1.17       0.2401    
x                    -2609.2727   11676.6755      -0.22       0.8232    
y                     1704.8097   11881.4542       0.14       0.8859    
z                       60.0807   35698.8093       0.00       0.9987    
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
======================================================================


======================================================================
Leaf Node 26 (Leaf ID 15) [n=106]
----------------------------------------------------------------------
Residual standard error: 2302.0627 on 96 degrees of freedom
Multiple R-squared:  0.1596,  Adjusted R-squared:  0.0808
F-statistic: 2.03 on 9 and 96 DF,  p-value: 4.4433e-2

Coefficients:
                       Estimate   Std. Error    t value     Pr(>|t|) 
(Intercept)          36746.7296  200488.2347       0.18       0.8546    
carat                 -301.4440    6152.7362      -0.05       0.9609    
cut                   -460.2091     260.1826      -1.77       0.0769 .  
color                  -36.2874     139.6024      -0.26       0.7949    
clarity               -151.9387     159.0387      -0.96       0.3394    
depth                 -313.5761    3529.3768      -0.09       0.9292    
table                 -226.7676     163.5333      -1.39       0.1655    
x                     -627.9815   12400.1016      -0.05       0.9596    
y                     2018.0743   12921.4556       0.16       0.8759    
z                      255.5905   43802.9790       0.01       0.9953    
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
======================================================================



=== Leaf 0 leaf_summary(0) ===
======================================================================
Leaf Node 0 (Leaf ID 0) [n=1315]
----------------------------------------------------------------------
Residual standard error: 153.3667 on 1305 degrees of freedom
Multiple R-squared:  0.2775,  Adjusted R-squared:  0.2725
F-statistic: 55.69 on 9 and 1305 DF,  p-value: < 2.2e-16

Coefficients:
                       Estimate   Std. Error    t value     Pr(>|t|) 
(Intercept)           -543.1376    2910.0985      -0.19       0.8519    
carat                   62.4231     986.2875       0.06       0.9495    
cut                    -25.0783       5.1499      -4.87     1.118e-6 ***
color                   -9.0927       1.9872      -4.58     4.750e-6 ***
clarity                  0.1176       2.3059       0.05       0.9593    
depth                  -10.4484      45.0513      -0.23       0.8166    
table                    0.6016       3.2086       0.19       0.8513    
x                     1221.9024     355.1129       3.44     5.798e-4 ***
y                     -874.0673     342.4732      -2.55       0.0107 *  
z                      136.5295    1044.3619       0.13       0.8960    
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
======================================================================
```

**What to read in the output:**

- **Test metrics:** the global training-mean baseline gives RMSE 3,926.26. At depth 4, mean leaves drop RMSE to 1,248.40 (MAE 690.52, R² 0.8988), and linear leaves drop it further to 1,182.90 (MAE 598.75, R² 0.9092) — about 5.2% lower RMSE and 13.3% lower MAE than mean leaves.
- **Full `summary()`:** the header reports the tree depth (`4`), total nodes (`31`, meaning 16 leaves and 15 internal splits), and active features (`9`). Each block below it corresponds to one leaf node (`Leaf Node 0` through `Leaf Node 26`), showing its training sample count `n`, residual standard error, R-squared, F-statistic, and a statsmodels-style coefficient table (`Estimate`, `Std. Error`, `t value`, `Pr(>|t|)`) with significance stars.
- **Single `leaf_summary(0)`:** isolates the first leaf (`Leaf Node 0`, `n=1315`, degrees of freedom `1305 = 1315 - 10`). In that region, `cut`, `color`, `x` and `y` have small p-values while `carat` has a large standard error because `carat`, `x`, `y` and `z` are strongly collinear inside a narrow size leaf.

Run the [complete study](https://github.com/jkapila/mltreelib/blob/main/examples/guides/real_tree_studies.py) to regenerate the chart.

```bash
uv run python examples/guides/real_tree_studies.py diamonds
```

![Test RMSE and errors across observed diamond prices for mean and linear leaves](../assets/guides/real-diamonds-leaves.svg)

**Next:** inspect the errors and leaf structure before relying on local linear slopes.

## 3. Interpret the errors

| Prediction | Test RMSE | Test MAE |
|---|---:|---:|
| Global training mean | 3,926.26 | — |
| Mean leaves (depth 4) | 1,248.40 | 690.52 |
| Linear leaves (depth 4) | 1,182.90 | 598.75 |

The left panel summarizes error with RMSE. The right panel shows prediction minus observed price. Look for systematic underpricing below the zero line and overpricing above it. Review residuals by cut, price band and support. A shallow linear tree can follow a trend inside a leaf. Check for extrapolation, collinearity and sudden changes at split boundaries before relying on that trend.

Use `summary()` and `leaf_summary(i)` to inspect support and coefficients, and `splits()` to review named feature conditions. Category conditions can display internal codes. Keep that mapping available before showing a rule to a reviewer.

**Next:** repeat the comparison on synthetic data with raw category labels and 15% missing numeric inputs, because the real training columns used so far had no missing values.

## 4. Repeat the comparison with missing inputs

The [regression comparison](https://github.com/jkapila/mltreelib/blob/main/examples/guides/regression_real_and_synthetic.py) first generates 5,000 records with raw category labels and 15% missing numeric inputs. It fits four `ModelTree` configurations on 3,000 rows, selects one using 1,000 validation rows, then checks it on 1,000 separate test rows. The test set contains 169 missing numeric values. No imputation or categorical encoder is applied.

```bash
uv run python examples/guides/regression_real_and_synthetic.py
```

| Synthetic configuration | Validation RMSE |
|---|---:|
| Mean leaves, depth 2 | 3.594 |
| Mean leaves, depth 4 | 2.704 |
| Linear leaves, depth 2 | 2.770 |
| Linear leaves, depth 4 | 2.633 |

For this seed and simulated data, linear leaves at depth 4 had the lowest validation RMSE. Its test RMSE was 2.933, against 6.385 for a training-mean baseline. This chooses among four settings on one synthetic problem. It does not establish the best model for other datasets.

![Synthetic model comparison and real held-out residuals](../assets/guides/regression-real-synthetic.svg)

The same configuration then fits 6,000 rows from the real [OpenML diamonds dataset 46923](https://www.openml.org/d/46923) and predicts 2,000 separate rows. The data has 53,940 total records. Download its parquet file to `datasets/diamonds.parquet`; see the [dataset manifest](https://github.com/jkapila/mltreelib/blob/main/datasets/MANIFEST.md). The example needs pandas, pyarrow and matplotlib.

On the fixed random split in the block above, test RMSE was 1,576 for linear leaves, 1,613 for mean leaves and 3,926 for the training-mean baseline (the same numbers as the table in section 3). These values use the dataset's price units and describe this split, not a benchmark result. The block reports RMSE only; add MAE in your own run if you need it. The selected real training columns contained no missing values, so the synthetic check supplies the missing-data exercise.

The real tree splits on named size measurements, including `x` and `y`, and exposes leaf support and coefficients. Inspect the feature definitions and leaf sample sizes before drawing business conclusions. Category values in fitted structures can appear as internal codes. Coefficients and paths describe the prediction model; they do not establish what happens if a business changes a feature.

For more on local slopes and the fair comparison setup, see the [model tree regression reference](../concepts/piecewise-linear-and-cubist.md).
