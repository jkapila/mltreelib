# ELO Benchmark Arenas

One scoreboard, one protocol, every model scored on the **same prepared data**. The arena evaluates models across **Rust, Python, and R** under a strict 300s process-group timeout.

Detailed runbook &amp; ModelTree experimentation guide: [`benchmarks/README.md`](<https://github.com/jkapila/mltreelib/blob/main/benchmarks/README.md>).

---

## The Protocols

### 1\. Regression Arena

- **Data**: 18 regression benchmark datasets, one stratified 80/20 split (seed 7), prepared once by `examples/elo_prepare.py`.
- **Matchups**: Per dataset, every unordered pair of models "plays" once; lower test RMSE wins (tie = draw).
- **Scores**: Evaluates 36 models across Rust, Python, and R (`benchmarks/elo_scores.csv`).
- **Reproduce**: `uv run python examples/elo_report.py` (or `bash examples/elo_run_resume.sh`).

### 2\. Classification Arena

- **Data**: 24 TabArena datasets (20 binary, 4 multiclass) from OpenML suite 457, stratified 80/20 split within class (seed 7), prepared once by `examples/elo_prepare_cla.py` with raw strings for categoricals and `<ds>.meta.json`.
- **Matchups**: Per dataset, round-robin pairwise comparisons based on test **log loss** (cross-entropy), K=32, starting rating 1000.
- **Scores**: Evaluates 28 models across Python, Rust, and R (`benchmarks/elo_cla_scores.csv`), including RuleQuest `C5.0 (tree)` and `C5.0 (rules)` in R, `HistGradientBoosting`, `sklearn GBM`, `LightGBM`, `CatBoost`, `CTBoost`, `ChimeraBoost`, `EBM`, `ranger`, `bart`, `glmtree`. *(Note: ModelTree classification is currently undergoing internal development in Rust and will be attached once complete).*
- **Reproduce**: `uv run python examples/elo_report_cla.py` (or `bash examples/elo_run_cla.sh`).

---

## Metadata Taxonomy

Every score tracked in the arena includes two core architectural dimensions: - **`lang`**: Model implementation language — **`Rust`**, **`Python`**, or **`R`**. - **`cat_handling`**: - **`internal`** (*native / as-is*): Direct native consumption of categoricals without driver encoding (`ModelTree`, `LightGBM`, `CatBoost`, `CTBoost`, `ChimeraBoost`, `EBM`, `HistGradientBoosting`, `modeltrees`, R factors like `rpart`, `cubist`, `C5.0`, `ctree`, `glmtree`). - **`external`** (*driver-encoded*): Requires driver-side ordinal encoding fit on train (`sklearn RF`, `sklearn GBM`, `ExtraTrees`, `sklearn tree`, `forust`, `perpetual`, `MARS`). - **`none`** (*numeric-only*): Strictly requires numeric matrices; refuses categorical columns (`statsmodels`, `trust-free`, R `grf`).

---

## The Regression Board

```
== Elo (round-robin on per-dataset RMSE, K=32, start 1000) ==
+-----------+---------------------+------------------------+--------+------------+-----------+
|    ELO    |    FIT TIME (±sd)   |         MODEL          |  LANG  |    CAT     |    DATA   |
+-----------+---------------------+------------------------+--------+------------+-----------+
|   1690.0  |   12.01 s (±17.76)  |      ChimeraBoost      | Python |  internal  |   18/18   |
|   1622.3  |   18.24 s (±23.58)  |         ranger         |   R    |  internal  |   16/18   |
|   1618.1  |   19.20 s (±33.35)  |       sklearn RF       | Python |  external  |   18/18   |
|   1568.1  |   13.02 s (±33.28)  |       ExtraTrees       | Python |  external  |   18/18   |
|   1532.3  |   23.17 s (±31.44)  |        grf_llf         |   R    |    none    |   16/18   |
|   1470.6  |     0.35 s (±0.22)  |     LightGBM(200)      | Python |  internal  |   18/18   |
|   1439.3  |    7.20 s (±23.69)  |    perpetual (Rust)    |  Rust  |  external  |   17/18   |
|   1425.5  |     1.04 s (±1.10)  |     CatBoost(200)      | Python |  internal  |   18/18   |
|   1350.1  |     0.95 s (±1.91)  |  HistGradientBoosting  | Python |  internal  |   17/18   |
|   1319.3  |   65.08 s (±68.98)  |          EBM           | Python |  internal  |   15/18   |
|   1299.2  |     1.22 s (±0.86)  |     forust (Rust)      |  Rust  |  external  |   18/18   |
|   1210.8  |     0.76 s (±1.83)  |        CTBoost         | Python |  internal  |   17/18   |
|   1168.4  |   27.91 s (±39.58)  |          bart          |   R    |  internal  |   17/18   |
|   1166.7  |   18.06 s (±23.80)  |         grf_rf         |   R    |    none    |   18/18   |
|   1050.1  |   13.77 s (±20.73)  |     ModelTree(all)     |  Rust  |  internal  |   16/18   |
|   1030.9  |    7.00 s (±13.24)  |      sklearn GBM       | Python |  external  |   18/18   |
|   976.4   |   20.22 s (±47.48)  |         lmtree         |   R    |  internal  |   14/18   |
|   972.9   |   15.29 s (±51.54)  |   modeltrees lmtree    | Python |  internal  |   14/18   |
|   958.1   |   29.02 s (±57.11)  |        glmtree         |   R    |  internal  |   14/18   |
|   928.3   |     5.95 s (±8.90)  |   ModelTree(winnow)    |  Rust  |  internal  |   16/18   |
|   892.7   |    6.23 s (±15.19)  |         cubist         |   R    |  internal  |   17/18   |
|   873.1   |   20.49 s (±31.78)  |   MARS (mars-earth)    | Python |  external  |   10/18   |
|   761.5   |   20.60 s (±51.49)  |         mboost         |   R    |  internal  |   15/18   |
|   697.6   |     0.33 s (±0.68)  |    statsmodels OLS     | Python |    none    |    8/18   |
|   690.7   |   10.19 s (±27.50)  |         earth          |   R    |  internal  |   17/18   |
|   681.8   |     0.09 s (±0.10)  |          KNN           | Python |  external  |   18/18   |
|   655.7   |   21.17 s (±59.09)  |         ctree          |   R    |  internal  |   17/18   |
|   611.2   |    14.37 s (±7.33)  |       trust-free       | Python |    none    |    4/18   |
|   584.5   |     0.09 s (±0.23)  |     sklearn Linear     | Python |  external  |   18/18   |
|   547.3   |   10.80 s (±21.74)  |    ModelTree(path)     |  Rust  |  internal  |   18/18   |
|   539.8   |     0.34 s (±0.57)  |      sklearn tree      | Python |  external  |   18/18   |
|   527.1   |   17.48 s (±43.02)  |    statsmodels RLM     | Python |    none    |    8/18   |
|   482.0   |    5.34 s (±20.14)  |    ModelTree(mean)     |  Rust  |  internal  |   18/18   |
|   368.3   |   22.52 s (±59.15)  |  statsmodels QuantReg  | Python |    none    |    7/18   |
|   289.3   |     1.34 s (±3.71)  |         rpart          |   R    |  internal  |   17/18   |
+-----------+---------------------+------------------------+--------+------------+-----------+
```

TabArena anchor (different protocol - reference only):

````
Reading it:

- **Stock configs only** — nothing tuned per dataset. The top of the board is
  *forests and boosted ensembles with strong feature handling* (grf's
  local-linear forest leads; forust/perpetual are boosted trees).
- **Our flavors reflect the current defaults** (the split grid at 0.001,
  ridge-linear leaves): `ModelTree(all)` 914.8, `winnow` 650.7, `mean`
  649.7, `path` 545.8. The field anchor predates the split-grid default, so
  the live comparison is the round deltas — see **Running a round** below
  and `benchmarks/elo_round.log`: the grid trades differently per flavor
  (`path`/`mean` gain, `all`/`winnow` pay), and the manifests in
  `benchmarks/rounds/` record exactly which settings produced each move.
- `FIT mean±sd` = fit seconds across all scored datasets (mean ± 1 sd);
  `DATA` = datasets where the model has a score — matchups only
  happen where both models have one. Shortfalls are honest failures, not
  skates: mars-earth hits the 180s-per-algorithm budget on wide sets (12/18),
  and grf_llf/earth skip fc2008's NAs.

TabArena numbers are an **anchor, not a comparison** — their protocol
(tuned+ensembled, different splits) differs; shown only for scale:
`1955.0 Kumo-Tabular (best overall)`, `1411.0 LightGBM tuned+ens (best
tree-based)`.

## The wide table (model × dataset)

Every score in the field — RMSE per model per dataset, best per column
bolded. Click any column header to sort (the table is interactive):

<!-- generated by examples/elo_table.py - do not edit by hand -->


<div id="elo-interactive">
<style>
#elo-interactive { overflow-x: auto; }
#elo-interactive table { border-collapse: collapse; font-size: 0.75rem; white-space: nowrap; font-family: var(--md-text-font-family, monospace); }
#elo-interactive th, #elo-interactive td { padding: 2px 10px; border-bottom: 1px solid var(--md-default-fg-color--lightest, #ddd); text-align: right; background: var(--md-default-bg-color, #fff); box-sizing: border-box; }
#elo-interactive th { cursor: pointer; user-select: none; position: sticky; top: 0; z-index: 2; }
#elo-interactive th:hover { color: var(--md-accent-fg-color, #06c); }
#elo-interactive td.n { text-align: right; }
#elo-interactive tr:hover td { background: var(--md-default-fg-color--lightest, #f5f5f5); }
/* freeze panes: the three key columns stay put while datasets scroll */
#elo-interactive th.k1, #elo-interactive td:nth-child(1) { position: sticky; left: 0; z-index: 1; width: 72px; min-width: 72px; }
#elo-interactive th.k2, #elo-interactive td:nth-child(2) { position: sticky; left: 72px; z-index: 1; width: 176px; min-width: 176px; }
#elo-interactive th.k3, #elo-interactive td:nth-child(3) { position: sticky; left: 248px; z-index: 1; width: 200px; min-width: 200px; }
#elo-interactive th.k1, #elo-interactive th.k2, #elo-interactive th.k3 { z-index: 3; }
#elo-interactive .c { color: var(--md-default-fg-color--light, #999); font-size: 0.68rem; }
</style>
<table>
<thead><tr>
<th class='k1 k' data-t='0'>ELO</th>
<th class='k2 k' data-t='1'>FIT TIME (±sd)</th>
<th class='k3 k' data-t='2'>MODEL</th><th data-t='3'>abalone</th><th data-t='4'>boston</th><th data-t='5'>concrete</th><th data-t='6'>fc2008</th><th data-t='7'>ozone</th><th data-t='8'>airfoil</th><th data-t='9'>ta_concrete</th><th data-t='10'>diamonds</th><th data-t='11'>fiat500</th><th data-t='12'>food_delivery</th><th data-t='13'>healthcare</th><th data-t='14'>houses</th><th data-t='15'>miami</th><th data-t='16'>protein</th><th data-t='17'>qsar_fish</th><th data-t='18'>qsar_tid11</th><th data-t='19'>superconductivity</th><th data-t='20'>wine</th>
</tr></thead>
<tbody><tr><td class='k n'>1577</td><td class='k n'>15.41±24.59s</td><td class='k n'><b>grf_llf</b> <span class=c>17/18</span></td><td class=n>2.31</td><td class=n>3.75</td><td class=n>6.29</td><td class=n>·</td><td class=n><b>3.77</b></td><td class=n>2.03</td><td class=n>5.07</td><td class=n>536.61</td><td class=n>695.74</td><td class=n>7.55</td><td class=n>4564.69</td><td class=n>0.23</td><td class=n>102864.92</td><td class=n><b>3.59</b></td><td class=n>0.97</td><td class=n><b>0.86</b></td><td class=n><b>8.96</b></td><td class=n>0.64</td></tr><tr><td class='k n'>1526</td><td class='k n'>0.67±0.55s</td><td class='k n'><b>LightGBM(200)</b> <span class=c>18/18</span></td><td class=n>2.43</td><td class=n>3.47</td><td class=n>5.85</td><td class=n>196.57</td><td class=n>4.09</td><td class=n><b>1.62</b></td><td class=n>4.29</td><td class=n><b>0.00</b></td><td class=n>716.88</td><td class=n><b>7.42</b></td><td class=n>5121.26</td><td class=n>0.22</td><td class=n>93946.87</td><td class=n>3.92</td><td class=n>0.97</td><td class=n>0.90</td><td class=n>9.64</td><td class=n><b>0.63</b></td></tr><tr><td class='k n'>1481</td><td class='k n'>0.36±0.31s</td><td class='k n'><b>CatBoost(200)</b> <span class=c>18/18</span></td><td class=n>2.33</td><td class=n><b>2.54</b></td><td class=n>5.74</td><td class=n><b>168.89</b></td><td class=n>3.82</td><td class=n>1.66</td><td class=n><b>4.05</b></td><td class=n><b>0.00</b></td><td class=n><b>678.25</b></td><td class=n>7.44</td><td class=n>4664.68</td><td class=n>0.22</td><td class=n><b>89876.80</b></td><td class=n>3.92</td><td class=n><b>0.95</b></td><td class=n>0.91</td><td class=n>10.24</td><td class=n>0.65</td></tr><tr><td class='k n'>1464</td><td class='k n'>6.50±13.12s</td><td class='k n'><b>perpetual (Rust)</b> <span class=c>18/18</span></td><td class=n>2.29</td><td class=n>3.44</td><td class=n>5.78</td><td class=n>220.86</td><td class=n>3.88</td><td class=n>2.35</td><td class=n>4.75</td><td class=n>0.00</td><td class=n>702.46</td><td class=n>7.47</td><td class=n>4533.55</td><td class=n><b>0.21</b></td><td class=n>91669.36</td><td class=n>3.78</td><td class=n>0.96</td><td class=n>1.02</td><td class=n>9.61</td><td class=n>0.66</td></tr><tr><td class='k n'>1331</td><td class='k n'>0.50±0.29s</td><td class='k n'><b>forust (Rust)</b> <span class=c>18/18</span></td><td class=n>2.45</td><td class=n>3.14</td><td class=n><b>5.59</b></td><td class=n>173.06</td><td class=n>3.94</td><td class=n>1.73</td><td class=n>4.43</td><td class=n>0.00</td><td class=n>744.09</td><td class=n>7.44</td><td class=n>5113.39</td><td class=n>0.22</td><td class=n>93768.85</td><td class=n>3.96</td><td class=n>1.01</td><td class=n>0.92</td><td class=n>9.87</td><td class=n>0.66</td></tr><tr><td class='k n'>1230</td><td class='k n'>11.47±17.45s</td><td class='k n'><b>grf_rf</b> <span class=c>18/18</span></td><td class=n>2.36</td><td class=n>3.27</td><td class=n>8.26</td><td class=n>228.61</td><td class=n>4.44</td><td class=n>2.78</td><td class=n>6.49</td><td class=n>599.99</td><td class=n>698.28</td><td class=n>7.54</td><td class=n>4614.00</td><td class=n>0.25</td><td class=n>124261.13</td><td class=n>4.00</td><td class=n>0.98</td><td class=n>1.01</td><td class=n>11.01</td><td class=n>0.69</td></tr><tr><td class='k n'>1217</td><td class='k n'>0.23±0.37s</td><td class='k n'><b>CTBoost</b> <span class=c>18/18</span></td><td class=n>2.48</td><td class=n>4.01</td><td class=n>6.38</td><td class=n>192.08</td><td class=n>4.54</td><td class=n>2.69</td><td class=n>4.91</td><td class=n>0.01</td><td class=n>703.96</td><td class=n>7.45</td><td class=n><b>4508.58</b></td><td class=n>0.23</td><td class=n>107014.09</td><td class=n>4.19</td><td class=n>0.99</td><td class=n>1.02</td><td class=n>11.41</td><td class=n>0.69</td></tr><tr><td class='k n'>980</td><td class='k n'>13.89±27.18s</td><td class='k n'><b>lmtree</b> <span class=c>16/18</span></td><td class=n>2.32</td><td class=n>4.05</td><td class=n>7.64</td><td class=n>237.92</td><td class=n>4.27</td><td class=n>2.87</td><td class=n>6.64</td><td class=n>876.88</td><td class=n>718.84</td><td class=n>7.51</td><td class=n>4791.58</td><td class=n>0.24</td><td class=n>112822.63</td><td class=n>·</td><td class=n>1.04</td><td class=n>1.12</td><td class=n>·</td><td class=n>0.71</td></tr><tr><td class='k n'>977</td><td class='k n'>1.19±1.53s</td><td class='k n'><b>cubist</b> <span class=c>18/18</span></td><td class=n>2.31</td><td class=n>3.98</td><td class=n>6.81</td><td class=n>204.32</td><td class=n>4.57</td><td class=n>2.43</td><td class=n>6.14</td><td class=n>608.85</td><td class=n>739.27</td><td class=n>7.53</td><td class=n>4535.60</td><td class=n>0.23</td><td class=n>114392.73</td><td class=n>4.43</td><td class=n>1.02</td><td class=n>1.11</td><td class=n>12.01</td><td class=n>0.72</td></tr><tr><td class='k n'>954</td><td class='k n'>10.39±19.45s</td><td class='k n'><b>glmtree</b> <span class=c>15/18</span></td><td class=n>2.31</td><td class=n>4.05</td><td class=n>7.42</td><td class=n>237.92</td><td class=n>4.27</td><td class=n>3.31</td><td class=n>6.13</td><td class=n>641.43</td><td class=n>718.84</td><td class=n>7.51</td><td class=n>4791.58</td><td class=n>0.24</td><td class=n>·</td><td class=n>·</td><td class=n>1.03</td><td class=n>1.12</td><td class=n>·</td><td class=n>0.71</td></tr><tr><td class='k n'>915</td><td class='k n'>70.56±238.75s</td><td class='k n'><b>ModelTree(all)</b> <span class=c>18/18</span></td><td class=n>2.35</td><td class=n>5.07</td><td class=n>6.82</td><td class=n>218.18</td><td class=n>5.76</td><td class=n>2.44</td><td class=n>5.83</td><td class=n><b>0.00</b></td><td class=n>2317.30</td><td class=n>7.46</td><td class=n>4639.11</td><td class=n>0.36</td><td class=n>1536911.90</td><td class=n>4.85</td><td class=n>1.13</td><td class=n>1.09</td><td class=n>256.25</td><td class=n>0.71</td></tr><tr><td class='k n'>892</td><td class='k n'>17.77±44.25s</td><td class='k n'><b>modeltrees lmtree</b> <span class=c>17/18</span></td><td class=n>2.32</td><td class=n>4.05</td><td class=n>7.64</td><td class=n>237.92</td><td class=n>4.27</td><td class=n>2.87</td><td class=n>6.64</td><td class=n>0.02</td><td class=n>718.84</td><td class=n>7.51</td><td class=n>4791.58</td><td class=n>0.24</td><td class=n>·</td><td class=n>4.85</td><td class=n>1.04</td><td class=n>1.12</td><td class=n>81.75</td><td class=n>0.71</td></tr><tr><td class='k n'>824</td><td class='k n'>143.31±463.47s</td><td class='k n'><b>MARS (mars-earth)</b> <span class=c>12/18</span></td><td class=n>2.27</td><td class=n>4.75</td><td class=n>7.11</td><td class=n>·</td><td class=n>4.10</td><td class=n>4.31</td><td class=n>6.38</td><td class=n><b>0.00</b></td><td class=n>713.60</td><td class=n>·</td><td class=n>5853.26</td><td class=n>·</td><td class=n>·</td><td class=n>·</td><td class=n>0.99</td><td class=n>1.30</td><td class=n>·</td><td class=n>0.73</td></tr><tr><td class='k n'>766</td><td class='k n'>1.46±2.38s</td><td class='k n'><b>ctree</b> <span class=c>18/18</span></td><td class=n>2.43</td><td class=n>4.02</td><td class=n>9.76</td><td class=n>276.40</td><td class=n>4.26</td><td class=n>3.58</td><td class=n>7.10</td><td class=n>603.51</td><td class=n>788.02</td><td class=n>7.87</td><td class=n>4850.64</td><td class=n>0.30</td><td class=n>140881.71</td><td class=n>4.59</td><td class=n>1.08</td><td class=n>1.15</td><td class=n>13.60</td><td class=n>0.75</td></tr><tr><td class='k n'>763</td><td class='k n'>0.26±0.37s</td><td class='k n'><b>earth</b> <span class=c>17/18</span></td><td class=n><b>2.26</b></td><td class=n>4.51</td><td class=n>7.06</td><td class=n>·</td><td class=n>4.23</td><td class=n>4.26</td><td class=n>6.20</td><td class=n>1094.57</td><td class=n>714.05</td><td class=n>7.99</td><td class=n>5724.24</td><td class=n>0.29</td><td class=n>145755.85</td><td class=n>5.03</td><td class=n>1.00</td><td class=n>1.30</td><td class=n>18.59</td><td class=n>0.73</td></tr><tr><td class='k n'>692</td><td class='k n'>0.13±0.20s</td><td class='k n'><b>sklearn tree</b> <span class=c>18/18</span></td><td class=n>3.16</td><td class=n>4.87</td><td class=n>7.93</td><td class=n>197.74</td><td class=n>5.46</td><td class=n>2.66</td><td class=n>5.66</td><td class=n><b>0.00</b></td><td class=n>841.83</td><td class=n>10.61</td><td class=n>6404.11</td><td class=n>0.32</td><td class=n>146810.51</td><td class=n>5.07</td><td class=n>1.27</td><td class=n>1.11</td><td class=n>11.36</td><td class=n>0.83</td></tr><tr><td class='k n'>651</td><td class='k n'>56.32±225.98s</td><td class='k n'><b>ModelTree(winnow)</b> <span class=c>18/18</span></td><td class=n>2.35</td><td class=n>5.07</td><td class=n>6.82</td><td class=n>231.47</td><td class=n>5.58</td><td class=n>2.39</td><td class=n>5.83</td><td class=n>0.05</td><td class=n>989.82</td><td class=n>7.86</td><td class=n>4778.18</td><td class=n>0.34</td><td class=n>3415089.19</td><td class=n>5.01</td><td class=n>1.13</td><td class=n>1.12</td><td class=n>4877.49</td><td class=n>0.73</td></tr><tr><td class='k n'>650</td><td class='k n'>0.15±0.20s</td><td class='k n'><b>ModelTree(mean)</b> <span class=c>18/18</span></td><td class=n>2.54</td><td class=n>3.97</td><td class=n>9.77</td><td class=n>262.32</td><td class=n>4.62</td><td class=n>4.69</td><td class=n>10.39</td><td class=n>0.23</td><td class=n>793.48</td><td class=n>7.88</td><td class=n>4865.40</td><td class=n>0.38</td><td class=n>188718.13</td><td class=n>5.36</td><td class=n>1.13</td><td class=n>1.27</td><td class=n>18.06</td><td class=n>0.76</td></tr><tr><td class='k n'>566</td><td class='k n'>0.22±0.32s</td><td class='k n'><b>rpart</b> <span class=c>18/18</span></td><td class=n>2.59</td><td class=n>4.31</td><td class=n>9.07</td><td class=n>279.94</td><td class=n>4.71</td><td class=n>4.43</td><td class=n>9.79</td><td class=n>1496.89</td><td class=n>823.83</td><td class=n>8.03</td><td class=n>5088.41</td><td class=n>0.38</td><td class=n>192132.40</td><td class=n>5.46</td><td class=n>1.12</td><td class=n>1.29</td><td class=n>18.91</td><td class=n>0.77</td></tr><tr><td class='k n'>546</td><td class='k n'>0.86±1.22s</td><td class='k n'><b>ModelTree(path)</b> <span class=c>18/18</span></td><td class=n>2.34</td><td class=n>4.40</td><td class=n>11.38</td><td class=n>270.69</td><td class=n>5.78</td><td class=n>4.12</td><td class=n>8.65</td><td class=n>0.27</td><td class=n>990.49</td><td class=n>7.86</td><td class=n>6573.19</td><td class=n>0.38</td><td class=n>1230887.17</td><td class=n>36.29</td><td class=n>1.14</td><td class=n>1.29</td><td class=n>18.98</td><td class=n>0.76</td></tr></tbody>
</table>
<script>
(function () {
  var t = document.querySelector('#elo-interactive table');
  if (!t) return;
  var dir = {};
  t.querySelectorAll('th').forEach(function (th) {
    th.addEventListener('click', function () {
      var i = +th.dataset.t, d = dir[i] = -(dir[i] || 1);
      var rows = Array.prototype.slice.call(t.tBodies[0].rows);
      rows.sort(function (a, b) {
        var x = a.cells[i].textContent.trim(), y = b.cells[i].textContent.trim();
        var nx = parseFloat(x), ny = parseFloat(y);
        return d * (isNaN(nx) || isNaN(ny) ? x.localeCompare(y) : nx - ny);
      });
      rows.forEach(function (r) { t.tBodies[0].appendChild(r); });
    });
  });
})();
</script>
</div>


## Development rounds (ours vs the frozen field)

The external field (R packages, LightGBM, CatBoost, CTBoost, forust,
perpetual, MARS, sklearn, modeltrees) is scored **once** and frozen in
`benchmarks/elo_baselines.csv` — *where we started*. Every development round
re-scores only the ModelTree variants on the same prepared splits:

```bash
uv run python examples/elo_round.py                 # all datasets
ELO_DATASETS=abalone,boston uv run python examples/elo_round.py   # a subset
uv run python examples/elo_round.py --freeze        # deliberately re-anchor
````

Each round prints `started (frozen baseline) -> reached` Elo for every ModelTree flavor, with per-fit timings in the scores file — so progress across iterations is visible instead of silently overwritten. The full from-scratch rebuild (all models, groups of 5 datasets, shareable intermediate state after each group) is `bash examples/elo_run_groups.sh`.

The field's raw scores (per dataset RMSE/MAE/fit time) live in [`benchmarks/elo_scores.csv`](<https://github.com/jkapila/mltreelib/blob/main/benchmarks/elo_scores.csv>); the longer research log is in `docs/field-notes.md`.

## Running a round (repeatable, with provenance)

One command re-scores **our four flavors** against the frozen field:

```
uv run python examples/elo_round.py                    # all 18 datasets
ELO_DATASETS=abalone,boston uv run python examples/elo_round.py   # subset
```

Every round records what created its numbers:

- `benchmarks/rounds/<stamp>/settings.json` — the **exact settings dict** each flavor was fit with (single-sourced: the manifest records the same dict `variants()` builds from, so record and reality cannot drift), the git SHA, the dataset list, every result row, and Elo before → after.
- `benchmarks/elo_round.log` — stdlib `logging` output (file + console), one line per round: `timestamp, stamp, sha,   dataset count, per-model Elo before->after`. This is the first place to look for "what did we change and what did it move".
- `benchmarks/elo_scores.csv` — the live board (our rows upserted); `benchmarks/elo_baselines.csv` — the frozen anchor ("where we started").

Tuned settings from `examples/hpo_optuna.py` print in the same dict shape — paste them into `SETTINGS` in `elo_round.py` to score a tuned flavor.
