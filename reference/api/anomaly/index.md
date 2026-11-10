# API reference — Anomaly &amp; density trees

Tree-based anomaly detection, density estimation, streaming mass scoring, collusive displacement (CoDisp), active human feedback, and contextual surprise.

## GritBot

`pythonclass GritBot((min_items=None, max_norm=None, sig_threshold=None, max_depth=None, check_cols=None))`

### GritBot.scan

`pythonscan((self, /, x, col_names=None, dtypes=None))`

## DensityEstimationTree

`pythonclass DensityEstimationTree((max_depth=8, min_samples_split=10, min_leaf_samples=5, min_density_gain=0.0, alpha=0.0))`

### DensityEstimationTree.fit

`pythonfit((self, /, x, col_names=None, dtypes=None))`

### DensityEstimationTree.predict\_anomaly\_score

`pythonpredict_anomaly_score((self, /, x))`

### DensityEstimationTree.predict\_density

`pythonpredict_density((self, /, x))`

## ExtendedIsolationTree

`pythonclass ExtendedIsolationTree((max_depth=None, extension_level=None, subsample_size=None, seed=42))`

### ExtendedIsolationTree.anomaly\_score

`pythonanomaly_score((self, /, x))`

### ExtendedIsolationTree.fit

`pythonfit((self, /, x, col_names=None, dtypes=None))`

### ExtendedIsolationTree.path\_length

`pythonpath_length((self, /, x))`

## HalfSpaceTree

`pythonclass HalfSpaceTree((max_depth=6, window_size=250, initial_mass=0.0, seed=42))`

### HalfSpaceTree.anomaly\_score

`pythonanomaly_score((self, /, data))`

### HalfSpaceTree.fit

`pythonfit((self, /, data))`

### HalfSpaceTree.insert

`pythoninsert((self, /, point))`

### HalfSpaceTree.mass\_score

`pythonmass_score((self, /, data))`

## RandomCutTree

`pythonclass RandomCutTree((sample_size=256, seed=42))`

### RandomCutTree.codisp

`pythoncodisp((self, /, data))`

### RandomCutTree.fit

`pythonfit((self, /, data))`

## ActiveAnomalyTree

`pythonclass ActiveAnomalyTree((max_depth=8, eta=0.1, prior_weight=1.0, seed=42))`

### ActiveAnomalyTree.anomaly\_score

`pythonanomaly_score((self, /, data))`

### ActiveAnomalyTree.fit

`pythonfit((self, /, data))`

### ActiveAnomalyTree.update\_feedback

`pythonupdate_feedback((self, /, point, is_anomaly, eta=0.5))`

## ContextAnomalyTree

`pythonclass ContextAnomalyTree((max_depth=5, alphabet_size=256, smoothing_discount=0.5))`

### ContextAnomalyTree.conditional\_prob

`pythonconditional_prob((self, /, context, symbol))`

### ContextAnomalyTree.fit

`pythonfit((self, /, sequences))`

### ContextAnomalyTree.insert

`pythoninsert((self, /, context, symbol))`

### ContextAnomalyTree.surprise\_score

`pythonsurprise_score((self, /, context, symbol))`
