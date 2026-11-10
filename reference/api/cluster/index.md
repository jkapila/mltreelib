# API reference — Clustering trees

Tree-based clustering and mixed-type search: TrieKNN, PCT (CLUS), CLTree, HDBSCAN Condensed Tree, and Unsupervised Random Forest (URF).

## TrieKNN

`pythonclass TrieKNN((task='classification', k=5, weights='distance', metric='euclidean', leaf_min_size=None))`

### TrieKNN.fit

`pythonfit((self, /, x, y, col_names=None, dtypes=None))`

### TrieKNN.predict

`pythonpredict((self, /, x))`

### TrieKNN.predict\_proba

`pythonpredict_proba((self, /, x))`

## PredictiveClusteringTree

`pythonclass PredictiveClusteringTree((max_depth=6, min_samples_split=10, min_samples_leaf=5, min_variance_reduction=0.0001, target_weights=None))`

### PredictiveClusteringTree.cluster\_rules

`pythoncluster_rules((self, /))`

### PredictiveClusteringTree.fit

`pythonfit((self, /, x, y=None, col_names=None, dtypes=None, target_names=None))`

### PredictiveClusteringTree.predict\_cluster

`pythonpredict_cluster((self, /, x))`

### PredictiveClusteringTree.predict\_prototype

`pythonpredict_prototype((self, /, x))`

## CLTree

`pythonclass CLTree((max_depth=6, min_samples_leaf=5, min_density=0.6))`

### CLTree.cluster\_bounding\_boxes

`pythoncluster_bounding_boxes((self, /))`

### CLTree.fit

`pythonfit((self, /, data, col_names=None))`

### CLTree.predict\_cluster

`pythonpredict_cluster((self, /, data))`

## HDBSCAN

`pythonclass HDBSCAN((min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0, max_cluster_size=None, metric='euclidean', cluster_selection_method='eom', allow_single_cluster=False))`

### HDBSCAN.condensed\_tree

`pythoncondensed_tree((self, /))`

### HDBSCAN.fit

`pythonfit((self, /, data))`

### HDBSCAN.predict

`pythonpredict((self, /, data))`

## UnsupervisedRandomForest

`pythonclass UnsupervisedRandomForest((n_trees=50, max_depth=6, min_samples_leaf=2, max_features=None, linkage='average', seed=42))`

### UnsupervisedRandomForest.apply

`pythonapply((self, /, data))`

### UnsupervisedRandomForest.fit

`pythonfit((self, /, data, n_clusters=2))`

### UnsupervisedRandomForest.pairwise\_distances

`pythonpairwise_distances((self, /))`
