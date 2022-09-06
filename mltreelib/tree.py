# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/04_trees.ipynb.

# %% auto 0
__all__ = ['Tree', 'CARTRegressionTree']

# %% ../nbs/04_trees.ipynb 3
from fastcore.utils import *
import numpy as np
import pandas as pd
import gc
import numba
from mltree.basetree import DecisionNode, BaseDecisionTree
from mltree.utils import *



# %% ../nbs/04_trees.ipynb 5
class Tree:
    "Tree class enable building all simpe single tree models"

    def __init__(self) -> None:
        pass

    def fit_formula(self,formula,data):
        "Fit using formula and dataset"
        pass

    def fit(self,X,y):
        "Sklearn style fit"
        pass

    def fit_dataset(self,dataset,X_cols,target_names):
        "Fit in native way"
        pass

    def predict(self,X):
        "Best predicitons"
        pass

    def predict_leaf(self,X):
        "Predicting vlaues"
        pass

    def predict_proba(self,X):
        "Prediction probabilities"
        pass
    
    def transform(self,X,value_type='prob'):
        "Provide raw outputs form the tree"
        pass
    
    def apply(self,X):
        pass

    def describe(self, strucutre=False, plot=False, eval=False,importance=None):
        "Everythign to know about the tree"
        pass

    def prune(self,complexity):
        "Prune the tree based on certain criteria"
        pass

    def save(self):
        "save tree"
        pass
    def load(self):
        "load a tree"
        pass
    

# %% ../nbs/04_trees.ipynb 6
class CARTRegressionTree(BaseDecisionTree):
    def _calculate_variance_reduction(self, y, y1, y2):
        var_tot = calculate_variance(y)
        var_1 = calculate_variance(y1)
        var_2 = calculate_variance(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)

        # Calculate the variance reduction
        variance_reduction = var_tot - (frac_1 * var_1 + frac_2 * var_2)

        return sum(variance_reduction)

    def _mean_of_y(self, y):
        value = np.mean(y, axis=0)
        if self.verbose: print('Leaf values:',value,value.shape,len(value.shape))
        return value
        # return value if len(value.shape) > 1 else value[0]
    
    def _cart_node_function(self,Xy, Xy1, Xy2, target,features):
        y = Xy[[target]].values
        y1 = Xy1[[target]].values
        y2 = Xy2[[target]].values
        impurity = self._calculate_variance_reduction(y,y1,y2)
        return impurity,None, None, None
    
    def _cart_leaf_function(self, Xy, target, features):
        mean_val = self._mean_of_y(Xy[target].values)
        return mean_val, None, None, None

    def fit(self, X, y):
        # output of both node function and leaf functionisin form leaf_value/impurity,estimator,weights,addons with inputs
        #inputs _node_function(Xy, Xy1,Xy2,target,features)
        self._node_function = self._cart_node_function
        #inputs _leaf_function(Xy, target, features)
        self._leaf_function = self._cart_leaf_function
        super(CARTRegressionTree, self).fit(X, y)


