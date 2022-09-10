mltreelib : Machine Learnign with Tree Based Library
================

<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

This package evovled from the attempt to make ***right*** kind of
**Decision Tress** which was ideated by many people like Hastie,
Tibshirani, Friedman, Quilan, Loh, Chaudhari.

## Install

``` sh
pip install mltreelib
```

## How to use

Create a sample data

``` python
import numpy as np
import pandas as pd
from mltreelib.base.data import Dataset
from mltreelib.tree import Tree
```

``` python
n_size = 1000
rnd = np.random.RandomState(1234)
dummy_data = pd.DataFrame({'numericfull':rnd.randint(1,500,size=n_size),
                            'unitint':rnd.randint(1,25,size=n_size),
                            'floatfull':rnd.random_sample(size=n_size),
                            'floatsmall':np.round(rnd.random_sample(size=n_size)+rnd.randint(1,25,size=n_size),2),
                            'categoryobj':rnd.choice(['a','b','c','d'],size=n_size),
                            'stringobj':rnd.choice(["{:c}".format(k) for k in range(97, 123)],size=n_size)})
dummy_data.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>numericfull</th>
      <th>unitint</th>
      <th>floatfull</th>
      <th>floatsmall</th>
      <th>categoryobj</th>
      <th>stringobj</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>304</td>
      <td>18</td>
      <td>0.908959</td>
      <td>8.56</td>
      <td>a</td>
      <td>c</td>
    </tr>
    <tr>
      <th>1</th>
      <td>212</td>
      <td>24</td>
      <td>0.348582</td>
      <td>14.35</td>
      <td>a</td>
      <td>g</td>
    </tr>
    <tr>
      <th>2</th>
      <td>295</td>
      <td>15</td>
      <td>0.392977</td>
      <td>21.98</td>
      <td>a</td>
      <td>y</td>
    </tr>
    <tr>
      <th>3</th>
      <td>54</td>
      <td>20</td>
      <td>0.720856</td>
      <td>5.33</td>
      <td>a</td>
      <td>q</td>
    </tr>
    <tr>
      <th>4</th>
      <td>205</td>
      <td>21</td>
      <td>0.897588</td>
      <td>23.03</td>
      <td>c</td>
      <td>k</td>
    </tr>
  </tbody>
</table>
</div>

Create a Dataset

``` python
dataset = Dataset(df=dummy_data)
print(dataset)
print('Pandas Data Frame        : ',np.round(dummy_data.memory_usage(deep=True).sum()*1e-6,2),'MB')
print('Dataset Structured Array : ',np.round(dataset.data.nbytes*1e-6/ 1024 * 1024,2),'MB')
dataset.data[:5]
```

    Dataset(df=Shape((1000, 6), reduce_datatype=True, encode_category=None, add_intercept=False, na_treatment=allow, copy=False, digits=None, n_category=None, split_ratio=None)
    Pandas Data Frame        :  0.15 MB
    Dataset Structured Array :  0.03 MB

    array([(304, 18, 0.9089594 ,  8.56, 'a', 'c'),
           (212, 24, 0.34858167, 14.35, 'a', 'g'),
           (295, 15, 0.39297667, 21.98, 'a', 'y'),
           ( 54, 20, 0.7208556 ,  5.33, 'a', 'q'),
           (205, 21, 0.89758754, 23.03, 'c', 'k')],
          dtype=[('numericfull', '<u2'), ('unitint', 'u1'), ('floatfull', '<f4'), ('floatsmall', '<f4'), ('categoryobj', 'O'), ('stringobj', 'O')])
