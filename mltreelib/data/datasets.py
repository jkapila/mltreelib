# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/01_APIS/01a_datasets.ipynb.

# %% auto 0
__all__ = ['get_dataset', 'show_data']

# %% ../../nbs/01_APIS/01a_datasets.ipynb 4
import pandas as pd
import numpy as np
import os, copy, sys
import tarfile
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import urlopen

# %% ../../nbs/01_APIS/01a_datasets.ipynb 5
# An important todo:
# 1) Make this more structured with giving all information about the data as in statsmodels and scikit leanr.
# 2) Add Rdataset things as well inspired by statsmodels
# 3) Use these data sets to show comparisons, testing and validations
# 4) Site Examples with these datasets

# %% ../../nbs/01_APIS/01a_datasets.ipynb 6
_datasets = ['adult','palmerpenguins-raw','palmerpenguins','california','titanic']

def get_dataset(task='regression', # define the need of task
                data_of='palmerpenguins',  # express which data is needed
                description=True, # weather pass description as and added output
                return_single_df = True, # return a single data frame or provide X,y behavior as with scikit-learn.
                target_name = None  # if target name is none the original mentioned target name would be used.
                ):

    if data_of =='adult':
        nms = 'age,workclass,fnlwgt,education,education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country,target'
        data_df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',header=None,
                        names=nms.split(','))
        data_df
        
    elif data_of == 'palmerpenguins-raw':
        data = []
        for uri in [
            "https://portal.edirepository.org/nis/dataviewer?packageid=knb-lter-pal.219.3&entityid=002f3893385f710df69eeebe893144ff",
            "https://portal.edirepository.org/nis/dataviewer?packageid=knb-lter-pal.220.3&entityid=e03b43c924f226486f2f0ab6709d2381",
            "https://portal.edirepository.org/nis/dataviewer?packageid=knb-lter-pal.221.2&entityid=fe853aa8f7a59aa84cdd3197619ef462"]:

            data.append(pd.read_csv(uri))
        
        data_df = pd.concat(data,axis=0).reset_index(drop=True)

    elif data_of == 'palmerpenguins':
        data = []
        for uri in [
            "https://portal.edirepository.org/nis/dataviewer?packageid=knb-lter-pal.219.3&entityid=002f3893385f710df69eeebe893144ff",
            "https://portal.edirepository.org/nis/dataviewer?packageid=knb-lter-pal.220.3&entityid=e03b43c924f226486f2f0ab6709d2381",
            "https://portal.edirepository.org/nis/dataviewer?packageid=knb-lter-pal.221.2&entityid=fe853aa8f7a59aa84cdd3197619ef462"]:

            data.append(pd.read_csv(uri))
        
        data_df = pd.concat(data,axis=0).reset_index(drop=True)
        data_df = data_df[['Species','Island','Culmen Length (mm)','Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)','Sex','Date Egg']].copy(deep=True)
        data_df.columns = ['species','island','bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g','sex','year']

    elif data_of =='california':
        data_df = pd.read_csv('https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv',sep=",")
        
    elif data_of == 'titanic':
        train = pd.read_csv('https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/titanic/train.csv')
        test = pd.read_csv('https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/titanic/test.csv')
        data_df = pd.concat((train,test),axis=0).reset_index(drop=True)


    return data_df

def show_data():
    """Lists all the data available with the package"""
    print('Currently available datasets are:')
    global _datasets
    print(_datasets)

    

