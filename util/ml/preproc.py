import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import glob, os
from tqdm import tqdm
from sklearn.decomposition import PCA

def train_PCA(X_train):
    # Fit the PCA model
    pca = PCA()
    pca.fit(X_train)
    return pca, np.mean(X_train), np.std(X_train)

def create_nonormlX(seed_dicts, PCloadings, corrlist, nVARS):
    # Find the number of most-correlated variables
    VARCORR = corrlist["Unnamed: 0"].iloc[:nVARS].values
    
    # Store the non-normalized X data and the variable sizes
    Xs = []
    varsizes = []
    for iseed in range(len(seed_dicts)):
        # Select the variables from the dictionary
        wantvars = [PCloadings[iseed][obj] for obj in VARCORR]
        # Size of the variables
        varsize = [obj.shape[1] for obj in wantvars]
        # Concatenate the variables
        X = np.concatenate(wantvars, axis=1)
        Xs.append(X)
        varsizes.append(varsize)
    return {'X':Xs, 'varsize':varsizes}

def myPCA_projection_sen(pca_dict=None,varname=None,toproj_flatvar=None,trainmean=None):
    projvar_transformed = np.dot(toproj_flatvar-trainmean,pca_dict[varname].components_.T)
    return projvar_transformed

def get_num_unique_var(filted_feature):
    TEST = [filted_feature[:i+1] for i in range(len(filted_feature))]
    storename_exp = []
    for obj in TEST:
        storename = []
        for i in range(len(obj)):
            storename.append(obj[i][0])
        storename_exp.append(storename)
    return [len(list(set(obj))) for obj in storename_exp]