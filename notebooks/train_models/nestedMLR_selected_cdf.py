import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import glob, os, gc
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Add the path to the directory containing the module
import sys
sys.path.append('../../')
from util.ml import baseline, metrics, nestedMLR
from properscoring import crps_ensemble  # For CRPS calculation
from sklearn.utils import resample  # For bootstrapping

print(sys.argv)
split=(str(sys.argv[1]))

# Find the folder name organized by seed number
seed_doc = sorted(glob.glob('../../datas/seed_revised_*/'))[0]

# Load the data
# Load the time series data
df = pd.read_csv(seed_doc +'X_train_ts_all.csv')
df_valid = pd.read_csv(seed_doc +'X_validation_ts_all.csv')
df_test = pd.read_csv(seed_doc +'X_test_ts_all.csv')

# Find the name for each column
column_names = ([obj.split('_step_')[0] for obj in df.columns])
# Unique names in the column name list
unique_names = list(set(column_names))
# Remove strings with large_scale
unique_names_filt = [var for var in unique_names if "large_scale" not in var]

# Now we read in the y data for every fold
y_train = []
y_val = []
for i in range(7):
    y_train.append(baseline.load_pickle(f'../../datas/proc/sfs/y/ytrain_split_{i}.pkl'))
    y_val.append(baseline.load_pickle(f'../../datas/proc/sfs/y/yval_split_{i}.pkl'))

pcs_train = baseline.load_pickle(f'../../datas/proc/sfs/pcsall_train.pkl')
pcs_val = baseline.load_pickle(f'../../datas/proc/sfs/pcsall_valid.pkl')

filted_feature = baseline.load_pickle('../../datas/proc/sfs/results/best_linear_cdf_feature.pkl')

def select_subset_X(pcs_train,pcs_val,y_train,iseed,target_cat,selected_pcs):
    X_train_subset = [pcs_train[iseed][sel_var][:] for sel_var, _ in selected_pcs]
    X_val_subset = [pcs_val[iseed][sel_var][:] for sel_var, _ in selected_pcs]
    return X_train_subset, X_val_subset

X_train_subset, X_val_subset = [],[]
for i in range(7):
    X_train, X_val = select_subset_X(pcs_train,pcs_val,y_train,i,0,filted_feature)
    X_train_subset.append(X_train)
    X_val_subset.append(X_val)

sizes = [obj.shape[1] for obj in X_train_subset[0]]

models, val_rmses = [],[]
for i in tqdm(range(1, 21)):
    model = nestedMLR.nestedMLR_var_global(sizes[:i],4, 2.25)
    model.fit(np.hstack(X_train_subset[split][:i]), y_train[0]['cdf'],2000)
    models.append(model)
    y_pred = model.predict(np.hstack(X_val_subset[split][:i]))
    val_rmses.append(mean_squared_error(y_val[split]['cdf'], y_pred, squared=False))

baseline.save_models({'model': models, 'val_rmses': val_rmses}, f'../../datas/proc/sfs/results/nestedMLR_cdf_{split}.pkl')