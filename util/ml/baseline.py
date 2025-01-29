from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import pickle

def filter_columns(df,criteria="^(convective_precipitation_|large_scale_snowfall_)"):
    # Remove columns with specific patterns
    columns_to_remove = df.filter(regex=criteria).columns
    df = df.drop(columns=columns_to_remove)
    return df

def filt_with_feature_importance(indexFI_20, indexFI_30, indexFI_40, data_dict):
    # Filter the input data with the feature importance, 20 variables
    Xtrain_filtered_20 = data_dict['Xtrain_20'].iloc[:, indexFI_20]
    Xvalid_filtered_20 = data_dict['Xvalid_20'].iloc[:, indexFI_20]

    # Filter the input data with the feature importance, 30 variables
    Xtrain_filtered_30 = data_dict['Xtrain_30'].iloc[:, indexFI_30]
    Xvalid_filtered_30 = data_dict['Xvalid_30'].iloc[:, indexFI_30]

    # Filter the input data with the feature importance, 40 variables
    Xtrain_filtered_40 = data_dict['Xtrain_40'].iloc[:, indexFI_40] 
    Xvalid_filtered_40 = data_dict['Xvalid_40'].iloc[:, indexFI_40] 

    # Store in a new dictionary
    data_dict_filt = {}
    data_dict_filt['Xtrain_20'] = Xtrain_filtered_20
    data_dict_filt['Xvalid_20'] = Xvalid_filtered_20
    data_dict_filt['Xtrain_30'] = Xtrain_filtered_30
    data_dict_filt['Xvalid_30'] = Xvalid_filtered_30
    data_dict_filt['Xtrain_40'] = Xtrain_filtered_40
    data_dict_filt['Xvalid_40'] = Xvalid_filtered_40
    data_dict_filt['ytrain_cdf'] = data_dict['ytrain_cdf']
    data_dict_filt['ytrain_max'] = data_dict['ytrain_max']
    data_dict_filt['yvalid_cdf'] = data_dict['yvalid_cdf']
    data_dict_filt['yvalid_max'] = data_dict['yvalid_max']
    return data_dict_filt

def filt_index_no_conv_snow(index_40,index_30,index_20,filtindex_40,filtindex_30,filtindex_20):
    filt_index_40 = []
    for obj in index_40:
        if obj in filtindex_40:
            continue
        else:
            if obj>39 and obj<85:
                filt_index_40.append(obj-2)
            elif obj>85:
                filt_index_40.append(obj-5)
            else:
                filt_index_40.append(obj)

    filt_index_30 = []
    for obj in index_30:
        if obj in filtindex_30:
            continue
        else:
            if obj>39:
                filt_index_30.append(obj-2)
            else:
                filt_index_30.append(obj)

    filt_index_20 = []
    for obj in index_20:
        if obj in filtindex_20:
            continue
        else:
            if obj>39:
                filt_index_20.append(obj-2)
            else:
                filt_index_20.append(obj)
    return filt_index_20, filt_index_30, filt_index_40

# Normalize the input data with mean and standard deviation with sklearn StandardScaler
def normalize_data(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X_normalized = scaler.transform(X)
    return X_normalized


# Train a linear regression model
def train_linear_regression(X, y):
    reg = linear_model.LinearRegression()
    reg.fit(X, y)
    return reg

def train_rf(X, y, seed):
    # Define the random forest model
    rf = RandomForestRegressor(random_state=seed)
    
    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],        # Number of trees
        'max_depth': [1, 10, 20],          # Maximum depth of the trees
        'min_samples_split': [2, 5, 10],      # Minimum samples to split a node
        'min_samples_leaf': [1, 2, 4]         # Minimum samples in a leaf
        }
    
    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                               cv=5, scoring='neg_mean_squared_error', 
                               verbose=0, n_jobs=-1)
    
    # Fit the model
    grid_search.fit(normalize_data(X), y)
    return grid_search.best_estimator_

# A function to save models with pickle
def save_models(models, filename):
    with open(filename,'wb') as f:
        pickle.dump(models, f)

# A function to load files saved with pickle
def load_pickle(file_path):
    import pickle
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def flatten(xss):
    return [x for xs in xss for x in xs]