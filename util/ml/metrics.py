from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import numpy as np
from natsort import natsorted
from scipy.stats import genextreme

# Revert transformation
def revert_cdf(cluster, transformedWINDS):
    outWINDS = np.zeros(transformedWINDS.shape)
    for i in range(15):
        clusterz = cluster[i]
        cdf_values = (1 - np.exp(-transformedWINDS[:,i]))
        outWINDS[:,i] = genextreme.ppf(cdf_values, clusterz[0], loc=clusterz[1], scale=clusterz[2])
    return outWINDS

def r2_score_f(y_true,y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    r2 = 1-np.sum((y_true-y_pred)**2)/np.sum((y_true-np.mean(y_true))**2)
    return r2

def score_model(model, X, y):
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2,r2analysis = r2_score_f(y, y_pred)
    return {"RMSE":rmse, "MAE":mae, "R2":r2, "r2analysis":r2analysis}

# Evaluate the linear regression model
def evaluate_model(model_cdf,model_max,data_dict,varnum):
    store_results = []
    for ind in range(len(data_dict)):
        # Initialize the scaler
        scaler = StandardScaler()
        # Fit the scaler only on the training set
        scaler.fit(data_dict[ind][f'Xtrain_{int(varnum)}'])
        # Transform the training, validation, and test sets
        X_train_scaled = scaler.transform(data_dict[ind][f'Xtrain_{int(varnum)}'])
        X_val_scaled = scaler.transform(data_dict[ind][f'Xvalid_{int(varnum)}'])

        # Evaluate the model for training set
        score_cdf = score_model(model_cdf[ind], X_train_scaled, data_dict[ind]['ytrain_cdf'])
        score_max = score_model(model_max[ind], X_train_scaled, data_dict[ind]['ytrain_max'])

        # Evaluate the model for validation set
        score_cdf_val = score_model(model_cdf[ind], X_val_scaled, data_dict[ind]['yvalid_cdf'])
        score_max_val = score_model(model_max[ind], X_val_scaled, data_dict[ind]['yvalid_max'])

        # Store the results in dictionary
        data_store = {
            "train_cdf": score_cdf,
            "train_max": score_max,
            "val_cdf": score_cdf_val,
            "val_max": score_max_val
        }
        store_results.append(data_store)
    return store_results

def evaluate_cdfmodel(model_cdf,data_dict,varnum):
    store_results = []
    for ind in range(len(data_dict)):
        # Initialize the scaler
        scaler = StandardScaler()
        # Fit the scaler only on the training set
        scaler.fit(data_dict[ind][f'Xtrain_{int(varnum)}'])
        # Transform the training, validation, and test sets
        X_train_scaled = scaler.transform(data_dict[ind][f'Xtrain_{int(varnum)}'])
        X_val_scaled = scaler.transform(data_dict[ind][f'Xvalid_{int(varnum)}'])

        # Evaluate the model for training set
        score_cdf = score_model(model_cdf[ind], X_train_scaled, data_dict[ind]['ytrain_cdf'])

        # Evaluate the model for validation set
        score_cdf_val = score_model(model_cdf[ind], X_val_scaled, data_dict[ind]['yvalid_cdf'])

        # Store the results in dictionary
        data_store = {
            "train_cdf": score_cdf,
            "val_cdf": score_cdf_val,
        }
        store_results.append(data_store)
    return store_results

def evaluate_maxmodel(model_max,data_dict,varnum):
    store_results = []
    for ind in range(len(data_dict)):
        # Initialize the scaler
        scaler = StandardScaler()
        # Fit the scaler only on the training set
        scaler.fit(data_dict[ind][f'Xtrain_{int(varnum)}'])
        # Transform the training, validation, and test sets
        X_train_scaled = scaler.transform(data_dict[ind][f'Xtrain_{int(varnum)}'])
        X_val_scaled = scaler.transform(data_dict[ind][f'Xvalid_{int(varnum)}'])

        # Evaluate the model for training set
        score_max = score_model(model_max[ind], X_train_scaled, data_dict[ind]['ytrain_max'])

        # Evaluate the model for validation set
        score_max_val = score_model(model_max[ind], X_val_scaled, data_dict[ind]['yvalid_max'])

        # Store the results in dictionary
        data_store = {
            "train_max": score_max,
            "val_max": score_max_val
        }
        store_results.append(data_store)
    return store_results

from sklearn.metrics import mean_squared_error
def get_model_and_test(filted_feature,pcs_train,pcs_test,pcs_val,y_train,y_val,y_test,target_cat):
    y_tests = [y_test[target_cat] for i in range(7)]
    TEST = [filted_feature[:i+1] for i in range(len(filted_feature))]
    storeout = []
    for j in range(len(TEST)):
        storestuff = []
        for i in range(7):
            final_X_train = np.hstack([pcs_train[i][sel_var][:, [pc_idx]] for sel_var, pc_idx in TEST[j]])
            final_X_val = np.hstack([pcs_val[i][sel_var][:, [pc_idx]] for sel_var, pc_idx in TEST[j]])
            final_X_test = np.hstack([pcs_test[i][sel_var][:, [pc_idx]] for sel_var, pc_idx in TEST[j]])
            final_model = linear_model.LinearRegression()
            final_model.fit(final_X_train, y_train[i][target_cat])
            y_pred_train = final_model.predict(final_X_train)
            y_pred_val = final_model.predict(final_X_val)
            y_pred_test = final_model.predict(final_X_test)
            storestuff.append({'train_rmse': mean_squared_error(y_train[i][target_cat], y_pred_train, squared=False),
                               'val_rmse': mean_squared_error(y_val[i][target_cat], y_pred_val, squared=False),
                               'test_rmse': mean_squared_error(y_tests[i], y_pred_test, squared=False),
                               'model': final_model,
                               'ypred_train': y_pred_train,
                               'ypred_val': y_pred_val,
                               'ypred_test': y_pred_test})
        storeout.append(storestuff)
    return storeout #modelnum, splitnum

def calc_convertedCDF_rmse(store_XXX_cdf,clusters,y_train,y_val,y_test,varsretained,fourier,varhier,typedata):
    TESTstore = []
    for splix in range(7):
        TSTsot = []
        for j in range(15):
            if typedata=='val':
                ydata = y_val
                Xname = 'ypred_val'
            elif typedata=='test':
                ydata = y_test
                Xname = 'ypred_test'
            else:
                ydata = y_train
                Xname = 'ypred_train'

            ytrue = ydata[splix]['max'][:,j]
            ypred = (revert_cdf(clusters,store_XXX_cdf[varsretained]['XXXs'][fourier][varhier][splix][Xname]))[:,j]
            mask = ~np.isnan(ypred)
            rmse= mean_squared_error(ytrue[mask].flatten(),ypred[mask].flatten(), squared=False)
            TSTsot.append(rmse)
        TESTstore.append(TSTsot)
    return TESTstore