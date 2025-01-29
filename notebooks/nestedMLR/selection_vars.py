import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import mean_absolute_error, mean_squared_error


#for name_of_variable_levels in [vars_selected_40, vars_selected_30, vars_selected_20]:
def extract_timeseries(name_of_variable_levels, all_series='yes'): 
    # shape all_data as the timeseries data
    X_all_vars = {}

    for var in name_of_variable_levels:
        var, stat = var.rsplit("_", 1)
        if var == 'sea_surface_temperature':
            continue
        #for stat in stats:
        var_stat = f'{var}_{stat}'
        #if var_stat == f'instantaneous_10m_wind_gust_{stat}':
        #storm_data = []
        if all_series == 'yes':
            var_temp_non_eu = pd.read_csv(f'data/time_series_1h_non_EU/{var}/{var}_{stat}.csv')
            # reverse the order of the columns
            var_temp_non_eu = var_temp_non_eu.iloc[:, ::-1]
            var_temp_eu = pd.read_csv(f'data/time_series_1h_EU/{var}/{var}_{stat}.csv')
            var_temp = pd.concat([var_temp_non_eu, var_temp_eu], axis=1)
        else:
            var_temp = pd.read_csv(f'data/time_series_1h_non_EU/{var}/{var}_{stat}.csv')
        var_temp = var_temp.drop(columns=['Unnamed: 0', 'storm_index'])
        # select only the 12 first hours (represented by the 12 first columns)
        if all_series == 'no':
            var_temp = var_temp.iloc[:, :12]
        #var_temp_2 = var_temp.drop(columns=['storm_index'])
        var_temp_reshape = var_temp.to_numpy().reshape((var_temp.shape[0]*var_temp.shape[1]))
        X_all_vars[f'{var_stat}'] = var_temp_reshape

    # Assuming all_vars is populated as a dictionary
    df_X_all_vars = pd.DataFrame(X_all_vars)

    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(df_X_all_vars)
    return df_X_all_vars, scaled_X

def extract_timeseries_y(all_series='yes'):
    y_all = {}

    var = 'instantaneous_10m_wind_gust'
    stat = 'max'
    var_stat = f'{var}_{stat}'
    if all_series == 'yes':
        var_temp_non_eu = pd.read_csv(f'data/time_series_1h_non_EU/{var}/{var}_{stat}.csv')
        # reverse the order of the columns
        var_temp_non_eu = var_temp_non_eu.iloc[:, ::-1]
        var_temp_eu = pd.read_csv(f'data/time_series_1h_EU/{var}/{var}_{stat}.csv')
        var_temp = pd.concat([var_temp_non_eu, var_temp_eu], axis=1)
    else:
        var_temp = pd.read_csv(f'data/time_series_1h_EU/{var}/{var}_{stat}.csv')
    var_temp = var_temp.drop(columns=['Unnamed: 0', 'storm_index', 'storm_name','start_date'])
    # select only the 12 first hours (represented by the 12 first columns)
    if all_series == 'no':
        var_temp = var_temp.iloc[:, :12]
    #var_temp_2 = var_temp.drop(columns=['storm_index'])
    var_temp_reshape = var_temp.to_numpy().reshape((var_temp.shape[0]*var_temp.shape[1]))
    y_all[f'{var_stat}'] = var_temp_reshape

    df_y = pd.DataFrame(y_all)

    return df_y

def feature_selection(df_X_all_vars, scaled_X, df_y, model):
    # Initialize the sequential feature selector
    sfs = SequentialFeatureSelector(
                                    model,  # Use the same model as before
                                    n_features_to_select=5,  # Select 5 features
                                    direction='forward',  # Forward selection
                                    n_jobs=-1  # Use all available cores
                                    )   

    # Fit the selector

    try: 
        df_y = df_y.to_numpy().ravel()
    except:
        print('y is already a numpy array')
    sfs.fit(scaled_X, df_y)

    # Get the selected feature indices
    selected_indices = sfs.get_support(indices=True)

    # Get the names of the selected features
    selected_features = df_X_all_vars.columns[selected_indices]
    print("Selected features:", selected_features)
    return selected_features

def prepare_training_data(transposed_data, storm_index_training, updated_columns):
    """
    Prepare training data by filtering and concatenating rows based on storm indices.

    Parameters:
        transposed_data (pd.DataFrame): The input DataFrame containing data to filter.
        storm_index_training (list): List of storm indices to filter the data by.
        updated_columns (list): List of column names to update the resulting DataFrame.

    Returns:
        pd.DataFrame: The processed training data with specified columns and without 'storm_number'.
    """
    X_train = []
    
    for storm_index in storm_index_training:
        temp = transposed_data[transposed_data['storm_number'] == storm_index]
        X_train.append(temp)
    
    # Concatenate the filtered data
    X_train = np.concatenate(X_train, axis=0)
    
    # Convert to DataFrame
    X_train = pd.DataFrame(X_train, columns=updated_columns)
    
    # Drop the 'storm_number' column
    try:
        X_train = X_train.drop(columns=['storm_number'])
    except KeyError:
        X_train = X_train.drop(columns=['storm_number_PCA_1'])
        print('No storm_number column to drop, which is not fine')
    
    return X_train

def process_y_data(y_all_cdf, storm_index_training):
    """
    Process the y_all_cdf DataFrame to filter, sort, and convert it into a NumPy array 
    for training purposes.

    Parameters:
        y_all_cdf (pd.DataFrame): DataFrame containing storm-related data, including 'storm_name' and 'storm_number'.
        storm_index_training (list or iterable): List of storm indices to include in the training data.

    Returns:
        np.ndarray: Processed NumPy array of the training data without 'storm_name' and 'storm_number'.
    """
    # Drop 'storm_name' column
    try:
        y_all_cdf = y_all_cdf.drop(columns=['storm_name'])
    except KeyError:
        print('No storm_name column to drop')
    
    # Filter rows where 'storm_number' is in storm_index_training
    y_train = y_all_cdf[y_all_cdf['storm_number'].isin(storm_index_training)]
    
    # Sort by 'storm_number' and reset index
    y_train = y_train.sort_values(by=['storm_number']).reset_index(drop=True)
    
    # Drop 'storm_number' column
    y_train = y_train.drop(columns=['storm_number'])
    
    # Convert to NumPy array
    y_train_np = y_train.to_numpy()
    
    return y_train_np
    
def evaluate_models(y_true, predictions_dict):
    """
    Evaluate RMSE, MAE, and transformed RMSE for multiple models.

    Parameters:
    - y_true (array-like): Ground truth (true) target values.
    - predictions_dict (dict): Dictionary where keys are model names and values are predictions.

    Returns:
    - results (dict): Dictionary containing RMSE, MAE, and transformed RMSE metrics for each model.
    """
    results = {}
    y_true_real = 1 - np.exp(-y_true)  # Transform y_true to real units

    for model_name, predictions in predictions_dict.items():
        # Calculate RMSE and MAE
        rmse = root_mean_squared_error(y_true, predictions)
        mae = mean_absolute_error(y_true, predictions)

        # Transform predictions to real units
        predictions_real = 1 - np.exp(-predictions)

        # Calculate transformed RMSE
        rmse_real = root_mean_squared_error(y_true_real, predictions_real)

        # Calculate return RMSE
        return_rmse = round((1 / (1 - rmse_real)) / 2, 2)

        # Store results for the current model
        results[model_name] = {
            "RMSE": rmse,
            "MAE": mae,
            "Transformed RMSE": rmse_real,
            "Return RMSE": return_rmse
        }
    
    return results