import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import rasterio

masking_value = 0#2**0.5

def preprocess_X_train_data(X):
    # Reshape the X array
    reshape = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    
    # Fit the StandardScaler to the reshaped X array
    X_train_mean = np.nanmean(reshape, axis=0)
    X_train_std = np.nanstd(reshape, axis=0)
    #reshape_std = (reshape - X_train_mean) / X_train_std
    
    reshape_std = StandardScaler().fit_transform(reshape) #do it over the training set for normalizing 

    # Create the mask indicating where NaN values were originally located
    mask_X = np.isnan(X).reshape(X.shape[0], X.shape[1], X.shape[2])
    
    # Replace NaN values with -1e9 in X
    reshape_std[np.isnan(reshape_std)] = masking_value
    
    # Reshape the data back to the original shape for X
    X_processed = reshape_std.reshape(X.shape[0], X.shape[1], X.shape[2])

    return X_processed, X_train_mean, X_train_std

def preprocess_X_other_data(X, X_train_mean, X_train_std):
    # Reshape the X array
    reshape = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    
    # Fit the StandardScaler to the reshaped X array
    #X_train_mean = np.nanmean(reshape, axis=0)
    #X_train_std = np.nanstd(reshape, axis=0)
    reshape_std = (reshape - X_train_mean) / X_train_std
    
    #reshape_std = StandardScaler().fit_transform(reshape) #do it over the training set for normalizing 

    # Create the mask indicating where NaN values were originally located
    mask_X = np.isnan(X).reshape(X.shape[0], X.shape[1], X.shape[2])
    
    # Replace NaN values with -1e9 in X
    reshape_std[np.isnan(reshape_std)] = masking_value
    
    # Reshape the data back to the original shape for X
    X_processed = reshape_std.reshape(X.shape[0], X.shape[1], X.shape[2])

    return X_processed

def preprocess_y_data(y, dataset_type):
    # Reshape the y array

    if dataset_type == '_all_stat':
        reshape = y.reshape(y.shape[0] * y.shape[1], y.shape[2])

        mask_y = np.isnan(y).reshape(y.shape[0], y.shape[1], y.shape[2])

        # Replace NaN values with -1e9 in X
        reshape[np.isnan(reshape)] = masking_value
        
        # Reshape the y array
        y_processed = reshape.reshape(y.shape[0], y.shape[1], y.shape[2])
    else:

    #reshape = y.reshape(y.shape[0] * y.shape[1], 1)

    # Fit the StandardScaler to the y array
    #reshape_std = StandardScaler().fit_transform(reshape)

    # Create the mask indicating where NaN values were originally located
        mask_y = np.isnan(y).reshape(y.shape[0], y.shape[1])

        # Replace NaN values with -1e9 in X
        y[np.isnan(y)] = masking_value
        
        # Reshape the y array
        y_processed = y.reshape(y.shape[0], y.shape[1], 1)
    
    return y_processed, mask_y

def process_storm_data(y_all_3d, y_all_3d_non_eu, number_of_steps_eu, number_of_steps_non_eu, print_info=False):
    """
    Process storm data for both EU and non-EU datasets.

    Parameters:
    - y_all_3d (ndarray): 3D array for EU data.
    - y_all_3d_non_eu (ndarray): 3D array for non-EU data.
    - number_of_steps_eu (int): The number of steps to retain in the processed DataFrame for EU data.
    - number_of_steps_non_eu (int): The number of steps to retain in the processed DataFrame for non-EU data.
    - print (bool): Whether to print information about the storms being removed or not.

    Returns:
    - eu_results[0 to 3]: Processed DataFrames for EU [0 = y_max, 1 = y_min, 2 = y_mean, 3 = y_std].
    - non_eu_results[0 to 3]: Processed DataFrames for non-EU [0 = y_max, 1 = y_min, 2 = y_mean, 3 = y_std].
    """
    
    def reverse_non_nan(row):
        non_nan_values = row.dropna().values[::-1]
        num_padding = len(row) - len(non_nan_values)
        reversed_padded = np.concatenate([non_nan_values, [np.nan] * num_padding])
        return pd.Series(reversed_padded, index=row.index)
    
    def add_column_first(df, column_name, column_values):
        df[column_name] = column_values
        cols = [column_name] + [col for col in df.columns if col != column_name]
        return df[cols]
    
    def process_single_dataset(y_data, number_of_steps, reverse):
        y_max = pd.DataFrame(y_data[:, :, 0])
        y_min = pd.DataFrame(y_data[:, :, 1])
        y_mean = pd.DataFrame(y_data[:, :, 2])
        y_std = pd.DataFrame(y_data[:, :, 3])

        # Reverse non-NaN values in each row for non-EU data

        if reverse==True:
            y_max = y_max.apply(reverse_non_nan, axis=1)
            y_min = y_min.apply(reverse_non_nan, axis=1)
            y_mean = y_mean.apply(reverse_non_nan, axis=1)
            y_std = y_std.apply(reverse_non_nan, axis=1)
        
        # Add a 'storm_index' column as the first column
        #for df in [y_max, y_min, y_mean, y_std]:
            #df['storm_index'] = df.index + 1
            #df = add_column_first(df, 'storm_index', df.index + 1)
        
        # Add a column with the storm index
        y_max['storm_index'] = y_max.index+1
        y_min['storm_index'] = y_min.index+1
        y_mean['storm_index'] = y_mean.index+1
        y_std['storm_index'] = y_std.index+1

        # Call the function to add a 'storm_index' column
        y_max = add_column_first(y_max, 'storm_index', y_max.index + 1)
        y_min = add_column_first(y_min, 'storm_index', y_min.index + 1)
        y_mean = add_column_first(y_mean, 'storm_index', y_mean.index + 1)
        y_std = add_column_first(y_std, 'storm_index', y_std.index + 1)
        
        # Retain only the specified number of steps (plus storm_index)
        y_max = y_max.iloc[:, 0:number_of_steps + 1]
        y_min = y_min.iloc[:, 0:number_of_steps + 1]
        y_mean = y_mean.iloc[:, 0:number_of_steps + 1]
        y_std = y_std.iloc[:, 0:number_of_steps + 1]
        
        # Remove rows with NaN values
        y_max = y_max.dropna()
        y_min = y_min.dropna()
        y_mean = y_mean.dropna()
        y_std = y_std.dropna()
        
        return y_max, y_min, y_mean, y_std
    
    # Process both datasets
    eu_results = process_single_dataset(y_all_3d, number_of_steps_eu, reverse=False)
    non_eu_results = process_single_dataset(y_all_3d_non_eu, number_of_steps_non_eu, reverse=True)

    if print_info == True:

        # Compare the storms in both datasets
        print(f'EU dataset had {eu_results[0].shape[0]} storms.')
        print(f'Non-EU dataset had {non_eu_results[0].shape[0]} storms.')

        # Remove the storms that are not in both datasets
        eu_storms = eu_results[0]['storm_index'].values
        non_eu_storms = non_eu_results[0]['storm_index'].values
        common_storms = np.intersect1d(eu_storms, non_eu_storms)
        eu_results = [df[df['storm_index'].isin(common_storms)] for df in eu_results]
        non_eu_results = [df[df['storm_index'].isin(common_storms)] for df in non_eu_results]

        # Print the storms that were removed
        removed_storms = np.setdiff1d(eu_storms, common_storms)
        print(f'{len(removed_storms)} storms were removed from the EU dataset.')
        removed_storms = np.setdiff1d(non_eu_storms, common_storms)
        print(f'{len(removed_storms)} storms were removed from the non-EU dataset.')

        print(f'EU dataset is 1st step in EU first, and non-EU dataset landfall step is 1st (meaning the step {number_of_steps_non_eu} hours before landfall is last).')
    
    return eu_results[0], eu_results [1], eu_results[2], eu_results[3], non_eu_results[0], non_eu_results[1], non_eu_results[2], non_eu_results[3]

def tif_to_dataframe(tif_file, date_climatology=None, band=1):
    """
    Converts a .tif file into a Pandas DataFrame with wind speed values, longitude, and latitude.

    Parameters:
    - tif_file (str): Path to the .tif file.
    - band (int): The band number to read from the .tif file (default is 1).
    
    Returns:
    - DataFrame: A Pandas DataFrame with columns ['wind_speed', 'longitude', 'latitude'].
    """
    try:
        # Open the .tif file
        with rasterio.open(tif_file) as dataset:
            # Read the specified band
            data = dataset.read(band)  
            
            # Get the affine transformation
            transform = dataset.transform
            
            # No-data value
            nodata_value = dataset.nodata

        # Create a list to store data
        data_list = []

        # Loop through each pixel
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                # Get the wind speed value
                value = data[row, col]
                
                # Skip if it's a no-data value
                if value == nodata_value or value is None:
                    continue
                
                # Get the coordinates for this pixel
                x, y = rasterio.transform.xy(transform, row, col, offset='center')
                
                # Append the value and coordinates
                data_list.append({f'wind_speed_{date_climatology}': value, 'longitude': x, 'latitude': y})

        # Convert the list into a DataFrame
        df = pd.DataFrame(data_list)
        
        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        #return None