import xarray as xr
import numpy as np
import pandas as pd
import sys
from datetime import datetime
#from guppy import hpy
import os

dataset_path = '/work/FAC/FGSE/IDYST/tbeucler/default/fabien/repos/cleaner_version/data/datasets_1h'
track_path = '/work/FAC/FGSE/IDYST/tbeucler/default/fabien/repos/cleaner_version/pre_processing/tracks/ALL_TRACKS/tracks_1h'

# Créer une instance de heapy
#hp = hpy()
# DOESN'T WORK 10m_v_component_of_wind, 2m_temperature, 10m_u_component_of_wind, 2m_dewpoint_temperature for 2020: too large
# Define a function to open datasets and concatenate them
def open_and_concatenate(year, variable, months, way, level=0):
    try:
        datasets = [xr.open_dataset(f'{way}{variable}/ERA5_{year}-{month}_{variable}.nc') for month in months]
    except:
        datasets = [xr.open_dataset(f'{way}{variable}/ERA5_{year}_{month:02d}_{variable}.nc') for month in months]
        
    if variable == 'geopotential' and level != 0:
        datasets = [dataset.sel(level=level) for dataset in datasets]
    for i, dataset in enumerate(datasets):
        # Rename variables if they differ (e.g., 'ISHF' vs 'sshf')
        if 'ISHF' in dataset.variables:
            datasets[i] = dataset.rename({'ISHF': 'sshf'})
        # Get the list of coordinates in the dataset
        keys = list(dataset.coords.keys())
        
        # Check if the specific coordinates are present
        if 'valid_time' in keys:
            #print(f"Dataset {i}: Renaming 'valid_time' to 'time'")
            
            # Rename 'valid_time' to 'time'
            datasets[i] = dataset.rename({'valid_time': 'time'})
            # Remove expver coordinate
            if 'expver' in keys:
                #print(f"Dataset {i}: Removing 'expver' coordinate")
                datasets[i] = datasets[i].drop_vars('expver')
                datasets[i] = datasets[i].drop_vars('number')
        else:
            #print(f"Dataset {i}: 'valid_time' not found, skipping")
            pass
        dim = 'time'
    return xr.concat(datasets, dim=dim), dim


# Define a function to calculate statistics
def calculate_statistics(data_array):
    return {
        'mean': np.mean(data_array),
        'min': np.min(data_array),
        'max': np.max(data_array),
        'std': np.std(data_array),
    }

# Function to log processing details
def log_processing(variable, year, level, storm_number):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f'Processed variable: {variable}, Year: {year}, Level: {level}, Timestamp: {timestamp}, Storm number:{storm_number}'
    with open(f'{dataset_path}/processing_log.txt', 'a') as log_file:
        log_file.write(log_message + '\n')

# Créer une instance de heapy
#hp = hpy()

# Function to check if all CSV files exist
def all_csv_files_exist(variable, year, level):
    directory = f'{dataset_path}/{variable}'
    if not os.path.exists(directory):
        return False

    for storm_dir in os.listdir(directory):
        storm_path = os.path.join(directory, storm_dir)
        if os.path.isdir(storm_path):
            for stat in ['mean', 'min', 'max', 'std']:
                file_path = os.path.join(storm_path, f'{stat}_{storm_dir.split("_")[1]}_{level}.csv')
                if not os.path.exists(file_path):
                    return False
    return True

# Main function to process data
def process_data(variable, year, level=0):

    year = int(year)
    year_next = year + 1
    month_act = [10, 11, 12]
    month_next = [1, 2, 3]

    if variable == 'geopotential':
        way = '/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ECMWF/ERA5_hourly_PL/'
    elif variable == 'sea_surface_temperature':
        way = '/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ECMWF/ERA5_hourly_PL/SL/'
    else:
        way = '/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ECMWF/ERA5/SL/'

    # Open and concatenate datasets
    if year == 1990:
        dataset_act, dim = open_and_concatenate(str(year), variable, month_next, way, level)
        dataset_next, dim = open_and_concatenate(str(year_next), variable, month_next, way, level)
        dataset = xr.concat([dataset_act, dataset_next], dim=dim)
        dataset = dataset.chunk({dim: 10})
    elif year == 2021:
        dataset, dim = open_and_concatenate(str(year), variable, month_next, way, level)
    else:
        dataset_act, dim = open_and_concatenate(str(year), variable, month_act, way, level)
        dataset_next, dim = open_and_concatenate(str(year_next), variable, month_next, way, level)
        dataset = xr.concat([dataset_act, dataset_next], dim=dim)
        dataset = dataset.chunk({dim: 10})

    # Determine the specific variable to extract
    specific_var = next(var for var in dataset.variables if var not in ['longitude', 'latitude', 'time', 'level','number', 'valid_time', 'expver'])

    # Import all tracks and convert dates
    dates = pd.read_csv(f'/work/FAC/FGSE/IDYST/tbeucler/default/fabien/repos/cleaner_version/pre_processing/tracks/storm_dates.csv', 
                    parse_dates=['start_date', 'end_date'])
    dates['year'] = dates['start_date'].dt.year
    try:
        dates.drop(columns=['Unnamed: 0'], inplace=True)
    except:
        pass

    # Find the indices for storms within the specified timeframe
    if year == 1990:
        index_start_october = dates[(dates['start_date'].dt.month <= 3) & (dates['start_date'].dt.year == year)].index[0]
        index_end_march = dates[(dates['end_date'].dt.month <= 3) & (dates['end_date'].dt.year == year_next)].index[0]
    elif year == 2021:
        index_start_october = dates[(dates['start_date'].dt.month <= 3) & (dates['start_date'].dt.year == year)].index[0]
        index_end_march = dates[(dates['end_date'].dt.year == 2021)].index[0]
    else:
    # Chercher start_october dans year, sinon chercher dès janvier de year_next
        index_start_october = dates[((dates['start_date'].dt.month >= 10) & (dates['start_date'].dt.year == year)) | ((dates['start_date'].dt.year == year_next) & (dates['start_date'].dt.month >= 1))].index[0]
        index_end_march_first = dates[((dates['end_date'].dt.month <= 3) & (dates['end_date'].dt.year == year_next))].index
        #print(index_start_october, index_end_march_first, '3rd condition start_october + index_end_march_first')
        if len(index_end_march_first) > 0:
            index_end_march = index_end_march_first[-1]
            #print(index_end_march, 'index_end_march 1st condition of 2nd condition')
        else:
            # Si year_next ne renvoie rien, chercher la dernière instance de tempête dans year
            index_end_march = dates[((dates['end_date'].dt.year == year) & (dates['end_date'].dt.month <= 12))].index[-1]
            #print(index_end_march, 'index_end_march 2nd condition of 2nd condition')
    # Process each storm
    for i in range(index_start_october, index_end_march + 1):
        storm_number = dates.at[i, 'storm_index']
        track = pd.read_csv(f'{track_path}/storm_{storm_number}.csv')
        start_date = dates.at[i, 'start_date']
        end_date = dates.at[i, 'end_date']
        if dim == 'valid_time':
            storm_data = dataset[specific_var].sel(valid_time=slice(start_date, end_date))
        else:
            storm_data = dataset[specific_var].sel(time=slice(start_date, end_date))

        # Initialize lists to store statistics
        stats = {'mean': [], 'min': [], 'max': [], 'std': []}
        #, 'skewness': [], 'kurtosis': []

        # Process each time step
        if dim == 'valid_time':
            time = storm_data.valid_time
        else:
            time = storm_data.time
        for t_index in range(0, len(time)):#, time_step in enumerate(storm_data.time):
            #data_slice = storm_data.sel(time=time_step).values

            # Extract coordinates for the current time step

            lon_e_temp = track.loc[t_index, 'lon_east']
            lon_w_temp = track.loc[t_index, 'lon_west']
            lat_s_temp = track.loc[t_index, 'lat_south']
            lat_n_temp = track.loc[t_index, 'lat_north']

            lon_test = np.asanyarray(storm_data.longitude[:])
            lat_test = np.asanyarray(storm_data.latitude[:])

            closest_lon_w = np.abs(lon_test - lon_w_temp).argmin()
            closest_lon_e = np.abs(lon_test - lon_e_temp).argmin()
            closest_lat_s = np.abs(lat_test - lat_s_temp).argmin()
            closest_lat_n = np.abs(lat_test - lat_n_temp).argmin()

            closest_lon_w_coor = lon_test[closest_lon_w]
            closest_lon_e_coor = lon_test[closest_lon_e]
            closest_lat_s_coor = lat_test[closest_lat_s]
            closest_lat_n_coor = lat_test[closest_lat_n]

            # Use .roll to handle the 0°/360° boundary
            if closest_lon_w_coor > 100 and closest_lon_e_coor < 100:
                roll_shift = {'longitude': int(round(closest_lon_e_coor, 0)), 'longitude': int(round(closest_lon_w_coor, 0))}
                storm_data_rolled = storm_data.roll(roll_shift, roll_coords=True)
            else:
                storm_data_rolled = storm_data

            # Slice the dataset based on the rolled longitudes and latitudes
            if dim == 'valid_time':
                temp_ds_time = storm_data_rolled.isel(valid_time=t_index)
            else:
                temp_ds_time = storm_data_rolled.isel(time=t_index)
            #[specific_var].isel(time=t_index)
            temp_ds = temp_ds_time.sel(latitude=slice(closest_lat_n_coor, closest_lat_s_coor),
                                        longitude=slice(closest_lon_w_coor, closest_lon_e_coor)).values

            # Calculate statistics for the sliced data
            step_stats = calculate_statistics(temp_ds)
            for key in stats:
                stats[key].append(step_stats[key])

        # Save statistics to CSV files
        for key in stats:
            directory = f'{dataset_path}/{variable}/storm_{storm_number}'
            if not os.path.exists(directory):
                os.makedirs(directory)
            pd.DataFrame(stats[key]).to_csv(f'{directory}/{key}_{storm_number}_{level}.csv')

        # Log the processing details
        log_processing(variable, year, level, storm_number)

if __name__ == '__main__':
    variable = sys.argv[1]
    year = sys.argv[2]
    level = int(sys.argv[3])
    process_data(variable, year, level)

'''if __name__ == '__main__':
    variable = sys.argv[1]
    year = sys.argv[2]
    level = int(sys.argv[3])

    if not all_csv_files_exist(variable, year, level):
        process_data(variable, year, level)
    else:
        print(f'All CSV files for variable: {variable}, year: {year}, and level: {level} already exist. Skipping processing.')'''

# Obtenir un instantané de l'utilisation de la mémoire
#h = hp.heap()

# Imprimer l'information d'utilisation de la mémoire
#print(h)