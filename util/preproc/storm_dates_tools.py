import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

'''operating_system = 'win'

if operating_system == 'win':
    os.chdir('C:/Users/fabau/OneDrive/Documents/GitHub/master-project-cleaned/')
else:
    os.chdir('/Users/fabianaugsburger/Documents/GitHub/master-project-cleaned/')'''

# go back to scratch, dates are not correct

def process_dates(original_storm_dates, degree, path_tracks, factor):
    # drop all columns except the 1st one 

    transformed_storm_dates = original_storm_dates.drop(original_storm_dates.columns[1:], axis=1)

    # then split the column 'Time&Longitude&Latitude' into 3 columns

    transformed_storm_dates[['Time&Longitude&Latitude','Longitude','Latitude','whateverthatis']] = transformed_storm_dates['Time&Longitude&Latitude'].str.split(' ', n=3, expand=True)
    transformed_storm_dates.rename(columns={'Time&Longitude&Latitude': 'Time'}, inplace=True)

    # extract each line that starts with POINT_NUM (it represents the number of points in the track)
    # Get the number of steps stored in 'Longitude' and 'Latitude' with the line that starts with 'POINT_NUM' in the column 'Time'
    steps_longitude = transformed_storm_dates[transformed_storm_dates['Time'].str.startswith('POINT_NUM')]['Longitude'].to_frame().reset_index(drop=True)
    steps_latitude = transformed_storm_dates[transformed_storm_dates['Time'].str.startswith('POINT_NUM')]['Latitude'].to_frame().reset_index(drop=True)
    # Replace spaces (' ') with actual empty strings ('') in 'Longitude'
    steps_longitude = steps_longitude.replace(' ', '')

    # Use steps stored in Longitude where it's not an empty string, otherwise use the value in Latitude
    steps = pd.DataFrame(np.where(steps_longitude != '', steps_longitude, steps_latitude), columns=['nb_steps'])

    # extract each line that starts with TRACK_ID (it represents the index of the storm)

    index = transformed_storm_dates[transformed_storm_dates['Time'].str.startswith('TRACK_ID')]['Longitude'].to_frame().reset_index(drop=True)
    index.rename(columns={'Longitude': 'storm_index'}, inplace=True)
    # substract 24 from the index to start at 1 and the storm names
    index['storm_index'] = pd.to_numeric(index['storm_index']) - 24

    # extract the names of the storms
    index['name'] = transformed_storm_dates[transformed_storm_dates['Time'].str.startswith('TRACK_ID')]['whateverthatis'].to_frame().reset_index(drop=True)
    index['name'] = index['name'].str.split(' 0 ', n=1).str[1].str.strip()
    index['name'] = index['name'].apply(lambda x: x.replace('NAME', '', 1).strip() if x.startswith('NAME') else x)
    # remove the C3S_STORM_TRACKS_ERA5 for the storm 95
    index_storm_95 = index.index[index['storm_index'] == 95].values[0]
    # Update the 'name' column in place
    index.loc[index_storm_95, 'name'] = index.loc[index_storm_95, 'name'].replace('C3S_STORM_TRACKS_ERA5', '', 1)

    # extract the date of start and end of the storm
    # THIS PART IS NOT THE START DATE FOR WHATEVER REASON
    '''start_date = transformed_storm_dates[transformed_storm_dates['Time'].str.startswith('TRACK_ID')]['whateverthatis'].to_frame().reset_index(drop=True)
    start_date = start_date['whateverthatis'].str.split(' ', n=1).str[0].str.strip().to_frame()
    start_date = start_date.rename(columns={'whateverthatis': 'start_date'})'''

    # find the index of the row where 'Time' is "POINT_NUM"
    start_indexes = transformed_storm_dates.index[transformed_storm_dates['Time'] == 'POINT_NUM'].to_list()
    start_date = []
    for start_index in start_indexes:
        previous_row_values = transformed_storm_dates.iloc[start_index + 1]['Time']
        start_date.append(previous_row_values)


    # same for the end date
    # find the index of the row where 'Time' is "TRACK_ID"
    end_indexes = transformed_storm_dates.index[transformed_storm_dates['Time'] == 'TRACK_ID'].to_list()
    # remove the first value in end_indexes and add the last index of the transformed_storm_dates
    end_indexes.pop(0)

    last_index = transformed_storm_dates.index[0]
    end_indexes.append(last_index)
    # Get the values of the row before "TRACK_ID"
    end_date = []
    for end_index in end_indexes:
            previous_row_values = transformed_storm_dates.iloc[end_index - 1]['Time']
            end_date.append(previous_row_values)

    end_indexes.pop(-1)

    last_index = transformed_storm_dates.index[-1]
    end_indexes.append(last_index)

    # transform the end date and start date into the format 'YYYY-MM-DDTHH:MM:SS'
    end_date_formatted = pd.to_datetime(end_date,format="%Y%m%d%H")
    end_date_formatted = end_date_formatted.strftime('%Y-%m-%dT%H:%M:%S')
    end_date_formatted = pd.DataFrame(end_date_formatted, columns=['end_date'])

    start_date_formatted = pd.to_datetime(start_date,format="%Y%m%d%H")
    start_date_formatted = start_date_formatted.strftime('%Y-%m-%dT%H:%M:%S')
    start_date_formatted = pd.DataFrame(start_date_formatted, columns=['start_date'])

    # create one dataframe with index, name, start_date, end_date and nb_steps
    final_storm_dates = pd.concat([index, start_date_formatted], axis=1)
    final_storm_dates['end_date'] = end_date_formatted
    final_storm_dates = pd.concat([final_storm_dates, steps], axis=1)

    # double check if the nb_steps actually corresponds of steps of 3 hours between start and end date
    test = pd.DataFrame(columns=['start_date'])

    test['start_date'] = pd.to_datetime(final_storm_dates['start_date'])
    test['end_date'] = pd.to_datetime(final_storm_dates['end_date'])
    test['diff'] = test['end_date'] - test['start_date']
    test['diff'] = test['diff'] / np.timedelta64(1, 'h')
    test['diff'] = test['diff'] / 3
    test['diff'] = test['diff'].astype(int)+1
    test['nb_steps'] = final_storm_dates['nb_steps'].astype(int)

    if test['diff'].equals(test['nb_steps']):
        print('The number of steps is correct')
    else:
        print('The number of steps is not correct at storm number: ')
        print(test[test['diff'] != test['nb_steps']])

    # creation of the square with a certain degree of latitude and longitude

    number_of_storms = len(final_storm_dates)

    for storm in range(0,number_of_storms):

        lon_east = []
        lon_west = []
        lat_north = []
        lat_south = []

        if storm == 95 and end_indexes[-1]==last_index:
            end_indexes[storm] = end_indexes[storm] + 1

        for step in range(start_indexes[storm], end_indexes[storm]-1):

            # get the latitude and longitude of the storm
            lat = float(transformed_storm_dates['Latitude'][step+1])
            lon = float(transformed_storm_dates['Longitude'][step+1])

            lon_east_temp = (lon + degree) % 360
            lon_west_temp = (lon - degree + 360 ) % 360
            lat_north_temp = (lat + degree)
            lat_south_temp = (lat - degree)

            lon_east.append(round(lon_east_temp, 6))
            lon_west.append(round(lon_west_temp, 6))
            lat_north.append(round(lat_north_temp, 6))
            lat_south.append(round(lat_south_temp, 6))

        # create the square
        storm_str = str(storm+1)
        square_name = f'storm_{storm_str}'
        square = pd.DataFrame({'lon_east': lon_east, 'lon_west': lon_west, 'lat_north': lat_north, 'lat_south': lat_south})

        # Define the folder and file path
        folder = 'tracks_3h/'
        file_name = f'{square_name}.csv'  # Specify the file name
        full_path = os.path.join(path_tracks, folder)

        # Create the folder if it doesn't exist
        os.makedirs(full_path, exist_ok=True)

        # Define the full path to the file, including the file name
        file_path = os.path.join(full_path, file_name)

        # Save the square in a CSV file in the folder 'tracks_3h'
        # If the file doesn't exist, it will be created
        square.to_csv(file_path, index=False)

    # now the same but the conversion from 0-360 to -180-180

    for storm in range(0,number_of_storms):
        
            lon_east = []
            lon_west = []
            lat_north = []
            lat_south = []
        
            for step in range(start_indexes[storm], end_indexes[storm]-1):
                # get the latitude and longitude of the storm
                lat = float(transformed_storm_dates['Latitude'][step+1])
                lon = float(transformed_storm_dates['Longitude'][step+1])
        
                lon_east_temp = (lon + degree) % 360
                lon_west_temp = (lon - degree + 360 ) % 360
                lat_north_temp = (lat + degree)
                lat_south_temp = (lat - degree)
        
                if lon_east_temp > 180:
                    lon_east_temp = lon_east_temp - 360
                if lon_west_temp > 180:
                    lon_west_temp = lon_west_temp - 360
        
                lon_east.append(round(lon_east_temp, 6))
                lon_west.append(round(lon_west_temp, 6))
                lat_north.append(round(lat_north_temp, 6))
                lat_south.append(round(lat_south_temp, 6))
        
            # create the square
            storm_str = str(storm+1)
            square_name = f'storm_{storm_str}'
            square = pd.DataFrame({'lon_east': lon_east, 'lon_west': lon_west, 'lat_north': lat_north, 'lat_south': lat_south})
        
            # Define the folder and file path
            folder = 'tracks_3h_GIS_friendly/'
            file_name = f'{square_name}.csv'  # Specify the file name
            full_path = os.path.join(path_tracks, folder)
        
            # Create the folder if it doesn't exist
            os.makedirs(full_path, exist_ok=True)
        
            # Define the full path to the file, including the file name
            file_path = os.path.join(full_path, file_name)
        
            # Save the square in a CSV file in the folder 'tracks_3h'
            # If the file doesn't exist, it will be created
            square.to_csv(file_path, index=False)

    # now extrapolate the steps from 3 hours to 1 hour
    
    for storm in range(0,number_of_storms):

        lon_east, lon_west, lat_north, lat_south = pd.read_csv(f'{path_tracks}tracks_3h/storm_{storm+1}.csv').values.T

        # interpolate the values for the latitudes coordinates first

        lat_north_1h = interpolate_vector(lat_north, factor)
        lat_south_1h = interpolate_vector(lat_south, factor)

        # interpolate the values for the longitudes coordinates

        lon_east_1h = np.round(interpolate_longitude(lon_east, factor),6)
        lon_west_1h = np.round(interpolate_longitude(lon_west, factor),6)

        # create the square
        storm_str = str(storm+1)
        square_name = f'storm_{storm_str}'
        square = pd.DataFrame({'lon_east': lon_east_1h, 'lon_west': lon_west_1h, 'lat_north': lat_north_1h, 'lat_south': lat_south_1h})

        # Define the folder and file path
        folder = 'tracks_1h/'
        file_name = f'{square_name}.csv'  # Specify the file name
        full_path = os.path.join(path_tracks, folder)

        # Create the folder if it doesn't exist
        os.makedirs(full_path, exist_ok=True)

        # Define the full path to the file, including the file name
        file_path = os.path.join(full_path, file_name)

        # Save the square in a CSV file in the folder 'tracks_3h'
        # If the file doesn't exist, it will be created
        square.to_csv(file_path, index=False)

    return final_storm_dates


# the function to interpolate the values

def interpolate_vector(data, factor):
    n = len(data)
    # X interpolation points. For factor=4, it is [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, ...]
    x = np.linspace(0, n - 1, (n - 1) * factor + 1)
    xp = np.arange(n)
    # Interpolate
    return np.round(np.interp(x, xp, np.asarray(data)),6)

#Interpolates longitude values with consideration for the 360-degree wrap-around

def interpolate_longitude(lon_list, factor):

    interp_list = []

    for j in range(len(lon_list) - 1):
        current, next_value = lon_list[j], lon_list[j + 1]

        if abs(current - next_value) > 300:
            # Handling wrap-around cases
            if current > next_value:
                delta_before_360 = 360 - current
                delta_after_360 = next_value
                sum_deltas = delta_before_360 + delta_after_360
                delta_factor = sum_deltas / factor
            else:
                delta_before_360 = 360 - next_value
                delta_after_360 = current
                sum_deltas = delta_before_360 + delta_after_360
                delta_factor = -sum_deltas / factor

            interp_list.append(current)
            for k in range(1, 3):
                new_value = current + k * delta_factor
                if new_value > 360:
                    new_value -= 360
                elif new_value < 0:
                    new_value += 360
                interp_list.append(new_value)
        else:
            # Standard interpolation
            delta = (next_value - current) / factor
            interp_list.append(current)
            interp_list.extend([current + k * delta for k in range(1, 3)])

    # Append the last value
    interp_list.append(lon_list[-1])
    return interp_list

def extract_first_step(storm_dates, path_tracks, folder, number_of_storms):
    first_steps = []
    for storm in range(1,number_of_storms+1):
        i = storm
        storm = pd.read_csv(f'{path_tracks}{folder}/storm_{storm}.csv')
        # make the column step croissant
        storm = storm.sort_values(by='step')
        try:
            first_step = storm['step'].iloc[0]
            first_steps.append(first_step)
        except:
            first_step = -1
            first_steps.append(first_step)
            print(f'Storm {i} is empty')
    try:
        storm_dates = pd.read_csv(storm_dates)
    except:
        storm_dates = storm_dates
    storm_dates['first_step_in_eu'] = first_steps

    # this code isn't consistant, because it checks by order of index in the 1st part, and the second it checks by storm index

    landfall_eu = []
    for storm in range(1,len(storm_dates)+1):
        start_date = storm_dates[storm_dates['storm_index'] == storm]['start_date']
        first_step_in_eu = storm_dates[storm_dates['storm_index'] == storm]['first_step_in_eu']

        if first_step_in_eu.values == -1:
            landfall_eu_temp = -1 #landfall_eu.at[storm] = -1
        else:
            landfall_eu_temp = pd.to_datetime(start_date) + first_step_in_eu*np.timedelta64(1, 'h')
            # reconverting the date to the format 'YYYY-MM-DDTHH:MM:SS'
            landfall_eu_temp = landfall_eu_temp.dt.strftime('%Y-%m-%dT%H:%M:%S')
            landfall_eu_temp = landfall_eu_temp.values[0]
        
        landfall_eu.append(landfall_eu_temp)

    storm_dates['landfall_eu'] = landfall_eu
        
    return storm_dates, pd.DataFrame(first_steps), landfall_eu

def process_dates_complete(original_storm_dates, degree, path_tracks, factor):
    
    storm_dates = process_dates(original_storm_dates, degree, path_tracks, factor)
    storm_dates, first_steps, landfall_eu = extract_first_step(storm_dates, path_tracks, 'tracks_1h_EU', len(storm_dates))

    storm_dates['nb_steps'] = storm_dates['nb_steps'].astype(int)
    storm_dates['total_steps_1h'] = storm_dates['nb_steps']*factor-2
    storm_dates['nb_steps_1h_before_landfall'] = storm_dates['first_step_in_eu']-1

    return storm_dates, first_steps, landfall_eu