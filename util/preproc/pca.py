import pandas as pd
import numpy as np
import extraction_squares
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def get_storms_data(data, storm_index):
        return data.loc[data['storm_index'].isin(storm_index)]

def to_pca_components(path, variables, seed_number, threshold=0.98):

    # Get the storm indices from a non-EU dataset
    x_storms = pd.read_csv(f'{path}/data/time_series_1h_non_EU/instantaneous_10m_wind_gust/instantaneous_10m_wind_gust_max.csv')
    index_storm_EU = x_storms['storm_index'].copy()
    # reduce by 1 to match the index of the storm in the EU dataset
    index_storm_EU = index_storm_EU - 1

    # Split the storm indices into training, test and validation based on the storm index
    storm_index_training, storm_index_test, storm_index_validation = extraction_squares.split_storm_numbers(index_storm_EU, 0.15, seed_number)

    # order the index of the storms
    try:
        storm_index_training.sort_values()
        storm_index_test.sort_values()
        storm_index_validation.sort_values()
    except:
        storm_index_training.sort()
        storm_index_test.sort()
        storm_index_validation.sort()

    print("Storm indices for training:", storm_index_training)
    print("Storm indices for test:", storm_index_test)
    print("Storm indices for validation:", storm_index_validation)

    # add 1 to the storm index to match the index of the storm in the timeseries dataset
    storm_index_training = [x+1 for x in storm_index_training]
    storm_index_test = [x+1 for x in storm_index_test]
    storm_index_validation = [x+1 for x in storm_index_validation]

    # Data is stored by statistics
    stats = ['max','min','mean','std']
    # Load the data
    for var in variables['variables']:
        if var == 'sea_surface_temperature':
             print('sea_surface_temperature is not taken into account')
             continue
        for stat in stats:
            var_name = f'{var}_{stat}'
            var_data = pd.read_csv(f'{path}/data/time_series_1h_non_EU/{var}/{var}_{stat}.csv')

            # Select the data for the training set
            var_training = get_storms_data(var_data, storm_index_training)
            # Same for the test set and validation set
            var_test = get_storms_data(var_data, storm_index_test)
            var_validation = get_storms_data(var_data, storm_index_validation)

            # drop unneccessary columns
            var_training = var_training.drop(columns=['storm_index','Unnamed: 0'])
            var_test = var_test.drop(columns=['storm_index','Unnamed: 0'])
            var_validation = var_validation.drop(columns=['storm_index','Unnamed: 0'])

            # standardize the data by row
            #var_training = var_training.T
            scaler = StandardScaler()
            scaler.fit(var_training)
            var_training_scaler = scaler.transform(var_training)

            #var_test = var_test.T
            scaler = StandardScaler()
            scaler.fit(var_test)
            var_test_scaler = scaler.transform(var_test)

            #var_validation = var_validation.T
            scaler = StandardScaler()
            scaler.fit(var_validation)
            var_validation_scaler = scaler.transform(var_validation)

            # Train the PCA model
            pca = PCA()
            pca_test = PCA()
            pca_validation = PCA()
            pca.fit(var_training_scaler)
            pca_test.fit(var_test_scaler)
            pca_validation.fit(var_validation_scaler)

            # get the number of components needed to explain x% of the variance
            explained_variance = pca.explained_variance_ratio_.cumsum()
            explained_variance_test = pca_test.explained_variance_ratio_.cumsum()
            explained_variance_validation = pca_validation.explained_variance_ratio_.cumsum()

            # Find the index of the first value meeting the threshold
            first_above_idx = np.argmax(explained_variance >= threshold)
            first_above_idx_test = np.argmax(explained_variance_test >= threshold)
            first_above_idx_validation = np.argmax(explained_variance_validation >= threshold)

            # Filter values
            n_components = explained_variance[(explained_variance < threshold) | (np.arange(len(explained_variance)) == first_above_idx)]
            n_components = explained_variance[:len(n_components)]

            n_components_test = explained_variance_test[(explained_variance_test < threshold) | (np.arange(len(explained_variance_test)) == first_above_idx_test)]
            n_components_test = explained_variance_test[:len(n_components_test)]

            n_components_validation = explained_variance_validation[(explained_variance_validation < threshold) | (np.arange(len(explained_variance_validation)) == first_above_idx_validation)]
            n_components_validation = explained_variance_validation[:len(n_components_validation)]

            print('We want to keep', threshold*100,'% of the variance, so we need', n_components.shape[0], 'components')

            # Transform the data
            scores = pca.transform(var_training_scaler)
            scores_threshold = scores[:, :n_components.shape[0]]

            scores_test = pca_test.transform(var_test_scaler)
            scores_threshold_test = scores_test[:, :n_components_test.shape[0]]

            scores_validation = pca_validation.transform(var_validation_scaler)
            scores_threshold_validation = scores_validation[:, :n_components_validation.shape[0]]

            # what we keep for the input of ML model is the scores_98
            scores_threshold = pd.DataFrame(scores_threshold)
            scores_threshold.rename(columns=lambda x: 'PCA_'+str(x+1), inplace=True)
            scores_threshold.to_csv(f'{path}/data/PCA_scores/training_{var_name}.csv')

            scores_threshold_test = pd.DataFrame(scores_threshold_test)
            scores_threshold_test.rename(columns=lambda x: 'PCA_'+str(x+1), inplace=True)
            scores_threshold_test.to_csv(f'{path}/data/PCA_scores/test_{var_name}.csv')

            scores_threshold_validation = pd.DataFrame(scores_threshold_validation)
            scores_threshold_validation.rename(columns=lambda x: 'PCA_'+str(x+1), inplace=True)
            scores_threshold_validation.to_csv(f'{path}/data/PCA_scores/validation_{var_name}.csv')

            print(f'PCA scores for {var_name} have been saved')
'''
    # how many components are needed to explain 98% of the variance
    pca.explained_variance_ratio_.cumsum()
    print('Explained variance ratio:', pca.explained_variance_ratio_.cumsum())
    # get the number of components needed to explain 98% of the variance
    explained_variance = pca.explained_variance_ratio_.cumsum()
    # take the first value that is greater than 0.98
    threshold = 0.98
    # Find the index of the first value meeting the threshold
    first_above_idx = np.argmax(explained_variance >= threshold)

    # Filter values
    n_components = explained_variance[(explained_variance < threshold) | (np.arange(len(explained_variance)) == first_above_idx)]
    n_components = explained_variance[:len(n_components)]

    #n_components = explained_variance[explained_variance > 0.98]
    print('We want to keep', threshold*100,'% of the variance, so we need', n_components.shape[0], 'components')
    print('Explained variance ratio shape:', pca.explained_variance_ratio_.shape)
    eigenvalues = pca.explained_variance_
    print('Eigenvalues :', eigenvalues)
    print('Eigenvalues shape:', len(eigenvalues))

    # transform the data
    scores = pca.transform(t2m_dewpoint_training)
    print('Scores shape:', scores.shape)
    scores_98 = scores[:, :n_components.shape[0]]
    print('Scores shape:', scores_98.shape)
    print('The scores are :',scores_98 )

    # get the eigenvectors to reconstruct the data
    eigenvectors_98 = pca.components_[:n_components.shape[0],:]
    mean = pca.mean_

    # reconstruct the data
    reconstructed_data = np.dot(scores_98, eigenvectors_98) + mean

    # what we keep for the input of ML model is the scores_98
    scores_98 = pd.DataFrame(scores_98)
    scores_98.rename(columns=lambda x: 'PCA_'+str(x+1), inplace=True)
    scores_98.to_csv('data/PCA_scores/test_2m_dewpoint_temperature.csv')
    # what we keep for the output of ML model is percentile of the wind speed

    # plot the original data and the reconstructed data
    for j in range(0,10):
        plt.plot(t2m_dewpoint_training.iloc[j,:])
        plt.plot(reconstructed_data[j,:])
        plt.legend(['Original data', 'Reconstructed data'])
        # invert the x-axis
        plt.gca
        plt.gca().invert_xaxis()
        plt.show()
        plt.plot(t2m_dewpoint_training.iloc[0,:])
    plt.plot(reconstructed_data[0,:])
    plt.legend(['Original data', 'Reconstructed data'])
    # invert the x-axis
    plt.gca
    plt.gca().invert_xaxis()
    plt.show()

    print('Length of PCA components:',len(pca.components_))
    print('Shape of PCA components:', pca.components_[:].shape)
    #print(pca.components_[:])

    # get the eigenvectors
    for i in range(0,3):
        plt.plot(pca.components_[i,...])
        plt.show()
'''