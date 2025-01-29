import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import os, glob
import sys
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
from scipy.optimize import shgo,dual_annealing, differential_evolution, basinhopping
from scipy.fftpack import fft, ifft, fftshift, ifftshift

class FNO1DWithShift(BaseEstimator, RegressorMixin):
    def __init__(self, n_features, modes, width, seed=None, bounds=None):
        # Initialization (similar to before)
        self.n_features = n_features
        self.modes = modes  # Number of modes to retain
        self.width = width  # Width of the Fourier layer
        self.seed = seed
        self.bounds = bounds
        np.random.seed(seed)
        
        # Initialize Fourier weights and biases
        self.fourier_weights = [np.random.randn(modes, width) for _ in range(n_features)]
        self.fourier_biases = [np.random.randn(width) for _ in range(n_features)]
        
        # Initialize final linear weights and biases
        self.linear_weights = np.random.randn(n_features * width, 15)
        self.linear_biases = np.random.randn(15)

    def unpack_params(self, params):
        # Unpack parameters (same as before)
        idx = 0
        fourier_weights = []
        fourier_biases = []
        
        for _ in range(self.n_features):
            weights = params[idx:idx + self.modes * self.width].reshape(self.modes, self.width)
            idx += self.modes * self.width
            biases = params[idx:idx + self.width]
            idx += self.width
            fourier_weights.append(weights)
            fourier_biases.append(biases)
        
        linear_weights = params[idx:idx + self.n_features * self.width * 15].reshape(self.n_features * self.width, 15)
        idx += self.n_features * self.width * 15
        linear_biases = params[idx:idx + 15]
        
        return fourier_weights, fourier_biases, linear_weights, linear_biases

    def forward(self, X, params):
        # Unpack parameters
        fourier_weights, fourier_biases, linear_weights, linear_biases = self.unpack_params(params)
        
        smoothed_features = []
        for i in range(self.n_features):
            X_feature = X[i]  # Extract the i-th feature (shape: sample)
            X_feature_fft = fftshift(fft(X_feature))  # FFT and shift (shape: sample)
        
            # Apply filtering: Retain only the desired modes
            filtered_fft = np.zeros_like(X_feature_fft, dtype=complex)
            mid = len(X_feature_fft) // 2  # Center index
            filtered_fft[mid - self.modes // 2: mid + self.modes // 2] = X_feature_fft[mid - self.modes // 2: mid + self.modes // 2]
        
            # Transform filtered FFT with learned weights and biases
            filtered_transformed = np.real(filtered_fft[mid - self.modes // 2: mid + self.modes // 2] @ fourier_weights[i]) + fourier_biases[i]
        
            # Inverse FFT
            X_feature_smoothed = ifft(ifftshift(filtered_transformed)).real  # IFFT with inverse shift (shape: sample, width)
            smoothed_features.append(X_feature_smoothed)
    
        # Combine smoothed features and pass through the linear layer
        smoothed_features = np.hstack(smoothed_features)  # (sample, n_features * width)
        output = smoothed_features @ linear_weights + linear_biases  # (sample, 15)
        return output

    def loss_fn(self, params, X, y):
        # Loss function (same as before)
        y_pred = self.forward(X, params)
        return np.mean((y - y_pred) ** 2)

    def fit(self, X, y, maxiter):
        # Fit method (same as before)
        init_params = np.hstack(
            [w.ravel() for w in self.fourier_weights] +
            [b.ravel() for b in self.fourier_biases] +
            [self.linear_weights.ravel(), self.linear_biases.ravel()]
        )
        bounds = [(-self.bounds, self.bounds) for _ in range(len(init_params))]
        #res = dual_annealing(self.loss_fn, bounds, args=(X, y), maxiter=maxiter, seed=self.seed)
        res = minimize(self.loss_fn, init_params, args=(X, y), method='L-BFGS-B')
        self.opt_params = res.x
        return self

    def predict(self, X):
        # Predict method (same as before)
        return self.forward(X, self.opt_params)

    def smooth_eigenvectors(self, eigenvectors):
        """
        Smooth eigenvectors using fftshift and the learned Fourier weights.

        Args:
        - eigenvectors: 2D array of shape (n_eigenvectors, n_features)

        Returns:
        - smoothed_eigenvectors: 2D array of smoothed eigenvectors
        """
        # Ensure the model is trained
        if not hasattr(self, 'opt_params'):
            raise ValueError("The model has not been trained yet. Call `fit` first.")

        # Extract learned parameters
        fourier_weights, fourier_biases, _, _ = self.unpack_params(self.opt_params)
        
        smoothed_eigenvectors = []
        
        for i in range(self.n_features):
            eigvec_feature = eigenvectors[:, i]  # Extract the i-th eigenvector
            eigvec_fft = fftshift(fft(eigvec_feature))  # FFT and shift
            
            # Apply filtering
            filtered_fft = np.zeros_like(eigvec_fft)
            mid = len(eigvec_fft) // 2  # Center index
            filtered_fft[mid - self.modes // 2: mid + self.modes // 2] = eigvec_fft[mid - self.modes // 2: mid + self.modes // 2]
            
            # Transform with learned weights and biases
            eigvec_transformed = np.real(filtered_fft[:self.modes] @ fourier_weights[i]) + fourier_biases[i]
            eigvec_smoothed = ifft(ifftshift(eigvec_transformed)).real  # IFFT with inverse shift
            smoothed_eigenvectors.append(eigvec_smoothed)
        
        return np.column_stack(smoothed_eigenvectors)