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
    
class nestedMLR_var_global(BaseEstimator, RegressorMixin):
    def __init__(self, sizes, seed=None, bounds=None):
        self.seed = seed
        np.random.seed(seed)
        # Initialize sizes
        self.sizes = sizes
        self.bounds = bounds
        # Initialize the weights and biases
        self.weights = {
            f'dense{i+1}': np.random.randn(size) for i, size in enumerate(sizes)
        }
        self.biases = {
            f'dense{i+1}': np.random.randn(1) for i in range(len(sizes))
        }
        self.weights['output'] = np.random.randn(len(sizes), 15)
        self.biases['output'] = np.random.randn(15)

    def unpack_params(self, params):
        """
        Unpack the parameters into weights and biases
        """
        idx = 0
        weights = {}
        biases = {}
        for i, size in enumerate(self.sizes):
            layer_name = f'dense{i+1}'
            weights[layer_name] = params[idx:idx+size].reshape(size, 1)
            idx += size
            biases[f'dense{i+1}'] = params[idx:idx+1]
            idx += 1
        weights['output'] = params[idx:idx+len(self.sizes)*15].reshape(len(self.sizes), 15)
        idx += len(self.sizes)*15
        biases['output'] = params[idx:idx+15]
        return weights, biases
    
    def forward(self, X, params):
        """
        Forward pass of the model
        """
        # Extract weights and biases from the params
        weights, biases = self.unpack_params(params)
        # Split input features
        brchindex = np.cumsum([0]+self.sizes)
        Xs = [X[:, brchindex[i]:brchindex[i+1]] for i in range(len(self.sizes))]
        # Forward pass
        dense = [Xs[i] @ weights[f'dense{i+1}'] + biases[f'dense{i+1}'] for i in range(len(self.sizes))]
        # Combine the branches
        bestdense = np.hstack(dense)
        output = bestdense @ weights['output'] + biases['output']
        return output
    
    def loss_fn(self, params, X, y):
        """
        Loss function
        """
        y_pred = self.forward(X, params)
        loss = np.mean((y - y_pred) ** 2)
        return loss
    
    def fit(self, X, y, maxiter):
        """
        Fit the model
        """
        # Flatten initial parameters: weights and biases
        nested_list = [x for pair in zip([self.weights[f'dense{i+1}'].ravel() for i in range(len(self.sizes))],[self.biases[f'dense{i+1}'] for i in range(len(self.sizes))]) for x in pair]
        flattened_list = [item for sublist in nested_list for item in (sublist if isinstance(sublist, list) else [sublist])]
        # Flatten the output weights and biases
        init_params = np.hstack(
            flattened_list + [self.weights['output'].ravel(), self.biases['output']]
        )

        # Define bounds for the parameters (you may need to adjust these based on your problem)
        bounds = [(-self.bounds,self.bounds) for _ in range(len(init_params))]

        # Optimize the parameters using SHGO
        #res = shgo(self.loss_fn, bounds, args=(X, y), options={'disp': True}, sampling_method='simplicial')
        #res = dual_annealing(self.loss_fn, bounds, args=(X, y), maxiter=maxiter, seed=self.seed)
        res = minimize(self.loss_fn, init_params, args=(X, y), method='L-BFGS-B')
        self.opt_params = res.x
        return self
    
    def predict(self, X):
        """
        Predict the output
        """
        return self.forward(X, self.opt_params)
    
class nestedMLR_var_global_l1l2(BaseEstimator, RegressorMixin):
    def __init__(self, sizes, seed=None, bounds=None,reg_type=None, reg_lambda=0.0):
        self.seed = seed
        np.random.seed(seed)
        # Initialize sizes
        self.sizes = sizes
        self.bounds = bounds
        # Initialize weights regularization
        self.reg_type = reg_type  # Regularization type: 'l1' or 'l2'
        self.reg_lambda = reg_lambda  # Regularization strength
        # Initialize the weights and biases
        self.weights = {
            f'dense{i+1}': np.random.randn(size) for i, size in enumerate(sizes)
        }
        self.biases = {
            f'dense{i+1}': np.random.randn(1) for i in range(len(sizes))
        }
        self.weights['output'] = np.random.randn(len(sizes), 15)
        self.biases['output'] = np.random.randn(15)

    def unpack_params(self, params):
        """
        Unpack the parameters into weights and biases
        """
        idx = 0
        weights = {}
        biases = {}
        for i, size in enumerate(self.sizes):
            layer_name = f'dense{i+1}'
            weights[layer_name] = params[idx:idx+size].reshape(size, 1)
            idx += size
            biases[f'dense{i+1}'] = params[idx:idx+1]
            idx += 1
        weights['output'] = params[idx:idx+len(self.sizes)*15].reshape(len(self.sizes), 15)
        idx += len(self.sizes)*15
        biases['output'] = params[idx:idx+15]
        return weights, biases
    
    def forward(self, X, params):
        """
        Forward pass of the model
        """
        # Extract weights and biases from the params
        weights, biases = self.unpack_params(params)
        # Split input features
        brchindex = np.cumsum([0]+self.sizes)
        Xs = [X[:, brchindex[i]:brchindex[i+1]] for i in range(len(self.sizes))]
        # Forward pass
        dense = [Xs[i] @ weights[f'dense{i+1}'] + biases[f'dense{i+1}'] for i in range(len(self.sizes))]
        # Combine the branches
        bestdense = np.hstack(dense)
        output = bestdense @ weights['output'] + biases['output']
        return output
    
    def loss_fn(self, params, X, y):
        """
        Loss function
        """
        #-----------------------------------------------
        # MSE loss term
        #-----------------------------------------------
        y_pred = self.forward(X, params)
        mse_loss = np.mean((y - y_pred) ** 2)
        #-----------------------------------------------
        # Regularization term
        #-----------------------------------------------
        # Extract weights and biases for regularization
        weights, _ = self.unpack_params(params)
        weight_list = [weights[f'dense{i+1}'].ravel() for i in range(len(self.sizes))] + [weights['output'].ravel()]
        
        # Compute the regularization term
        if self.reg_type == 'l2':
            reg_term = self.reg_lambda * sum(np.sum(w ** 2) for w in weight_list)
        elif self.reg_type == 'l1':
            reg_term = self.reg_lambda * sum(np.sum(np.abs(w)) for w in weight_list)
        else:
            reg_term = 0.0  # No regularization
        return mse_loss + reg_term
    
    def fit(self, X, y, maxiter):
        """
        Fit the model
        """
        # Flatten initial parameters: weights and biases
        nested_list = [x for pair in zip([self.weights[f'dense{i+1}'].ravel() for i in range(len(self.sizes))],[self.biases[f'dense{i+1}'] for i in range(len(self.sizes))]) for x in pair]
        flattened_list = [item for sublist in nested_list for item in (sublist if isinstance(sublist, list) else [sublist])]
        # Flatten the output weights and biases
        init_params = np.hstack(
            flattened_list + [self.weights['output'].ravel(), self.biases['output']]
        )

        # Define bounds for the parameters (you may need to adjust these based on your problem)
        bounds = [(-self.bounds,self.bounds) for _ in range(len(init_params))]

        # Optimize the parameters using SHGO
        #res = shgo(self.loss_fn, bounds, args=(X, y), options={'disp': True}, sampling_method='simplicial')
        #res = dual_annealing(self.loss_fn, bounds, args=(X, y), maxiter=maxiter, seed=self.seed)
        # Optimize the parameters
        res = minimize(self.loss_fn, init_params, args=(X, y), method='L-BFGS-B')
        #self.opt_params = res.x
        self.opt_params = res.x
        return self
    
    def predict(self, X):
        """
        Predict the output
        """
        return self.forward(X, self.opt_params)

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import minimize


class VAESklearnDynamic(BaseEstimator, RegressorMixin):
    def __init__(self, sizes, latent_sizes, output_dim, coeff=0.5, seed=None, bounds=None):
        """
        Parameters:
        - sizes: List of input sizes for each branch.
        - latent_sizes: List of latent sizes for each branch.
        - output_dim: Dimension of the regression output.
        - coeff: Weighting factor for reconstruction loss.
        - seed: Random seed for reproducibility.
        """
        self.sizes = sizes
        self.latent_sizes = latent_sizes
        self.output_dim = output_dim
        self.coeff = coeff
        self.seed = seed
        self.bound = bounds
        np.random.seed(seed)

        # Initialize weights and biases dynamically
        self.weights = {
            f'encoder{i+1}_mean': np.random.randn(sizes[i], latent_sizes[i])
            for i in range(len(sizes))
        }
        self.weights.update({
                f'encoder{i+1}_logvar': np.random.randn(sizes[i], latent_sizes[i])
            for i in range(len(sizes))
        })
        self.weights['decoder'] = np.random.randn(sum(latent_sizes), output_dim)

        self.biases = {
            f'encoder{i+1}_mean': np.random.randn(latent_sizes[i])
            for i in range(len(sizes))
        }
        self.biases.update({
            f'encoder{i+1}_logvar': np.random.randn(latent_sizes[i])
            for i in range(len(sizes))
        })
        self.biases['decoder'] = np.random.randn(output_dim)

    def reparameterize(self, mu, log_var):
        std = np.exp(0.5 * log_var)
        eps = np.random.randn(*std.shape)
        return mu + eps * std

    def forward(self, X, params=None):
        if params is not None:
            self.unpack_params(params)

        # Split input into branches
        brchindex = np.cumsum([0] + self.sizes)
        branches = [
            X[:, brchindex[i]:brchindex[i+1]]
            for i in range(len(self.sizes))
        ]

        # Encoder
        z = []
        for i, branch in enumerate(branches):
            mu = branch @ self.weights[f'encoder{i+1}_mean'] + self.biases[f'encoder{i+1}_mean']
            log_var = branch @ self.weights[f'encoder{i+1}_logvar'] + self.biases[f'encoder{i+1}_logvar']
            z.append(self.reparameterize(mu, log_var))

        # Decoder
        z_concat = np.hstack(z)
        output = z_concat @ self.weights['decoder'] + self.biases['decoder']
        return output, z

    def loss_fn(self, params, X, y):
        reconstructed_x, z = self.forward(X, params)

        # Reconstruction loss
        recon_loss = np.sum(np.abs(reconstructed_x - y))

        # KL divergence loss
        kl_loss = 0
        for i, zi in enumerate(z):
            mu = zi @ self.weights[f'encoder{i+1}_mean']
            log_var = zi @ self.weights[f'encoder{i+1}_logvar']
            kl_loss += -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))

        return self.coeff * recon_loss + (1 - self.coeff) * kl_loss

    def fit(self, X, y, maxiter=1000):
        """
        Fit the model using global optimization with dual annealing.
        Parameters:
        - X: Input data of shape (n_samples, input_dim).
        - y: Target data of shape (n_samples, output_dim).
        - maxiter: Maximum number of iterations for the optimizer.
        """
        # Flatten weights and biases into a single parameter array
        init_params = np.hstack([
            self.weights[f'encoder{i+1}_mean'].ravel() for i in range(len(self.sizes))] + 
            [self.weights[f'encoder{i+1}_logvar'].ravel() for i in range(len(self.sizes))] + 
            [self.weights['decoder'].ravel()] + 
            [self.biases[f'encoder{i+1}_mean'] for i in range(len(self.sizes))] + 
            [self.biases[f'encoder{i+1}_logvar'] for i in range(len(self.sizes))] + 
            [self.biases['decoder']]
            )
        # Define bounds for the parameters
        bounds = [(-self.bound, self.bound) for _ in range(len(init_params))]  # Adjust bounds as needed
        
        # Optimize parameters using dual annealing
        res = dual_annealing(self.loss_fn,bounds,args=(X, y),maxiter=maxiter,seed=self.seed)
        
        # Unpack the optimized parameters
        self.unpack_params(res.x)
        self.opt_params = res.x  # Store optimized parameters for reference
        return self

    def predict(self, X):
        reconstructed_x, _ = self.forward(X)
        return reconstructed_x

    def unpack_params(self, params):
        idx = 0

        def extract(shape):
            nonlocal idx
            size = np.prod(shape)
            val = params[idx:idx + size].reshape(shape)
            idx += size
            return val

        # Unpack weights and biases
        for i in range(len(self.sizes)):
            self.weights[f'encoder{i+1}_mean'] = extract((self.sizes[i], self.latent_sizes[i]))
            self.weights[f'encoder{i+1}_logvar'] = extract((self.sizes[i], self.latent_sizes[i]))
        self.weights['decoder'] = extract((sum(self.latent_sizes), self.output_dim))

        for i in range(len(self.sizes)):
            self.biases[f'encoder{i+1}_mean'] = extract((self.latent_sizes[i],))
            self.biases[f'encoder{i+1}_logvar'] = extract((self.latent_sizes[i],))
        self.biases['decoder'] = extract((self.output_dim,))

from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import dual_annealing
import numpy as np

class nestedMLR_VED(BaseEstimator, RegressorMixin):
    def __init__(self, sizes, latent_dims, seed=None, bounds=None, reg_type=None, reg_coeff=0.0):
        """
        VED model with reparameterization trick.
        
        Parameters:
        - sizes: list of int, input feature sizes for each branch.
        - latent_dims: tuple of int, latent dimensions for the two branches.
        - seed: int, random seed for reproducibility.
        - bounds: float, optimization bounds for the parameters.
        """
        self.sizes = sizes
        self.latent_dims = latent_dims
        self.seed = seed
        np.random.seed(seed)
        self.bounds = bounds
        self.reg_type = reg_type
        self.reg_coeff = reg_coeff

    def unpack_params(self, params):
        """
        Unpack parameters into branch encoders and decoders.
        """
        idx = 0
        weights, biases = {}, {}

        # Branch encoder parameters
        for i, size in enumerate(self.sizes):
            layer_name = f'branch{i+1}'
            latent_dim = self.latent_dims[i]
            weights[layer_name] = {
                'mu': params[idx:idx+size*latent_dim].reshape(size, latent_dim),
                'logvar': params[idx+size*latent_dim:idx+2*size*latent_dim].reshape(size, latent_dim),
            }
            biases[layer_name] = {
                'mu': params[idx+2*size*latent_dim:idx+2*size*latent_dim+latent_dim],
                'logvar': params[idx+2*size*latent_dim+latent_dim:idx+2*size*latent_dim+2*latent_dim],
            }
            idx += 2*size*latent_dim + 2*latent_dim

        # Decoder parameters
        weights['decoder'] = params[idx:idx+sum(self.latent_dims)*15].reshape(sum(self.latent_dims), 15)
        biases['decoder'] = params[idx+sum(self.latent_dims)*15:idx+sum(self.latent_dims)*15+15]
        return weights, biases

    def forward(self, X, params):
        """
        Forward pass with reparameterization trick.
        """
        weights, biases = self.unpack_params(params)
        branch_outputs, mus, logvars = [], [], []

        # Process each branch
        branch_indices = np.cumsum([0] + self.sizes)
        for i in range(len(self.sizes)):
            branch_input = X[:, branch_indices[i]:branch_indices[i+1]]
            mu = branch_input @ weights[f'branch{i+1}']['mu'] + biases[f'branch{i+1}']['mu']
            logvar = branch_input @ weights[f'branch{i+1}']['logvar'] + biases[f'branch{i+1}']['logvar']
            std = np.exp(0.5 * logvar)
            eps = np.random.randn(*mu.shape)
            z = mu + eps * std
            branch_outputs.append(z)
            mus.append(mu)
            logvars.append(logvar)

        # Combine branch outputs
        combined = np.hstack(branch_outputs)

        # Decoder
        output = combined @ weights['decoder'] + biases['decoder']
        return output, mus, logvars
    
    def regularization_loss(self, params):
        """
        Compute L1 or L2 regularization loss.
        """
        if self.reg_type is None or self.reg_coeff == 0.0:
            return 0.0

        if self.reg_type == 'L1':
            return self.reg_coeff * np.sum(np.abs(params))
        elif self.reg_type == 'L2':
            return self.reg_coeff * np.sum(params**2)
        else:
            raise ValueError("reg_type must be either 'L1' or 'L2'.")

    def loss_fn(self, params, X, y, coeff=0.5):
        """
        Loss function: reconstruction loss + KL divergence.
        """
        y_pred, mus, logvars = self.forward(X, params)
        recon_loss = np.mean(np.abs(y - y_pred))  # Reconstruction loss
        kl_loss = sum(
            -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))
            for mu, logvar in zip(mus, logvars)
        )  # KL divergence
        reg_loss = self.regularization_loss(params)
        return coeff * recon_loss + (1 - coeff) * kl_loss + reg_loss

    def fit(self, X, y, maxiter=1000, coeff=0.5):
        """
        Fit the model using dual annealing for global optimization.
        """
        # Initialize random parameters for branches and decoder
        total_params = sum(
            2 * size * dim + 2 * dim
            for size, dim in zip(self.sizes, self.latent_dims)
        ) + sum(self.latent_dims) * 15 + 15
        init_params = np.random.randn(total_params)

        # Define bounds
        bounds = [(-self.bounds, self.bounds) for _ in range(total_params)]

        # Optimize using dual annealing
        res = dual_annealing(
            self.loss_fn,
            bounds,
            args=(X, y, coeff),
            maxiter=maxiter,
            seed=self.seed,
        )
        self.opt_params = res.x
        return self

    def predict(self, X):
        """
        Predict the output.
        """
        return self.forward(X, self.opt_params)[0]

from scipy.fftpack import fft, ifft, fftshift, ifftshift