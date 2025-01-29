import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_SFFS(remaining_vars, pcs_train, pcs_val, y_train, y_val, target_cat, best_val_rmse):
    
    # Track RMSEs
    rmse_log = []  # List to log RMSE values for each iteration
    selected_pcs = []  # List to store selected PCs
    
    while remaining_vars:  
        best_var = None
        best_pc_index = None
        best_mean_val_rmse = best_val_rmse
        best_mean_train_rmse = None
        
        # Test each variable
        for var in remaining_vars:
            nPC = pcs_train[0][var].shape[1]  # Number of PCs for this variable
            
            # Test each PC of the variable
            for pc_index in range(nPC):
                train_scores = []
                mean_score = []
                
                # Evaluate using all seeds
                for iseed in range(7):
                    # Prepare data for the current candidate PC
                    candidate_features = [pcs_train[iseed][sel_var][:, [pc_idx]] for sel_var, pc_idx in selected_pcs]
                    candidate_features.append(pcs_train[iseed][var][:, [pc_index]])
                    X_train_subset = np.hstack(candidate_features)
                    X_val_subset = np.hstack([pcs_val[iseed][sel_var][:, [pc_idx]] for sel_var, pc_idx in selected_pcs] +[pcs_val[iseed][var][:, [pc_index]]])
                
                    # Train a model and evaluate
                    model = linear_model.LinearRegression()
                    model.fit(X_train_subset, y_train[iseed][target_cat])
                    y_pred = model.predict(X_val_subset)
                    
                    # Training RMSE
                    y_train_pred = model.predict(X_train_subset)
                    train_rmse = mean_squared_error(y_train[iseed][target_cat], y_train_pred, squared=False)
                    #train_rmse = -r2_score_f(y_train[iseed][target_cat], y_train_pred)
                    train_scores.append(train_rmse)
                    
                    # Calculate the validation RMSE
                    val_rmse = mean_squared_error(y_val[iseed][target_cat], y_pred, squared=False)
                    #val_rmse = -r2_score_f(y_val[iseed][target_cat], y_pred)
                    mean_score.append(val_rmse)
                
                # Compute the mean training RMSE across seeds
                mean_train_rmse = np.mean(train_scores)
                # Compute the mean validation RMSE across seeds
                mean_val_rmse = np.mean(mean_score)

                # Update the best PC if this one performs better
                if mean_val_rmse < best_mean_val_rmse:
                    best_mean_val_rmse = mean_val_rmse
                    best_mean_train_rmse = mean_train_rmse
                    best_var = var
                    best_pc_index = pc_index

        # Check if we found a PC that improves validation RMSE
        if best_var and best_mean_val_rmse < best_val_rmse:
            # Add the best-performing PC to the selected set
            selected_pcs.append((best_var, best_pc_index))
            remaining_vars.remove(best_var)
            best_val_rmse = best_mean_val_rmse
            
            # Log RMSEs for this iteration
            rmse_log.append({
                "selected_pc": f"{best_var}_PC{best_pc_index + 1}",
                "train_rmse": best_mean_train_rmse,
                "val_rmse": best_mean_val_rmse,
                })
            print(f"Selected PC: {best_var}_PC{best_pc_index + 1}, Train RMSE: {best_mean_train_rmse}, Val RMSE: {best_mean_val_rmse}")
        else:
            print("No improvement. Stopping feature selection.")
            break
    
    # Train the final model using all selected PCs and all training data
    final_X_train = np.hstack(
        [pcs_train[0][sel_var][:, [pc_idx]] for sel_var, pc_idx in selected_pcs]
        )
    final_model = linear_model.LinearRegression()
    final_model.fit(final_X_train, y_train[0][target_cat])

    return final_model, selected_pcs, rmse_log