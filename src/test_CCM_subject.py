# Author @Tenzin Sangpo Choedon

import logging
import os
from pathlib import Path
import random
from typing import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from scipy.interpolate import make_interp_spline
from tqdm import tqdm # for showing progress bar in for loops
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import mean_squared_error
import statistics
### Just to remove warnings to prettify the notebook. 
import warnings
warnings.filterwarnings("ignore")
# import jdc
import time
import multiprocessing as mp
from statsmodels.tsa.stattools import acf
import pyEDM

######## START CCM CODE #########

# Computing "Causality" (Correlation between True and Predictions)
class ccm:
    def __init__(self, X, Y, tau=1, E=2, L=5000):
        '''
        X: timeseries for variable X that could cause Y
        Y: timeseries for variable Y that could be caused by X
        tau: time lag
        E: shadow manifold embedding dimension
        L: time period/duration to consider (longer = more data)
        We're checking for X -> Y
        '''
        self.X = X
        self.Y = Y
        self.tau = tau
        self.E = E
        self.L = L        
        self.My = self.shadow_manifold(Y) # shadow manifold for Y (we want to know if info from X is in Y)
        self.t_steps, self.dists = self.get_distances(self.My) # for distances between points in manifold    

# %%add_to ccm
    def shadow_manifold(self, X):
        """
        Given
            X: some time series vector
            tau: lag step
            E: shadow manifold embedding dimension
            L: max time step to consider - 1 (starts from 0)
        Returns
            {t:[t, t-tau, t-2*tau ... t-(E-1)*tau]} = Shadow attractor manifold, dictionary of vectors
        """
        X = X[:L] # make sure we cut at L
        M = {t:[] for t in range((self.E-1) * self.tau, self.L)} # shadow manifold
        for t in range((self.E-1) * self.tau, self.L):
            x_lag = [] # lagged values
            for t2 in range(0, self.E-1 + 1): # get lags, we add 1 to E-1 because we want to include E
                x_lag.append(X[t-t2*self.tau])            
            M[t] = x_lag
        return M
    
    # get pairwise distances between vectors in X
    def get_distances(self, Mx):
        """
        Args
            Mx: The shadow manifold from X
        Returns
            t_steps: timesteps
            dists: n x n matrix showing distances of each vector at t_step (rows) from other vectors (columns)
        """

        # we extract the time indices and vectors from the manifold Mx
        # we just want to be safe and convert the dictionary to a tuple (time, vector)
        # to preserve the time inds when we separate them
        t_vec = [(k, v) for k,v in Mx.items()]
        t_steps = np.array([i[0] for i in t_vec])
        vecs = np.array([i[1] for i in t_vec])
        dists = distance.cdist(vecs, vecs)    
        return t_steps, dists

    # %%add_to ccm
    def get_nearest_distances(self, t, t_steps, dists):
        """
        Args:
            t: timestep of vector whose nearest neighbors we want to compute
            t_teps: time steps of all vectors in Mx, output of get_distances()
            dists: distance matrix showing distance of each vector (row) from other vectors (columns). output of get_distances()
            E: embedding dimension of shadow manifold Mx 
        Returns:
            nearest_timesteps: array of timesteps of E+1 vectors that are nearest to vector at time t
            nearest_distances: array of distances corresponding to vectors closest to vector at time t
        """
        t_ind = np.where(t_steps == t) # get the index of time t
        dist_t = dists[t_ind].squeeze() # distances from vector at time t (this is one row)
        
        # get top closest vectors
        nearest_inds = np.argsort(dist_t)[1:self.E+1 + 1] # get indices sorted, we exclude 0 which is distance from itself
        nearest_timesteps = t_steps[nearest_inds] # index column-wise, t_steps are same column and row-wise 
        nearest_distances = dist_t[nearest_inds]  
        
        return nearest_timesteps, nearest_distances

    # %%add_to ccm
    def predict(self, t):
        """
        Args
            t: timestep at Mx to predict Y at same time step
        Returns
            Y_true: the true value of Y at time t
            Y_hat: the predicted value of Y at time t using Mx
        """
        eps = 0.000001 # epsilon minimum distance possible
        t_ind = np.where(self.t_steps == t) # get the index of time t
        dist_t = self.dists[t_ind].squeeze() # distances from vector at time t (this is one row)    
        nearest_timesteps, nearest_distances = self.get_nearest_distances(t, self.t_steps, self.dists)    
        
        # get weights
        u = np.exp(-nearest_distances/np.max([eps, nearest_distances[0]])) # we divide by the closest distance to scale
        w = u / np.sum(u)
        
        # get prediction of X
        X_true = self.X[t] # get corresponding true X
        X_cor = np.array(self.X)[nearest_timesteps] # get corresponding Y to cluster in Mx
        X_hat = (w * X_cor).sum() # get X_hat
        
        return X_true, X_hat

    # %%add_to ccm
    def causality(self):
        '''
        Args:
            None
        Returns:
            correl: how much self.X causes self.Y. correlation between predicted Y and true Y
        '''

        # run over all timesteps in M
        # X causes Y, we can predict X using My
        # X puts some info into Y that we can use to reverse engineer X from Y        
        X_true_list = []
        X_hat_list = []

        for t in list(self.My.keys()): # for each time step in My
            X_true, X_hat = self.predict(t) # predict X from My
            X_true_list.append(X_true)
            X_hat_list.append(X_hat) 

        x, y = X_true_list, X_hat_list
        r, p = pearsonr(x, y)        

        return r, p


    # %%add_to ccm
    def visualize_cross_mapping(self):
        """
        Visualize the shadow manifolds and some cross mappings
        """
        # we want to check cross mapping from Mx to My and My to Mx

        f, axs = plt.subplots(1, 2, figsize=(12, 6))        
        
        for i, ax in zip((0, 1), axs): # i will be used in switching Mx and My in Cross Mapping visualization
            #===============================================
            # Shadow Manifolds Visualization

            X_lag, Y_lag = [], []
            for t in range(1, len(self.X)):
                X_lag.append(X[t-tau])
                Y_lag.append(Y[t-tau])    
            X_t, Y_t = self.X[1:], self.Y[1:] # remove first value

            ax.scatter(X_t, X_lag, s=5, label='$M_x$')
            ax.scatter(Y_t, Y_lag, s=5, label='$M_y$', c='y')

            #===============================================
            # Cross Mapping Visualization

            A, B = [(self.Y, self.X), (self.X, self.Y)][i]
            cm_direction = ['Mx to My', 'My to Mx'][i]

            Ma = self.shadow_manifold(A)
            Mb = self.shadow_manifold(B)

            t_steps_A, dists_A = self.get_distances(Ma) # for distances between points in manifold
            t_steps_B, dists_B = self.get_distances(Mb) # for distances between points in manifold

            # Plot cross mapping for different time steps
            timesteps = list(Ma.keys())
            for t in np.random.choice(timesteps, size=3, replace=False):
                Ma_t = Ma[t]
                near_t_A, near_d_A = self.get_nearest_distances(t, t_steps_A, dists_A)

                for i in range(E+1):
                    # points on Ma
                    A_t = Ma[near_t_A[i]][0]
                    A_lag = Ma[near_t_A[i]][1]
                    ax.scatter(A_t, A_lag, c='b', marker='s')

                    # corresponding points on Mb
                    B_t = Mb[near_t_A[i]][0]
                    B_lag = Mb[near_t_A[i]][1]
                    ax.scatter(B_t, B_lag, c='r', marker='*', s=50)  

                    # connections
                    ax.plot([A_t, B_t], [A_lag, B_lag], c='r', linestyle=':') 

            ax.set_title(f'{cm_direction} cross mapping. time lag, tau = {tau}, E = 2')
            ax.legend(prop={'size': 14})

            ax.set_xlabel('$X_t$, $Y_t$', size=15)
            ax.set_ylabel('$X_{t-1}$, $Y_{t-1}$', size=15)               
        plt.show()       


    # %%add_to ccm
    def plot_ccm_correls(self):
        """
        Args
            X: X time series
            Y: Y time series
            tau: time lag
            E: shadow manifold embedding dimension
            L: time duration
        Returns
            None. Just correlation plots
        """
        M = self.shadow_manifold(self.Y) # shadow manifold
        t_steps, dists = self.get_distances(M) # for distances

        ccm_XY = ccm(X, Y, tau, E, L) # define new ccm object # Testing for X -> Y
        ccm_YX = ccm(Y, X, tau, E, L) # define new ccm object # Testing for Y -> X

        X_My_true, X_My_pred = [], [] # note pred X | My is equivalent to figuring out if X -> Y
        Y_Mx_true, Y_Mx_pred = [], [] # note pred Y | Mx is equivalent to figuring out if Y -> X

        for t in range(tau, L):
            true, pred = ccm_XY.predict(t)
            X_My_true.append(true)
            X_My_pred.append(pred)    

            true, pred = ccm_YX.predict(t)
            Y_Mx_true.append(true)
            Y_Mx_pred.append(pred)        

        # # plot
        figs, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        # predicting X from My
        r, p = np.round(pearsonr(X_My_true, X_My_pred), 4)
        
        axs[0].scatter(X_My_true, X_My_pred, s=10)
        axs[0].set_xlabel('$X(t)$ (observed)', size=12)
        axs[0].set_ylabel('$\hat{X}(t)|M_y$ (estimated)', size=12)
        axs[0].set_title(f'tau={tau}, E={E}, L={L}, Correlation coeff = {r}')

        # predicting Y from Mx
        r, p = np.round(pearsonr(Y_Mx_true, Y_Mx_pred), 4)
        
        axs[1].scatter(Y_Mx_true, Y_Mx_pred, s=10)
        axs[1].set_xlabel('$Y(t)$ (observed)', size=12)
        axs[1].set_ylabel('$\hat{Y}(t)|M_x$ (estimated)', size=12)
        axs[1].set_title(f'tau={tau}, E={E}, L={L}, Correlation coeff = {r}')
        plt.show()

######## END CCM CODE #########

# Test convergence for a single channel pair
def plot_convergence(Title, filename, L_range, Es, taus, Xs, Ys):
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    
    states=["a) Non-seizure", "b) Pre-ictal", "c) Ictal"]
    for idx, s in enumerate(states):
        X = Xs[idx]
        Y = Ys[idx]
        Xhat_My, Yhat_Mx = [], [] # correlation list
        for L in L_range: 
            ccm_XY = ccm(X, Y, taus[idx], Es[idx], L) # define new ccm object # Testing for X -> Y
            ccm_YX = ccm(Y, X, taus[idx], Es[idx], L) # define new ccm object # Testing for Y -> X    
            Xhat_My.append(ccm_XY.causality()[0]) 
            Yhat_Mx.append(ccm_YX.causality()[0]) 
    
        axs[idx].plot(L_range, Xhat_My, label='$\hat{X}(t)|M_y$')
        axs[idx].plot(L_range, Yhat_Mx, label='$\hat{Y}(t)|M_x$')
        axs[idx].set_title(f"{s} state E{Es[idx]}_tau{taus[idx]}")
        axs[idx].set_xlabel('L', size=12)
        axs[idx].set_ylabel('correl', size=12)
        axs[idx].legend()
    # plot convergence as L->inf. Convergence is necessary to conclude causality
    fig.suptitle(Title)
    plt.tight_layout()
    plt.savefig(filename+".png")
    plt.close()

# Find the overall optimal L through convergence analysis
def plot_overall_convergence(output_filename, L_range, E_range, tau_range):
    print("### Convergence Analysis ###")
    
    # Load all causality matrices with different parameter sets
    X = np.load(output_filename+'.npz')
    plt.figure(figsize=(14,10))
    
    # Style X->Y and Y->X differently to preserve directionality
    style_X_Y = dict(linestyle='-',   marker='o', markersize=4, linewidth=1.5)
    style_Y_X = dict(linestyle='--',  marker='s', markersize=4, linewidth=1.5)
            
    for E in E_range:
        for tau in tau_range:            
            # Correlation list
            Xhat_My, Yhat_Mx = [], []
            
            # Compute mean causality score of matrix across L values
            for L in L_range:
                data = X[f'L{L}_E{E}_tau{tau}']
                corrX_Y = np.mean(data[np.triu_indices_from(data, k=1)])
                corrY_X  = np.mean(data[np.tril_indices_from(data, k=-1)])
                Xhat_My.append(corrX_Y)
                Yhat_Mx.append(corrY_X)
            
            # Plot the mean causality values across L values
            plt.plot(L_range, Xhat_My, label='$\hat{X}(t)|M_y$'+f'_E{E}_tau{tau}', **style_X_Y)
            plt.plot(L_range, Yhat_Mx, label='$\hat{Y}(t)|M_x$'+f'_E{E}_tau{tau}', **style_Y_X)
            print('plot done for E={E}, tau={tau}')
    
    # Plot the complete figure for causality score across library length L
    plt.title('Convergence analysis of causality values')
    plt.xlabel('Library length (L)', size=12)
    plt.ylabel('Causality values', size=12)
    plt.legend(prop={'size': 6}, loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=2)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_filename+f'_overall-convergence.png', bbox_inches='tight')
    plt.close()

# Find the tau for the first zero crossing in autocorrelation function
def first_zero_crossing(auto_corr_vals, tau_range):
    
    # Locate among values
    for i in range(1, len(auto_corr_vals)):
        isPrevPos = auto_corr_vals[i-1] > 0
        isAfterNeg = auto_corr_vals[i] <= 0
        
        # Discover first zero crossing
        if isPrevPos and isAfterNeg:
            return tau_range[i]
    
    # Discover no zero crossing
    return None

# Find the overall optimal tau through autocorrelation
def plot_autocorrelation(output_filename, L, E_range, tau_range):
    print("### Autocorrelation ###")
    
    # Load all causality matrices with different parameter sets
    X = np.load(output_filename+'.npz')
    plt.figure(figsize=(9,5))
    
    # Style X->Y and Y->X differently to preserve directionality
    style_X_Y = dict(linestyle='-',   marker='o', markersize=4, linewidth=1.5)
    style_Y_X = dict(linestyle='--',  marker='s', markersize=4, linewidth=1.5)
    
    # All potential optimal tau values from autocorrelation plots
    opt_taus=[]
    
    for E in E_range:
        # Correlation list
        Xhat_My, Yhat_Mx = [], []
        
        # Compute mean causality score of matrix across tau values
        for tau in tau_range:                
            data = X[f'L{L}_E{E}_tau{tau}']
            corrX_Y = np.mean(data[np.triu_indices_from(data, k=1)])
            corrY_X  = np.mean(data[np.tril_indices_from(data, k=-1)])
            Xhat_My.append(corrX_Y)
            Yhat_Mx.append(corrY_X)
        
        # Compute autocorrelation values for different time lag (tau)
        auto_corrX_Y = acf(Xhat_My, nlags=len(Xhat_My)-1)
        auto_corrY_X = acf(Yhat_Mx, nlags=len(Yhat_Mx)-1)
        
        # Plot the auto correlation values across tau values
        plt.plot(tau_range, auto_corrX_Y, label='$\hat{X}(t)|M_y$'+f'_L{L}_E{E}', **style_X_Y)
        plt.plot(tau_range, auto_corrY_X, label='$\hat{Y}(t)|M_x$'+f'_L{L}_E{E}', **style_Y_X)
        print(f'plot done for E={E}')
        
        # Find optimal tau through first zero crossing
        opt_tauX_Y = first_zero_crossing(auto_corrX_Y, tau_range)
        opt_tauY_X = first_zero_crossing(auto_corrY_X, tau_range)
        opt_taus.append(opt_tauX_Y)
        opt_taus.append(opt_tauY_X)
        print(f"Optimal taus: X->Y {opt_tauX_Y} and Y->X {opt_tauY_X}")
    
    # Select overall tau as the most frequently occurring tau value during zero crossing
    counter = Counter(opt_taus)
    overall_tau = counter.most_common(1)[0][0]
    print(f"Overall optimal tau: {overall_tau}")
    plt.figtext(0.1, 0.01, f"Overall optimal tau: {overall_tau}", ha='left', va='bottom', fontsize=8)
    
    # Plot the complete figure for autocorrelation
    plt.title('Autocorrelation of causality values')
    plt.xticks(tau_range)
    plt.xlabel('Time lag (tau)')
    plt.ylabel('Autocorrelation')
    plt.legend(prop={'size': 16}, loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(output_filename+f'_autocorrelation.png', bbox_inches='tight')
    plt.close()
    
    return overall_tau

# Find the overall optimal E through simplex projection
def plot_simplex(output_filename, limit_channels, L, tau, E_range, X):
    print("### Simplex Projection ###")
    
    plt.figure(figsize=(9,5))
    
    # All potential optimal E values from simplex projection
    opt_Es = []
    
    # Perform simplex projection for each channel
    for idx, i in enumerate(limit_channels):
        
        # The data points of ith channel
        df = pd.DataFrame({
            # "Time": np.concatenate((np.arange(start_index, end_index),np.concatenate((np.arange(0, start_index), np.arange(end_index, X.shape[1]))))),
            # f"Ch{i}": np.concatenate((X[i, start_index:end_index], np.concatenate((X[i, :start_index], X[i, end_index:]))))
            "Time": np.arange(end_index-start_index),
            f"Ch{i}": X[i, start_index:end_index]
        })
        
        # All correlation values from simplex projection
        simplex_vals = []
        
        for E in E_range:
            
            # Compute Simplex projection
            res = pyEDM.Simplex(
                dataFrame=df,
                lib=f'1 {L}',                                         # Train embedding with L data points
                pred=f'{L+1} {L*2 if L*2<=end_index-start_index else end_index-start_index}', # Test predictions with remaining data points OLD: if L*2<=end_index-start_index else end_index-start_index
                tau=tau,                                              # Fixed tau
                E=E,                                                  
                Tp=1,                                                 # Prediction Horizon
                columns=f'Ch{i}',                                     # Data points for library
                target=f'Ch{i}',                                      # Data points for prediction
                showPlot=False
            )
            
            # Create a mask for finite numbers to ensure infs and NaNs are filtered out
            o_val = np.array(res['Observations'])
            p_val = np.array(res['Predictions'])
            mask = np.isfinite(p_val) & np.isfinite(o_val)
            masked_o = o_val[mask]
            masked_p = p_val[mask]
            
            # Correlation between masked observation and predictions
            simplex_corr, _ = pearsonr(masked_o, masked_p)
            
            # Final simplex value for E is the mean of the correlations
            simplex_vals.append(simplex_corr.mean())
        
        # Plot the correlation values of simplex projection across E values for ith channel
        plt.plot(E_range, simplex_vals, marker='o', label=f'Ch{i}')
        
        # Find the E with highest prediction skill
        opt_E = E_range[np.argmax(simplex_vals)]
        print(f"Optimal E: {opt_E} for Channel {i}")
        opt_Es.append(opt_E)
        
    # Select the most frequently occuring value of E
    counter = Counter(opt_Es)
    overall_E = counter.most_common(1)[0][0]       
    print(f"Overall optimal E: {overall_E}")
    plt.figtext(0.1, 0.01, f"Overall optimal E: {overall_E}", ha='left', va='bottom', fontsize=8)
    
    # Plot the estimated Es for all channels
    plt.xlabel('Embedding dimension (E)')
    plt.ylabel('Prediction skill (correl)')
    plt.title('Simplex Projection')
    plt.legend(prop={'size': 16}, loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_filename+"_overall-simplex-projection.png", bbox_inches='tight')
    
    return overall_E

# Plot heatmap for single causality matrix
def plot_heatmap(L, E, tau, output_filename, limit_channels, type):
    
    # Load all causality matrices
    X = np.load(output_filename+'.npz')
    
    # The selected causality matrix
    data = X[f'L{L}_E{E}_tau{tau}']
    
    # Create figure and heatmap
    plt.figure(figsize=(10, 8))
    channel_arr = [f"Ch{i}" for i in limit_channels]
    sns.heatmap(data, annot=True, cmap="coolwarm", square=True, vmin=0,vmax=1, xticklabels=channel_arr, yticklabels=channel_arr)
    
    # Plot the causality matrix as a heatmap
    plt.title(f'{type} correls heatmaps for L{L}_E{E}_tau{tau}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(output_filename+f'-heatmap.png')
    plt.close()

# Plot heatmaps for multiple causality matrices
def plot_heatmaps(output_subj_dir, L, E, tau, output_filenames, limit_channels):
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    channel_arr = [f"Ch{i}" for i in limit_channels]
    states=["a) Non-seizure", "b) Pre-ictal", "c) Ictal"]
    
    for idx, (s, output_filename) in enumerate(zip(states, output_filenames)):
        # Load all causality matrices
        X = np.load(output_filename+'.npz')
        
        # Current causality matrix
        data = X[f'L{L}_E{E}_tau{tau}']
            
        # Heatmap of the causality matrix
        sns.heatmap(data, annot=False, cmap="coolwarm", square=True,vmin=0,vmax=1,xticklabels=channel_arr, yticklabels=channel_arr, ax=axs[idx], cbar=False)
            
        axs[idx].set_title(f"{s} state")
    
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(axs[2].get_children()[0], cax=cbar_ax)
    cbar.set_label('Causality', fontsize=12)
    
    fig.suptitle(f'Causality heatmaps with L{L}_E{E}_tau{tau}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(output_subj_dir+f"/{output_subj_dir.split('/')[-1]}-causality-heatmaps.png")
    plt.close()

# SCRAPS
def plot_boxplots(control_file, ictal_file, pre_ictal_file, subject_dir, L, E_range, tau_range):
    X_c = np.load(control_file+'.npz')
    X_ic = np.load(ictal_file+'.npz')
    X_pre = np.load(pre_ictal_file+'.npz')
    
    nrows = len(E_range)
    ncols = len(tau_range)   
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    
    for i, E in enumerate(E_range):
        for j, tau in enumerate(tau_range):
            c = X_c[f'L{L}_E{E}_tau{tau}']
            ic = X_ic[f'L{L}_E{E}_tau{tau}']
            pre = X_pre[f'L{L}_E{E}_tau{tau}']
            mx = [c.flatten(), pre.flatten(), ic.flatten()]
            ax = axes[i][j] if nrows > 1 else (axes[j] if ncols > 1 else axes)
            ax.boxplot(mx, labels=["Interictal", "Pre-ictal", "Ictal"], showfliers=False)
            ax.set_title(f'E={E}, τ={tau}')
            ax.set_ylabel('correls')

            for idx, group in enumerate(mx, start=1):
                ax.plot(np.random.normal(idx, 0.05, size=len(group)), group, 'o', alpha=1, markersize=0.8)

    fig.suptitle(f'Correls Distributions L={L}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(subject_dir+ f'/correl-distribution-boxplots.png')
    plt.close()

# SCRAPS
def plot_boxplot(control_file, ictal_file, pre_ictal_file, subject_dir, L, E, tau):
    X_c = np.load(control_file+'.npz')
    X_ic = np.load(ictal_file+'.npz')
    X_pre = np.load(pre_ictal_file+'.npz')
        
    plt.figure(figsize=(10, 8))
    
    c = X_c[f'L{L}_E{E}_tau{tau}']
    ic = X_ic[f'L{L}_E{E}_tau{tau}']
    pre = X_pre[f'L{L}_E{E}_tau{tau}']
    mx = [c.flatten(), pre.flatten(), ic.flatten()]
    plt.boxplot(mx, labels=["Interictal", "Pre-ictal", "Ictal"], showfliers=False)
    plt.set_ylabel('correls')

    for idx, group in enumerate(mx, start=1):
        plt.plot(np.random.normal(idx, 0.05, size=len(group)), group, 'o', alpha=1, markersize=0.8)

    plt.title(f'Correls Distributions L{L}_E{E}_tau{tau}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(subject_dir+ f'/correl-distribution-boxplot.png')
    plt.close()

# NOT IN USE: Compute CCM over sliding window
def compute_ccm_over_window(limit_channels, X_c, output_filename):
    start_time = time.perf_counter()
    
    step_size, window_size = 50000, 50000
    
    ccm_correls = np.zeros((len(limit_channels), len(limit_channels)))
    # Apply CCM per non-overlapping window of the time series
    for start in range(0, X_c.shape[1] - window_size + 1, step_size):
        
        ccm_correls = np.zeros((len(limit_channels), len(limit_channels)))
        
        # Reading channels of control file
        for idx, i in enumerate(limit_channels[:-1]):
            X0 = X_c[i, start:start + window_size]
            
            for jdx, j in enumerate(limit_channels[idx+1:], start=idx+1):
                Y0 = X_c[j, start:start + window_size]
                
                # plot_convergence(filename+"-convergence", X0, Y0)            
                
                # Apply CCM to channel pair 
                ccm_XY = ccm(X0, Y0, tau, E, L) # define new ccm object # Testing for X -> Y
                ccm_YX = ccm(Y0, X0, tau, E, L) # define new ccm object # Testing for Y -> X
        
                ccm_XY_corr = ccm_XY.causality()[0]
                ccm_YX_corr = ccm_YX.causality()[0]
                
                ccm_correls[idx, jdx] = ccm_XY_corr # X -> Y over triangle
                ccm_correls[jdx, idx] = ccm_YX_corr # Y -> X under triangle
        
        plot_heatmap(ccm_correls,  f"{output_filename}/{start}-{start+window_size}-ccm_heatmap.png", limit_channels)
        print(f"Done with {output_filename} for {start}-{start+window_size}")

    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.5f} seconds")

# Combine multiple files into a continous file for a subject
def combine_samples(subject, patient_files):
    tot_len = 0
    max_N = 0
    
    # Compute the total length of patient files
    for filename in patient_files:        
        
        with np.load(os.path.join(proc_data_dir, subject, filename), mmap_mode='r') as X_p:
            # Add each length
            tot_len += X_p['arr'].shape[1]
            
            # Update maximum number of channels
            if X_p['arr'].shape[0] > max_N:
                max_N = X_p['arr'].shape[0]
                
    # Combined time series matrix of samples
    X_ps = np.zeros((max_N, tot_len))
    curr_len = 0
    
    # Combine the time series of data samples
    for filename in patient_files:
        print(f"Combining patient file: {filename}")

        # Load data sample
        X_p = np.load(os.path.join(proc_data_dir, subject, filename))['arr']
        
        # Add data
        X_ps[:X_p.shape[0]:, curr_len:curr_len+X_p.shape[1]] = X_p
        curr_len+=X_p.shape[1]
    
    return X_ps, tot_len

# FIX: Plot distribution of asymmetry index values as boxplot
def plot_boxplot_asymm(asymm_idx_subjects, output_dir, L, E, tau):
    
    plt.figure(figsize=(10, 8))
    
    # Load the asymmetry index from the list of dictionaries
    c = np.array([dict['control'] for dict in asymm_idx_subjects])
    pre = np.array([dict['preictal'] for dict in asymm_idx_subjects])
    ic = np.array([dict['ictal'] for dict in asymm_idx_subjects])
    mx = [c, pre, ic]
    
    # Plot the distribution
    plt.boxplot(mx, labels=["Control", "Pre-ictal", "Ictal"], showfliers=False)
    plt.ylabel('Asymmetry Index')
    
    # Customize spacing between groups and color
    xs = []
    colors = plt.cm.tab10.colors

    # Add x with spacings
    for idx, group in enumerate(mx, start=1):
        x = np.random.normal(loc=idx, scale=0.05, size=len(group))
        xs.append(x)
    
    # Plot for each subject
    for i in range(len(mx[0])):
        
        # All x points across states
        x_s = [xs[group_idx][i] for group_idx in range(3)]
        
        # All asymmetry index value for same subject across states
        y_s = [mx[group_idx][i] for group_idx in range(3)]
        
        # Choose unique color
        color = colors[i % len(colors)]
        
        # Plot the points and corresponding lines between
        plt.plot(x_s, y_s, 'o', color=color, alpha=1, markersize=8)
        plt.plot(x_s, y_s, color=color,linestyle='-', alpha=0.7, linewidth=2)
    
    # Statistics text under figure
    ymin, _ = plt.ylim()
    text_y = ymin - 0.07 * (plt.ylim()[1] - ymin)

    # Add statistics under each group
    for idx, group in enumerate(mx, start=1):
        mean = np.mean(group)
        sd = np.std(group)
        median = np.median(group)
        iqr = np.percentile(group, 75) - np.percentile(group, 25)
        text=(f"Mean={mean:.2f}\nMedian={median:.2f}\nIQR={iqr:.2f}\nSD={sd:.2f}")
        plt.text(idx, text_y, text, ha='center', va='top', fontsize=10)

    plt.title(f'Asymmetry Index Distributions L{L}_E{E}_tau{tau}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir+ f'/Overall-asymmetry-index-distribution.png')
    plt.close()

# Compute asymmetry index for each causality matrix in the different states
def compute_asymm_idx(subject, output_filenames, L, E, tau):
    
    # List of asymmetry index values
    asymm_idxs=[]
    
    # Compute asymm idx for each state
    for output_filename in output_filenames:
        
        # Load causality matrix
        X = np.load(output_filename+'.npz')
        data = X[f'L{L}_E{E}_tau{tau}']
        
        # Calculate asymm index for the causality matrix
        asymm_idx = np.linalg.norm(data - data.T, 'fro')
        asymm_idxs.append(asymm_idx)
        print(f'Done computing asymmetry index for {output_filename}\n')
    
    # Dictionary of asymmetry index values for all state
    asymm_row = {'subject_id': subject, 'control': asymm_idxs[0], 'preictal': asymm_idxs[1], 'ictal': asymm_idxs[2]}
    print(f'Done computing asymmetry index for patient specific files {subject}\n')
    return asymm_row

# SCRAPS
def compute_pair_asymm_format(subject, output_filenames, L, E, tau):

    max_asymm_pair_states=[]    
    for output_filename in output_filenames:
        X = np.load(output_filename+'.npz')
        data = X[f'L{L}_E{E}_tau{tau}']
        asymm_pairs_list = []
        for i in range(data.shape[0]-1):
            for j in range(i+1, data.shape[0]):
                corr_X_Y = data[i,j]
                corr_Y_X = data[j,i]
                asymm = np.abs(corr_X_Y-corr_Y_X)
                asymm_pairs_list.append({'key': f'{i+1}, {j+1}', 'value': asymm})
        max_asymm_pair = sorted(asymm_pairs_list, key=lambda x: x['value'], reverse=True)[0]
        print(max_asymm_pair)
        max_asymm_pair_states.append(f"Ch({max_asymm_pair['key']}): {round(max_asymm_pair['value'], 3)}")
        print(f'Done computing asymmetry pairs for {output_filename}\n')
    
    asymm_row={'subject_id': subject, 'control': max_asymm_pair_states[0], 'preictal': max_asymm_pair_states[1], 'ictal': max_asymm_pair_states[2]}
    print(f'Done computing asymmetry pairs for patient specific files {subject}\n')
    return asymm_row

# Identify the top_N channels with the higher asymmetry value for each state
def compute_ch_asymm(subject, output_filenames, L, E, tau, top_N, format=False):

    # List of channels with max asymmetry for each state
    max_asymm_ch_states=[]    
    
    # Find highest asymmetry for each state
    for output_filename in output_filenames:
        
        # Load causality matrix
        X = np.load(output_filename+'.npz')
        data = X[f'L{L}_E{E}_tau{tau}']
        
        # List of channels and corresponding asymmetry value
        asymm_ch_list = []
        for i in range(data.shape[0]-1):
            
            # Total causal outflow
            outF = np.sum(data[i,:])
            
            # Total causal inflow
            inF = np.sum(data[:,i])
            
            # Asymmetry value is quantified with the difference between outflow and inflow
            asymm = np.abs(outF-inF)
            asymm_ch_list.append({'key': f'{i+1}', 'value': asymm})
        
        # Get top N channel and value pair with highest asymmetry value
        max_asymm_ch = sorted(asymm_ch_list, key=lambda x: x['value'], reverse=True)[:top_N]
        
        # Select this dominant channel for the state
        if format and top_N==1:
            max_asymm_ch_states.append(f"Ch({max_asymm_ch[0]['key']}): {round(max_asymm_ch[0]['value'], 3)}")
            print(max_asymm_ch)
            print(f'Done computing asymmetry channel for {output_filename}\n')
        else:
            max_asymm_ch_states.append(max_asymm_ch)
            print(max_asymm_ch_states)
            print(f'Done computing {top_N} asymmetry channel for {output_filename}\n')
    
    # Dictionary of the total dominant asymmetric channels for each state
    asymm_row={'subject_id': subject, 'control': max_asymm_ch_states[0], 'preictal': max_asymm_ch_states[1], 'ictal': max_asymm_ch_states[2]}
    print(f'Done finding dominant asymmetric channel for patient specific files {subject}\n')
    return asymm_row

# SCRAPS
def compute_pair_asymm(subject, output_filenames, L, E, tau):
    
    # rows = []
    max_asymm_pair_states=[]
    
    for output_filename in output_filenames:
        X = np.load(output_filename+'.npz')
        data = X[f'L{L}_E{E}_tau{tau}']
        asymm_pairs_list = []
        for i in range(data.shape[0]-1):
            for j in range(i+1, data.shape[0]):
                corr_X_Y = data[i,j]
                corr_Y_X = data[j,i]
                asymm = np.abs(corr_X_Y-corr_Y_X)
                asymm_pairs_list.append({'key': f'{i+1}, {j+1}', 'value': asymm})
        max_asymm_pair = sorted(asymm_pairs_list, key=lambda x: x['value'], reverse=True)[:3]
        # max(asymm_pairs_list, key=lambda x: x['value'])]
        # max_asymm_pair = [d['key'] for d in sorted(asymm_pairs_list, key=lambda x: x['value'], reverse=True)[:3]]
        print(max_asymm_pair)
        max_asymm_pair_states.append(max_asymm_pair)
        print(f'Done computing asymmetry pairs for {output_filename}\n')
    
    # df = pd.DataFrame(rows)
    # with open(f'{output_filename}-asymm.md', 'w') as f:
    #     f.write(df.to_markdown(index=False))
    # df.to_csv(f'{output_dir_subj}/asymmetry-indices.csv')
    asymm_row=[]
    for i in range(np.array(max_asymm_pair_states).shape[1]):
        asymm_row.append({'subject_id': subject, 'control': max_asymm_pair_states[0][i], 'preictal': max_asymm_pair_states[1][i], 'ictal': max_asymm_pair_states[2][i]})
    # asymm_row={'subject_id': subject, 'control': max_asymm_pair_states[0], 'preictal': max_asymm_pair_states[1], 'ictal': max_asymm_pair_states[2]}
    print(f'Done computing asymmetry pairs for patient specific files {subject}\n')
    return asymm_row

# SCRAPS
def plot_frequency_asymm_pair(asymm_pairs_subjects, output_dir, L, E, tau):
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))  # 1 row, 3 columns
    # states_top_3 = []
    
    # Create horisontal bar chart of frequency for channel pair for each state
    for idx, state in enumerate(['control', 'preictal', 'ictal']):
        state_pairs = np.array([dict[state] for dict in asymm_pairs_subjects])
        freq_pairs = []
        
        # Compute the frequency of the channel pair
        for pair in state_pairs:
            i = pair['key'].split(', ')[0]
            j = pair['key'].split(', ')[1]
            asymm_value = pair['value']
            exist = False
            
            # Check if the pair is already registered
            for p in freq_pairs:
                if p['pair'] == f'{i}, {j}' or p['pair'] == f'{j}, {i}':
                    p['freq'] += 1
                    p['sum'] += asymm_value
                    exist = True
                    break
                
            # Register the channel pair the first time
            if not exist:
                freq_pairs.append({'pair': i+", "+j, 'freq': 1,'sum': asymm_value})
        
        # Compute the top 3 most frequently occuring channel pair
        top_3 = sorted(freq_pairs, key=lambda x: x['freq'], reverse=True)[:3]
        # states_top_3.append({'state': state, 'top3': top_3})
        
        # Plot the subplot for the state
        y = np.array([dict['pair'] for dict in top_3])
        x = np.array([dict['freq'] for dict in top_3])
        bars=axs[idx].barh(y, x, color='skyblue')
        axs[idx].set_xlabel('Frequency')
        axs[idx].set_title(f'{state.capitalize()} state')
        axs[idx].set_yticks(np.arange(len(y)))
        axs[idx].set_yticklabels(y)
        
        for bar, pair_data in zip(bars, top_3):
            mean_asymm = pair_data['sum'] / pair_data['freq']
            axs[idx].text(bar.get_width() *0.5, bar.get_y() + bar.get_height()/2,
                        f'{mean_asymm:.2f}', va='center', fontsize=9)
    
    fig.suptitle(f'Frequency of highest asymmetry channel pair for L{L}_E{E}_tau{tau}', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir+ f'/Overall-asymmetry-pairs-freqs.png')
    plt.close()

# Plot the top 3 frequent dominant asymmetric channels for each state
def plot_frequency_asymm_ch(asymm_subjects, output_dir, L, E, tau):
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))  # 1 row, 3 columns
    
    # Create horisontal bar chart of frequency for channel pair for each state
    for idx, state in enumerate(['control', 'preictal', 'ictal']):
        
        # Get the channels with highest asymmetry for that state
        # state_dom_ch = np.array([dict[state] for dict in asymm_subjects])
        state_dom_ch = np.array([in_dict for o_dict in asymm_subjects for in_dict in o_dict[state]])
        freq_dom_ch = []
        print(state_dom_ch)
        
        # Compute the frequency of the channel pair
        for pair in state_dom_ch:
            print(type(pair))
            ch = pair['key']
            asymm_value = pair['value']
            exist = False
            
            # Check if the pair is already registered
            for p in freq_dom_ch:
                if p['ch'] == f'{ch}':
                    p['freq'] += 1
                    p['sum'] += asymm_value
                    exist = True
                    break
                
            # Register the channel pair the first time
            if not exist:
                freq_dom_ch.append({'ch': f'{ch}', 'freq': 1,'sum': asymm_value})
        
        # Compute the top 3 most frequently occuring channels
        top_3 = sorted(freq_dom_ch, key=lambda x: x['freq'], reverse=True)[:3]
        
        y = np.array([dict['ch'] for dict in top_3])
        x = np.array([dict['freq'] for dict in top_3])
        
        # Plot the horisontal bar charts for the state
        bars=axs[idx].barh(y, x, color='skyblue')
        axs[idx].set_xlabel('Frequency')
        axs[idx].set_title(f'{state.capitalize()} state')
        axs[idx].set_yticks(np.arange(len(y)))
        axs[idx].set_yticklabels(y)
        
        # Display the mean value for individual bar chart
        for bar, pair_data in zip(bars, top_3):
            mean_asymm = pair_data['sum'] / pair_data['freq']
            axs[idx].text(bar.get_width() *0.5, bar.get_y() + bar.get_height()/2, f'{mean_asymm:.2f}', va='center', fontsize=9)
    
    fig.suptitle(f'Frequency of highest asymmetry channel for L{L}_E{E}_tau{tau}', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir+ f'/Overall-asymmetry-channel-freqs.png')
    plt.close()

# Compute ccm on selected channels
def compute_ccm(limit_channels, X, L, E, tau):
    start_time = time.perf_counter()
    print(f"The L: {L}")
    print(f"The E: {E}")
    print(f"The tau: {tau}")
    print(f"Starts: {start_index} and Ends: {end_index}")
        
    #Variable Initialization
    Xhat_My_f = []
    Yhat_Mx_f = []
    ccm_correls = np.zeros((len(limit_channels), len(limit_channels)))
    
    #Reading channels
    for idx, i in enumerate(limit_channels[:-1]):
        Xhat_My, Yhat_Mx = [], []
        X0 = X[i-1,start_index:end_index]
        for jdx, j in enumerate(limit_channels[idx+1:], start=idx+1):
            Y0=X[j-1,start_index:end_index]
            
            #Applying CCM to channel pair 
            ccm_XY = ccm(X0, Y0, tau, E, L) # define new ccm object # Testing for X -> Y
            ccm_YX = ccm(Y0, X0, tau, E, L) # define new ccm object # Testing for Y -> X
            
            ccm_XY_corr = ccm_XY.causality()[0]
            ccm_YX_corr = ccm_YX.causality()[0]
            
            ccm_correls[idx, jdx] = ccm_XY_corr # X -> Y over triangle
            ccm_correls[jdx, idx] = ccm_YX_corr # Y -> X under triangle
            
            Xhat_My.append(ccm_XY_corr)
            Yhat_Mx.append(ccm_YX_corr)

        Xhat_My_f.append(Xhat_My)
        Yhat_Mx_f.append(Yhat_Mx)

    # X_Y_arr = np.zeros((len(Yhat_Mx_f)+1,len(Yhat_Mx_f)+1))
    # for i in range(len(X_Y_arr)-1):
    #     for j in range(0,len(Yhat_Mx_f[i])):
    #         X_Y_arr[i,i+1+j] = Yhat_Mx_f[i][j]
    #         X_Y_arr[i+1+j,i] = Xhat_My_f[i][j]
    
    # plot_heatmap(ccm_correls,  output_filename+"-ccm_heatmap.png", limit_channels)
    # print(f"Done with {output_filename}")
        
    end_time = time.perf_counter()
    return ccm_correls

# Save ccm results
def compute_across_params(L_range, E_range, tau_range, output_filename, limit_channels, X):
    if os.path.exists(output_filename+'.npz'):
        print("Hellos")
        print(f"Already exists {output_filename}")
        # Old param values
        data = dict(np.load(output_filename+'.npz', allow_pickle=True))
        
        # New param values
        for L in L_range:
                for E in E_range:
                    for tau in tau_range:
                        data.update({f'L{L}_E{E}_tau{tau}':compute_ccm(limit_channels, X, L, E, tau)})
        
        np.savez(output_filename, **data)
        print(f"Done updating {output_filename}")
    else:
        all_matrix = {}
        if len(X[0,:]) >= end_index - start_index:
            for L in L_range:
                for E in E_range:
                    for tau in tau_range:
                        # Compute ccm on control file
                        all_matrix[f'L{L}_E{E}_tau{tau}']  = compute_ccm(limit_channels, X, L, E, tau)
            np.savez_compressed(output_filename, **all_matrix)
            print(f"Done creating {output_filename}")

# Compute ccm results for all states for each subject
def ccm_subject(subject, proc_data_dir, output_dir):
    
    # Start processing subject
    print(f"Starting subject {subject}")
    subject_dir = Path(proc_data_dir + "/" + subject)
    
    # Limit channels for parameter testing
    limit_channels = [2, 4, 6, 7]
    
    # Selected patient files
    control_file = os.path.join(subject_dir, "control-data.npz")
    patient_ictal_files = [f for f in os.listdir(subject_dir) if os.path.isfile(os.path.join(subject_dir, f)) and f.split("-")[0]=="ictal"]
    patient_pre_ictal_files = [f for f in os.listdir(subject_dir) if os.path.isfile(os.path.join(subject_dir, f)) and f.split("-")[0]=="pre"]
    
    # Output paths
    output_dir_subj = output_dir + '/' + subject
    os.makedirs(output_dir_subj, exist_ok=True)
    output_filename_c=output_dir_subj+"/control-file"
    output_filename_ic = output_dir_subj + '/patient-ictal-file'
    output_filename_pre = output_dir_subj + '/patient-pre-ictal-file'
    
    # Load control data
    X_c = np.load(os.path.join(proc_data_dir, subject, "nonses", control_file))['arr']
    
    # Parameters range
    L_range = [6000, 7000, 8000, 9000, 10000]
    E_range = [2,3, 4, 5]
    tau_range=[1,2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Observed optimal parameter values (L, E, tau) = (10 000, 4, 4)
    opt_L = L_range[4]
    opt_tau = tau_range[3]
    opt_E = E_range[2]
    
    # Length decides amount of datapoints in the window
    global end_index
    global start_index
        
    # Tune CCM parameters for subject chb01 on control data
    if(subject == "chb01"):
                
        # 1. Compute ccm on control file across params
        start_index = X_c.shape[1]//2
        end_index = X_c.shape[1] - 1
        compute_across_params(L_range, E_range, tau_range,output_filename_c+"_parameter-testing", limit_channels, X_c)
        
        # 2. Plot overall convergence of control file to decide L
        plot_overall_convergence(output_filename_c+"_parameter-testing", L_range, E_range, tau_range)
        opt_L = L_range[4]      # Observed from convergence plot
        
        # 3. Plot autocorrelation of control file to decide tau
        opt_tau=plot_autocorrelation(output_filename_c+"_parameter-testing", opt_L, E_range, tau_range)
        
        # 4. Plot simplex projection of control file to decide E
        opt_E=plot_simplex(output_filename_c+"_parameter-testing", limit_channels, opt_L, opt_tau, E_range, X_c)
        
    # 5. Individual convergence checks for each state for a single channel pair
    if not os.path.exists(output_dir_subj+"/test-convergences.png"):
        Xs, Ys = [], []         # Channel data for each state
        opt_Es = [4, 4, 4]      # Different Es: [4, 3, 3]
        opt_taus = [4, 4, 4]    # Different taus: [4, 5, 5]
        i, j = 1, 2
        
        # Extract channel i and j data from control data
        start_index= random.randint(opt_L, X_c.shape[1] - opt_L - 1)
        end_index = start_index + opt_L
        Xs.append(X_c[i, start_index:end_index])
        Ys.append(X_c[j, start_index:end_index])
        
        # Extract channel i and j data from preictal data
        X_pre, pre_len = combine_samples(subject, patient_pre_ictal_files)
        start_index= random.randint(opt_L, pre_len - opt_L - 1)
        end_index = start_index + opt_L
        Xs.append(X_pre[i, start_index:end_index])
        Ys.append(X_pre[j, start_index:end_index])
        
        # Extract channel i and j data from ictal data
        X_ic, ic_len = combine_samples(subject, patient_ictal_files)
        start_index= random.randint(opt_L, ic_len - opt_L - 1)
        end_index = start_index + opt_L
        Xs.append(X_ic[i, start_index:end_index])
        Ys.append(X_ic[j, start_index:end_index])
        
        # Plot convergence check
        plot_convergence(f"Convergence for Ch({i}, {j})", output_dir_subj+"/test-convergences", L_range, opt_Es, opt_taus, Xs, Ys)
    
    # 6. Compute ccm on control file for the fixed parameter set
    start_index= random.randint(opt_L, X_c.shape[1] - opt_L - 1)
    end_index = start_index + opt_L
    compute_across_params([opt_L], [opt_E],[opt_tau],output_filename_c, [i for i in range(1, 24)], X_c)
    
    # 7. Compute ccm on ictal files for the fixed parameter set
    X_ic, ic_len = combine_samples(subject, patient_ictal_files)
    start_index= random.randint(opt_L, ic_len - opt_L - 1)
    end_index = start_index + opt_L
    compute_across_params([opt_L], [opt_E],[opt_tau],output_filename_ic, [i for i in range(1, 24)], X_ic)
    
    # 8. Compute ccm on preictal files for the fixed parameter set
    X_pre, pre_len = combine_samples(subject, patient_pre_ictal_files)
    start_index= random.randint(opt_L, pre_len - opt_L - 1)
    end_index = start_index + opt_L
    compute_across_params([opt_L], [opt_E],[opt_tau],output_filename_pre, [i for i in range(1, 24)], X_pre)
                
    # Test over window
    # OLD: compute_ccm_over_window(limit_channels, X_c, output_dir_subj)

# Evaluate all subjects
def eval_subjects(subjects):
    
    # Parameters range
    L_range = [6000, 7000, 8000, 9000, 10000]
    E_range = [2,3, 4, 5]
    tau_range=[1,2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Observed optimal parameter values 
    opt_L = L_range[4]
    opt_tau = tau_range[3]
    opt_E = E_range[2]
    
    # Lists of asymmetry indices and dominant channels
    asymm_idx_subjects = []
    asymm_ch_highest = []
    asymm_ch_highest_sub = []
    
    for subject in subjects:
        
        # Output paths
        output_dir_subj = output_dir + '/' + subject
        os.makedirs(output_dir_subj, exist_ok=True)
        output_filename_c=output_dir_subj+"/control-file"
        output_filename_ic = output_dir_subj + '/patient-ictal-file'
        output_filename_pre = output_dir_subj + '/patient-pre-ictal-file'
        
        # Plot heatmaps of each causality matrix corresponding to each state
        plot_heatmaps(output_dir_subj, opt_L, opt_E, opt_tau, [output_filename_c, output_filename_pre, output_filename_ic], [i for i in range(1, 24)])
        
        # Compute asymmetry index for each subject
        asymm_idx_subjects.append(compute_asymm_idx(subject, [output_filename_c, output_filename_pre, output_filename_ic], opt_L, opt_E, opt_tau))
        
        # Identify only the highest asymmetric channel in each state
        asymm_ch_highest.append(compute_ch_asymm(subject, [output_filename_c, output_filename_pre, output_filename_ic], opt_L, opt_E, opt_tau, 1, True))
        
        # Identify the top N highest asymmetric channel in each state
        asymm_ch_highest_sub.append(compute_ch_asymm(subject, [output_filename_c, output_filename_pre, output_filename_ic], opt_L, opt_E, opt_tau, 1, False))
    
    # Table of the asymmetry indices for all subjects
    df = pd.DataFrame(asymm_idx_subjects)
    df.to_excel(f'{output_dir}/Overall-asymmetry-index.xlsx')
    
    # Table of the dominant asymmetric channels for all subjects
    df2 = pd.DataFrame(asymm_ch_highest)
    df2.to_excel(f'{output_dir}/Overall-asymmetry-channels.xlsx')
    
    # Plot distribution of asymmetry indices across all subjects
    plot_boxplot_asymm(asymm_idx_subjects, output_dir, opt_L, opt_E, opt_tau)
    
    # Plot frequency of dominant asymmetric channels across all subjects
    plot_frequency_asymm_ch(asymm_ch_highest_sub, output_dir, opt_L, opt_E, opt_tau)
    

# CCM Parameters
np.random.seed(1)
L=10000      # length of time period
tau=1       # time lag
E=2         # embedding dimensions

# Select chunk
start_index = 0
end_index = start_index + L
n_samples = 1

# Get the parent directory
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)

# Define the relative paths
proc_data_dir = os.path.join(parent_dir, 'processed_data')
output_dir = os.path.join(parent_dir, 'output_data')
os.makedirs(output_dir, exist_ok=True)

# list_subjects = [f"chb{str(i).zfill(2)}" for i in [1, 2, 3,4, 5, 6, 8, 9, 10, 23]]
list_subjects = [f"chb{str(i).zfill(2)}" for i in [1, 2, 3,4, 5, 6, 8, 9, 10, 23]]
num_cores = mp.cpu_count()
args_list = [(subject, proc_data_dir, output_dir) for subject in list_subjects]

# ccm_subject("chb01", proc_data_dir, output_dir)

eval_subjects(list_subjects)

# for subject in list_subjects:
#     ccm_subject(subject, proc_data_dir, output_dir)

# ccm_subject("chb03", proc_data_dir, output_dir)

# pool = mp.Pool(8)
# results = pool.starmap(ccm_subject,args_list)
