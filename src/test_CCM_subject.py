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
def plot_convergence(filename, L_range, X, Y):
    L_range = range(6000, 8000, 200) # L values to test
    tau = 1
    E = 2

    Xhat_My, Yhat_Mx = [], [] # correlation list
    for L in L_range: 
        ccm_XY = ccm(X, Y, tau, E, L) # define new ccm object # Testing for X -> Y
        ccm_YX = ccm(Y, X, tau, E, L) # define new ccm object # Testing for Y -> X    
        Xhat_My.append(ccm_XY.causality()[0]) 
        Yhat_Mx.append(ccm_YX.causality()[0]) 
    
    # plot convergence as L->inf. Convergence is necessary to conclude causality
    plt.figure(figsize=(5,5))
    plt.plot(L_range, Xhat_My, label='$\hat{X}(t)|M_y$')
    plt.plot(L_range, Yhat_Mx, label='$\hat{Y}(t)|M_x$')
    plt.xlabel('L', size=12)
    plt.ylabel('correl', size=12)
    plt.legend(prop={'size': 16})    
    plt.savefig(filename)
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
            "Time": np.arange(end_index-start_index),
            f"Ch{i}": X[i,start_index:end_index]
        })
        
        # All correlation values from simplex projection
        simplex_vals = []
        
        for E in E_range:
            
            # Compute Simplex projection
            res = pyEDM.Simplex(
                dataFrame=df,
                lib=f'1 {L}',                                         # Train embedding with L data points
                pred=f'{L+1} {L*2 if L*2<=end_index else end_index}', # Test predictions with remaining data points
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
    print(f"Overall optimal E: {opt_E}")
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
    plt.savefig(output_filename+f'-heatmaps.png')
    plt.close()

# Plot heatmaps for multiple causality matrices
def plot_heatmaps(L, E_range, tau_range, output_filename, limit_channels, C, type):
    
    # Load all causality matrices
    X = np.load(output_filename+'.npz')
    
    # Create a grid of plots for different (E, tau)
    nrows = len(E_range)
    ncols = len(tau_range)    
    fig, axes = plt.subplots(nrows, ncols, figsize=(len(limit_channels)*ncols, len(limit_channels)*nrows))
    channel_arr = [f"Ch{i}" for i in limit_channels]
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
    
    for i, E in enumerate(E_range):
        for j, tau in enumerate(tau_range):
            
            # Current causality matrix
            data = X[f'L{L}_E{E}_tau{tau}']
            
            # Heatmap of the causality matrix
            sns.heatmap(data, annot=True, cmap="coolwarm", square=True,vmin=0,vmax=1,xticklabels=channel_arr, yticklabels=channel_arr, ax=axes[i][j], cbar=(i == 0 and j == 0), cbar_ax=cbar_ax if (i==0 and j==0) else None)
            
            # Add title of this sub plot
            axes[i][j].set_title(f'L = {L} E={E}, τ={tau}')
            
    fig.suptitle(f'{type} correls heatmaps', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(output_filename+f'-heatmaps.png')
    plt.close()
 
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

def compute_ccm_over_window(limit_channels, X_c, output_filename):
    # Stores all correlations
    Xhat_My_f = []
    Yhat_Mx_f = []

    start_time = time.perf_counter()
    
    step_size, window_size = 50000, 50000
    
    ccm_correls = np.zeros((len(limit_channels), len(limit_channels)))
    # Apply CCM per non-overlapping window of the time series
    for start in range(0, X_c.shape[1] - window_size + 1, step_size):
        
        ccm_correls = np.zeros((len(limit_channels), len(limit_channels)))
        
        # Reading channels of control file
        for idx, i in enumerate(limit_channels[:-1]):
            X0 = X_c[i, start:start + window_size]
            Xhat_My1 = []
            Yhat_Mx1 = []
            
            for jdx, j in enumerate(limit_channels[idx+1:], start=idx+1):
                Y0 = X_c[j, start:start + window_size]
                Xhat_My, Yhat_Mx = [], []
                
                # plot_convergence(filename+"-convergence", X0, Y0)            
                
                # Apply CCM to channel pair 
                ccm_XY = ccm(X0, Y0, tau, E, L) # define new ccm object # Testing for X -> Y
                ccm_YX = ccm(Y0, X0, tau, E, L) # define new ccm object # Testing for Y -> X
        
                ccm_XY_corr = ccm_XY.causality()[0]
                ccm_YX_corr = ccm_YX.causality()[0]
                
                ccm_correls[idx, jdx] = ccm_XY_corr # X -> Y over triangle
                ccm_correls[jdx, idx] = ccm_YX_corr # Y -> X under triangle
                
                # Xhat_My.append(ccm_XY_corr)
                # Yhat_Mx.append(ccm_YX_corr)
                
                # print("PLOTTING")
                # ccm_XY.plot_ccm_correls()
                # ccm_YX.plot_ccm_correls()
                
            # Xhat_My1.append(Xhat_My)
            # Yhat_Mx1.append(Yhat_Mx)
            
        # Xhat_My_f.append(Xhat_My1)
        # Yhat_Mx_f.append(Yhat_Mx1)
        
        plot_heatmap(ccm_correls,  f"{output_filename}/{start}-{start+window_size}-ccm_heatmap.png", limit_channels)
        print(f"Done with {output_filename} for {start}-{start+window_size}")

    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.5f} seconds")

def combine_samples(subject, patient_files):
    tot_len = 0
    
    # Compute the total length of seizures
    for filename in patient_files:        
        with np.load(os.path.join(proc_data_dir, subject, filename), mmap_mode='r') as X_p:
            tot_len += X_p['arr'].shape[1]
    
    # Combined time series matrix of seizures
    X_ps = np.zeros((23, tot_len))
    curr_len = 0
    
    # Combine the time series of seizure samples
    for filename in patient_files:
        print(f"Combining patient file: {filename}")

        # Load seizure data
        X_p = np.load(os.path.join(proc_data_dir, subject, filename))['arr']
        
        # Add seizure data
        X_ps[:, curr_len:curr_len+X_p.shape[1]] = X_p
        curr_len+=X_p.shape[1]
    return X_ps, tot_len

def compute_asymm_idx(subject, output_filenames, L, E, tau):
    
    # rows = []
    asymm_idxs=[]
    
    for output_filename in output_filenames:
        X = np.load(output_filename+'.npz')
        data = X[f'L{L}_E{E}_tau{tau}']
        asymm_idx = np.linalg.norm(data - data.T, 'fro')
        # rows.append({'Type': output_filename.split("/")[-1] ,'asymm_idx': asymm_idx})
        asymm_idxs.append(asymm_idx)
        print(f'Done computing asymmetry index for {output_filename}\n')
                
    # df = pd.DataFrame(rows)
    # with open(f'{output_filename}-asymm.md', 'w') as f:
    #     f.write(df.to_markdown(index=False))
    # df.to_csv(f'{output_dir_subj}/asymmetry-indices.csv')
    
    asymm_row = {'subject_id': subject, 'interictal': asymm_idxs[0], 'preictal': asymm_idxs[1], 'ictal': asymm_idxs[2]}
    print(f'Done computing asymmetry index for patient specific files {subject}\n')
    return asymm_row

def compute_metrics(output_filename, C, L, E_range, tau_range):
    X = np.load(output_filename+'.npz')
    rows = []
    
    for E in E_range:
        for tau in tau_range:
            data = X[f'L{L}_E{E}_tau{tau}']
            diff = data - C
            # num_links = np.sum(diff>0)
            # var = np.var(data)
            # asymm_idx = np.linalg.norm(data - data.T, 'fro') / np.linalg.norm(data, 'fro')
            asymm_idx = np.linalg.norm(data - data.T, 'fro')
            # rows.append({'L': L, 'E': E, 'tau': tau, 'num_links': num_links, 'variance': var, 'asymm_idx': asymm_idx})
            rows.append({'L': L, 'E': E, 'tau': tau,'asymm_idx': asymm_idx})
            
    df = pd.DataFrame(rows)
    with open(f'{output_filename}-metrics.md', 'w') as f:
        f.write(df.to_markdown(index=False))
    print(f'Done computing metrics for {output_filename}\n')

def compute_high_outflow(output_dir_subj, output_filenames, L, E, tau, limit_channels):
    
    outflow_row = []
    for i in range(len(limit_channels)):
        
        outflows = []
        
        for output_filename in output_filenames:
            X = np.load(output_filename+'.npz')
            data = X[f'L{L}_E{E}_tau{tau}']
            outflows.append(np.sum((data[i, :])))

        outflow_row.append({'channel': limit_channels[i], 'interictal': outflows[0], 'preictal': outflows[1], 'ictal': outflows[2]})
    
    df = pd.DataFrame(outflow_row)
    df.to_csv(f'{output_dir_subj}/causal-outflows.csv')
    print(f' Outflow L{L}_E{E}_tau{tau} for {output_dir_subj}\n')

def compute_reference_matrix(output_filename, L, E_range, tau_range, limit_channels):
    X = np.load(output_filename+'.npz')
    sum_mx = np.zeros((len(limit_channels), len(limit_channels)))
    num_mx = 0
    for E in E_range:
        for tau in tau_range:
            num_mx += 1
            data = X[f'L{L}_E{E}_tau{tau}']
            sum_mx += data
    mean_mx = sum_mx / num_mx
    plot_heatmap(mean_mx, output_filename+"-reference_matrix.png", limit_channels, 'Inter-ictal reference correls heatmap')
    return mean_mx   

def compute_ccm(limit_channels, output_filename, X, L, E, tau):
    start_time = time.perf_counter()
    print(f"The L: {L}")
    print(f"The E: {E}")
    print(f"The tau: {tau}")
    print(f"The signal length ends at: {end_index}")
        
    #Variable Initialization
    Xhat_My_f = []
    Yhat_Mx_f = []
    ccm_correls = np.zeros((len(limit_channels), len(limit_channels)))
    
    #Reading channels
    for idx, i in enumerate(limit_channels[:-1]):
        Xhat_My, Yhat_Mx = [], []
        X0 = X[i,start_index:end_index]
        for jdx, j in enumerate(limit_channels[idx+1:], start=idx+1):
            Y0=X[j,start_index:end_index]
            
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
    
def compute_across_params(L_range, E_range, tau_range, output_filename, limit_channels, X):
    if os.path.exists(output_filename+'.npz'):
        print(f"Already exists {output_filename}")
        # Old param values
        data = dict(np.load(output_filename+'.npz', allow_pickle=True))
        
        # New param values
        for L in L_range:
                for E in E_range:
                    for tau in tau_range:
                        data.update({f'L{L}_E{E}_tau{tau}':compute_ccm(limit_channels,output_filename, X, L, E, tau)})
        
        np.savez(output_filename, **data)
        print(f"Done updating {output_filename}")
    else:
        all_matrix = {}
        if len(X[0,:]) >= start_index + end_index:
            for L in L_range:
                for E in E_range:
                    for tau in tau_range:
                        # Compute ccm on control file
                        all_matrix[f'L{L}_E{E}_tau{tau}'] = compute_ccm(limit_channels,output_filename, X, L, E, tau)
        
            np.savez_compressed(output_filename, **all_matrix)
            print(f"Done creating {output_filename}")

def ccm_subject(subject, proc_data_dir, output_dir):
    
    print(f"Starting subject {subject}")
    subject_dir = Path(proc_data_dir + "/" + subject)
    limit_channels = [1, 4, 5, 7, 8, 9, 12, 13, 18, 21]
    
    # Selected control fil
    control_file = os.path.join(subject_dir, "control-data.npz")
    
    # Selected patient files
    patient_ictal_files = [f for f in os.listdir(subject_dir) if os.path.isfile(os.path.join(subject_dir, f)) and f.split("-")[0]=="ictal"]
    patient_pre_ictal_files = [f for f in os.listdir(subject_dir) if os.path.isfile(os.path.join(subject_dir, f)) and f.split("-")[0]=="pre"]
    
    # Output paths
    output_dir_subj = output_dir + '/' + subject
    os.makedirs(output_dir_subj, exist_ok=True)
    output_filename_c=output_dir_subj+"/control-file"
    output_filename_ic = output_dir_subj + '/patient-ictal-file'
    output_filename_pre = output_dir_subj + '/patient-pre-ictal-file'
    
    # Load non-seizure data
    X_c = np.load(os.path.join(proc_data_dir, subject, "nonses", control_file))['arr']
    
    # Length is limited by max sample size
    global end_index
    
    # Parameters range
    L_range = [6000, 7000, 8000, 9000, 10000]
    E_range = [2,3, 4, 5]
    tau_range=[1,2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Observed optimal parameter values 
    opt_L = L_range[4]
    opt_tau = tau_range[3]
    opt_E = E_range[2]
    
    # Tune CCM parameters for subject chb01
    if(subject == "chb01"):
        
        # STEP 1: Compute ccm on control file across params
        end_index = start_index + X_c.shape[1]//2
        # compute_across_params(L_range, E_range, tau_range,output_filename_c, limit_channels, X_c)
        
        # STEP 2: Plot overall convergence of control file to decide L
        # plot_overall_convergence(output_filename_c, L_range, E_range, tau_range)
        opt_L = L_range[4]      # Observed from convergence plot
        
        # STEP 3: Plot autocorrelation of control file to decide tau
        # opt_tau = plot_autocorrelation(output_filename_c, opt_L, E_range, tau_range)
        
        # STEP 4: Plot simplex projection of control file to decide E
        # opt_E = plot_simplex(output_filename_c, limit_channels, opt_L, opt_tau, E_range, X_c)
        
        # Parameter tuning yields (L, E, tau) = (10 000, 4, 4)
        
        # STEP 5: Compute ccm on patient ictal and pre-ictal files for the fixed parameter set
        X_ic, ic_len = combine_samples(subject, patient_ictal_files)
        X_pre, pre_len = combine_samples(subject, patient_pre_ictal_files)
        
        # Compute ccm on ictal file across params
        end_index = start_index + ic_len//2
        compute_across_params([opt_L], [opt_E],[opt_tau],output_filename_ic, limit_channels, X_ic)
        
        # Compute ccm on pre ictal file across params
        end_index = start_index + pre_len // 2   
        compute_across_params([opt_L], [opt_E],[opt_tau],output_filename_pre, limit_channels, X_pre)
    
    else:                
        # Compute ccm on inter ictal file
        end_index = start_index + X_c.shape[1]//2
        compute_across_params([opt_L], [opt_E],[opt_tau],output_filename_c, limit_channels, X_c)
    
    # STEP 5: Compute ccm on patient specific files for the fixed parameter set
    X_ic, ic_len = combine_samples(subject, patient_ictal_files)
    X_pre, pre_len = combine_samples(subject, patient_pre_ictal_files)
    
    # Compute ccm on ictal file
    end_index = start_index + ic_len//2
    compute_across_params([opt_L], [opt_E],[opt_tau],output_filename_ic, limit_channels, X_ic)
    
    # Compute ccm on pre ictal file
    end_index = start_index + pre_len // 2   
    compute_across_params([opt_L], [opt_E],[opt_tau],output_filename_pre, limit_channels, X_pre)
    
    # STEP 6: Plot heatmaps of causality matrices for the patient-specific files
    # OLD: plot_heatmaps(L_range[4], E_range, tau_range, output_filename_c, limit_channels, ref_mx, 'Inter-ictal')
    # OLD: plot_heatmaps(L_range[4], E_range, tau_range, output_filename_pre, limit_channels, ref_mx, 'Pre-ictal')
    # OLD: plot_heatmaps(L_range[4], E_range, tau_range, output_filename_ic, limit_channels, ref_mx, 'Ictal')  
    plot_heatmap(opt_L, opt_E, opt_tau, output_filename_c, limit_channels, 'Inter-ictal') 
    plot_heatmap(opt_L, opt_E, opt_tau, output_filename_pre, limit_channels, 'Pre-ictal')
    plot_heatmap(opt_L, opt_E, opt_tau, output_filename_ic, limit_channels, 'Ictal')
    
    # STEP 7: Compute metrics of causality matrix with fixed params for the patient-specific files
    # compute_metrics(output_filename_ic, ref_mx, L_range[4], E_range, tau_range)
    # compute_metrics(output_filename_pre, ref_mx, L_range[4], E_range, tau_range)
    # compute_metrics(output_filename_c, ref_mx, L_range[4], E_range, tau_range)
    # compute_asymm_idx(subject, [output_filename_c, output_filename_pre, output_filename_ic], opt_L, opt_E, opt_tau)
    
    # STEP 8: Compute outflow for channels for fixed parameters
    # compute_high_outflow(output_filename_c, L_range[4], E_range[1], tau_range[1], limit_channels)
    # compute_high_outflow(output_filename_ic, L_range[4], E_range[1], tau_range[1], limit_channels)
    # compute_high_outflow(output_filename_pre, L_range[4], E_range[1], tau_range[1], limit_channels)
    
    # STEP 9: Box plots across states for each parameter set
    # plot_boxplots(output_filename_c, output_filename_ic, output_filename_pre, output_dir_subj, L_range[4], E_range, tau_range)
            
    # OLD: test over window
    # OLD: compute_ccm_over_window(limit_channels, X_ps, output_dir_subj)

def eval_subjects(subjects):
    
    limit_channels = [1, 4, 5, 7, 8, 9, 12, 13, 18, 21]
    
    # Parameters range
    L_range = [6000, 7000, 8000, 9000, 10000]
    E_range = [2,3, 4, 5]
    tau_range=[1,2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Observed optimal parameter values 
    opt_L = L_range[4]
    opt_tau = tau_range[3]
    opt_E = E_range[2]
    
    asymm_subject = {}
    
    for subject in subjects:
        
        # Output paths
        output_dir_subj = output_dir + '/' + subject
        os.makedirs(output_dir_subj, exist_ok=True)
        output_filename_c=output_dir_subj+"/control-file"
        output_filename_ic = output_dir_subj + '/patient-ictal-file'
        output_filename_pre = output_dir_subj + '/patient-pre-ictal-file'
        
        asymm_subject.append(compute_asymm_idx(subject, [output_filename_c, output_filename_pre, output_filename_ic], opt_L, opt_E, opt_tau))
        compute_high_outflow(output_dir_subj, [output_filename_c, output_filename_pre, output_filename_ic], opt_L, opt_E, opt_tau, limit_channels)
        plot_boxplot(output_filename_c, output_filename_ic, output_filename_pre, output_dir_subj, opt_L, opt_E, opt_tau)
        
        # Compute frequency of top 3 of each state
        
    df = pd.DataFrame(asymm_subject)
    df.to_excel(f'{output_dir}/Overall-asymmetry-index.xlsx')
        

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

list_subjects = [f"chb{str(i).zfill(2)}" for i in range(2, 25)]
num_cores = mp.cpu_count()
args_list = [(subject, proc_data_dir, output_dir) for subject in list_subjects]

# ccm_subject("chb01", proc_data_dir, output_dir)

pool = mp.Pool(8)
results = pool.starmap(ccm_subject,args_list)
