# Author @Tenzin Sangpo Choedon

import logging
import os
from pathlib import Path
import random
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

def plot_overall_convergence(output_filename, L_range, E_range, tau_range):
    X = np.load(output_filename+'.npz')
    
    plt.figure(figsize=(9,5))
    
    style_X = dict(linestyle='-',   marker='o', markersize=4, linewidth=1.5)
    style_Y = dict(linestyle='--',  marker='s', markersize=4, linewidth=1.5)
        
    for E in E_range:
        for tau in tau_range:
            
            print(f"The E: {E}")
            print(f"The tau: {tau}")
            Xhat_My, Yhat_Mx = [], [] # correlation list
            
            for L in L_range:
                print(f"The L: {L}")
                data = X[f'L{L}_E{E}_tau{tau}']
                corrX_Y = np.mean(data[np.triu_indices_from(data, k=1)])
                corrY_X  = np.mean(data[np.tril_indices_from(data, k=-1)])
                # ccm_XY = ccm(X, Y, tau, E, L) # define new ccm object # Testing for X -> Y
                # ccm_YX = ccm(Y, X, tau, E, L) # define new ccm object # Testing for Y -> X    
                # Xhat_My.append(ccm_XY.causality()[0]) 
                # Yhat_Mx.append(ccm_YX.causality()[0]) 
                Xhat_My.append(corrX_Y)
                Yhat_Mx.append(corrY_X)
            
            plt.plot(L_range, Xhat_My, label='$\hat{X}(t)|M_y$'+f'_E{E}_tau{tau}', **style_X)
            plt.plot(L_range, Yhat_Mx, label='$\hat{Y}(t)|M_x$'+f'_E{E}_tau{tau}', **style_Y)
            print('a plot done')
            
    plt.xlabel('L', size=12)
    plt.ylabel('correl', size=12)
    plt.legend(prop={'size': 16}, loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(output_filename+f'_overall-convergence.png')
    plt.close()

def plot_autocorrelation(output_filename, L, E_range, tau_range):
    # X = np.load(output_filename+'.npz')
    # c_vals=[]
    # for tau in tau_range:
    #     c_mx = X[f'L{L}_E{E}_tau{tau}']
    #     c_vals.append(c_mx[0][1])
    # auto_corr = acf(c_vals, nlags=len(c_vals)-1)
    # plt.stem(range(len(auto_corr)), auto_corr, basefmt="")
    # plt.title('Autocorrelation of causality values')
    # plt.xlabel('Lag')
    # plt.ylabel('Autocorrelation')
    # plt.savefig(output_filename+f'_autocorrelation.png')
    # plt.close()        
    
    X = np.load(output_filename+'.npz')
    plt.figure(figsize=(9,5))
    
    style_X = dict(linestyle='-',   marker='o', markersize=4, linewidth=1.5)
    style_Y = dict(linestyle='--',  marker='s', markersize=4, linewidth=1.5)
        
    for E in E_range:
        Xhat_My, Yhat_Mx = [], [] # correlation list
        for tau in tau_range:                
            data = X[f'L{L}_E{E}_tau{tau}']
            corrX_Y = np.mean(data[np.triu_indices_from(data, k=1)])
            corrY_X  = np.mean(data[np.tril_indices_from(data, k=-1)])
            Xhat_My.append(corrX_Y)
            Yhat_Mx.append(corrY_X)
        auto_corrX_Y = acf(Xhat_My, nlags=len(Xhat_My)-1)
        auto_corrY_X = acf(Yhat_Mx, nlags=len(Yhat_Mx)-1)
        plt.plot(tau_range, auto_corrX_Y, label='$\hat{X}(t)|M_y$'+f'_L{L}_E{E}', **style_X)
        plt.plot(tau_range, auto_corrY_X, label='$\hat{Y}(t)|M_x$'+f'_L{L}_E{E}', **style_Y)
        print(f'plot for E={E} done')
            
    plt.title('Autocorrelation of causality values')
    plt.xlabel('Time lag')
    plt.ylabel('Autocorrelation')
    plt.legend(prop={'size': 16}, loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(output_filename+f'_autocorrelation.png')
    plt.close()    

def plot_heatmap(matrix, filename, limit_channels, title):
    
    plt.figure(figsize=(10, 8))
    channel_arr = [f"Ch{i}" for i in limit_channels]
    sns.heatmap(matrix, annot=True, cmap="coolwarm", square=True, vmin=0,vmax=1, xticklabels=channel_arr, yticklabels=channel_arr)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_heatmaps(L, E_range, tau_range, output_filename, limit_channels, C, type):
    X = np.load(output_filename+'.npz')
    
    nrows = len(E_range)
    ncols = len(tau_range)    
    fig, axes = plt.subplots(nrows, ncols, figsize=(len(limit_channels)*ncols, len(limit_channels)*nrows))
    channel_arr = [f"Ch{i}" for i in limit_channels]
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
    
    for i, E in enumerate(E_range):
        for j, tau in enumerate(tau_range):
            # plot_heatmap(data, output_filename+f'-L{L}_E{E}_tau{tau}-heatmap.png', limit_channels, 'Correl heatmap')
            # plot_heatmap(data-C, output_filename+f'-L{L}_E{E}_tau{tau}-heatmap.png', limit_channels, '(correl - reference) heatmap')
            
            data = X[f'L{L}_E{E}_tau{tau}']
            # sns.heatmap(data-C, annot=True, cmap="coolwarm", square=True,vmin=-1,vmax=1,xticklabels=channel_arr, yticklabels=channel_arr, ax=axes[i][j], cbar=(i == 0 and j == 0), cbar_ax=cbar_ax if (i==0 and j==0) else None)
            sns.heatmap(data, annot=True, cmap="coolwarm", square=True,vmin=0,vmax=1,xticklabels=channel_arr, yticklabels=channel_arr, ax=axes[i][j], cbar=(i == 0 and j == 0), cbar_ax=cbar_ax if (i==0 and j==0) else None)
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

def compute_high_outflow(output_filename, L, E, tau, limit_channels):
    X = np.load(output_filename+'.npz')
    
    data = X[f'L{L}_E{E}_tau{tau}']
    outflow_row = []
    
    # FIX: correct outflow?
    for i in range(len(limit_channels)):
        outflow_row.append({'channel': limit_channels[i], 'outflow': np.sum((data[i, :]))})

    df = pd.DataFrame(outflow_row)
    with open(f'{output_filename}-channel-outflows.md', 'w') as f:
        f.write(df.to_markdown(index=False))
    print(f' Outflow L{L}_E{E}_tau{tau} for {output_filename}\n')

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

def ccm_subject(subject, proc_data_dir, output_dir):
    
    print(f"Starting subject {subject}")
    limit_channels = [1, 4, 5, 7, 8, 9, 12, 13, 18, 21]
    
    # Output paths
    output_dir_subj = output_dir + '/' + subject
    os.makedirs(output_dir_subj, exist_ok=True)
    output_filename_c=output_dir_subj+"/control-file"
    output_filename_ic = output_dir_subj + '/patient-ictal-file'
    output_filename_pre = output_dir_subj + '/patient-pre-ictal-file'
        
    # Parameters range
    L_range = [6000, 7000, 8000, 9000, 10000]
    E_range = [2,3]
    tau_range=[1,2]
    
    # STEP 1: Plot overall convergence of control file to determine L
    plot_overall_convergence(output_filename_c, L_range, E_range, tau_range)
    
    # STEP 2: Plot autocorrelation of time lagged control time series to determine tau
    plot_autocorrelation(output_filename_c, L_range[4], E_range, tau_range)
    
    # STEP 3: Compute markovs process to determine E
    
    # OLD: Compute the reference which is the inter-ictal average matrix for the selected L
    # ref_mx = compute_reference_matrix(output_filename_c, L_range[4], E_range, tau_range, limit_channels)
    
    # STEP 4: Plot heatmaps for the causality matrix with optimal param set of all files
    # plot_heatmaps(L_range[4], [E_range[0]], [tau_range[0]], output_filename_c, limit_channels, ref_mx, 'Inter-ictal')
    # plot_heatmaps(L_range[4], [E_range[0]], [tau_range[0]], output_filename_ic, limit_channels, ref_mx, 'Ictal')
    # plot_heatmaps(L_range[4], [E_range[0]], [tau_range[0]], output_filename_pre, limit_channels, ref_mx, 'Pre-ictal')
    
    # STEP 5: Compute asymmetry index (metrics) of causality matrix with optimal param set for files
    # compute_metrics(output_filename_ic, ref_mx, L_range[4], [E_range[0]], [tau_range[0]])
    # compute_metrics(output_filename_pre, ref_mx, L_range[4], [E_range[0]], [tau_range[0]])
    # compute_metrics(output_filename_c, ref_mx, L_range[4], [E_range[0]], [tau_range[0]])
    
    # STEP 6: Compute outflow for channels of causality matrix with optimal param set for files
    # compute_high_outflow(output_filename_c, L_range[4], [E_range[0]], [tau_range[0]], limit_channels)
    # compute_high_outflow(output_filename_ic, L_range[4], [E_range[0]], [tau_range[0]], limit_channels)
    # compute_high_outflow(output_filename_pre, L_range[4], [E_range[0]], [tau_range[0]], limit_channels)
    
    # STEP 7: Box plots across states for optimal parameter set
    # plot_boxplots(output_filename_c, output_filename_ic, output_filename_pre, output_dir_subj, L_range[4], [E_range[0]], [tau_range[0]])
    

# Get the parent directory
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)

# Define the relative paths
proc_data_dir = os.path.join(parent_dir, 'processed_data')
output_dir = os.path.join(parent_dir, 'output_data')
os.makedirs(output_dir, exist_ok=True)

list_subjects = [f"chb{str(i).zfill(2)}" for i in range(1, 25)]
num_cores = mp.cpu_count()
args_list = [(subject, proc_data_dir, output_dir) for subject in list_subjects]

ccm_subject("chb01", proc_data_dir, output_dir)

# pool = mp.Pool(8)
# results = pool.starmap(ccm_subject,args_ses_list)
# pool.close()
# pool.join()
