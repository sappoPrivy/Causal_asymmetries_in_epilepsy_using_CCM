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

list_subjects = [f"chb{str(i).zfill(2)}" for i in [1, 2, 3,4, 5, 6, 8, 9, 10, 23]]
num_cores = mp.cpu_count()
args_list = [(subject, proc_data_dir, output_dir) for subject in list_subjects]

eval_subjects(list_subjects)
