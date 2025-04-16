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

######## MODIFY FOLLOWING CODES #########

def plot_convergence(filename, X, Y):
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


def plot_heatmap(matrix, filename, limit_channels):
    
    plt.figure(figsize=(10, 8))
    channel_arr = [f"Ch{i}" for i in limit_channels]
    sns.heatmap(matrix, annot=True, cmap="coolwarm", square=True,xticklabels=channel_arr, yticklabels=channel_arr)
    plt.title("CCM heatmap")
    plt.tight_layout()
    plt.savefig(filename)
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

def compute_ccm(limit_channels, output_filename, X):
    start_time = time.perf_counter()
    print(f"The new L: {L}")
    print(f"The new end_index: {end_index}")
    if len(X[0,:]) >= start_index + end_index:
        
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

        X_Y_arr = np.zeros((len(Yhat_Mx_f)+1,len(Yhat_Mx_f)+1))
        for i in range(len(X_Y_arr)-1):
            for j in range(0,len(Yhat_Mx_f[i])):
                X_Y_arr[i,i+1+j] = Yhat_Mx_f[i][j]
                X_Y_arr[i+1+j,i] = Xhat_My_f[i][j]

        # Save the array
        np.savez_compressed(output_filename, X_Y_arr)
        
        plot_heatmap(ccm_correls,  output_filename+"-ccm_heatmap.png", limit_channels)
        print(f"Done with {output_filename}")
        
    end_time = time.perf_counter()

def ccm_subject(subject, proc_data_dir, output_dir):
    
    print(f"Starting subject {subject}")
    subject_dir = Path(proc_data_dir + "/" + subject)
    limit_channels = [1, 4, 5, 7, 8, 9, 12, 13, 18, 21]
    
    # Selected controle fil
    control_file = os.path.join(subject_dir,"nonses", "nonn-sample-1.npz")
    
    # Selected patient files
    patient_files = [f for f in os.listdir(os.path.join(subject_dir,"ses")) if os.path.isfile(os.path.join(subject_dir,"ses", f)) and f.split("-")[0]=="nonn"]
    # random_files = random.sample(all_files, min(n_samples, len(all_files)))
    
    # Output paths
    output_dir_subj = output_dir + '/' + subject
    os.makedirs(output_dir_subj, exist_ok=True)
    output_filename_c=output_dir_subj+"/nonn-control-file"
    output_filename_p = output_dir_subj + '/nonn-patient-file'
    
    # Load non-seizure data
    X_c = np.load(os.path.join(proc_data_dir, subject, "nonses", control_file))['arr']
    
    # Testing convergence on control file
    # for idx, ch1 in enumerate(limit_channels[:-1]):
    #     for ch2 in limit_channels[idx+1:]:
    # plot_convergence(output_filename_c+f"-small-Ch({ch1,ch2})-convergence", X_c[ch1], X_c[ch2])
    
    tot_len = 0
    
    # Compute the total length of seizures
    for filename in patient_files:        
        with np.load(os.path.join(proc_data_dir, subject, "ses", filename), mmap_mode='r') as X_p:
            tot_len += X_p['arr'].shape[1]
    
    # Combined time series matrix of seizures
    X_ps = np.zeros((23, tot_len))
    curr_len = 0
    
    # Combine the time series of seizure samples
    for filename in patient_files:
        print(f"Combining patient file: {filename}")

        # Load seizure data
        X_p = np.load(os.path.join(proc_data_dir, subject, "ses", filename))['arr']
        
        # Add seizure data
        X_ps[:, curr_len:curr_len+X_p.shape[1]] = X_p
        curr_len+=X_p.shape[1]
    
    # Length is limited by max sample size
    global end_index
      
    # Compute ccm on control file
    end_index = start_index + X_c.shape[1]//2
    compute_ccm(limit_channels,output_filename_c, X_c)
    
    # Compute the ccm on conjoined patient file
    end_index = start_index + tot_len//2
    compute_ccm(limit_channels,output_filename_p, X_ps)
    
    # test over window
    # compute_ccm_over_window(limit_channels, X_ps, output_dir_subj)


# CCM Parameters
np.random.seed(1)
L=6000      # length of time period
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

list_subjects = [f"chb{str(i).zfill(2)}" for i in range(1, 25)]
num_cores = mp.cpu_count()
args_list = [(subject, proc_data_dir, output_dir) for subject in list_subjects]

ccm_subject("chb01", proc_data_dir, output_dir)

# pool = mp.Pool(8)
# results_ses = pool.starmap(ccm1,args_ses_list)
# pool.close()
# pool.join()

# pool = mp.Pool(8)
# results_nonses = pool.starmap(ccm1,args_nonses_list)
# pool.close()
# pool.join()
