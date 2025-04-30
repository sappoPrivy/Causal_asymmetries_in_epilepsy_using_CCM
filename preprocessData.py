from functools import partial
import logging
import os
import random
import re
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fooof import FOOOF, FOOOFGroup
import time
import multiprocessing as mp
from multiprocessing import Pool
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

class edf_data:
    def __init__(self, filename, subject, data_dir):
        self.filename = filename
        self.subject = subject
        self.data_dir = data_dir
        self.raw = mne.io.read_raw_edf(os.path.join(data_dir,subject, filename), preload = True)
        self.data = self.raw.get_data()

    def annotations(self):
        # Extract annotations
        print("----- EXTRACT ANNOTATIONS ----\n")
        seizure_onsets = [] 
        seizure_offsets = []
        seizure_descriptions = []
        findFile = False

        with open(os.path.join(data_dir, self.subject, self.subject+"-summary.txt"), "r") as f:
            for l in f:
                if l.strip() == "File Name: "+self.filename: 
                    findFile = True
                    continue
                if findFile:
                    if l.strip() == '': break;    
                    if l.startswith("Seizure") and "Start Time:" in l:
                        seizure_onsets.append(float(l.split(":")[1].strip().split(" ")[0]))
                    if l.startswith("Seizure") and "End Time:" in l:
                        seizure_offsets.append(float(l.split(":")[1].strip().split(" ")[0]))
            
        seizure_durations = [offset-onset for onset, offset in zip(seizure_onsets, seizure_offsets)]
        seizure_descriptions = [("seizure_"+ str((i+1))) for i in range(len(seizure_onsets))]

        # Creating annotation object
        annots = mne.Annotations(onset=seizure_onsets, duration=seizure_durations, description=seizure_descriptions)

        # Saving annotation object for data
        self.raw.set_annotations(annots)
        # print(annots)
        
        for i in range(len(seizure_onsets)):
            print(f"Seizure event: {annots.onset[i], annots.duration[i]}")

        # # Extract events from annotations
        # print("----- EXTRACT EVENTS ----\n")
        # events, event_id = mne.events_from_annotations(self.raw)
        # print(events)
    
    # Band-pass Filtering
    def filtering(self):    
        print("----- FILTERING ----\n")
        self.unfiltered = self.raw.copy()
        self.raw.filter(l_freq=0.5, h_freq=40)

    # Segmenting into epochs
    def segmenting(self):
        print("----- SEGMENTING ----\n")
        # epochs = mne.Epochs(raw_edf, events, tmin=0, tmax=2.0, preload=True, baseline=(0,0))
        # epochs = mne.Epochs(raw_edf, events, event_id, tmin=0, tmax=2, baseline=None, preload=True)
        self.epochs = mne.make_fixed_length_epochs(self.raw, duration=2.0, preload=True)

    # Computing basic features
    def compute_features(self, data):
        print("----- BASIC FEATURE EXTRACTION ----\n")
        features = []
        mean=0
        variance=0
        rms=0
        for epoch in data:
            channel_features=[]
            for channel in epoch:
                mean = sum(channel) / len(channel)
                for x in channel:
                    variance += (x-mean) ** 2
                    rms += x**2
                variance /= len(channel)
                std = np.sqrt(variance)
                rms = np.sqrt(rms/len(channel))
                channel_features.append([mean, variance, std, rms])
            features.append(channel_features)
        return np.array(features)

    # Compute exponent of aperiodic component
    def compute_aperiodic_slope(self, freqs, psds, fm):    
        start = time.time()
        psds_reshape = psds.reshape(-1, psds.shape[-1])
        fm.fit(freqs, psds_reshape)
        exps = fm.get_params('aperiodic_params', 'exponent')
        exps_reshape = exps.reshape(psds.shape[:-1])
        end = time.time()
        print(f"Time for computing exponent: {end - start:.4f} seconds\n")
        return exps_reshape

    # Compute Power Spectral Density
    def psd(self):
        epoch_spectrum = self.normalized.compute_psd(method="multitaper")
        psds, freqs = epoch_spectrum.get_data(return_freqs=True)
        
        fm = FOOOFGroup()

        if os.path.exists(os.path.join(dummy_data_dir,"aperiodic_exps.npy")):
            lambda_matrix = np.load(os.path.join(dummy_data_dir,"aperiodic_exps.npy"))
        else:
            lambda_matrix = self.compute_aperiodic_slope(freqs, psds, fm)
            np.save(os.path.join(dummy_data_dir,"aperiodic_exps.npy"), lambda_matrix)
            
        # Display aperiodic slope
        _, num_channels = lambda_matrix.shape
        col = [f"Ch_{i+1}" for i in range(num_channels)]
        lambda_matrix_pd = pd.DataFrame(lambda_matrix, columns = col)
        print(f"Aperiodic exponent:\n{lambda_matrix_pd}")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot all channels
        for ch in range(lambda_matrix.shape[1]):
            ax.scatter(range(len(lambda_matrix[:, ch])), lambda_matrix[:, ch], label=f'Ch {ch+1}')

        # Get first seizure onset epoch
        epoch_start = int(self.raw.annots.onset[0] // 2)
        epoch_end = int((self.raw.annots.onset[0] + self.raw.annots.duration[0]) // 2)

        ax.axvspan(epoch_start, epoch_end, color='red', alpha=0.3, label="Seizure")

        ax.axvline(x=epoch_start, color='red', linestyle='--', linewidth=2, label="Seizure Start")
        ax.axvline(x=epoch_end, color='red', linestyle='--', linewidth=2, label="Seizure End")

        # Labels and title
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Aperiodic Exponent")
        ax.set_title("Aperiodic Exponent Over Epochs")
        ax.legend()

        # ax.set_xlim(epoch_start, epoch_start + 50)
        plt.ion() 
        plt.show()
    
    def plot_edf(self):    
        # Plot the data
        print("----- PLOTTING DATA ----\n")
        self.epochs.plot(n_epochs=5, scalings=dict(eeg=10e-5), events=True)
        self.normalized.plot(block=True, n_epochs=5, scalings=dict(eeg=1), events=True)
        self.unfiltered.plot(block=True, scalings=dict(eeg=10e-5), start=self.raw.annotations.onset[0], duration=self.raw.annotations.duration[0])
        self.raw.plot(block=True, scalings=dict(eeg=10e-5), start=self.raw.annotations.onset[0], duration=self.raw.annotations.duration[0])

def plot_fooof_exponents():
    # Loading the datafile
    edf = edf_data("chb01_16.edf", "chb01", data_dir)

    # Annotations
    edf.annotations()
    # edf.raw.annotations.save(os.path.join(proc_dir_subj,filename.stem +"_annotations.csv"), overwrite=True)

    # Preprocessing
    edf.filtering()
    edf.segmenting()
    normalized = edf.epochs.copy()
    edf.normalized = normalized.apply_function(standardization, 'all')
    edf.psd()

# Normalization
def standardization(data):
    mean = np.mean(data)
    std = np.std(data)

    if std == 0: 
        return np.zeros_like(data)
    print("done")
    # Vector operations
    return (data - mean) / std 
    
def preprocess_subject(subject, data_dir, proc_data_dir):
    logging.debug(f"Starting subject {subject}")
    
    num_ses=0
    num_nonses=0
    num_pre_ses=0
    
    subject_dir = Path(data_dir + "/" + subject)
    num_files = 2
    edf_files = [f for f in os.listdir(subject_dir) if f.endswith('.edf') and os.path.isfile(os.path.join(subject_dir, f))]
    # edf_files_sorted = sorted(edf_files, key=lambda f: int(f.split('_')[1].split('.')[0]))
    # random_edf_files = random.sample(edf_files, num_files)
            
    for filename in edf_files:
                
        # Loading the datafile
        edf = edf_data(filename, subject, data_dir)
        
        # Output dirs
        proc_dir_subj = proc_data_dir + '/' + subject
        os.makedirs(proc_dir_subj, exist_ok=True)

        # Annotations
        edf.annotations()
        # edf.raw.annotations.save(os.path.join(proc_dir_subj,filename.stem +"_annotations.csv"), overwrite=True)

        # Preprocessing
        edf.filtering()
        
        # Get the sampling frequency (in Hz)
        sfreq = edf.raw.info['sfreq']
        
        if len(edf.raw.annotations.onset):
            # edf.plot_edf()
            
            # Loop through each seizure and crop the data
            for idx, ann in enumerate(edf.raw.annotations):
                # Convert annotation start and stop time (in seconds)
                start_time = ann['onset']
                end_time = start_time + ann['duration']
                
                # Crop the data around the seizure
                start_sample = int(start_time * sfreq)
                stop_sample = int(end_time * sfreq)
                
                # Crop the data around the seizure
                seizure_data = edf.raw.get_data(start=start_sample, stop=stop_sample)
                
                num_ses += 1
                proc_filename = proc_dir_subj + '/ictal-data-' + str(num_ses)
                
                # Save the seizure data as a .npy file
                np.savez_compressed(file=proc_filename, arr=seizure_data)
                
                print("Saved seizure to "+proc_filename)
                
                # Crop the data around the pre seizure
                start_pre_sample = int((start_time-30) * sfreq)
                stop_pre_sample = int(start_time * sfreq)
                
                # Crop the data around the pre seizure
                pre_seizure_data = edf.raw.get_data(start=start_pre_sample, stop=stop_pre_sample)
                
                num_pre_ses += 1
                proc_pre_filename = proc_dir_subj + '/pre-ictal-data-' + str(num_pre_ses)
                
                # Save the pre seizure data as a .npy file
                np.savez_compressed(file=proc_pre_filename, arr=pre_seizure_data)
                
                print("Saved pre seizure to "+proc_pre_filename)
            
        else:
            if filename=="chb01_01.edf":
                num_nonses += 1
                proc_filename = proc_dir_subj + '/control-data' #+ str(num_nonses)
                # Save preprocessed data
                np.savez_compressed(file=proc_filename, arr=edf.raw.get_data())
                print("Saved non seizure to "+proc_filename)

np.random.seed(1)

# Get the parent directory
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)

# Define the relative paths
proc_data_dir = os.path.join(parent_dir, 'processed_data')
dummy_data_dir = os.path.join(parent_dir, 'dummy_data')
data_dir = os.path.join(parent_dir, os.path.join('data', "chbmit-1.0.0.physionet.org"))

# Create directories
os.makedirs(proc_data_dir, exist_ok=True)
os.makedirs(dummy_data_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Temporary test file
# subject = 'chb01'
# preprocess(subject)

# All subjects
# list_subjects = [f"chb{str(i).zfill(2)}" for i in range(1, 25)]
# args_list = list_subjects
# num_cores = mp.cpu_count()
# preprocess_partial = partial(preprocess_subject, data_dir=data_dir, proc_data_dir=proc_data_dir)
# pool = mp.Pool(num_cores)
# result = pool.map(preprocess_partial, args_list)
# pool.close()
# pool.join()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    list_subjects = [f"chb{str(i).zfill(2)}" for i in range(1, 25)]
    num_cores = mp.cpu_count()
    args_list = [(subject, data_dir, proc_data_dir) for subject in list_subjects]
    
    # Not correctly working
    # with mp.Pool(num_cores//2, maxtasksperchild=1) as pool:
    #     pool.starmap(preprocess_subject, args_list, chunksize=1)
    
    # plot_fooof_exponents()
    
    preprocess_subject("chb01",data_dir, proc_data_dir)
    
    # for args in args_list:
    #     preprocess_subject(*args)
