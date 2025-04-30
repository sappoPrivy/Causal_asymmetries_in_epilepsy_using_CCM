# Author @Tenzin Sangpo Choedon

from functools import partial
import logging
import os
import random
import re
import sys
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

class Subject:
    def __init__(self, subject, data_dir):
        self.subject = subject
        self.data_dir = data_dir
        self.control_file = ""
        self.seizure_samples = []

    # BEGIN WITH FIXING THIS
    def process_samples(self):
        print("----- EXTRACT ANNOTATIONS ----\n")
        findFile = False

        with open(os.path.join(data_dir, self.subject, self.subject+"-summary.txt"), "r") as f:
            for l in f:
                if l.strip().split(": ")[0] == "File Name": 
                    filename = l.strip().split(": ")[1]
                    print(filename)
                    findFile = True
                    num_seizure = 0
                    seizure_onsets = [] 
                    seizure_offsets = []
                    continue
                if findFile:
                    if l.strip() == '': 
                        
                        if num_seizure == 0:
                            if not self.control_file:
                                self.control_file = filename
                                print(f"Controle file: {filename}")
                        else:                          
                            seizure_sample = {
                                "filename": filename,
                                "num_seizures": num_seizure,
                                "onsets": seizure_onsets,
                                "durations":[offset-onset for onset, offset in zip(seizure_onsets, seizure_offsets)],
                                "descriptions": [("seizure_"+ str((i+1))) for i in range(len(seizure_onsets))]
                            }
                            print(seizure_sample)
                            self.seizure_samples.append(seizure_sample)
                        findFile= False;    
                    if l.strip().split(": ")[0] == "Number of Seizures in File":
                        num_seizure = int(l.strip().split(": ")[1])
                    if l.startswith("Seizure") and "Start Time:" in l:
                        seizure_onsets.append(float(l.split(":")[1].strip().split(" ")[0]))
                    if l.startswith("Seizure") and "End Time:" in l:
                        seizure_offsets.append(float(l.split(":")[1].strip().split(" ")[0]))
        

def preprocess_subject(subject, data_dir, proc_data_dir):
    logging.debug(f"Starting subject {subject}")
    
    num_ses=0
    num_pre_ses=0
    
    # Creating the subject for the preprocessing
    s0 = Subject(subject, data_dir)
    
    # Preparing subject for segmenting into seizure, pre-seizure samples
    s0.process_samples()
    
    # Output dirs
    proc_dir_subj = proc_data_dir + '/' + subject
    os.makedirs(proc_dir_subj, exist_ok=True)
    
    try:
        # Load and preprocess the non-seizure sample
        control_data = mne.io.read_raw_edf(os.path.join(data_dir,subject, s0.control_file), preload = True)
        control_data.filter(l_freq=0.5, h_freq=40)
        proc_filename_c = proc_dir_subj + '/control-data'
        np.savez_compressed(file=proc_filename_c, arr=control_data.get_data())
    
    except Exception:
        print("ERROR: no control file")
    
    # Running through all the files containing seizures
    for sample in s0.seizure_samples:
                
        # Loading the file containing seizure
        patient_data = mne.io.read_raw_edf(os.path.join(data_dir,subject, sample['filename']), preload = True)
        
        patient_data.filter(l_freq=0.5, h_freq=40)
        sfreq = patient_data.info['sfreq']
                
        # Loop through each seizure and crop the data
        for i in range(sample['num_seizures']):
            # Convert annotation start and stop time (in seconds)
            start_time = sample['onsets'][i]
            end_time = start_time + sample['durations'][i]
            
            # Crop the data around the seizure
            start_sample = int(start_time * sfreq)
            stop_sample = int(end_time * sfreq)
            
            # Crop the data around the seizure
            seizure_data = patient_data.get_data(start=start_sample, stop=stop_sample)
            
            num_ses += 1
            proc_filename = proc_dir_subj + '/ictal-data-' + str(num_ses)
            
            # Save the seizure data as a .npy file
            np.savez_compressed(file=proc_filename, arr=seizure_data)
            
            print("Saved seizure to "+proc_filename)
            
            # Crop the data around the pre seizure
            start_pre_sample = int((start_time-30) * sfreq)
            stop_pre_sample = int(start_time * sfreq)
            
            # Crop the data around the pre seizure
            pre_seizure_data = patient_data.get_data(start=start_pre_sample, stop=stop_pre_sample)
            
            num_pre_ses += 1
            proc_pre_filename = proc_dir_subj + '/pre-ictal-data-' + str(num_pre_ses)
            
            # Save the pre seizure data as a .npy file
            np.savez_compressed(file=proc_pre_filename, arr=pre_seizure_data)
            
            print("Saved pre seizure to "+proc_pre_filename)

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

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    list_subjects = [f"chb{str(i).zfill(2)}" for i in range(3, 25)]
    num_cores = mp.cpu_count()
    args_list = [(subject, data_dir, proc_data_dir) for subject in list_subjects]
    
    # Not correctly working
    # with mp.Pool(num_cores//2, maxtasksperchild=1) as pool:
    #     pool.starmap(preprocess_subject, args_list, chunksize=1)
    
    preprocess_subject("chb24",data_dir, proc_data_dir)
    
    # for args in args_list:
    #     preprocess_subject(*args)
