import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fooof import FOOOF, FOOOFGroup
import time

# Temporary test file
file = 'chb01_16.edf'

# Loading the datafile
raw_edf = mne.io.read_raw_edf(file, preload = True)

# Querying the raw object through info object
print("----- DATA INFO ----\n"+ str(raw_edf.info))

# Extracting data
data = raw_edf.get_data()
print(f"Shape of data: (num_channels, num_samples) = {data.shape}")
print("----- DATA AS ARRAY ----\n" + str(data))

# Saving data
print("----- SAVING DATA ----\n")
np.save(file="my_data.npy", arr=data)

# Extract annotations
print("----- EXTRACT ANNOTATIONS ----\n")
seizure_onsets = [] 
seizure_offsets = []
seizure_descriptions = []
findFile = False

with open("chb01-summary.txt", "r") as f:
    for l in f:
        if l.strip() == "File Name: chb01_16.edf": 
            findFile = True
            continue
        if findFile:
            if l.strip() == '': break;    
            if l.startswith("Seizure Start Time:"):
                seizure_onsets.append(float(l.split(":")[1].strip().split(" ")[0]))
            if l.startswith("Seizure End Time:"):
                seizure_offsets.append(float(l.split(":")[1].strip().split(" ")[0]))
    
seizure_durations = [offset-onset for onset, offset in zip(seizure_onsets, seizure_offsets)]
seizure_descriptions = [("seizure_"+ str((i+1))) for i in range(len(seizure_onsets))]

# Creating annotation object
annots = mne.Annotations(onset=seizure_onsets, duration=seizure_durations, description=seizure_descriptions)

# Saving annotation object for data
raw_edf.set_annotations(annots)
print(annots)
print(f"Seizure event: {annots.onset[0], annots.duration[0]}")
annots.save("annotations.csv", overwrite=True)

# Extract events from annotations
print("----- EXTRACT EVENTS ----\n")
events, event_id = mne.events_from_annotations(raw_edf)
print(events)

# Band-pass Filtering
print("----- FILTERING ----\n")
unfiltered_edf = raw_edf.copy()
raw_edf.filter(l_freq=0.5, h_freq=40)

# Segmenting into epochs
print("----- SEGMENTING ----\n")
# epochs = mne.Epochs(raw_edf, events, tmin=0, tmax=2.0, preload=True, baseline=(0,0))
# epochs = mne.Epochs(raw_edf, events, event_id, tmin=0, tmax=2, baseline=None, preload=True)
epochs = mne.make_fixed_length_epochs(raw_edf, duration=2.0, preload=True)
index = int(annots.onset[0] // 2)
epochs[index:].plot(n_epochs=5, scalings=dict(eeg=10e-5), events=True)

# Normalization
print("----- NORMALIZATION ----\n")
def normalization(data):
    datapoints = data.size
    mean = sum(data)/datapoints
    variance = 0
    for n in data:
        variance += (n-mean)*(n-mean)
    variance /= datapoints
    std = np.sqrt(variance)
    for x in range(datapoints):
        data[x] = (data[x]-mean)/std
    print('done')
    return data

normalized_epochs = epochs.copy()
normalized_epochs.apply_function(normalization, 'all')

# Computing basic features
print("----- FEATURE EXTRACTION ----\n")
def compute_features(data):
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

# feature_matrix = compute_features(normalized_epochs)

# Compute Power Spectral Density
epoch_spectrum = normalized_epochs.compute_psd(method="multitaper")
psds, freqs = epoch_spectrum.get_data(return_freqs=True)

# Compute exponent of aperiodic component
def compute_aperiodic_slope(freqs, psds):
    # fm = FOOOF()
    # features = []
    # # freq_range=[1, 40]
    # for epoch in range(psds.shape[0]):
    #     print(f"Epoch {epoch} gives exps:")
    #     channel_features=[]
    #     for channel in range(psds.shape[1]):
    #         fm.fit(freqs, psds[epoch, channel,:])
    #         exp = fm.get_params('aperiodic_params', 'exponent')
    #         # print(f"Epoch {epoch} gives exps:{exp}")
    #         channel_features.append(exp)
    #     features.append(channel_features)
    # return np.array(features)
    
    start = time.time()
    fm = FOOOFGroup()
    psds_reshape = psds.reshape(-1, psds.shape[-1])
    fm.fit(freqs, psds_reshape)
    exps = fm.get_params('aperiodic_params', 'exponent')
    exps_reshape = exps.reshape(psds.shape[:-1])
    end = time.time()
    print(f"Time for computing exponent: {end - start:.4f} seconds\n")
    return exps_reshape

if os.path.exists("aperiodic_exps.npy"):
    lambda_matrix = np.load("aperiodic_exps.npy")
else:
    lambda_matrix = compute_aperiodic_slope(freqs, psds)
    np.save("aperiodic_exps.npy", lambda_matrix)
    
# Display aperiodic slope
_, num_channels = lambda_matrix.shape
col = [f"Ch_{i+1}" for i in range(num_channels)]
lambda_matrix_pd = pd.DataFrame(lambda_matrix, columns = col)
print(f"Aperiodic exponent:\n{lambda_matrix_pd}")

plt.figure(figsize=(10, 6))
for ch in range(lambda_matrix.shape[1]):
    plt.plot(range(index,index+30), lambda_matrix[index:index+30, ch], label=f'Ch {ch+1}')
plt.xlabel("Epoch")
plt.ylabel("Aperiodic exp")
plt.title("Aperiodic exp over epochs")
plt.legend()
plt.show()

# Plot the data
print("----- PLOTTING DATA ----\n")
normalized_epochs[index:].plot(block=True, n_epochs=5, scalings=dict(eeg=1), events=True)
unfiltered_edf.plot(block=True, scalings=dict(eeg=10e-5), start=annots.onset[0], duration=annots.duration[0])
raw_edf.plot(block=True, scalings=dict(eeg=10e-5), start=annots.onset[0], duration=annots.duration[0])
