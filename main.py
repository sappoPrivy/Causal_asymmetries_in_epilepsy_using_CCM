import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

feature_matrix = compute_features(epochs)

# Display features
num_epochs, num_channels, num_features = feature_matrix.shape
m_mx = feature_matrix[:,:, 0]
v_mx = feature_matrix[:,:, 1]
std_mx = feature_matrix[:,:, 2]
rms_mx = feature_matrix[:,:, 3]

col = [f"Ch_{i+1}" for i in range(num_channels)]
feature_matrix_reshape = feature_matrix.reshape(num_epochs, -1)
mean_matrix = pd.DataFrame(m_mx, columns = col)
variance_matrix = pd.DataFrame(v_mx, columns = col)
std_matrix = pd.DataFrame(std_mx, columns = col)
rms_matrix = pd.DataFrame(rms_mx, columns = col)

print(f"MEAN:\n{mean_matrix.head()}")
print(f"VARIANCE:\n{variance_matrix.head()}")
print(f"STANDARD DEVIATION:\n{std_matrix.head()}")
print(f"RMS:\n{rms_matrix.head()}")


# Plot the data
print("----- PLOTTING DATA ----\n")
normalized_epochs[index:].plot(block=True, n_epochs=5, scalings=dict(eeg=1), events=True)
unfiltered_edf.plot(block=True, scalings=dict(eeg=10e-5), start=annots.onset[0], duration=annots.duration[0])
raw_edf.plot(block=True, scalings=dict(eeg=10e-5), start=annots.onset[0], duration=annots.duration[0])
