import os
import mne
import numpy
import matplotlib

file = 'chb01_03.edf'
raw_edf = mne.io.read_raw_edf(file, preload = True)

print(raw_edf.info)

events = mne.find_events(raw_edf, stim_channel="F7-T7", output='onset')

data = raw_edf.get_data()
print(data)

raw_edf.plot(block=True, scalings=dict(eeg=10e-5), duration=40)

# Normalization

# Filtering

# Segmentation: epochs
