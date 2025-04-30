from fooof import FOOOF, FOOOFGroup
from fooof.utils.download import load_fooof_data
from fooof.plts.annotate import plot_annotated_model
# Import the model object
from specparam import SpectralModel

# Download example data files needed for this example
freqs = load_fooof_data('freqs.npy', folder='data')
spectrum = load_fooof_data('spectrum.npy', folder='data')

# Initialize a FOOOF object
fm = FOOOF()

# Set the frequency range to fit the model
freq_range = [1, 40]

# Report: fit the model, print the resulting parameters, and plot the reconstruction
fm.fit(freqs, spectrum, freq_range)

fm.plot(plot_aperiodic=True, plt_log=False)

print(fm)

# Plot annotated model of aperiodic parameters
plot_annotated_model(fm, annotate_peaks=False, annotate_aperiodic=True, plt_log=True)

# Extract parameters, indicating sub-selections of parameters
exp = fm.get_params('aperiodic_params', 'exponent')

print(exp)

# # Plot the aperiodic fits for Group 1
# plot_aperiodic_fits(aps1, freq_range, control_offset=True)

# # Plot the aperiodic fits for both groups
# plot_aperiodic_fits([aps1, aps2], freq_range,
#                     control_offset=True, log_freqs=True,
#                     labels=labels, colors=colors)

# # Compare the aperiodic parameters between groups
# plot_aperiodic_params([aps1, aps2], labels=labels, colors=colors)