"""
This code plot raw trace of recordings from each channel
to determine which channel could be an EMG channel

"""

import matplotlib
matplotlib.use('Agg')

import shutil
import os
import tables
import numpy as np
from clustering import *
import sys
from matplotlib.backends.backend_pdf import PdfPages
import pylab as plt
import matplotlib.cm as cm
from scipy.spatial.distance import mahalanobis
import blech_waveforms_datashader
import easygui
import math

# Read blech.dir, and cd to that directory
#f = open('blech.dir', 'r')
dir_name = easygui.diropenbox(msg = 'Choose a directory with a lick file, hit cancel to stop choosing')
#for line in f.readlines():
#	dir_name.append(line)
#f.close()
os.chdir(dir_name)#[:-1])

# Get the names of all files in the current directory, and find the .params and hdf5 (.h5) file
file_list = os.listdir('./')
hdf5_name = ''
params_file = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files
	if files[-6:] == 'params':
		params_file = files

# pull channels num
channels = [f for f in file_list if f[:3] == 'amp']
electrode_nums = [i[-6:-4] for i in channels] #int(sys.argv[1]) - 1 


# Read the .params file
f = open(params_file, 'r')
params = []
for line in f.readlines():
	params.append(line)
f.close()

# Assign the parameters to variables
max_clusters = int(params[0])
num_iter = int(params[1])
thresh = float(params[2])
num_restarts = int(params[3])
voltage_cutoff = float(params[4])
max_breach_rate = float(params[5])
max_secs_above_cutoff = int(params[6])
max_mean_breach_rate_persec = float(params[7])
wf_amplitude_sd_cutoff = int(params[8])
bandpass_lower_cutoff = float(params[9])
bandpass_upper_cutoff = float(params[10])
spike_snapshot_before = float(params[11])
spike_snapshot_after = float(params[12])
sampling_rate = float(params[13])

# Open up hdf5 file, and load this electrode number
hf5 = tables.open_file(hdf5_name, 'r')


fig, axes = plt.subplots(math.ceil(len(electrode_nums)/6), 6, sharey=True,
                        sharex=True, figsize=(8,8))
# make 1d for easier access
axes = np.ravel(axes)
for i, electrode_num in enumerate(electrode_nums):
    electrode_num = int(electrode_num)
    exec("raw_el = hf5.root.raw.electrode"+str(electrode_num)+"[:]")
    filt_el = get_filtered_electrode(raw_el, freq = [bandpass_lower_cutoff, bandpass_upper_cutoff], 
                                     sampling_rate = sampling_rate)
    # Delete raw electrode recording from memory
    del raw_el

    # Calculate the 3 voltage parameters
    breach_rate = float(len(np.where(filt_el>voltage_cutoff)[0])*int(sampling_rate))/len(filt_el)
    test_el = np.reshape(filt_el[:int(sampling_rate)*int(len(filt_el)/sampling_rate)], (-1, int(sampling_rate)))
    breaches_per_sec = [len(np.where(test_el[i] > voltage_cutoff)[0]) for i in range(len(test_el))]
    breaches_per_sec = np.array(breaches_per_sec)
    secs_above_cutoff = len(np.where(breaches_per_sec > 0)[0])
    if secs_above_cutoff == 0:
    	mean_breach_rate_persec = 0
    else:
    	mean_breach_rate_persec = np.mean(breaches_per_sec[np.where(breaches_per_sec > 0)[0]])
    
    # And if they all exceed the cutoffs, assume that the headstage fell off mid-experiment
    recording_cutoff = int(len(filt_el)/sampling_rate)
    if breach_rate >= max_breach_rate and \
        secs_above_cutoff >= max_secs_above_cutoff and \
        mean_breach_rate_persec >= max_mean_breach_rate_persec:
    	# Find the first 1 second epoch where the number of cutoff breaches 
        # is higher than the maximum allowed mean breach rate 
    	recording_cutoff = np.where(breaches_per_sec > max_mean_breach_rate_persec)[0][0]

    # Dump a plot showing where the recording was cut off at
    axes[i].plot(np.arange(test_el.shape[0]), np.mean(test_el, axis = 1))
    axes[i].plot((recording_cutoff, recording_cutoff), (np.min(np.mean(test_el, axis = 1)), 
                                                    np.max(np.mean(test_el, axis = 1))), 
             'k-', linewidth = 4.0)
#    axes[i].set_xlabel('Recording time (secs)')
#    axes[i].set_ylabel('Average voltage recorded per sec (microvolts)')
axes = np.reshape(axes, (math.ceil(len(electrode_nums)/6), 6))
plt.tight_layout()
fig.savefig('raw_traces.png', bbox_inches='tight')
plt.close("all")

hf5.close()
ax[i].plot([0,1], [0, 1], c=f"C{i}")
# reshape to initial dimensions
ax = np.reshape(ax, (2, 2))

exec("raw_el = hf5.root.raw.electrode"+str(electrode_num)+"[:]")

# High bandpass filter the raw electrode recordings
filt_el = get_filtered_electrode(raw_el, freq = [bandpass_lower_cutoff, bandpass_upper_cutoff], sampling_rate = sampling_rate)

# Delete raw electrode recording from memory
del raw_el



# Dump a plot showing where the recording was cut off at
fig = plt.figure()
plt.plot(np.arange(test_el.shape[0]), np.mean(test_el, axis = 1))
plt.plot((recording_cutoff, recording_cutoff), (np.min(np.mean(test_el, axis = 1)), np.max(np.mean(test_el, axis = 1))), 'k-', linewidth = 4.0)
plt.xlabel('Recording time (secs)')
plt.ylabel('Average voltage recorded per sec (microvolts)')
plt.title('Recording cutoff time (indicated by the black horizontal line)')
fig.savefig('./Plots/%i/Plots/cutoff_time.png' % electrode_num, bbox_inches='tight')
plt.close("all")

hf5.close()
