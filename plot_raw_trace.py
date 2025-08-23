"""
This code plot raw trace of recordings from each channel
to determine which channel could be an EMG channel

May not need for this ==> Run blech_clust.py first to create the params file

"""

import matplotlib
matplotlib.use('Agg')
import os
from pathlib import Path
import tables
import numpy as np
import pandas as pd
from clustering import *
import sys
import pylab as plt
import easygui
import math
# Necessary modules to read dat files
import read_file
from write_file import make_powershell_parallel_script
from util_tools import *

# Get name of directory with the data files
try:
    dir_name = sys.argv[1]
    if dir_name == '-f':
        dir_name = easygui.diropenbox('Select the dir path where data are saved')
except:
    dir_name = easygui.diropenbox('Select the dir path where data are saved')

# dir_name = '/media/jianyoulin/Vol_A/Christina_train_day_saccharin/train1/CM53_CTATrain_h2o_sac_240925_101207/'
os.chdir(dir_name)

# list all files in the directory
file_list = os.listdir('./')

# Create hdf5 file, and make groups for raw data, raw emgs, digital outputs and digital inputs, and close
hf5 = tables.open_file(Path(dir_name).name +'.h5', 'w', title = Path(dir_name).name)
hf5.create_group('/', 'raw')
hf5.create_group('/', 'raw_emg')
hf5.create_group('/', 'digital_in')
hf5.create_group('/', 'digital_out')
hf5.close()

# Get the amplifier ports used
ports = list(set(f[4] for f in file_list if f[:3] == 'amp'))
# Sort the ports in alphabetical order
ports.sort()
print("Used Ports: {}".format(ports))

# Pull out the digital input channels used, and convert them to integers
dig_in = list(set(f[10:12] for f in file_list if f[:9] == 'board-DIN'))
for i in range(len(dig_in)):
    print(dig_in[i][:]) 
    dig_in[i] = int(dig_in[i][:])
dig_in.sort()
print("Used dig-ins: {}".format(dig_in))

# get all signal channels. then divided into electrodes and emgs
all_chs = []
for port in ports:
    port_chs = list(set(int(f[6:9]) for f in file_list if f[:5] == 'amp-{}'.format(port)))
    all_chs.append(port_chs)
sig_chs = np.concatenate(all_chs)

# Get the electrode channels on each port
e_channels = {}
for port in ports:
    e_channels[port] = list(set(int(f[6:9]) for f in file_list if f[:5] == 'amp-{}'.format(port)))

emg_port = ports[0]
emg_channels=[]

# Create arrays for each electrode
read_file.create_hdf_arrays(Path(dir_name).name+'.h5', ports, dig_in, e_channels, emg_port, emg_channels)

read_file.read_files(Path(dir_name).name+'.h5', ports, dig_in, e_channels, emg_port, emg_channels)

# Load the param values
voltage_cutoff = 3000 #float(params[4])
max_breach_rate = 2 #float(params[5])
max_secs_above_cutoff = 20 #int(params[6])
max_mean_breach_rate_persec = 40 #float(params[7])
# wf_amplitude_sd_cutoff = int(params[8])
bandpass_lower_cutoff = 300 #float(params[9])
bandpass_upper_cutoff = 3000 #float(params[10])
# spike_snapshot_before = float(params[11])
# spike_snapshot_after = float(params[12])
sampling_rate = 30000 #float(params[13])

total_chs = len(sig_chs)
print(f'Total channels: {total_chs}')
# for port in ports:
#     total_chs = total_chs + len(e_channels[port])

# Open up hdf5 file, and load this electrode number
hf5 = tables.open_file(Path(dir_name).name +'.h5', 'r')
fig_path = os.path.join(dir_name, 'raw_traces.png')

fig, axes = plt.subplots(math.ceil(total_chs/6), 6, sharey=True,
                        sharex=True, figsize=(8,8))
# make 1d for easier access
axes = np.ravel(axes)
for ch_i, ch in enumerate(sig_chs): #range(total_chs): #, electrode_num in enumerate(electrode_nums):
    print(f'Channels {ch} of {total_chs}')
    if ch in e_channels[emg_port]: #electrode_nums:
        exec("raw_el = hf5.root.raw.electrode"+str(ch)+"[:]")
    if ch in emg_channels: #emg_nums:
        emg_index = emg_channels.index(ch)
        exec("raw_el = hf5.root.raw_emg.emg"+str(emg_index)+"[:]")
    s = int(len(raw_el) * 0.3)
    e = int(s * 2)
    limited_trace = raw_el[s:e]
    filt_el = get_filtered_electrode(limited_trace, freq = [bandpass_lower_cutoff, bandpass_upper_cutoff],
                                     sampling_rate = sampling_rate)
    # Delete raw electrode recording from memory
    del raw_el
    del limited_trace

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
    axes[ch_i].plot(np.arange(test_el.shape[0]), np.mean(test_el, axis = 1))
    axes[ch_i].plot((recording_cutoff, recording_cutoff), (np.min(np.mean(test_el, axis = 1)), 
                                                    np.max(np.mean(test_el, axis = 1))), 
             'k-', linewidth = 4.0)
    # Place text in the top right corner
    axes[ch_i].text(0.98, 0.98, f'Ch {ch}',
            horizontalalignment='right',
            verticalalignment='top',
            transform=axes[ch_i].transAxes, fontsize=10)
    axes[ch_i].set_xlim(60,540)
    axes[ch_i].set_ylim(-0.2,0.2)
#    axes[i].set_xlabel('Recording time (secs)')
#    axes[i].set_ylabel('Average voltage recorded per sec (microvolts)')
axes = np.reshape(axes, (math.ceil(total_chs/6), 6))
plt.tight_layout()
fig.savefig(fig_path, bbox_inches='tight')
plt.close("all")

hf5.close()
os.remove(Path(dir_name).name +'.h5')

