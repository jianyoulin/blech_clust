"""
need to run nosepoke_emg_make_array.py first which save data in HDF5 
to be used for plotting here

"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

import tables
import easygui
import sys
import os
import shutil

from scipy.signal import butter, filtfilt, periodogram
from scipy.signal import find_peaks
from numpy.fft import fft, rfft
from scipy.signal import spectrogram

###
# Define util functions
def filt_emg(data, low=15.0, high=300.0):
    """
    data: 1-d array
    low: lowpass frequency
    high: highpass frequency
    """
    # Get coefficients for Butterworth filters
    m, n = butter(2, 2.0*high/1000.0, 'highpass')
    c, d = butter(2, 2.0*low/1000.0, 'lowpass')
    emg_filt = filtfilt(m, n, data)
    env = filtfilt(c, d, np.abs(emg_filt))
    return emg_filt, env
###

# get path for blech_clust folder
blech_clust_dir = os.getcwd()

# Get name of directory with the data files
try:
    dir_name = sys.argv[1]
    if dir_name == '-f':
        dir_name = easygui.diropenbox('Select the dir path where data are saved')
except:
    dir_name = easygui.diropenbox('Select the dir path where data are saved')

# for testing, comment out is not in testing mode
dir_name = 'G:\\in_vivo_recordings_licking\\read_digitalins_cue_nosepoke_touch\\TG43_30min_30Strials_2S8p6S30p_240528_092631\\temp_files'

# Change to that directory
os.chdir(dir_name)

# Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for file in file_list:
    if file[-2:] == 'h5':
        hdf5_name = file

# open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')
# read in nosepoke and emg signals
npokes_dig_in = hf5.list_nodes('/nosepoke_lengths')
# read in emgs
emgs_dig_in = hf5.list_nodes('/emg_data') # # of emg channels

# make folders to save EMG-related figures
# Make directory to store the raster plots. Delete and remake the directory if it exists
try:
    shutil.rmtree("emgs")
except:
	pass
os.mkdir("emgs")

n_emg_chs = len(emgs_dig_in)
emgs = [hf5.list_nodes(f'/emg_data/emg{i}') for i in range(n_emg_chs)]
n_rows = n_emg_chs + 3 if n_emg_chs > 1 else 3
colors = ['r', 'g', 'b']
labels = ['CH1', 'CH2', 'CH2-CH1']
for taste in range(len(npokes_dig_in)): # loops through tastes
    for trial in range(npokes_dig_in[taste].shape[0]): # loop through trials
        fig, axes = plt.subplots(n_rows, 1, sharex=True, figsize=(20, 10))
        
        # merge emg sginals from different channel
        # add a differential emg signals if more than 1 channel used
        emg_merged = []
        for j, emg in enumerate(emgs):
            emg_merged.append(emg[taste][trial,:])
        emg_merged.append(emg_merged[0] - emg_merged[1])
        
        for j, emg in enumerate(emg_merged):
            emg_filt, env = filt_emg(emg) #[taste][trial,:])
            pokes = npokes_dig_in[taste][trial,:]
            t_min = min(len(env), len(pokes))
            t = np.arange(t_min)
            peaks, _ = find_peaks(env, distance=100, height=0.2*np.ptp(env))
            # distance in find_peaks function as the minimal samples between 2 peaks
            color_ = colors[j]
            axes[j].plot(t, emg_filt[:t_min], c = color_, label = f'Filtered EMG ({labels[j]})', alpha=0.5)
            if j == 0:
                axes[n_rows-2].plot(t, pokes[:t_min], label = 'Nose Poke')
            axes[n_rows-1].plot(t, env[:t_min], c =color_, label = f'EMG Envelope ({labels[j]})', alpha=0.8)
            axes[n_rows-1].plot(peaks, env[peaks], "x", c =color_)
            axes[n_rows-2].scatter(peaks, pokes[peaks]+j*0.05, s = 10, c =color_, marker='o', alpha=0.8)
        for i in range(n_rows): 
            axes[i].legend(loc = 'best')
            # axes[i].set_xlim(5000, 10000)
        plt.tight_layout()
        fig.suptitle(f'{taste = } -- {trial = }')
        fig.savefig(f'./emgs/taste{taste}_Trial{trial}.png')
        plt.close("all")
    
# Plotting frequency power over time
# 1) do auto-correlation first to see if there is any periodic activity
for taste in range(len(npokes_dig_in)): # loops through tastes
    for trial in range(npokes_dig_in[taste].shape[0]): # loop through trials
        fig, axes = plt.subplots(1, 1, sharex=True, figsize=(6, 4))
        
        # merge emg sginals from different channel
        # add a differential emg signals if more than 1 channel used
        emg_merged = []
        for j, emg in enumerate(emgs):
            emg_merged.append(emg[taste][trial,:])
        emg_merged.append(emg_merged[0] - emg_merged[1])
        
        for j, emg in enumerate(emg_merged):
            _, env = filt_emg(emg) #[taste][trial,:])
            t = np.arange(len(env))
            color_ = colors[j]
            
            N = env.shape[0]
            dt = 0.001
            lags = np.arange(-len(env) + 1, len(env))    # Compute the lags for the full autocovariance vector
                                                  # ... and the autocov for L +/- 100 indices
            ac = 1 / N * np.correlate(env - env.mean(), env - env.mean(), mode='full')
            inds = abs(lags) <= 1000               # Find the lags that are within 100 time steps
            axes.plot(lags[inds] * dt, ac[inds], c =color_, label = f'Filtered EMG ({labels[j]})')       # ... and plot them
            axes.set_xlabel('Lag [s]')                     # ... with axes labelled
            axes.set_ylabel('Autocovariance')
            axes.legend(loc='best')
            
        plt.tight_layout()
        fig.savefig(f'./emgs/taste{taste}_Trial{trial}_autocorrelation.png')
        plt.close("all")

# 2) plot frequency power over time
n_rows = n_emg_chs + 1 if n_emg_chs > 1 else n_emg_chs
for taste in range(len(npokes_dig_in)): # loops through tastes
    for trial in range(npokes_dig_in[taste].shape[0]): # loop through trials
        fig, axes = plt.subplots(n_rows, 1, sharex=True, figsize=(20, 10))
        
        # merge emg sginals from different channel
        # add a differential emg signals if more than 1 channel used
        emg_merged = []
        for j, emg in enumerate(emgs):
            emg_merged.append(emg[taste][trial,:])
        emg_merged.append(emg_merged[0] - emg_merged[1])
        
        for j, emg in enumerate(emg_merged):
            
            _, env = filt_emg(emg) #[taste][trial,:])
            t = np.arange(len(env))

            Fs = 1 / dt               # Define the sampling frequency,
            interval = int(Fs)        # ... the interval size,
            overlap = int(Fs * 0.99)  # ... and the overlap intervals

            f, t, Sxx = spectrogram(
                env,                  # Provide the signal,
                fs=Fs,                # ... the sampling frequency,
                nperseg=interval,     # ... the length of a segment,
                noverlap=overlap)     # ... the number of samples to overlap,
            
            axes[j].pcolormesh(t, f, Sxx, shading='gouraud')             
            axes[j].set_title(f'Filtered EMG ({labels[j]})', fontsize = 10)
            if j == n_rows-1:
                axes[j].set_xlabel('time [s]')                     # ... with axes labelled
            axes[j].set_ylabel('Frequency [Hz]')
            axes[j].set_ylim([0, 20])  # ... set the frequency range,
        plt.tight_layout()
        fig.savefig(f'./emgs/taste{taste}_Trial{trial}_frequency_power.png')
        plt.close("all")
hf5.close()