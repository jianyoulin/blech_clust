"""
# only used for experiment with IR beambreak to detect licking  
# run this code after blech_clust.py and blech_process.py (i.e., after waveform clustering)
# this code read in digin information for cue and nose poke
# split and align "emg signals" based on the cue
# Save data into hdf5 file; nosepoke_trains, nosepoke_lengths, emg_data
"""
# Necessary python modules
import easygui
import os
import tables
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from pathlib import Path
import json
import math

# Necessary blech_clust modules
import read_file

# get path for blech_clust folder
blech_clust_dir = os.getcwd()

# Get name of directory with the data files
try:
    dir_name = sys.argv[1]
    if dir_name == '-f':
        dir_name = easygui.diropenbox('Select the dir path where data are saved')
except:
    dir_name = easygui.diropenbox('Select the dir path where data are saved')

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

# What are the signals to intan digins?
dig_in_nodes = hf5.list_nodes('/digital_in') # get number of the digins being used
digin_nums = [dig_in_nodes[i]._v_pathname.split('/')[-1].split('_')[-1] for i in range(len(dig_in_nodes))]
digin_nums = np.sort([int(i) for i in digin_nums])
#dig_in_names = [dig_in_nodes[i]._v_pathname.split('/')[-1] for i in range(len(dig_in_nodes))]
digin_signals = easygui.multenterbox(msg = 'Fill in what signals feeding into each Intan Digital Input', 
                                     fields = [f'dig_in_{i}' for i in digin_nums],
                                     values = ['LED_cue', 'Nose_pokes', 'Position2_cue', 'Position6_cue'])
print(f'{digin_nums = }')
print(f'{digin_signals = }')
# And print them to a blech_params file
f = open(Path(dir_name).name+'_digin_information.txt', 'w')
for i, j in enumerate(digin_signals):
    print(f'dig_in_{digin_nums[i]}:{j}', file=f)
f.close()

# get digins and params
for file in file_list:
    if 'digin_info' in file:
        digin_info_file = file
    if file[-4:] == 'json':
        json_file = file

# read in digin num-signal match
digin_map = {}
with open (digin_info_file, 'r') as f:
    for i in range(len(dig_in_nodes)):
        di, signal = f.readline().split(':')
        print(di, signal)
        digin_map[signal.rstrip()] = di

# Get sampling rate of recording
sampling_rate = np.fromfile('info.rhd', dtype = np.dtype('float32'))
sampling_rate = int(sampling_rate[2])   
print(f'{sampling_rate = }')

# Grab the names of the arrays containing digital inputs, and pull the data into a numpy array
#dig_in_nodes = hf5.list_nodes('/digital_in')
dig_in = [] # [cue, nosepokes, touches]
for signal, di in digin_map.items():
    print(signal, di)
    exec("dig_in.append(hf5.root.digital_in.%s[:])" % di)
dig_in = np.array(dig_in)

# Get the times for taste delivery, lick, cue - take the start of the stimulus pulse as the time of delivery
dig_on = []
for i in range(dig_in.shape[0]): # digin 9,11,12,13 are led cue, nose_pokes, position2_cue, position6_cue array
	dig_on.append(np.where(dig_in[i,:] == 1)[0])

start_points, end_points = [], [] # for each taste delivery on and off time
trial_start_points, trial_end_points = [], [] # for each trial on and off time
#on_times = dig_on[3]
for on_times in dig_on: #[:len(digin_tastes)]:
    #####
    if len(on_times) > 3:
        on_times_diff = np.diff(on_times)
        # for each burst of on and off in the digins
        starts = on_times[np.where(on_times_diff > int(sampling_rate/1000))[0]+1]
        starts = np.insert(starts, 0, on_times[0])
        ends = on_times[np.where(on_times_diff > int(sampling_rate/1000))[0]]
        ends = np.append(ends, on_times[-1])
        # on and off for each trial (with longer delays)
        t_starts = on_times[np.where(on_times_diff > 15*sampling_rate)[0]+1]
        t_starts = np.insert(t_starts, 0, on_times[0])
        t_ends = on_times[np.where(on_times_diff > 15*sampling_rate)[0]]
        t_ends = np.append(t_ends, on_times[-1])
    else:
        starts, ends = np.array([0]), np.array([0])
        t_starts, t_ends = np.array([0]), np.array([0])

    start_points.append(np.array(starts))
    end_points.append(np.array(ends))
    trial_start_points.append(np.array(t_starts)) # trial_start time
    print(trial_start_points[-1])
    trial_end_points.append(np.array(t_ends))
    print(trial_end_points[-1])


# read in experiment info
f = open(json_file)
params = json.load(f)
trial_list = params['trial_list']

max_trial_dur = params['max_lick_time'] # np.max(np.array(digin_trial_times))
# Ask the user for the pre and post stimulus durations to be pulled out, and convert to integers
duration_ = easygui.multenterbox(msg = 'What are the signal durations pre stimulus that you want to pull out', 
                                 fields = ['Pre trial (ms)', 'post trial (ms)'],
                                 values = [2000, int((max_trial_dur+2)*1000)])
pre_trial, post_trial = int(duration_[0]), int(duration_[1])


# Arange nose poke and touch signal aligning the cue delivery (trial starting) time
# Delete the response_train node in the hdf5 file if it exists, and then create it
try:
	hf5.remove_node('/nosepoke_trains', recursive = True)
except:
	pass
hf5.create_group('/', 'nosepoke_trains')

# Then create nose poke array for each trial 
# (# trials x trial duration (ms)) - use start of digital input pulse as the time of cue (trial) starts
# 11 as intaninput for nose poke IR (lick)

# which channel to be used for poke trains trial-wise  
trial_channels = []
for i, sig in enumerate(digin_signals): # sig = signals
    if 'Position' in sig:
        trial_channels.append(i)
        
def padding_zeros(list):
    """
    make aeach row the same length of columns
    taking the max number of columns
    list: a list contains lists with different lengths
    """
    lens = [len(i) for i in list]
    max_len = np.max([len(i) for i in list])
    arr = np.zeros((len(list), max_len))
    for i in range(len(list)):
        arr[i, :lens[i]] = list[i]
    return arr

spout_pokes = {}
for i in trial_channels:
    spout_pokes[f'dig_in_{digin_nums[i]}'] = []

    # digin_11, 2nd list of start_points list
    for trial in range(len(trial_start_points[i])): # number of cue presented
        cue_dur = trial_end_points[i][trial] - trial_start_points[i][trial]# in ms unit
        cue_dur = math.ceil(cue_dur/30)
        nosepokes = np.zeros((pre_trial + cue_dur))
        # Get the lick times around the start of taste delivery
        poke_times = np.where((start_points[1][:] <= trial_start_points[i][trial] + post_trial*int(sampling_rate/1000))* \
                              (start_points[1][:] >= trial_start_points[i][trial] - pre_trial*int(sampling_rate/1000)))[0]
        poke_times = start_points[1][poke_times]
        poke_times = poke_times - trial_start_points[i][trial]
        poke_times = (poke_times/int(sampling_rate/1000)).astype(int) + pre_trial
        nosepokes[poke_times] = 1
        spout_pokes[f'dig_in_{digin_nums[i]}'].append(nosepokes) 
    nosepoke_train = padding_zeros(spout_pokes[f'dig_in_{digin_nums[i]}'])    
    print(f'nosepoke array shape = {nosepoke_train.shape}')
    nosepoke_array = hf5.create_array(f'/nosepoke_trains/', f'dig_in_{digin_nums[i]}', nosepoke_train)
    hf5.flush()

# obtain durations for response 
# nose poke: how long the infared beam was interupted by tongue
# lick: how long the spout was touched by tongue
try:
	hf5.remove_node('/nosepoke_lengths', recursive = True)
except:
	pass
hf5.create_group('/', 'nosepoke_lengths')

pokes = dig_in[1, :] # nosepoke digin signal
pokes_lens = {}
for i in trial_channels: # digin channels used to trial alignment
    pokes_lens[f'dig_in_{digin_nums[i]}'] = []

    # digin_11, 2nd list of start_points list
    for trial in range(len(trial_start_points[i])): # number of cue presented
        ts = trial_start_points[i][trial] - pre_trial*int(sampling_rate/1000)
        te = trial_end_points[i][trial]
        this_trial_poke = pokes[ts:te][::30] # down-sampling to milliseconds
        pokes_lens[f'dig_in_{digin_nums[i]}'].append(this_trial_poke)
        print(f'{len(this_trial_poke) = }')
        
    nosepoke_lens = padding_zeros(pokes_lens[f'dig_in_{digin_nums[i]}'])    
    print(f'nosepoke lens array shape = {nosepoke_lens.shape}')
    nosepoke_array = hf5.create_array(f'/nosepoke_lengths/', f'dig_in_{digin_nums[i]}', nosepoke_lens)
    hf5.flush()

# And pull out emg data into this array
try:
	hf5.remove_node('/emg_data', recursive = True)
except:
	pass
hf5.create_group('/', 'emg_data')

# Grab the names of the arrays containing emg recordings
emg_nodes = hf5.list_nodes('/raw_emg')
emg_pathname = []
for node in emg_nodes:
    emg_pathname.append(node._v_pathname)
    
    
for i in range(len(emg_pathname)): # for each emg channel
    hf5.create_group('/emg_data', f'emg{i}')
    emgs = []
    exec("data = hf5.root.raw_emg.%s[:]" % emg_pathname[i].split('/')[-1])
    
    for channel in trial_channels: # for each digin trials
        emg_traces = []

        for trial in range(len(trial_start_points[channel])): # number of cue presented:
            ts = trial_start_points[channel][trial] - pre_trial*int(sampling_rate/1000)
            te = trial_end_points[channel][trial]
            # raw_emg_data = data[ts:te] #start_points[dig_in_channel_nums[j]][k]-durations[0]*30:start_points[dig_in_channel_nums[j]][k]+durations[1]*30]
            raw_emg_data = 0.195*(data[ts:te]) # *0.195 so to convert to microVolt
			# Downsample the raw data by averaging the 30 samples per millisecond, and assign to emg_data
            # emg_data[emg#, n_tastes, n_trials]
			#emg_data[i, j, k, :] = np.mean(raw_emg_data.reshape((-1, 30)), axis = 1)
            end = len(raw_emg_data) - len(raw_emg_data)%30 # 
            emg_traces.append(np.mean(raw_emg_data[:end].reshape((-1, 30)), axis = 1))
        emgs.append(padding_zeros(emg_traces))
        emg_array = hf5.create_array(f'/emg_data/emg{i}', f'dig_in_{digin_nums[channel]}', padding_zeros(emg_traces))
        print(f'{emgs[-1].shape = }') # = {np.array(emg_traces).shape}')
    
        hf5.flush()
    
hf5.close()


            
# Save the emg_data
# np.save('emg_data.npy', emg_data)


# for i, trial in enumerate(trial_list): #range(len(trial_start_points[0])): # number of cue presented
#     # Get the lick times around the start of taste delivery
#     trial_type = 1 if trial == '2' else 2
#     ts = trial_start_points[trial_type][i//2] - pre_trial*int(sampling_rate/1000)
#     te = trial_start_points[trial_type][i//2] + post_trial*int(sampling_rate/1000)
#     # for nose_poke
#     this_trial_poke = pokes[ts:te][::30] # down-sampling to milliseconds
#     pokes_lens[trial].append(this_trial_poke)

# print(f'nosepoke array shape = {np.array(touch_lens).shape}')
# nosepoke_lengths = hf5.create_array(f'/nosepoke_trains/', 'nosepoke_lengths', np.array(nosepoke_lens))
# hf5.flush()


# # first split nose poke by spout_position cue                                                                
# nosepoke_train = [] # digin_11, 2nd list of start_points list
# for trial in range(len(trial_start_points[0])): # number of cue presented
#     nosepokes = np.zeros((pre_trial + post_trial))
#     # Get the lick times around the start of taste delivery
#     poke_times = np.where((start_points[1][:] <= trial_start_points[0][trial] + post_trial*int(sampling_rate/1000))* \
#                           (start_points[1][:] >= trial_start_points[0][trial] - pre_trial*int(sampling_rate/1000)))[0]
#     poke_times = start_points[1][poke_times]
#     poke_times = poke_times - trial_start_points[0][trial]
#     poke_times = (poke_times/int(sampling_rate/1000)).astype(int) + pre_trial
#     nosepokes[poke_times] = 1
#     nosepoke_train.append(nosepokes)
# print(f'nosepoke array shape = {np.array(nosepoke_train).shape}')
# nosepoke_array = hf5.create_array(f'/response_train/', 'nose_pokes', np.array(nosepoke_train))
# hf5.flush()

# # Then create touch sensor array for each trial 
# # (# trials x trial duration (ms)) - use start of digital input pulse as the time of cue (trial) starts
# # 12 as intaninput for touch sensor
# touch_train = [] # digin_11, 3rd list of start_points list
# for trial in range(len(trial_start_points[0])): # number of cue presented
#     touches = np.zeros((pre_trial + post_trial))
#     # Get the lick times around the start of taste delivery
#     touch_times = np.where((start_points[2][:] <= trial_start_points[0][trial] + post_trial*int(sampling_rate/1000))* \
#                           (start_points[2][:] >= trial_start_points[0][trial] - pre_trial*int(sampling_rate/1000)))[0]
#     touch_times = start_points[2][touch_times]
#     touch_times = touch_times - trial_start_points[0][trial]
#     touch_times = (touch_times/int(sampling_rate/1000)).astype(int) + pre_trial
#     touches[touch_times] = 1
#     touch_train.append(touches)
# print(f'touch array shape = {np.array(touch_train).shape}')
# touch_array = hf5.create_array(f'/response_train/', 'touch_signals', np.array(touch_train))
# hf5.flush()







# # Get the names of all files in this directory
# file_list = os.listdir('./')

# # Create hdf5 file, and make groups for raw data, raw emgs, digital outputs and digital inputs, and close
# # hf5 = tables.open_file(hdf5_name[-1]+'.h5', 'w', title = hdf5_name[-1])
# hf5 = tables.open_file(Path(dir_name).name +'_digins.h5', 'w', title = Path(dir_name).name)
# hf5.create_group('/', 'digital_in')
# hf5.close()

# # Pull out the digital input channels used, and convert them to integers
# dig_in = list(set(f[10:12] for f in file_list if f[:9] == 'board-DIN'))
# for i in range(len(dig_in)):
#     print(dig_in[i][:]) 
#     dig_in[i] = int(dig_in[i][:])
# dig_in.sort()
# print("Used dig-ins: {}".format(dig_in))

# # What are the signals to intan digins?
# digin_signals = easygui.multenterbox(msg = 'Fill in what signals feeding into each Intan Digital Input', 
#                                      fields = [f'dig_in_{i}' for i in dig_in],
#                                      values = ['LED_cue', 'Nose_pokes', 'Touches'])

# # Read the amplifier sampling rate from info.rhd - look at Intan's website for structure of header files
# sampling_rate = np.fromfile('info.rhd', dtype = np.dtype('float32'))
# sampling_rate = int(sampling_rate[2])   

# # Check with user to see if the right inputs and sampling rate were identified. 
# # Screw user if something was wrong, and terminate reading information
# check = easygui.ynbox(msg = 'Sampling rate: ' + str(sampling_rate) + ' Hz' + '\n' + 'Digital inputs on Intan board: ' + str(dig_in), title = 'Check parameters from your recordings!')

# # Go ahead only if the user approves by saying yes
# if check:
#     pass
# else:
#     print("Well, if you don't agree, blech_clust can't do much!")
#     sys.exit()

# # Create arrays for each electrode
# read_file.create_hdf_digin_arrays(Path(dir_name).name+'_digins.h5', dig_in)

# # Read data files, and append to electrode arrays
# read_file.read_digin_files(Path(dir_name).name+'_digins.h5', dig_in)

# # And print them to a blech_params file
# f = open(Path(dir_name).name+'_digin_information.txt', 'w')
# for i, j in enumerate(digin_signals):
#     print(f'dig_in_{dig_in[i]}: {j}', file=f)
# f.close()

# print("Digin information has been read and saved in {}".format(Path(dir_name).name +'_digins.h5'))


# # align nose poke and toches to trial start (indicated by cue start)
# # Ask for the directory where the hdf5 file sits, and change to that directory
# dir_name = easygui.diropenbox()
# os.chdir(dir_name)




# nosepoke2_lens, touch_lens = [], [] # nosepoke length
# pokes = dig_in[1, :]

# # touches = dig_in[2, :] # touch sensor input
# for trial in range(len(trial_start_points[0])): # number of cue presented
#     # Get the lick times around the start of taste delivery
#     ts = trial_start_points[0][trial] - pre_trial*int(sampling_rate/1000)
#     te = trial_start_points[0][trial] + post_trial*int(sampling_rate/1000)
#     # for nose_poke
#     this_trial_poke = pokes[ts:te][::30] # down-sampling to milliseconds
#     nosepoke_lens.append(this_trial_poke)
#     # for touch length
#     this_trial_touch = touches[ts:te][::30] # down-sampling to milliseconds
#     touch_lens.append(this_trial_touch)    

# print(f'nosepoke array shape = {np.array(touch_lens).shape}')
# nosepoke_lengths = hf5.create_array(f'/response_train/', 'nosepoke_lengths', np.array(nosepoke_lens))
# hf5.flush()
# touch_lengths = hf5.create_array(f'/response_train/', 'touch_lengths', np.array(touch_lens))
# hf5.flush()


