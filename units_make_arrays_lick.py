# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:44:35 2019

@author: jiany
"""

# Necessary python modules
import easygui
import os
import tables
import sys
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import shutil
import pickle

read = True
if not read:
    # Get name of directory with the data files and Change to that directory
    dir_name = easygui.diropenbox()
    os.chdir(dir_name)
    
    # Get the names of all files in this directory
    file_list = os.listdir('./')
    
    # Grab directory name to create the name of the hdf5 file
    hdf5_name = dir_name.split('\\') #str.split(dir_name, '\')
    
    # Create hdf5 file, and make groups for raw data, raw emgs, digital outputs and digital inputs, and close
    hf5 = tables.open_file(hdf5_name[-1]+'.h5', 'w', title = hdf5_name[-1])
    hf5.create_group('/', 'digital_in')
    #hf5.close()
    
    # Pull out the digital input channels used, and convert them to integers
    dig_in = list(set(f[11:13] for f in file_list if f[:9] == 'board-DIN'))
    for i in range(len(dig_in)):
    	dig_in[i] = int(dig_in[i][0])
    dig_in.sort()
    
    # Create EArrays in hdf5 file
    atom = tables.IntAtom()
    
    #hf5 = tables.open_file(hdf5_name[-1]+'.h5', 'r+')
    for i in dig_in[0:4]:
    	dig_inputs = hf5.create_earray('/digital_in', 'dig_in_%i' % i, atom, (0,))
    #hf5.close()
    
    # Read digital inputs, and append to the respective hdf5 arrays
    #hf5 = tables.open_file(hdf5_name[-1]+'.h5', 'r+')
    for i in dig_in[0:4]:
    	inputs = np.fromfile('board-DIN-%02d'%i + '.dat', dtype = np.dtype('uint16'))
    	exec("hf5.root.digital_in.dig_in_"+str(i)+".append(inputs[:])")
    hf5.flush()
    hf5.close()



# Ask for the directory where the hdf5 file sits, and change to that directory
dir_name = easygui.diropenbox()
os.chdir(dir_name)

# Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
    if files[-2:] == 'h5':
        hdf5_name = files
    if files[:3] == 'exp':
        exp_params_file = files
    if files[-6:] == 'params':
        params_file = files
		
# Read the .params file
f = open(params_file, 'r')
params = []
for line in f.readlines():
	params.append(line)
f.close()

# Get sampling rate of recording
sampling_rate = float(params[13])

print(params_file)
print(sampling_rate)  

# Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

# read param file to get taste list
ts = {'A':'Acid', 'W':'Water', 'S':'Sucrose', 'M':'MSG', 'N':'NaCl', 'X':'SN_Mixture', 'r':'rWater', 'L':'LiCl', 'Q':'QHCl'}
# get the exp param file

with open (exp_params_file, 'r') as f:
    digin_tastes, digin_trial_times = [], []
    for i in range(6):
        this_line = f.readline()
        if 'valve' in this_line:
            digin_tastes.append(this_line.split('=')[1][0])
            digin_trial_times.append(float(this_line.split('=')[2][0:3]))
    digin_tastes = [ts[t] for t in digin_tastes]
    print(list(zip(digin_tastes, digin_trial_times)))

# Grab the names of the arrays containing digital inputs, and pull the data into a numpy array
#dig_in[0] taste delivery array; dig_in[1] lick time array
dig_in_nodes = hf5.list_nodes('/digital_in')
dig_in = []
dig_in_pathname = []
for node in dig_in_nodes:
	dig_in_pathname.append(node._v_pathname)
	exec("dig_in.append(hf5.root.digital_in.%s[:])" % dig_in_pathname[-1].split('/')[-1])
dig_in = np.array(dig_in)

# Get the times for taste delivery, lick, cue - take the start of the stimulus pulse as the time of delivery
dig_on = []
trial_dig_on = []
for i in range(dig_in.shape[0]): # digin 4, 5, 6 are light cue, nose_poke, lick array, so minus 3
	dig_on.append(np.where(dig_in[i,:] == 1)[0])

start_points, end_points = [], [] # for each taste delivery on and off time
trial_start_points, trial_end_points = [], [] # for each trial on and off time
#start_pointss, end_pointss = [], [] # for each taste delivery on and off time
#trial_start_pointss, trial_end_pointss = [], [] # for each trial on and off time
for on_times in dig_on: #[:len(digin_tastes)]:
    
    #####
    if len(on_times) > 3:
        on_times_diff = np.diff(on_times)
        starts = on_times[np.where(on_times_diff > int(sampling_rate/1000))[0]+1]
        starts = np.insert(starts, 0, on_times[0])
        ends = on_times[np.where(on_times_diff > int(sampling_rate/1000))[0]]
        ends = np.insert(ends, -1, on_times[-1])
    
        t_starts = on_times[np.where(on_times_diff > 15*sampling_rate)[0]+1]
        t_starts = np.insert(t_starts, 0, on_times[0])
        t_ends = on_times[np.where(on_times_diff > 15*sampling_rate)[0]]
        t_ends = np.insert(t_ends, -1, on_times[-1])
    else:
        starts, ends = np.array([0]), np.array([0])
        t_starts, t_ends = np.array([0]), np.array([0])
# =============================================================================
#     start_pointss.append(np.array(starts))
#     end_pointss.append(np.array(ends))
#     trial_start_pointss.append(np.array(t_starts))
#     trial_end_pointss.append(np.array(t_ends))	
# 
#     
#     start, end = [], []
#     t_start, t_end = [], []
#     try:
#         start.append(on_times[0]) # Get the start of the first trial
#         t_start.append(on_times[0])
#     except:
#         pass # Continue without appending anything if this port wasn't on at all
#         
#     for j in range(len(on_times) - 1):
#         if np.abs(on_times[j] - on_times[j+1]) > int(sampling_rate/1000):
#             end.append(on_times[j])
#             start.append(on_times[j+1])
# 
#         if np.abs(on_times[j] - on_times[j+1]) > 15*sampling_rate: # 15*30000 are the criterion that separates delivery trial by trial
#             t_end.append(on_times[j])
#             t_start.append(on_times[j+1])
# 
#     try:
#         end.append(on_times[-1]) # append the last trial which will be missed by this method
#         t_end.append(on_times[-1])
#     except:
#         pass # Continue without appending anything if this port wasn't on at all
# 
# =============================================================================
    start_points.append(np.array(starts))
    end_points.append(np.array(ends))
    trial_start_points.append(np.array(t_starts))
    trial_end_points.append(np.array(t_ends))	
    
    
##########################
# Show the user the number of trials on each digital input channel, and ask them to confirm
check = easygui.ynbox(msg = 'Digital input channels: ' + str(dig_in_pathname) + '\n' + 'No. of trials: ' + str([len(trials) for trials in trial_start_points]),
                      title = 'Check and confirm the number of trials detected on digital input channels')
# Go ahead only if the user approves by saying yes
if check:
	pass
else:
	print("Well, if you don't agree, blech_clust can't do much!")
	sys.exit()

# Ask the user which digital input channels should be used for getting spike train data, and convert the channel numbers into integers for pulling stuff out of change_points
dig_in_channels = easygui.multchoicebox(msg = 'Which digital input channels should be used to produce spike train data trial-wise?',
                                        choices = ([path for path in dig_in_pathname]))
dig_in_channel_nums = []
for i in range(len(dig_in_pathname)):
	if dig_in_pathname[i] in dig_in_channels:
		dig_in_channel_nums.append(i)

# Ask the user which digital input channels should be used for conditioning the stimuli channels above (laser channels for instance)
licks = easygui.multchoicebox(msg = 'Which digital input channels were used for LICK inputs? Click clear all and continue if you did not use lasers',
                              choices = ([path for path in dig_in_pathname]))
lick_nums = []
if licks:
	for i in range(len(dig_in_pathname)):
		if dig_in_pathname[i] in licks:
			lick_nums.append(i)
            
# Ask the user which digital input channels should be used for conditioning the stimuli channels above (cue channels for instance)
cues = easygui.multchoicebox(msg = 'Which digital input channels were used for cue signaling? Click clear all and continue if you did not use cues',
                              choices = ([path for path in dig_in_pathname]))
cue_nums = []
if cues:
	for i in range(len(dig_in_pathname)):
		if dig_in_pathname[i] in cues:
			cue_nums.append(i)
############################

    
max_trial_dur = np.max(np.array(digin_trial_times))
# Ask the user for the pre and post stimulus durations to be pulled out, and convert to integers
duration_ = easygui.multenterbox(msg = 'What are the signal durations pre stimulus that you want to pull out', 
                                 fields = ['Pre trial (ms)', 'post trial (ms)'],
                                 values = [2000, int((max_trial_dur+2)*1000)])
pre_trial, post_trial = int(duration_[0]), int(duration_[1])


# arange spike time aligning the taste delivery time
# Delete the spike_trains node in the hdf5 file if it exists, and then create it
try:
	hf5.remove_node('/spike_trains', recursive = True)
except:
	pass
hf5.create_group('/', 'spike_trains')

# Get list of units under the sorted_units group. Find the latest/largest spike time amongst the units, 
# and get an experiment end time (to account for cases where the headstage fell off mid-experiment)
units = hf5.list_nodes('/sorted_units')
expt_end_time = 0
for unit in units:
	if unit.times[-1] > expt_end_time:
		expt_end_time = unit.times[-1]
#num_units = len(units)

#dig_in_channels = [0, 1, 2, 3] # 5 as intaninput for nose poke IR (lick)
# Go through the dig_in_channel_nums and make an array of spike trains of dimensions 
# (# trials x # units x trial duration (ms)) - use start of digital input pulse as the time of taste delivery
for i in dig_in_channel_nums: #range(len(dig_in_channel_nums)):
    spike_train = []
    for trial in range(len(trial_start_points[i])):
		# Skip the trial if the headstage fell off before it
        if trial_end_points[i][trial] >= expt_end_time:
            continue
		# Otherwise run through the units and convert their spike times to milliseconds
        else:
            spikes = np.zeros((len(units), pre_trial + post_trial)) 
			# Get the spike times around the end of taste delivery
            for k in range(len(units)):
                spike_times = np.where((units[k].times[:] <= trial_start_points[i][trial] + post_trial*int(sampling_rate/1000))* \
                                       (units[k].times[:] >= trial_start_points[i][trial] - pre_trial*int(sampling_rate/1000)))[0]
                spike_times = units[k].times[spike_times]
                spike_times = spike_times - trial_start_points[i][trial]
                spike_times = (spike_times/int(sampling_rate/1000)).astype(int) + pre_trial
				# Drop any spikes that are too close to the ends of the trial
                spike_times = spike_times[np.where((spike_times >= 0)*(spike_times < pre_trial+post_trial))[0]]
                spikes[k, spike_times] = 1
				#for l in range(durations[0] + durations[1]):
				#	spikes[k, l] = len(np.where((units[k].times[:] >= end_points[dig_in_channel_nums[i]][j] - (durations[0]-l)*30)*(units[k].times[:] < end_points[dig_in_channel_nums[i]][j] - (durations[0]-l-1)*30))[0])
					
		# Append the spikes array to spike_train 
        spike_train.append(spikes)
	# And add spike_train to the hdf5 file
    hf5.create_group('/spike_trains', f'dig_in_{i}')
    spike_array = hf5.create_array(f'/spike_trains/dig_in_{i}', 'spike_array', np.array(spike_train))
    hf5.flush()


# Then create taste delivery train for each dig_in_channel_nums 
# (# trials x trial duration (ms)) - use end of digital input pulse as the time of taste delivery
for i in dig_in_channel_nums: #range(len(dig_in_channel_nums)):
    taste_train = []
    for trial in range(len(trial_start_points[i])):
        tastes = np.zeros((pre_trial + post_trial))
		# Get the lick times around the start of taste delivery
        taste_times = np.where((start_points[i][:] <= trial_start_points[i][trial] + post_trial*int(sampling_rate/1000))* \
                              (start_points[i][:] >= trial_start_points[i][trial] - pre_trial*int(sampling_rate/1000)))[0]
        taste_times = start_points[i][taste_times]
        taste_times = taste_times - trial_start_points[i][trial]
        taste_times = (taste_times/int(sampling_rate/1000)).astype(int) + pre_trial
        tastes[taste_times] = 1
        taste_train.append(tastes)
        print(f'taste array shape = {tastes.shape}')
    taste_array = hf5.create_array(f'/spike_trains/dig_in_{i}', 'taste_array', np.array(taste_train))

    hf5.flush()

# Then create lick array for each dig_in_channel_nums 
# (# trials x trial duration (ms)) - use end of digital input pulse as the time of taste delivery
# 5 as intaninput for nose poke IR (lick)
for i in dig_in_channel_nums: #range(len(dig_in_channel_nums)):
    lick_train = []
    for trial in range(len(trial_start_points[i])):
        licks = np.zeros((pre_trial + post_trial))
		# Get the lick times around the start of taste delivery
        lick_times = np.where((start_points[lick_nums[0]][:] <= trial_start_points[i][trial] + post_trial*int(sampling_rate/1000))* \
                              (start_points[lick_nums[0]][:] >= trial_start_points[i][trial] - pre_trial*int(sampling_rate/1000)))[0]
        lick_times = start_points[lick_nums[0]][lick_times]
        lick_times = lick_times - trial_start_points[i][trial]
        lick_times = (lick_times/int(sampling_rate/1000)).astype(int) + pre_trial
        licks[lick_times] = 1
        lick_train.append(licks)
    #hf5.create_group('/spike_trains', f'dig_in_{i}')#'str.split(dig_in_channels[i], '/')[-1])
    print(f'lick array shape = {licks.shape}')
    lick_array = hf5.create_array(f'/spike_trains/dig_in_{i}', 'lick_array', np.array(lick_train))
    hf5.flush()

# Then create cue array for each dig_in_channel_nums 
# (# trials x trial duration (ms)) - use end of digital input pulse as the time of taste delivery
# 4 as intaninput for cue light 
for i in dig_in_channel_nums: #range(len(dig_in_channels)):
    cue_train = []
    for trial in range(len(trial_start_points[i])):
        cues = np.zeros((pre_trial + post_trial))
		# Get the lick times around the start of taste delivery
        cue_times = np.where((dig_on[cue_nums[0]][:] <= trial_start_points[i][trial] + post_trial*int(sampling_rate/1000))* \
                              (dig_on[cue_nums[0]][:] >= trial_start_points[i][trial] - pre_trial*int(sampling_rate/1000)))[0]
        cue_times = dig_on[cue_nums[0]][cue_times]
        cue_times = cue_times - trial_start_points[i][trial]
        cue_times = (cue_times/int(sampling_rate/1000)).astype(int) + pre_trial
        cues[cue_times] = 1
        cue_train.append(cues)
    print(f'cue light array shape = {cues.shape}')
    cue_array = hf5.create_array(f'/spike_trains/dig_in_{i}', 'cue_array', np.array(cue_train))
    hf5.flush()

#####
# Delete the spike_trains node in the hdf5 file if it exists, and then create it
cue_node = '/cue_align_data'
try:
	hf5.remove_node(cue_node, recursive = True)
except:
	pass
hf5.create_group(cue_node[0], cue_node[1:])
# Go through the cue_channel_nums and make an array of spike trains aligned with cue start 
# (# trials x # units x trial duration (ms)) - use start of digital input pulse as the time of taste delivery
pre_cue, post_cue = 2000, 15000

for i in cue_nums: #range(len(dig_in_channel_nums)):
    spike_train = []
    rewarded_trials = np.zeros((len(trial_start_points[i])))
    # merge all reward delivery from each dig_in_channel_nums
    rewarded_times = np.concatenate(tuple(start_points[di][:] for di in dig_in_channel_nums))
    rewarded_times = np.sort(rewarded_times)
    for trial in range(len(trial_start_points[i])):
		# Skip the trial if the headstage fell off before it
        if trial_end_points[i][trial] >= expt_end_time:
            continue
		# Otherwise run through the units and convert their spike times to milliseconds
        else:
            spikes = np.zeros((len(units), pre_cue + post_cue)) 
			# Get the spike times around the end of taste delivery
            for k in range(len(units)):
                spike_times = np.where((units[k].times[:] <= trial_start_points[i][trial] + post_cue*int(sampling_rate/1000))* \
                                       (units[k].times[:] >= trial_start_points[i][trial] - pre_cue*int(sampling_rate/1000)))[0]
                spike_times = units[k].times[spike_times]
                spike_times = spike_times - trial_start_points[i][trial]
                spike_times = (spike_times/int(sampling_rate/1000)).astype(int) + pre_cue
				# Drop any spikes that are too close to the ends of the trial
                spike_times = spike_times[np.where((spike_times >= 0)*(spike_times < pre_cue+post_cue))[0]]
                spikes[k, spike_times] = 1

        reward_t = np.where((rewarded_times <= trial_end_points[i][trial])* \
                      (rewarded_times >= trial_start_points[i][trial]))[0]
        rewarded_trials[trial] = len(reward_t)
			
		# Append the spikes array to spike_train in align with the start of cue (2000ms before and 15000ms after)
        spike_train.append(spikes)
        
	# And add spike_train to the hdf5 file
    cue_spike_array = hf5.create_array(cue_node, 'cue_spike_array', np.array(spike_train))
    cue_reward_array = hf5.create_array(cue_node, 'cue_reward_array', np.array(rewarded_trials)) # # of rewards per trial
    hf5.flush()

# obtain what taste being delivered on each trial
rewarded_times_digins = [start_points[di][:] for di in dig_in_channel_nums]
for i in cue_nums: #range(len(dig_in_channel_nums)):
    trial_order = []
    for trial in range(len(trial_start_points[i])):
        sum_ones = [] # temp list to save which digin has taste delivered
        for num, di in enumerate(dig_in_channel_nums):
            sum_ones.append(np.sum((rewarded_times_digins[num]>trial_start_points[i][trial])*(rewarded_times_digins[num]<trial_end_points[i][trial])))
        if np.sum(sum_ones) == 0:
            trial_order.append(np.nan)
        else:
            trial_order.append(np.argmax(sum_ones))
    print(trial_order)


# obtain time for all events on each trial (aligned to cue onset)
trial_events = {}
event_vars = ['spike_times', 'reward_times', 'lick_times', 'cue_times']

for i in cue_nums: #range(len(dig_in_channel_nums)):
#    spike_train, reward_train, lick_train = []
    
    # merge all reward delivery from each dig_in_channel_nums
    rewarded_times_digins = [start_points[di][:] for di in dig_in_channel_nums]
    rewarded_times = np.sort(rewarded_times)
    
    
    for trial in range(len(trial_start_points[i])):
        print(f'Trial{trial}')
        pre_cue_t = 2000
        cue_duration = (trial_end_points[i][trial] - trial_start_points[i][trial])/int(sampling_rate/1000)
        post_cue_t = int(10000 if cue_duration < 10000 else cue_duration + 2000)
        
		# Skip the trial if the headstage fell off before it
        if trial_end_points[i][trial] >= expt_end_time:
            continue
		# Otherwise run through the units and convert their spike times to milliseconds
        else:
            spikes = np.zeros((len(units), pre_cue_t + post_cue_t)) 
			# Get the spike times around the end of taste delivery
            for k in range(len(units)):
                spike_times = np.where((units[k].times[:] <= trial_start_points[i][trial] + post_cue_t*int(sampling_rate/1000))* \
                                       (units[k].times[:] >= trial_start_points[i][trial] - pre_cue_t*int(sampling_rate/1000)))[0]
                spike_times = units[k].times[spike_times]
                spike_times = spike_times - trial_start_points[i][trial]
                spike_times = (spike_times/int(sampling_rate/1000)).astype(int) + pre_cue_t
				# Drop any spikes that are too close to the ends of the trial
                spike_times = spike_times[np.where((spike_times >= 0)*(spike_times < pre_cue_t+post_cue_t))[0]]
                spikes[k, spike_times] = 1

        # get reward time aligned to the cue presentation
        if trial_order[trial] != np.nan:
            
            tastes = np.zeros((pre_cue_t + post_cue_t))
    		# Get the lick times around the start of taste delivery
            taste_times = np.where((rewarded_times <= trial_start_points[i][trial] + post_cue_t*int(sampling_rate/1000))* \
                                  (rewarded_times >= trial_start_points[i][trial] - pre_cue_t*int(sampling_rate/1000)))[0]
            taste_times = rewarded_times[taste_times]
            taste_times = taste_times - trial_start_points[i][trial]
            taste_times = (taste_times/int(sampling_rate/1000)).astype(int) + pre_cue_t
            tastes[taste_times] = 1
    
            # get lick time aligned to the cue presentation
            licks = np.zeros((pre_cue_t + post_cue_t))
    		# Get the lick times around the start of taste delivery
            lick_times = np.where((start_points[lick_nums[0]][:] <= trial_start_points[i][trial] + post_cue_t*int(sampling_rate/1000))* \
                                  (start_points[lick_nums[0]][:] >= trial_start_points[i][trial] - pre_cue_t*int(sampling_rate/1000)))[0]
            #print(lick_times)
            lick_times = start_points[lick_nums[0]][lick_times]
            lick_times = lick_times - trial_start_points[i][trial]
            lick_times = (lick_times/int(sampling_rate/1000)).astype(int) + pre_cue_t
            licks[lick_times] = 1
            
            # get cue time aligned to the cue presentation
            cues = np.zeros((pre_cue_t + post_cue_t))
    		# Get the lick times around the start of taste delivery
            cue_times = np.where((start_points[cue_nums[0]][:] <= trial_start_points[i][trial] + post_cue_t*int(sampling_rate/1000))* \
                                  (start_points[cue_nums[0]][:] >= trial_start_points[i][trial] - pre_cue_t*int(sampling_rate/1000)))[0]
            #print(lick_times)
            cue_times = start_points[cue_nums[0]][cue_times]
            cue_times = cue_times - trial_start_points[i][trial]
            cue_times = (cue_times/int(sampling_rate/1000)).astype(int) + pre_cue_t
            licks[cue_times] = 1
            
#        lick_train.append(licks)
        else:
            tastes = np.zeros((pre_cue_t + post_cue_t))
            licks = np.zeros((pre_cue_t + post_cue_t))
            cues = np.zeros((pre_cue_t + post_cue_t))
        # append data into a dictionary
        this_trial = {}
        this_trial[event_vars[0]] = spikes
        this_trial[event_vars[1]] = tastes
        this_trial[event_vars[2]] = licks
        this_trial[event_vars[3]] = cues
        trial_events[f'Trial{trial}'] = this_trial

with open('trial_events.pkl', 'wb') as f:
    pickle.dump(trial_events, f)
			

# Make directory to store the raster plots. Delete and remake the directory if it exists
try:
    shutil.rmtree("raster")
except:
	pass
os.mkdir("raster")

trains_dig_in = hf5.list_nodes('/spike_trains')
cue_align_data = hf5.list_nodes(cue_node)
with open('trial_events.pkl', 'rb') as pkl_file:
    trial_events = pickle.load(pkl_file)
        
# Plot PSTHs and rasters by digital input channels
for i, dig_in in enumerate(trains_dig_in): #trains_dig_in:
    # Now plot the rasters for this digital input channel and unit
    # Run through the trials
    spike_train, lick_train, taste_train, cue_train = dig_in.spike_array[:], dig_in.lick_array[:], dig_in.taste_array[:], dig_in.cue_array[:]
    digin_n = dig_in_channel_nums[i]
    for trial in range(len(trial_start_points[digin_n])):#len(lick_train)):#dig_in.spike_array[:].shape[0]):
        fig, axes = plt.subplots(nrows = 1, ncols=1, sharey='col',
                                 sharex=True, squeeze=False, figsize = (4*1, 4*1))

        x_licks = np.where(lick_train[trial, :] > 0.0)[0]#dig_in.spike_array[trial, unit, :] > 0.0)[0]
        x_tastes = np.where(taste_train[trial, :] > 0.0)[0]
        x_cues = np.where(cue_train[trial, :] > 0.0)[0]
        
        axes[0,0].vlines(x_cues, len(units)+1, len(units)+1.3, colors = 'green', label='cue on', linewidth = 2)
        axes[0,0].vlines(x_licks, -2, -1, colors = 'blue', label='licks', linewidth = 1)
        axes[0,0].vlines(x_tastes, -1, 0, colors = 'red', label='taste delivery', linewidth = 1)
        axes[0,0].vlines(pre_trial, 0, len(units), colors = 'orange', linewidth = 1)
        axes[0,0].vlines(pre_trial+1000*digin_trial_times[i], 0, len(units), 
                         colors = 'orange', linestyle = '--', linewidth = 1)

        for unit in range(len(units)):
            x_spikes = np.where(spike_train[trial, unit, :] > 0.0)[0]
            axes[0,0].vlines(x_spikes, unit, unit + 1, colors = 'black', linewidth = 0.5)
        axes[0,0].set(xticks=np.arange(0, pre_trial+post_trial+1, 1000),
                      xticklabels=(np.arange(0, pre_trial+post_trial+1, 1000)-2000)//1000)
        axes[0,0].set_title(f'{digin_tastes[i]}: Trial {trial}') 
        axes[0,0].set(xlabel='Time from taste delivery (s)', ylabel='Unit number')
        axes[0,0].legend(loc = 'upper right', fontsize=8)
        plt.tight_layout()
        fig.savefig(f'./raster/{digin_tastes[i]}_Trial{trial}.png')#./raster/'+str.split(dig_in._v_pathname, '/')[-1]+'/Unit%i.png' % (unit))
        plt.close("all")
###################3
# Plot PSTHs and rasters aligned to cue presentation

taste_colors = ['y', 'b', 'g', 'k']
for unit in range(len(units)):
    fig, axes = plt.subplots(nrows = 1, ncols=1, sharey='col',
                             sharex=True, squeeze=False, figsize = (4*3, 4*2))
    for t, trial in enumerate(trial_order): #len(lick_train)):#dig_in.spike_array[:].shape[0]):
        if trial >= 0:
            x_licks = np.where(trial_events[f'Trial{t}']['lick_times'][:] > 0.0)[0]#dig_in.spike_array[trial, unit, :] > 0.0)[0]
            x_tastes = np.where(trial_events[f'Trial{t}']['reward_times'][:] > 0.0)[0]
            x_spikes = np.where(trial_events[f'Trial{t}']['spike_times'][unit, :] > 0.0)[0]
            
            #axes[0,0].vlines(x_cues, len(units)+1, len(trial)+1.3, colors = 'green', label='cue on', linewidth = 2)
            axes[0,0].vlines(x_licks, t, t+1, colors = 'black', linewidth = 0.4, label='Licks')
            axes[0,0].vlines(x_tastes, t, t+1, colors = taste_colors[trial], label=f'{digin_tastes[trial]}', linewidth = 0.4)
            axes[0,0].vlines(x_spikes, t, t+1, colors = 'r', linewidth = 0.4, label='Spikes', alpha = 0.5)
        axes[0,0].set(xlim=(1000, 16000))
        axes[0,0].set(xticks=np.arange(1000, 16000+1, 1000),
                      xticklabels=(np.arange(1000, 16000+1, 1000)-2000)//1000)
        axes[0,0].set_title(f' Unit {unit}') 
        axes[0,0].set(xlabel='Time from cue presentation (s)', ylabel='Trials')
    current_handles, current_labels = plt.gca().get_legend_handles_labels()
    labels = list(np.unique(current_labels))
    legend_idx = [np.where(np.array(current_labels) == taste_n)[0][0] for taste_n in labels]
    axes[0,0].legend(np.array(current_handles)[legend_idx], 
                     np.array(current_labels)[legend_idx],
                     bbox_to_anchor=(1.09, 1), fontsize=8)
    plt.tight_layout()
    fig.savefig(f'./raster/unit{unit}_event_times.png')
    plt.close("all")
        
#####################        
# Plot PSTHs and rasters (aligned to cue and then taste onset) by digital input channels
bin_size = 250
for unit in range(len(units)):
    print('Plotting Unit {}: Rasters and PSTH'.format(unit))
    for i, dig_in in enumerate(trains_dig_in): #trains_dig_in:
        spike_train, lick_train, taste_train = dig_in.spike_array[:], dig_in.lick_array[:], dig_in.taste_array[:]
        digin_n = dig_in_channel_nums[i]
        #print(digin_n)
        
        this_digin_trials = np.where(np.array(trial_order) == i)[0]
        fig, axes = plt.subplots(nrows=2, ncols=2, sharey='row',# sharex=True,  
                                 squeeze=False, figsize = (4*2, 4*2))

        # plot rasters aligned to cue onset
        cue_spike_train = [] # collect spike array from 2000 and 4000
        for trial, t in enumerate(this_digin_trials): # in range(len(trial_start_points[digin_n])): 
            t =  this_digin_trials[trial]
            #x_licks = np.where(trial_events[f'Trial{t}']['lick_times'][:] > 0.0)[0]
            x_tastes = np.where(trial_events[f'Trial{t}']['reward_times'][:] > 0.0)[0]
            x_spikes = np.where(trial_events[f'Trial{t}']['spike_times'][unit, :] > 0.0)[0]
            cue_spike_train.append(trial_events[f'Trial{t}']['spike_times'][unit, :pre_trial+post_trial])
            
            axes[0,0].vlines(x_tastes, trial, trial+1, colors = 'red', label=f'{digin_tastes[i]}', linewidth = 0.6)
            axes[0,0].vlines(x_spikes, trial, trial+1, colors = 'gray', linewidth = 0.4, label='Spikes')
        axes[0,0].vlines(2000, 0, len(trial_start_points[digin_n]), colors='black', linestyles='dashed', linewidth = 1)
        axes[0,0].set(xlim=(0, pre_trial+post_trial))
        axes[0,0].set(xticks=np.arange(0, pre_trial+post_trial+1, 1000),
                      xticklabels=(np.arange(0, pre_trial+post_trial+1, 1000)-2000)//1000)
        axes[0,0].set_title(f' Unit {unit}') 
        axes[0,0].set(xlabel='Time from cue onset (s)', ylabel='Trials')
        current_handles, current_labels = axes[0,0].get_legend_handles_labels()
        labels = list(np.unique(current_labels))
        legend_idx = [np.where(np.array(current_labels) == taste_n)[0][0] for taste_n in labels]
        axes[0,0].legend(np.array(current_handles)[legend_idx], 
                         np.array(current_labels)[legend_idx],
                         #bbox_to_anchor=(1.09, 1),
                         loc = 'upper left', fontsize=8)

        # plot PSTH aligned to cue onset
        t_bins = np.arange(0, pre_trial+post_trial+1, bin_size)
        cue_spike_train = np.array(cue_spike_train)
        spike_rate = 1000*np.array([np.mean(cue_spike_train[:, t_bins[i]:t_bins[i+1]], axis =(0,1)) for i in range(len(t_bins)-1)])
        axes[1,0].bar(np.arange(len(spike_rate)), spike_rate, color='gray')
        axes[1,0].set(xticks=np.arange(len(spike_rate))[::1000//bin_size]-0.5,
                      xticklabels=np.round((t_bins[:-1][::1000//bin_size]-pre_trial)/1000, 2))
        axes[1,0].set(xlabel=f'Time from cue onset (s) - {bin_size}ms-bins', ylabel='Firing Rates')

        # plot rasters aligned to first taste onset
        for trial in range(len(trial_start_points[digin_n])):#len(lick_train)):#dig_in.spike_array[:].shape[0]):
#            x_licks = np.where(lick_train[trial, :] > 0.0)[0]#dig_in.spike_array[trial, unit, :] > 0.0)[0]
            x_tastes = np.where(taste_train[trial, :] > 0.0)[0]
            if trial == 1:
#                axes[0,0].vlines(x_licks, trial, trial+0.5, colors = 'blue', label='licks', linewidth = 0.5)
                axes[0,1].vlines(x_tastes, trial, trial+0.5, colors = 'red', label='taste delivery', linewidth = 0.5)
            else:
#                axes[0,0].vlines(x_licks, trial, trial+0.5, colors = 'blue', linewidth = 0.5)
                axes[0,1].vlines(x_tastes, trial, trial+0.5, colors = 'red', linewidth = 0.5)
            x_spikes = np.where(spike_train[trial, unit, :] > 0.0)[0]
            axes[0,1].vlines(x_spikes, trial, trial + 1, colors = 'gray', linewidth = 0.5)
        axes[0,1].vlines(2000, 0, len(trial_start_points[digin_n]), colors='black', linestyles='dashed', linewidth = 1)
        axes[0,1].set(xlim=(0, pre_trial+post_trial))
        axes[0,1].set(xticks=np.arange(0, pre_trial+post_trial+1, 1000),
                      xticklabels=(np.arange(0, pre_trial+post_trial+1, 1000)-2000)//1000)
        axes[0,1].set_title(f'{digin_tastes[i]}') 
        axes[0,1].set(xlabel='Time from taste delivery (s)')#, ylabel='Trials')
        axes[0,1].legend(loc = 'lower left', fontsize=8)
        
        # plot PSTH aligned to first taste onset of trials
        t_bins = np.arange(0, pre_trial+post_trial+1, bin_size)
        spike_rate = 1000*np.array([np.mean(spike_train[:, unit, t_bins[i]:t_bins[i+1]], axis =(0, 1)) for i in range(len(t_bins)-1)])
        axes[1,1].bar(np.arange(len(spike_rate)), spike_rate, color='gray')
        axes[1,1].set(xticks=np.arange(len(spike_rate))[::2]-0.5,
                      xticklabels=(t_bins[:-1][::2]-pre_trial)//1000)
        axes[1,1].set(xlabel=f'Time from taste delivery (s) - {bin_size}ms-bins')#, ylabel='Firing Rates')
        
        plt.tight_layout()
        fig.savefig(f'./raster/unit{unit}_{digin_tastes[i]}.png')#./raster/'+str.split(dig_in._v_pathname, '/')[-1]+'/Unit%i.png' % (unit))
        plt.close("all")

############################    
# Plot PSTHs and rasters by digital input channels
bin_size = 250; plot_bin = 500//bin_size
plot_lims = [-1000, int(1000*digin_trial_times[0])+1000]
for unit in range(len(units)): 
    fig, axes = plt.subplots(nrows=len(trains_dig_in), ncols=1, sharey='col', sharex=True,  
                             squeeze=False, figsize = (4*1, 4*2))
    for i, dig_in in enumerate(trains_dig_in): #trains_dig_in:
        spike_train = dig_in.spike_array[:]
        
        t_bins = np.arange(plot_lims[0]+pre_trial, plot_lims[1]+pre_trial, bin_size)
        spike_rate = 1000*np.array([np.mean(spike_train[:, unit, t_bins[t]:t_bins[t+1]], axis =(0, 1)) for t in range(len(t_bins)-1)])
        axes[i,0].bar(np.arange(len(spike_rate)), spike_rate, label=digin_tastes[i]) #, color='gray')
        axes[i,0].set(xticks=np.arange(len(spike_rate))[::plot_bin])#,
                      #xticklabels=(t_bins[:-1][::plot_bin]-pre_trial),
                      #rotation=30)
        axes[i,0].set_xticklabels(np.round((t_bins[:-1][::plot_bin]-pre_trial)/1000, 1), rotation = 90, fontsize=8)
        axes[i,0].set(ylabel='Firing Rates')
        if i == len(trains_dig_in)-1:
            axes[i,0].set_xlabel(f'Time from taste delivery (s) - {bin_size}ms-bins', fontsize=10)
        if i == 0:
            #axes[i,0].set_title(f'Unit {unit}') 
            axes[i,0].set_title('Unit: %i PSTH plot' % (unit) + '\n' + 'Electrode: %i, Single Unit: %i, RSU: %i, FS: %i' \
                                % (hf5.root.unit_descriptor[unit]['electrode_number'], 
                                   hf5.root.unit_descriptor[unit]['single_unit'], 
                                   hf5.root.unit_descriptor[unit]['regular_spiking'], 
                                   hf5.root.unit_descriptor[unit]['fast_spiking']), fontsize=10)	

        axes[i,0].legend(loc = 'upper right', fontsize=8)
        
    plt.tight_layout()
    fig.savefig(f'./raster/PSTH_unit{unit}.png')#./raster/'+str.split(dig_in._v_pathname, '/')[-1]+'/Unit%i.png' % (unit))
    plt.close("all")

# Plot heatmap aligned with cue presentation
cue_spike_array = hf5.root.cue_align_data.cue_spike_array[:]
cue_reward_array = hf5.root.cue_align_data.cue_reward_array[:]

# make firing rates into bin responses
window_size, step_size = 100, 100
t, u, b = cue_spike_array.shape
b = b-window_size+1

convolve_responses = np.empty(shape = (t,u,b), dtype = np.dtype('float64'))
window = np.ones(window_size)

for i in range(t):
    for j in range(u):
        convolve_responses[i,j,:] = 1000*((np.convolve(window, cue_spike_array[i, j, :], 'valid'))/window_size)
stepwise_response = convolve_responses[:, :, 0::step_size]

x_times = np.arange(0, cue_spike_array.shape[-1]+1, window_size)
for unit in range(1):#len(units)): 
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey='row',# sharex=True,  
                             squeeze=False, figsize = (4*2, 4*1))
    axes[0,0].imshow(stepwise_response[:, unit, :], aspect='auto')
    axes[0,0].vlines(np.where(x_times==pre_cue)[0], 0, t, colors='y')
    axes[0,0].set(xticks=np.arange(len(x_times))[::1000//window_size])#,
                  #xticklabels=((x_times[::1000//window_size]-pre_cue)//1000))#,
                  #rotation=90)
    axes[0,0].set_xticklabels(np.round((x_times[::1000//window_size]-pre_cue)/1000, 1), rotation = 90, fontsize=8)
    for trial in range(t):
        if cue_reward_array[trial] > 0:
            axes[0,0].scatter(np.where(x_times==pre_cue)[0], trial, color='r', s = 5)
    axes[0,0].set(ylabel='Trials')
    axes[0,0].set_xlabel('Time from cue presentation (s)', fontsize=10)
    axes[0,0].set_title('Unit: %i' % (unit) + '\n' + 'Electrode: %i, Single Unit: %i, RSU: %i, FS: %i' \
                            % (hf5.root.unit_descriptor[unit]['electrode_number'], 
                               hf5.root.unit_descriptor[unit]['single_unit'], 
                               hf5.root.unit_descriptor[unit]['regular_spiking'], 
                               hf5.root.unit_descriptor[unit]['fast_spiking']), fontsize=10)	

    ### plot only 1 sec before and after cue start time
    axes[0,1].imshow(stepwise_response[:, unit, 10:30], aspect='auto')
    axes[0,1].vlines(np.where(x_times==pre_cue-1000)[0], 0, t, colors='y')
    axes[0,1].set(xticks=np.arange(20)[::2])
    axes[0,1].set_xticklabels(np.arange(-1000, 1000, 200), rotation = 90, fontsize=8)
    for trial in range(t):
        if cue_reward_array[trial] > 0:
            axes[0,1].scatter(np.where(x_times==pre_cue-1000)[0], trial, color='r', s = 5)
    axes[0,1].set(ylabel='Trials')
    axes[0,1].set_xlabel('Time from cue presentation (s)', fontsize=10)
    axes[0,1].set_title('Unit: %i' % (unit) + 'from -1000 to 1000 ms', fontsize=10)
    
    plt.tight_layout()
    fig.savefig(f'./raster/cue_spikes_heatmap_unit{unit}.png')#./raster/'+str.split(dig_in._v_pathname, '/')[-1]+'/Unit%i.png' % (unit))
    plt.close("all")

hf5.close()

