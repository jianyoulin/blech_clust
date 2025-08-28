# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os

# Get name of directory with the data files
try:
    dir_name = sys.argv[1]
    if dir_name == '-f':
        dir_name = easygui.diropenbox('Select the dir path where data are saved')
except:
    dir_name = easygui.diropenbox('Select the dir path where data are saved')

os.chdir(dir_name)

# Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files

# Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

# Grab the names of the arrays containing digital inputs, and pull the data into a numpy array
dig_in_nodes = hf5.list_nodes('/digital_in')
dig_in = []
dig_in_pathname = []
for node in dig_in_nodes:
	dig_in_pathname.append(node._v_pathname)
	exec("dig_in.append(hf5.root.digital_in.%s[:])" % dig_in_pathname[-1].split('/')[-1])
dig_in = np.array(dig_in)

# Get the stimulus delivery times - take the end of the stimulus pulse as the time of delivery
# dig_on = []
# for i in range(len(dig_in)):
# 	dig_on.append(np.where(dig_in[i,:] == 1)[0])
# change_points = []
# for on_times in dig_on:
# 	changes = []
# 	for j in range(len(on_times) - 1):
# 		if np.abs(on_times[j] - on_times[j+1]) > 30:
# 			changes.append(on_times[j])
# 	try:
# 		changes.append(on_times[-1]) # append the last trial which will be missed by this method
# 	except:
# 		pass # Continue without appending anything if this port wasn't on at all
# 	change_points.append(changes)	

# # Get the stimulus delivery times - take the end of the stimulus pulse as the time of delivery
dig_on = []
for i in range(len(dig_in)):
	dig_on.append(np.where(dig_in[i,:] == 1)[0])
start_points = []
end_points = []
for on_times in dig_on:
	start = []
	end = []
	try:
		start.append(on_times[0]) # Get the start of the first trial
	except:
		pass # Continue without appending anything if this port wasn't on at all
	for j in range(len(on_times) - 1):
		if np.abs(on_times[j] - on_times[j+1]) > 30:
			end.append(on_times[j])
			start.append(on_times[j+1])
	try:
		end.append(on_times[-1]) # append the last trial which will be missed by this method
	except:
		pass # Continue without appending anything if this port wasn't on at all
	start_points.append(np.array(start))
	end_points.append(np.array(end))
    
# Show the user the number of trials on each digital input channel, and ask them to confirm
# check = easygui.ynbox(msg = 'Digital input channels: ' + str(dig_in_pathname) + '\n' + 'No. of trials: ' + str([len(changes) for changes in change_points]), title = 'Check and confirm the number of trials detected on digital input channels')
# # Go ahead only if the user approves by saying yes
# if check:
# 	pass
# else:
# 	print("Well, if you don't agree, blech_clust can't do much!")
# 	sys.exit()

check = easygui.ynbox(msg = 'Digital input channels: ' + str(dig_in_pathname) + '\n' + 'No. of trials: ' + str([len(changes) for changes in start_points]), title = 'Check and confirm the number of trials detected on digital input channels')
# Go ahead only if the user approves by saying yes
if check:
	pass
else:
	print("Well, if you don't agree, blech_clust can't do much!")
	sys.exit()

# Ask the user which digital input channels should be used for slicing out EMG arrays, and convert the channel numbers into integers for pulling stuff out of change_points
dig_in_channels = easygui.multchoicebox(msg = 'Which digital input channels should be used to slice out EMG data trial-wise?', choices = ([path for path in dig_in_pathname]))
dig_in_channel_nums = []
for i in range(len(dig_in_pathname)):
	if dig_in_pathname[i] in dig_in_channels:
		dig_in_channel_nums.append(i)

# Ask the user which digital input channels should be used for conditioning the stimuli channels above (laser channels for instance)
lasers = easygui.multchoicebox(msg = 'Which digital input channels were used for lasers? Click clear all and continue if you did not use lasers', choices = ([path for path in dig_in_pathname]))
laser_nums = []
if lasers:
	for i in range(len(dig_in_pathname)):
		if dig_in_pathname[i] in lasers:
			laser_nums.append(i)
            
# Ask the user for the pre and post stimulus durations to be pulled out, and convert to integers
durations = easygui.multenterbox(msg = 'What are the signal durations pre and post stimulus that you want to pull out', fields = ['Pre stimulus (ms)', 'Post stimulus (ms)'])
for i in range(len(durations)):
	durations[i] = int(durations[i])

# Grab the names of the arrays containing emg recordings
emg_nodes = hf5.list_nodes('/raw_emg')
emg_pathname = []
for node in emg_nodes:
	emg_pathname.append(node._v_pathname)

# Create a numpy array to store emg data by trials
# emg_data = np.ndarray((len(emg_pathname), len(dig_in_channels), len(change_points[dig_in_channel_nums[0]]), durations[0]+durations[1]))

# # And pull out emg data into this array
# for i in range(len(emg_pathname)):
# 	exec("data = hf5.root.raw_emg.%s[:]" % emg_pathname[i].split('/')[-1])
# 	for j in range(len(dig_in_channels)):
# 		for k in range(len(change_points[dig_in_channel_nums[j]])):
# 			raw_emg_data = data[change_points[dig_in_channel_nums[j]][k]-durations[0]*30:change_points[dig_in_channel_nums[j]][k]+durations[1]*30]
# 			raw_emg_data = 0.195*(raw_emg_data)
# 			# Downsample the raw data by averaging the 30 samples per millisecond, and assign to emg_data
#             # emg_data[emg#, n_tastes, n_trials]
# 			emg_data[i, j, k, :] = np.mean(raw_emg_data.reshape((-1, 30)), axis = 1)

# And pull out emg data into this array
emg_data = np.ndarray((len(emg_pathname), len(dig_in_channels), len(start_points[dig_in_channel_nums[0]]), durations[0]+durations[1]))

for i in range(len(emg_pathname)):
	exec("data = hf5.root.raw_emg.%s[:]" % emg_pathname[i].split('/')[-1])
	for j in range(len(dig_in_channels)):
		for k in range(len(start_points[dig_in_channel_nums[j]])):
			raw_emg_data = data[start_points[dig_in_channel_nums[j]][k]-durations[0]*30:start_points[dig_in_channel_nums[j]][k]+durations[1]*30]
			raw_emg_data = 0.195*(raw_emg_data)
			# Downsample the raw data by averaging the 30 samples per millisecond, and assign to emg_data
            # emg_data[emg#, n_tastes, n_trials]
			emg_data[i, j, k, :] = np.mean(raw_emg_data.reshape((-1, 30)), axis = 1)
            
# Save the emg_data
np.save('emg_data.npy', emg_data)

# Make conditional stimulus array for this digital input if lasers were used
max_trial_num = max([len(start_points[i]) for i in dig_in_channel_nums])
if laser_nums:
    laser_durs, laser_onsets = [], []
    n_lasers = []
    for i in range(len(dig_in_channels)): # number of tastes
        cond_array = np.zeros(len(start_points[dig_in_channel_nums[i]]))
        laser_start = np.zeros(len(start_points[dig_in_channel_nums[i]]))
        # Also make an array to note down the firing of the lasers one by one - for experiments where only 1 laser was fired at a time. This has 3 sorts of laser on conditions - each laser on alone, and then both on together
        laser_single = np.zeros((len(start_points[dig_in_channel_nums[i]]), 2))
        for j in range(len(start_points[dig_in_channel_nums[i]])):
            # Skip the trial if the headstage fell off before it - mark these trials by -1
    #        if start_points[dig_in_channel_nums[i]][j] >= expt_end_time:
    #            cond_array[j] = -1
            # Else run through the lasers and check if the lasers went off within 5 secs of the stimulus delivery time
            for laser in range(len(laser_nums)):
                on_trial = np.where(np.abs(start_points[laser_nums[laser]] - end_points[dig_in_channel_nums[i]][j]) <= 5*30000)[0]
                if len(on_trial) > 0: #If the lasers did go off around stimulus delivery
                    # Mark this laser appropriately in the laser_single array
                    laser_single[j, laser] = 1.0
                    # get the duration and start time in ms (from end of taste delivery) of the laser trial (as a multiple of 10 - so 53 gets rounded off to 50)
                    cond_array[j] = 10*int((end_points[laser_nums[laser]][on_trial][0] - \
                                            start_points[laser_nums[laser]][on_trial][0])/300)
                    
                    laser_start[j] = 10*int((start_points[laser_nums[laser]][on_trial][0] - \
                                             end_points[dig_in_channel_nums[i]][j])/300)
        laser_durs.append(cond_array)
        laser_onsets.append(laser_start)
        n_lasers.append(laser_single)

else:
	laser_durs = np.zeros(shape=(len(dig_in_channels), max_trial_num))
	laser_onsets = np.zeros(shape=(len(dig_in_channels), max_trial_num))
	n_lasers = np.zeros(shape=(len(dig_in_channels), max_trial_num, 2))

# Save the laser durations and onsets
# Write the conditional stimulus duration array to the hdf5 file
np.save('laser_durations.npy', np.array(laser_durs))
np.save('laser_onset_lag.npy', np.array(laser_onsets))
np.save('on_laser.npy', np.array(n_lasers))
print(np.array(laser_durs).shape)

# gather trial information for emg_BSA_segmentation
# First pull out the unique laser(duration,lag) combinations - these are the same irrespective of the unit and time
num_trials = np.array(laser_durs).shape[1]
num_units = 1
num_tastes = np.array(laser_durs).shape[0]
time = durations[0] + durations[1]
params = [250, 25]

laser = np.empty(shape = (int((time - params[0])/params[1]) + 1, num_units, num_tastes*num_trials, 2), dtype = float)
print(f'{laser.shape=}')

# Now fill in the responses and laser (duration,lag) tuples
for i in range(0, time - params[0] + params[1], params[1]):
	for j in range(num_units):
		for k in range(num_tastes):
			# If the lasers were used, get the appropriate durations and lags. Else assign zeros to both
			try:
				laser[int(i/params[1]), j, num_trials*k:num_trials*(k+1)] = np.vstack((laser_durs[k], laser_onsets[k])).T
			except:
				print('except')
				laser[int(i/params[1]), j, num_trials*k:num_trials*(k+1)] = np.zeros((num_trials, 2))

# unique_lasers = np.vstack({tuple(row) for row in laser[0, 0, :, :]})
unique_lasers = np.vstack([laser[0, 0, row, :] for row in range(laser.shape[2])])
#unique_lasers = unique_lasers[unique_lasers[:, 0].argsort(), :]
#unique_lasers = unique_lasers[unique_lasers[:, 1].argsort(), :]
unique_lasers = np.unique(unique_lasers, axis=0)
# Now get the sets of trials with these unique duration and lag combinations
trials = []
for i in range(len(unique_lasers)):
	this_trials = [j for j in range(laser.shape[2]) if np.array_equal(laser[0, 0, j, :], unique_lasers[i, :])]
	trials.append(this_trials)
trials = np.array(trials)
np.save('trials.npy', trials)
np.save('laser_combination_d_l.npy', unique_lasers)


hf5.close()
			
			
			
			











