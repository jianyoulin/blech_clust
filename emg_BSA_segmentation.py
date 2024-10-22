import tables
import numpy as np
import easygui
import os
import matplotlib.pyplot as plt

# Ask for the directory where the hdf5 file sits, and change to that directory
dir_name = easygui.diropenbox()
os.chdir(dir_name)

# Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files

# Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

# Grab the nodes for the available tastes
try:
    trains_dig_in = hf5.list_nodes('/spike_trains')
    num_trials = trains_dig_in[0].spike_array.shape[0]
    num_tastes = len(trains_dig_in)
except:
    print('This is a no-ephys recording, \nretrieve trial info from numpy npy files')
    

# Load the unique laser duration/lag combos and the trials that correspond to them from the ancillary analysis node
try:
    trials = hf5.root.ancillary_analysis.trials[:]
    unique_lasers = hf5.root.ancillary_analysis.laser_combination_d_l[:]
except:
    trials = np.load('trials.npy')
    unique_lasers = np.load('laser_combination_d_l.npy')
    laser_durs = np.load('laser_durations.npy')
    num_tastes, num_trials = laser_durs.shape

# Ask the user for the pre-stimulus time used
# pre_stim = easygui.multenterbox(msg = 'Enter the pre-stimulus time for the spike trains', fields = ['Pre stim (ms)'])
# pre_stim = int(pre_stim[0])
# Save the entire time window of BSA analysis instead

# Now run through the tastes, and stack up the BSA results for the EMG responses by trials
emg_BSA_results = hf5.root.emg_BSA_results.taste0_p[:, :, :]
for i in range(num_tastes - 1):
	exec("emg_BSA_results = np.vstack((emg_BSA_results[:], hf5.root.emg_BSA_results.taste" + str(i+1) + "_p[:, :, :]))")

# Now run through the consolidated array of emg_BSA_results and check for activity in the gape/LTP range
gapes = np.zeros((emg_BSA_results.shape[0], emg_BSA_results.shape[1]))
ltps = np.zeros((emg_BSA_results.shape[0], emg_BSA_results.shape[1]))
## Find the frequency with the maximum EMG power at each time point on each trial
#max_freq = np.argmax(emg_BSA_results[:, :, :], axis = 2)
## Gapes are anything upto 4.6 Hz
#gapes = np.array(max_freq <= 7, dtype = int)
## LTPs are from 5.95 Hz to 8.65 Hz
#ltps = np.array((max_freq >= 10)*(max_freq <= 16), dtype = int)
#Alternatively, gapes from 3.65-5.95 Hz (6-11). LTPs from 5.95 to 8.65 Hz (11-17) 
gapes = np.sum(emg_BSA_results[:, :, 6:11], axis = 2)/np.sum(emg_BSA_results[:, :, :], axis = 2)
ltps = np.sum(emg_BSA_results[:, :, 11:], axis = 2)/np.sum(emg_BSA_results[:, :, :], axis = 2)

# Also load up the array of significant trials (trials where the post-stimulus response is at least 4 stdev above the pre-stimulus response)
sig_trials = np.load('sig_trials.npy')
sig_trials = np.reshape(sig_trials, (sig_trials.shape[0]*sig_trials.shape[1]))

# Now arrange these arrays by laser condition X taste X time
final_emg_BSA_results = np.empty((len(trials), num_tastes, int(num_trials/len(trials)),  emg_BSA_results.shape[1], emg_BSA_results.shape[2]), dtype = float) 
final_gapes = np.empty((len(trials), num_tastes, int(num_trials/len(trials)),  gapes.shape[1]), dtype = float)
final_ltps = np.empty((len(trials), num_tastes, int(num_trials/len(trials)), ltps.shape[1]), dtype = float)
final_sig_trials = np.empty((len(trials), num_tastes, int(num_trials/len(trials))), dtype = float)

# Fill up these arrays
for i in range(len(trials)):
	for j in range(num_tastes):
		final_emg_BSA_results[i, j, :, :, :] = emg_BSA_results[trials[i][np.where((trials[i] >= num_trials*j)*(trials[i] < num_trials*(j+1)) == True)], :, :]
		final_gapes[i, j, :,  :] = gapes[trials[i][np.where((trials[i] >= num_trials*j)*(trials[i] < num_trials*(j+1)) == True)], :]
		final_ltps[i, j, :, :] = ltps[trials[i][np.where((trials[i] >= num_trials*j)*(trials[i] < num_trials*(j+1)) == True)], :]
		final_sig_trials[i, j, :] = sig_trials[trials[i][np.where((trials[i] >= num_trials*j)*(trials[i] < num_trials*(j+1)) == True)]]


# Save these arrays to file unde the /emg_analysis node
# Create an ancillary_analysis group in the hdf5 file, and write these arrays to that group
try:
	hf5.remove_node('/emg_bsa', recursive = True)
except:
	pass
hf5.create_group('/', 'emg_bsa')

try:
	hf5.remove_node('/emg_bsa/gapes')
	hf5.remove_node('/emg_bsa/ltps')
	hf5.remove_node('/emg_bsa/sig_trials')
#	hf5.remove_node('/ancillary_analysis/emg_BSA_results')
except:
	pass
hf5.create_array('/emg_bsa', 'gapes', final_gapes)
hf5.create_array('/emg_bsa', 'ltps', final_ltps)
hf5.create_array('/emg_bsa', 'sig_trials', final_sig_trials)
#hf5.create_array('/ancillary_analysis', 'emg_BSA_results', final_emg_BSA_results)
np.save('emg_BSA_results.npy', final_emg_BSA_results)

hf5.flush()

hf5.close()






