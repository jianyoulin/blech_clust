import tables
import numpy as np
import easygui
import os
import sys
import matplotlib.pyplot as plt

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
    print(trials)

# Ask the user for the pre-stimulus time used
pre_stim = easygui.multenterbox(msg = 'Enter the pre-stimulus time for the spike trains', 
                                fields = ['Pre stim (ms)'], 
                                values=[2000])
pre_stim = int(pre_stim[0])
# Save the entire time window of BSA analysis instead

# Now run through the tastes, and stack up the BSA results for the EMG responses by trials
trials_by_taste = []
emg_BSA_results = hf5.root.emg_BSA_results.taste0_p[:, :, :]
trials_by_taste.append(emg_BSA_results.shape[0])
for i in range(num_tastes - 1):
    exec("this_BSA_results = hf5.root.emg_BSA_results.taste" + str(i+1) + "_p[:, :, :]")
    trials_by_taste.append(this_BSA_results.shape[0])
    emg_BSA_results = np.vstack((emg_BSA_results[:], this_BSA_results[:]))
    # exec("emg_BSA_results = np.vstack((emg_BSA_results[:], hf5.root.emg_BSA_results.taste" + str(i+1) + "_p[:, :, :]))")

# Now run through the consolidated array of emg_BSA_results and check for activity in the gape/LTP range
gapes = np.zeros((emg_BSA_results.shape[0], emg_BSA_results.shape[1]))
ltps = np.zeros(shape=gapes.shape)
# gapes = np.zeros((emg_BSA_results.shape[0], emg_BSA_results.shape[1]))
# ltps = np.zeros((emg_BSA_results.shape[0], emg_BSA_results.shape[1]))

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
min_trials = min(trials_by_taste)
final_emg_BSA_results = np.empty((len(trials), num_tastes, int(min_trials/len(trials)),  emg_BSA_results.shape[1], emg_BSA_results.shape[2]), dtype = float) 
final_gapes = np.empty((len(trials), num_tastes, int(min_trials/len(trials)),  gapes.shape[1]), dtype = float)
final_ltps = np.empty((len(trials), num_tastes, int(min_trials/len(trials)), ltps.shape[1]), dtype = float)
final_sig_trials = np.empty((len(trials), num_tastes, int(min_trials/len(trials))), dtype = float)

# Fill up these arrays
cum_trials = np.insert(trials_by_taste, 0, 0)
cum_trials = np.cumsum(cum_trials)
min_laser_trials = final_emg_BSA_results.shape[2]
print(cum_trials)
for i in range(len(trials)): # number of laser conditions
    for j in range(num_tastes):
        trial_s, trial_e = cum_trials[j], min(cum_trials[j] + min_trials, cum_trials[j+1])
        print(trial_s, trial_e)
        # a = emg_BSA_results[trials[i][np.where((trials[i] >= trial_s)*(trials[i] < trial_e) == True)], :, :]
        # print(a.shape)
        final_emg_BSA_results[i, j, :, :, :] = emg_BSA_results[trials[i][np.where((trials[i] >= trial_s)*(trials[i] < trial_e) == True)], :, :][:min_laser_trials,:,:]
        final_gapes[i, j, :,  :] = gapes[trials[i][np.where((trials[i] >= trial_s)*(trials[i] < trial_e) == True)], :][:min_laser_trials,:]
        final_ltps[i, j, :, :] = ltps[trials[i][np.where((trials[i] >= trial_s)*(trials[i] < trial_e) == True)], :][:min_laser_trials,:]
        final_sig_trials[i, j, :] = sig_trials[trials[i][np.where((trials[i] >= trial_s)*(trials[i] < trial_e) == True)]][:min_laser_trials]

# # Fill up these arrays
# trials_by_taste.insert(0, 0)
# for i in range(len(trials)): # number of laser conditions
# 	for j in range(num_tastes):
# 		trial_s, trial_e = trials_by_taste[j], trials_by_taste[j+1]
# 		print(trial_s, trial_e)
# 		final_emg_BSA_results[i, j, :, :, :] = emg_BSA_results[trials[i][np.where((trials[i] >= num_trials*j)*(trials[i] < num_trials*(j+1)) == True)], :, :]
# 		final_gapes[i, j, :,  :] = gapes[trials[i][np.where((trials[i] >= num_trials*j)*(trials[i] < num_trials*(j+1)) == True)], :]
# 		final_ltps[i, j, :, :] = ltps[trials[i][np.where((trials[i] >= num_trials*j)*(trials[i] < num_trials*(j+1)) == True)], :]
# 		final_sig_trials[i, j, :] = sig_trials[trials[i][np.where((trials[i] >= num_trials*j)*(trials[i] < num_trials*(j+1)) == True)]]


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

# plot emg bsa gapes probability
n_lasers, n_tastes, n_trials, n_time = final_gapes.shape

# set boundaries for the duration of emg data to be plotted
time_start, time_end = -500, 2500
x_ticks = np.arange(time_end-time_start)[::500]
fig, ax = plt.subplots(1, n_lasers, sharey=True, squeeze=False, figsize=(4*n_lasers, 6))
for i in range(n_lasers):
    gapes_conc = np.concatenate([final_gapes[i, t,:,time_start+pre_stim:time_end+pre_stim] for t in range(n_tastes)])
    ax[0, i].imshow(gapes_conc, origin='lower', aspect='auto')
    ax[0, i].set_xticks(x_ticks, (x_ticks+time_start)/1000)
    ax[0, i].set_xlabel('Time from taste delivery (m)')
    ax[0, i].set_title(f'Laser_condition {i}')
    if i == 0:
        ax[0, i].set_ylabel('All trials')
fig.suptitle('Gapes')
plt.tight_layout()
plt.savefig('./gapes_BSA_probability.png')
plt.close()

# plot emg bsa licks probability
n_lasers, n_tastes, n_trials, n_time = final_ltps.shape
fig, ax = plt.subplots(1, n_lasers, sharey=True, squeeze=False, figsize=(4*n_lasers, 6))
for i in range(n_lasers):
    licks_conc = np.concatenate([final_ltps[i, t,:,time_start+pre_stim:time_end+pre_stim] for t in range(n_tastes)])
    ax[0, i].imshow(licks_conc, origin='lower', aspect='auto')
    ax[0, i].set_xticks(x_ticks, (x_ticks+time_start)/1000)
    ax[0, i].set_xlabel('Time from taste delivery (s)')
    ax[0, i].set_title(f'Laser_condition {i}')
    if i == 0:
        ax[0, i].set_ylabel('Trials')
fig.suptitle('Licks / LTPs')
plt.tight_layout()
plt.savefig('./LTPs_BSA_probability.png')
plt.close()

hf5.flush()

hf5.close()






