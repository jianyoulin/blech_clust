# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set(style="white", context="talk", font_scale=1.8)
#sns.set_color_codes(palette = 'colorblind')

# Ask the user for the hdf5 files that need to be plotted together
dirs = []
while True:
	dir_name = easygui.diropenbox(msg = 'Choose a directory with a hdf5 file, hit cancel to stop choosing')
	try:
		if len(dir_name) > 0:	
			dirs.append(dir_name)
	except:
		break

# Now run through the directories, and pull out the data
unique_lasers = []
gapes = []
ltps = []
sig_trials = []
pre_stim = []
gapes_Li = []
gape_trials_Li = []
first_gape_Li = []
emg_BSA_results = []
num_trials = 0
for dir_name in dirs:
	# Change to the directory
	os.chdir(dir_name)
	# Locate the hdf5 file
	file_list = os.listdir('./')
	hdf5_name = ''
	for files in file_list:
		if files[-2:] == 'h5':
			hdf5_name = files

	# Open the hdf5 file
	hf5 = tables.open_file(hdf5_name, 'r')

	# Pull the data from the /ancillary_analysis node
	unique_lasers.append(hf5.root.ancillary_analysis.laser_combination_d_l[:])
	gapes.append(hf5.root.ancillary_analysis.gapes[:])
	ltps.append(hf5.root.ancillary_analysis.ltps[:])
	sig_trials.append(hf5.root.ancillary_analysis.sig_trials[:])
#	gapes_Li.append(hf5.root.ancillary_analysis.gapes_Li[:])
#	gape_trials_Li.append(hf5.root.ancillary_analysis.gape_trials_Li[:])
#	first_gape_Li.append(hf5.root.ancillary_analysis.first_gape_Li[:])
    
#	emg_BSA_results.append(hf5.root.ancillary_analysis.emg_BSA_results[:])
	# Reading single values from the hdf5 file seems hard, needs the read() method to be called
	pre_stim.append(hf5.root.ancillary_analysis.pre_stim.read())
	# Also maintain a counter of the number of trials in the analysis
	num_trials += hf5.root.ancillary_analysis.gapes.shape[0]

	# Close the hdf5 file
	hf5.close()
	emg_result = np.load('emg_BSA_results.npy')
	emg_BSA_results.append(emg_result)

# Check if the number of laser activation/inactivation windows is same across files, raise an error and quit if it isn't
if all(unique_lasers[i].shape == unique_lasers[0].shape for i in range(len(unique_lasers))):
	pass
else:
	print("Number of inactivation/activation windows doesn't seem to be the same across days. Please check and try again")
	sys.exit()

# Now first set the ordering of laser trials straight across data files
laser_order = []
for i in range(len(unique_lasers)):
	# The first file defines the order	
	if i == 0:
		laser_order.append(np.arange(unique_lasers[i].shape[0]))
	# And everyone else follows
	else:
		this_order = []
		for j in range(unique_lasers[i].shape[0]):
			for k in range(unique_lasers[i].shape[0]):
				if np.array_equal(unique_lasers[0][j, :], unique_lasers[i][k, :]):
					this_order.append(k)
		laser_order.append(np.array(this_order))

# Now join up all the data into big numpy arrays, maintaining the laser order defined in laser_order
# If there's only one data file, set the final arrays to the only array read in
# Also get an array for the number of trials in every session - needed to plot the gape/ltp segmentation results
trials = []
if len(laser_order) == 1:
	trials.append(sig_trials[0].shape[2])
	gapes = gapes[0]
	ltps = ltps[0]
	sig_trials = sig_trials[0]
#	gapes_Li = gapes_Li[0][:, :, :, int(pre_stim[0]):]
#	gape_trials_Li = gape_trials_Li[0]
#	first_gape_Li = first_gape_Li[0]
	emg_BSA_results = emg_BSA_results[0]
	
else:
	trials = [sig_trials[i].shape[2] for i in range(len(sig_trials))]
	gapes = np.concatenate(tuple(gapes[i][laser_order[i], :, :, :] for i in range(len(gapes))), axis = 2)
	ltps = np.concatenate(tuple(ltps[i][laser_order[i], :, :, :] for i in range(len(ltps))), axis = 2)
	sig_trials = np.concatenate(tuple(sig_trials[i][laser_order[i], :, :] for i in range(len(sig_trials))), axis = 2)
	emg_BSA_results = np.concatenate(tuple(emg_BSA_results[i][laser_order[i], :, :, :, :] for i in range(len(emg_BSA_results))), axis = 2)
#	gapes_Li = np.concatenate(tuple(gapes_Li[i][laser_order[i], :, :, int(pre_stim[0]):] for i in range(len(gapes_Li))), axis = 2)
#	gape_trials_Li = np.concatenate(tuple(gape_trials_Li[i][laser_order[i], :, :] for i in range(len(gape_trials_Li))), axis = 2)
#	first_gape_Li = np.concatenate(tuple(first_gape_Li[i][laser_order[i], :, :] for i in range(len(first_gape_Li))), axis = 2)
	

# Ask the user for the directory to save plots etc in
dir_name = easygui.diropenbox(msg = 'Choose the output directory for EMG BSA analysis')
os.chdir(dir_name)

# Get the pre stimulus time in the experiments
pre_stim = int(pre_stim[0])

# Ask the user for the time limits to plot the results upto
time_limits = easygui.multenterbox(msg = 'Enter the time limits to be used in the plots', fields = ['Pre stim (ms)', 'Post stim (ms)'])
for i in range(len(time_limits)):
	time_limits[i] = int(time_limits[i])

# Get an array of x values to plot the average probability of gaping or licking across time
x = np.arange(gapes.shape[-1]) - pre_stim

# Get the indices of x that need to be plotted based on the chosen time limits
plot_indices = np.where((x >= time_limits[0])*(x <= time_limits[1]))[0]

# Ask the user for the names of the tastes in the dataset
tastes = easygui.multenterbox(msg = 'Enter the names of the tastes used in the experiments', fields = ['Taste{:d}'.format(i+1) for i in range(gapes.shape[1])])

#.................................
# Plots by laser condition
# Gapes
for i in range(gapes.shape[0]):
	fig = plt.figure()
	for j in range(gapes.shape[1]):
		plt.plot(x[plot_indices], np.mean(gapes[i, j, :, plot_indices].T, axis = 0), linewidth = 2.0, label = tastes[j])
	plt.xlabel('Time post stimulus (ms)')
	plt.ylabel('Trial averaged fraction of power in 3.65-5.95 Hz')
	plt.title('Gapes, Duration:%i ms, Lag:%i ms' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
	plt.legend(loc = 'upper left', fontsize = 10)
	fig.savefig('Gapes, laser condition%i.png' %(i+1), bbox_inches = 'tight')	
	plt.close('all')

#LTPs
for i in range(ltps.shape[0]):
	fig = plt.figure()
	for j in range(ltps.shape[1]):
		plt.plot(x[plot_indices], np.mean(ltps[i, j, :, plot_indices].T, axis = 0), linewidth = 2.0, label = tastes[j])
	plt.xlabel('Time post stimulus (ms)')
	plt.ylabel('Trial averaged fraction of power in 5.95-10 Hz')
	plt.title('LTPs, Duration:%i ms, Lag:%i ms' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
	plt.legend(loc = 'upper left', fontsize = 10)
	fig.savefig('LTPs, laser condition%i.png' %(i+1), bbox_inches = 'tight')
	plt.close('all')
	
#.................................

#.................................
# Plots by taste
# Gapes
for i in range(gapes.shape[1]):
	fig = plt.figure()
	for j in range(gapes.shape[0]):
		plt.plot(x[plot_indices], np.mean(gapes[j, i, :, plot_indices].T, axis = 0), linewidth = 2.0, label = 'Duration:%i, Lag:%i' % (unique_lasers[0][j, 0], unique_lasers[0][j, 1]))
	plt.xlabel('Time post stimulus (ms)')
	plt.ylabel('Trial averaged fraction of power in 3.65-5.95 Hz')
	plt.title('Gapes, Taste:%i' % (i+1))
	plt.legend(loc = 'upper left', fontsize = 10)
	fig.savefig('Gapes, taste%i.png' %(i+1), bbox_inches = 'tight')	
	plt.close('all')

#LTPs
for i in range(ltps.shape[1]):
	fig = plt.figure()
	for j in range(ltps.shape[0]):
		plt.plot(x[plot_indices], np.mean(ltps[j, i, :, plot_indices].T, axis = 0), linewidth = 2.0, label = 'Duration:%i, Lag:%i' % (unique_lasers[0][j, 0], unique_lasers[0][j, 1]))
	plt.xlabel('Time post stimulus (ms)')
	plt.ylabel('Trial averaged fraction of power in in 5.95-10 Hz')
	plt.title('LTPs, Taste:%i' % (i+1))
	plt.legend(loc = 'upper left', fontsize = 10)
	fig.savefig('LTPs, taste%i.png' %(i+1), bbox_inches = 'tight')	
	plt.close('all')
#.................................

# Save these arrays in the output folder
np.save('unique_lasers.npy', unique_lasers)
np.save('gapes.npy', gapes)
np.save('ltps.npy', ltps)
#np.save('gapes_Li.npy', gapes_Li)
#np.save('gape_trials_Li.npy', gape_trials_Li)
#np.save('first_gape_Li.npy', first_gape_Li)
np.save('emg_BSA_results.npy', emg_BSA_results)

"""
#.................................
# Plot the gaping results from the analysis in Li et al., 2016
# Plots by taste
for i in range(gape_trials_Li.shape[1]):
	# Plot gape probabilities first
	fig = plt.figure()
	plt.bar(np.arange(gape_trials_Li.shape[0]) + 1, np.mean(gape_trials_Li[:, i, :], axis = -1), 0.35)
	plt.xticks(np.arange(gape_trials_Li.shape[0]) + 1, [unique_lasers[0][j] for j in range(len(unique_lasers[0]))])
	plt.ylabel('Fraction of trials with gapes' + '\n' + 'according to Li et al., 2016')
	plt.title('Taste: %i, Trials: %i' % (i+1, gape_trials_Li.shape[2]))
	fig.savefig('Gape_probabilities_Li, taste%i.png' %(i+1), bbox_inches = 'tight')
	plt.close('all')

	# Plot time of first gape next
	fig = plt.figure()
	plt.bar(np.arange(first_gape_Li.shape[0]) + 1, np.array(np.ma.masked_equal(first_gape_Li[:, i, :], 0).mean(axis = -1)), 0.35, yerr = np.array(np.ma.masked_equal(first_gape_Li[:, i, :], 0).std(axis = -1))/np.sqrt(np.array(np.ma.masked_equal(first_gape_Li[:, i, :], 0).sum(axis = -1)))) 
	plt.xticks(np.arange(first_gape_Li.shape[0]) + 1, [unique_lasers[0][j] for j in range(len(unique_lasers[0]))])
	plt.ylabel('Mean length of gape bouts across trials (ms)' + '\n' + 'according to Li et al., 2016')
	plt.title('Taste: %i, Trials: %i' % (i+1, first_gape_Li.shape[2]))
	fig.savefig('Gape_durations_Li, taste%i.png' %(i+1), bbox_inches = 'tight')
	plt.close('all')
"""	
#.................................

# Ask the user for the parameters to use for emg segmentation
params = easygui.multenterbox(msg = 'Enter the parameters for EMG segmentation', fields = ['Minimum onset lag for gapes and ltps (ms) - usually 500', 'Minimum length of acceptable gape bout (ms) - usually 300', 'Minimum length of acceptable ltp bout (ms) - usually 150', 'Maximum length of broken gape bout (ms) - usually 100', 'Maximum length of broken ltp bout (ms) - usually 50'])
min_onset_lag = int(params[0])
min_gape_len = int(params[1])
min_ltp_len = int(params[2])
max_broken_gape = int(params[3])
max_broken_ltp = int(params[4])

#.................................
# Get 1.) Number of bouts, 2.) Start time of first bout, and 3.) Average length of bouts of gapes and LTPs on every trial

# First convert the gapes and ltps arrays to only the post-stimulus window so that emg segmentation doesn't look at the pre stim data
gapes = gapes[:, :, :, pre_stim:]
ltps = ltps[:, :, :, pre_stim:]

# Then convert the arrays of gapes and ltps to 0s and 1s - 1s if the power at that trial and time-point is > 0.5
gapes[gapes >= 0.5] = 1.0
gapes[gapes < 0.5] = 0.0
ltps[ltps >= 0.5] = 1.0
ltps[ltps < 0.5] = 0.0
 
# Run through the tastes and laser conditions

gape_segments = np.empty((gapes.shape[0], gapes.shape[1], gapes.shape[2], 3), dtype = float)
ltp_segments = np.empty((ltps.shape[0], ltps.shape[1], ltps.shape[2], 3), dtype = float)
for i in range(gapes.shape[0]):
	for j in range(gapes.shape[1]):
		for k in range(gapes.shape[2]):
			#if sig_trials[i, j, k] > 0.0:
			# Where does activity in the gape/LTP range happen on this trial
			gape_places = np.where(gapes[i, j, k, :time_limits[1]] > 0)[0] 
			ltp_places = np.where(ltps[i, j, k, :time_limits[1]] > 0)[0]

			# Drop any activity that's below the minimum onset lag
			# gape_places = gape_places[gape_places >= min_onset_lag]
			# ltp_places = ltp_places[ltp_places >= min_onset_lag]

			# If there's no activity (or just one time point of activity) in the gape range, mark this trial appropriately in gape_segments
			if len(gape_places) <= 1:
				gape_segments[i, j, k, :] = np.array([0, -1, -1])
			else:
				# Get the difference between the consecutive time points where gapes has 1s - if this difference is > 1, this indicates different bouts
				gape_diff = np.ediff1d(gape_places)
				# If gapes are broken by less than the max_broken_gape length, correct them
				gape_diff[np.where(gape_diff < max_broken_gape)[0]] = np.ones(len(np.where(gape_diff < max_broken_gape)[0]))
				# Check where this diff is > 1
				bout_changes = np.where(gape_diff > 1)[0]
				gape_bouts = []
				# Run through the bouts, and see if they are greater than the minimum acceptable length
				if len(bout_changes) == 0:
					if np.abs(gape_places[0] - gape_places[-1]) >= min_gape_len and gape_places[0] >= min_onset_lag:
						gape_segments[i, j, k, :]  = np.array([1, gape_places[0], np.abs(gape_places[0] - gape_places[-1])])
					else:
						gape_segments[i, j, k, :] = np.array([0, -1, -1])
				else:
					for l in range(len(bout_changes)):
						if l == len(bout_changes) - 1:
							if l == 0:
								if np.abs(gape_places[0] - gape_places[bout_changes[l]]) >= min_gape_len and gape_places[0] >= min_onset_lag:
									gape_bouts.append((gape_places[0], gape_places[bout_changes[l]]))
								if np.abs(gape_places[bout_changes[l] + 1] - gape_places[-1]) >= min_gape_len and gape_places[bout_changes[l] + 1] >= min_onset_lag:
									gape_bouts.append((gape_places[bout_changes[l] + 1], gape_places[-1]))
							else:
								if np.abs(gape_places[bout_changes[l] + 1] - gape_places[-1]) >= min_gape_len and gape_places[bout_changes[l] + 1] >= min_onset_lag:
									gape_bouts.append((gape_places[bout_changes[l] + 1], gape_places[-1]))
						else:
							if l == 0:
								if np.abs(gape_places[0] - gape_places[bout_changes[l]]) >= min_gape_len and gape_places[0] >= min_onset_lag:
									gape_bouts.append((gape_places[0], gape_places[bout_changes[l]]))
								if np.abs(gape_places[bout_changes[l] + 1] - gape_places[bout_changes[l+1]]) >= min_gape_len and gape_places[bout_changes[l] + 1] >= min_onset_lag:
									gape_bouts.append((gape_places[bout_changes[l] + 1], gape_places[bout_changes[l+1]]))
							else:
								if np.abs(gape_places[bout_changes[l] + 1] - gape_places[bout_changes[l+1]]) >= min_gape_len and gape_places[bout_changes[l] + 1] >= min_onset_lag:
									gape_bouts.append((gape_places[bout_changes[l] + 1], gape_places[bout_changes[l+1]]))
					# Now summarize the gape bouts in gape_segments
					if len(gape_bouts) == 0:
						gape_segments[i, j, k, :] = np.array([0, -1, -1])
					else:
						gape_segments[i, j, k, :] = np.array([len(gape_bouts), gape_bouts[0][0], np.mean([np.abs(gape_bouts[m][0] - gape_bouts[m][1]) for m in range(len(gape_bouts))])])

		
			# If there's no activity (or just one time point of activity) in the ltp range, mark this trial appropriately in ltp_segments
			if len(ltp_places) <= 1:
				ltp_segments[i, j, k, :] = np.array([0, -1, -1])
			else:
				# Get the difference between the consecutive time points where ltps has 1s - if this difference is > 1, this indicates different bouts
				ltp_diff = np.ediff1d(ltp_places)
				# If ltps are broken by less than the max_broken_ltp length, correct them
				ltp_diff[np.where(ltp_diff < max_broken_ltp)[0]] = np.ones(len(np.where(ltp_diff < max_broken_ltp)[0]))
				# Check where this diff is > 1
				bout_changes = np.where(ltp_diff > 1)[0]
				ltp_bouts = []
				# Run through the bouts, and see if they are greater than the minimum acceptable length
				if len(bout_changes) == 0:
					if np.abs(ltp_places[0] - ltp_places[-1]) >= min_ltp_len:
						ltp_segments[i, j, k, :]  = np.array([1, ltp_places[0], np.abs(ltp_places[0] - ltp_places[-1])])
					else:
						ltp_segments[i, j, k, :] = np.array([0, -1, -1])
				else:
					for l in range(len(bout_changes)):
						if l == len(bout_changes) - 1:
							if l == 0:
								if np.abs(ltp_places[0] - ltp_places[bout_changes[l]]) >= min_ltp_len and ltp_places[0] >= min_onset_lag:
									ltp_bouts.append((ltp_places[0], ltp_places[bout_changes[l]]))
								if np.abs(ltp_places[bout_changes[l] + 1] - ltp_places[-1]) >= min_ltp_len and ltp_places[bout_changes[l] + 1] >= min_onset_lag:
									ltp_bouts.append((ltp_places[bout_changes[l] + 1], ltp_places[-1]))
							else:
								if np.abs(ltp_places[bout_changes[l] + 1] - ltp_places[-1]) >= min_ltp_len and ltp_places[bout_changes[l] + 1] >= min_onset_lag:
									ltp_bouts.append((ltp_places[bout_changes[l] + 1], ltp_places[-1]))
						else:
							if l == 0:
								if np.abs(ltp_places[0] - ltp_places[bout_changes[l]]) >= min_ltp_len and ltp_places[0] >= min_onset_lag:
									ltp_bouts.append((ltp_places[0], ltp_places[bout_changes[l]]))
								if np.abs(ltp_places[bout_changes[l] + 1] - ltp_places[bout_changes[l+1]]) >= min_ltp_len and ltp_places[bout_changes[l] + 1] >= min_onset_lag:
									ltp_bouts.append((ltp_places[bout_changes[l] + 1], ltp_places[bout_changes[l+1]]))
							else:
								if np.abs(ltp_places[bout_changes[l] + 1] - ltp_places[bout_changes[l+1]]) >= min_ltp_len and ltp_places[bout_changes[l] + 1] >= min_onset_lag:
									ltp_bouts.append((ltp_places[bout_changes[l] + 1], ltp_places[bout_changes[l+1]]))
					# Now summarize the ltp bouts in ltp_segments
					if len(ltp_bouts) == 0:
						ltp_segments[i, j, k, :] = np.array([0, -1, -1])
					else:
						ltp_segments[i, j, k, :] = np.array([len(ltp_bouts), ltp_bouts[0][0], np.mean([np.abs(ltp_bouts[m][0] - ltp_bouts[m][1]) for m in range(len(ltp_bouts))])])

			# If this trial is 0 on sig_trials, mark it appropriately on gape_segments and ltp_segments
			#else:
			#	gape_segments[i, j, k, :] = np.array([0, -1, -1])
			#	ltp_segments[i, j, k, :] = np.array([0, -1, -1])
					
#.................................

# Save these arrays in the output folder
np.save('gape_segments.npy', gape_segments)
np.save('ltp_segments.npy', ltp_segments)

# Produce bar plots of the emg segmentation results
# First plot the gapes....................................................

# Plot of gape probabilities across tastes (different laser conditions on the same graph)
for i in range(gape_segments.shape[1]):
	fig = plt.figure()
	plt.bar(np.arange(gape_segments.shape[0]) + 1, np.sum(gape_segments[:, i, :, 0] > 0, axis = 1)/float(gape_segments.shape[2]), 0.35)
	plt.xticks(np.arange(gape_segments.shape[0]) + 1, [unique_lasers[0][j] for j in range(len(unique_lasers[0]))])
	plt.ylabel('Fraction of trials with gapes')
	plt.title('Taste: %i, Trials: %i' % (i+1, gape_segments.shape[2]))
	fig.savefig('Gape_probabilities, taste%i.png' %(i+1), bbox_inches = 'tight')
	plt.close('all')

# Plot of average onset times of gape bouts
for i in range(gape_segments.shape[1]):
	fig = plt.figure()
	plt.bar(np.arange(gape_segments.shape[0]) + 1, [np.mean(gape_segments[j, i, :, 1][np.where(gape_segments[j, i, :, 1] > 0)[0]]) for j in range(gape_segments.shape[0])], 0.35, yerr = [np.std(gape_segments[j, i, :, 1][np.where(gape_segments[j, i, :, 1] > 0)[0]])/np.sqrt(len(np.where(gape_segments[j, i, :, 1] > 0)[0])) for j in range(gape_segments.shape[0])])
	plt.xticks(np.arange(gape_segments.shape[0]) + 1, [unique_lasers[0][j] for j in range(len(unique_lasers[0]))])
	plt.ylabel('Mean onset time of gape bouts across trials (ms)')
	plt.title('Taste: %i, Trials: %i' % (i+1, gape_segments.shape[2]))
	fig.savefig('Gape_onset_times, taste%i.png' %(i+1), bbox_inches = 'tight')
	plt.close('all')
	
# Plot of average length of gape bouts
for i in range(gape_segments.shape[1]):
	fig = plt.figure()
	plt.bar(np.arange(gape_segments.shape[0]) + 1, [np.mean(gape_segments[j, i, :, 2][np.where(gape_segments[j, i, :, 2] > 0)[0]]) for j in range(gape_segments.shape[0])], 0.35, yerr = [np.std(gape_segments[j, i, :, 2][np.where(gape_segments[j, i, :, 2] > 0)[0]])/np.sqrt(len(np.where(gape_segments[j, i, :, 2] > 0)[0])) for j in range(gape_segments.shape[0])])
	plt.xticks(np.arange(gape_segments.shape[0]) + 1, [unique_lasers[0][j] for j in range(len(unique_lasers[0]))])
	plt.ylabel('Mean length of gape bouts across trials (ms)')
	plt.title('Taste: %i, Trials: %i' % (i+1, gape_segments.shape[2]))
	fig.savefig('Gape_durations, taste%i.png' %(i+1), bbox_inches = 'tight')
	plt.close('all')

# Then plot the ltps....................................................
for i in range(ltp_segments.shape[1]):
	fig = plt.figure()
	plt.bar(np.arange(ltp_segments.shape[0]) + 1, np.sum(ltp_segments[:, i, :, 0] > 0, axis = 1)/float(ltp_segments.shape[2]), 0.35)
	plt.xticks(np.arange(ltp_segments.shape[0]) + 1, [unique_lasers[0][j] for j in range(len(unique_lasers[0]))])
	plt.ylabel('Fraction of trials with ltps')
	plt.title('Taste: %i, Trials: %i' % (i+1, ltp_segments.shape[2]))
	fig.savefig('ltp_probabilities, taste%i.png' %(i+1), bbox_inches = 'tight')
	plt.close('all')

# Plot of average onset times of ltp bouts
for i in range(ltp_segments.shape[1]):
	fig = plt.figure()
	plt.bar(np.arange(ltp_segments.shape[0]) + 1, [np.mean(ltp_segments[j, i, :, 1][np.where(ltp_segments[j, i, :, 1] > 0)[0]]) for j in range(ltp_segments.shape[0])], 0.35, yerr = [np.std(ltp_segments[j, i, :, 1][np.where(ltp_segments[j, i, :, 1] > 0)[0]])/np.sqrt(len(np.where(ltp_segments[j, i, :, 1] > 0)[0])) for j in range(ltp_segments.shape[0])])
	plt.xticks(np.arange(ltp_segments.shape[0]) + 1, [unique_lasers[0][j] for j in range(len(unique_lasers[0]))])
	plt.ylabel('Mean onset time of ltp bouts across trials (ms)')
	plt.title('Taste: %i, Trials: %i' % (i+1, ltp_segments.shape[2]))
	fig.savefig('ltp_onset_times, taste%i.png' %(i+1), bbox_inches = 'tight')
	plt.close('all')

# Plot of average length of ltp bouts
for i in range(ltp_segments.shape[1]):
	fig = plt.figure()
	plt.bar(np.arange(ltp_segments.shape[0]) + 1, [np.mean(ltp_segments[j, i, :, 2][np.where(ltp_segments[j, i, :, 2] > 0)[0]]) for j in range(ltp_segments.shape[0])], 0.35, yerr = [np.std(ltp_segments[j, i, :, 2][np.where(ltp_segments[j, i, :, 2] > 0)[0]])/np.sqrt(len(np.where(ltp_segments[j, i, :, 2] > 0)[0])) for j in range(ltp_segments.shape[0])])
	plt.xticks(np.arange(ltp_segments.shape[0]) + 1, [unique_lasers[0][j] for j in range(len(unique_lasers[0]))])
	plt.ylabel('Mean length of ltp bouts across trials (ms)')
	plt.title('Taste: %i, Trials: %i' % (i+1, ltp_segments.shape[2]))
	fig.savefig('ltp_durations, taste%i.png' %(i+1), bbox_inches = 'tight')  
	plt.close('all')

#..........................................................................





