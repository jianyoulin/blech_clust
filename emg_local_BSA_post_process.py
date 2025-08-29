"""
Post processing cleanup of the mess of files created by emg_local_BSA_execute.py.
All the output files will be saved to p (named by tastes) and omega in the hdf5 file under the node emg_BSA_results

"""
# Import stuff
import numpy as np
import easygui
import os
import sys
import tables
    
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

# Delete the raw_emg node, if it exists in the hdf5 file, to cut down on file size
try:
	hf5.remove_node('/raw_emg', recursive = 1)
except:
	print("Raw EMG recordings have already been removed, so moving on ..")


# Load sig_trials.npy to get number of tastes
sig_trials = np.load('sig_trials.npy')
tastes = sig_trials.shape[0]

# Since number of trials can be unequal between tastes, ask the user for the number of trials for each taste
trials = easygui.multenterbox(msg = 'Enter the number of trials for each taste', fields = [str(i) for i in range(tastes)])
for i in range(len(trials)):
	trials[i] = int(trials[i])	

# Change to emg_BSA_results
os.chdir('emg_BSA_results')

# Add group to hdf5 file for emg BSA results
# Create an ancillary_analysis group in the hdf5 file, and write these arrays to that group
try:
	hf5.remove_node('/emg_BSA_results', recursive = True)
except:
	pass
hf5.create_group('/', 'emg_BSA_results')

# Omega doesn't vary by trial, so just pick it up from the 1st taste and trial, and delete everything else
omega = np.load('taste0_trial0_omega.npy')
os.system('rm *omega.npy')

# Add omega to the hdf5 file
atom = tables.Atom.from_dtype(omega.dtype)
om = hf5.create_carray('/emg_BSA_results', 'omega', atom, omega.shape)
om[:] = omega 
hf5.flush()

# Load one of the p arrays to find out the time length of the emg data
p = np.load('taste0_trial0_p.npy')
time_length = p.shape[0]

# Go through the tastes and trials
for i in range(tastes):
	# Make an array for posterior probabilities for each taste
	p = np.zeros((trials[i], time_length, 20))
	for j in range(trials[i]):
		p[j, :, :] = np.load('taste%i_trial%i_p.npy' % (i, j))
	# Save p to hdf5 file
	atom = tables.Atom.from_dtype(p.dtype)
	prob = hf5.create_carray('/emg_BSA_results', 'taste%i_p' % i, atom, p.shape)
	prob[:, :, :] = p
hf5.flush()

# Then delete all p files
os.system('rm *p.npy')

# And delete the emg_BSA_results directory
os.chdir('..')
os.system('rm -r emg_BSA_results')

# Close the hdf5 file
hf5.close()
