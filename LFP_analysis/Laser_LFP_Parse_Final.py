# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:06:32 2017

@author: Bradly
"""
#Import necessary tools
import numpy as np
import easygui
import tables
import os, sys

# Get name of directory where the data files and hdf5 file sits, 
# and change to that directory for processing
try:
    dir_name = sys.argv[1]
    if dir_name == '-f':
        dir_name = easygui.diropenbox('Select the dir path where data are saved')
except:
    dir_name = easygui.diropenbox('Select the dir path where data are saved')


#Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
    if files[-2:] == 'h5':
        hdf5_name = files

#Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')
trains_dig_in = hf5.list_nodes('/spike_trains')
lfp_nodes = hf5.list_nodes('/Parsed_LFP')

# Create a LFP_laser group in the hdf5 file (if one exists, remove and create new)
try:
    hf5.remove_node('/LFP_Lasers', recursive = True)
except:
    pass
hf5.create_group('/', 'LFP_Lasers')

# Ask the user about the type of units they want to do the calculations on (single or all units)
unit_type = easygui.multchoicebox(msg = 'Which type of units do you want to use?', choices = ('All units', 'Single units', 'Multi units', 'Custom choice'))
all_units = np.arange(trains_dig_in[0].spike_array.shape[1])
single_units = np.array([i for i in range(len(all_units)) if hf5.root.unit_descriptor[i]["single_unit"] == 1])
multi_units = np.array([i for i in range(len(all_units)) if hf5.root.unit_descriptor[i]["single_unit"] == 0])
chosen_units = []
if unit_type[0] == 'All units':
    chosen_units = all_units
elif unit_type[0] == 'Single units':
    chosen_units = single_units
elif unit_type[0] == 'Multi units':
    chosen_units = multi_units
else:
    chosen_units = easygui.multchoicebox(msg = 'Which units do you want to choose?', choices = ([i for i in all_units]))
    for i in range(len(chosen_units)):
        chosen_units[i] = int(chosen_units[i])
    chosen_units = np.array(chosen_units)

#Get taste and LFP information
num_tastes = len(hf5.list_nodes('/spike_trains'))
# Now make arrays to pull the data out
num_trials, _, time = trains_dig_in[0].spike_array.shape
num_units = len(chosen_units)
params = [250, 25]
laser = np.empty(shape = (int((time - params[0])/params[1]) + 1, num_units, num_tastes*num_trials, 2), dtype = float)

# Now fill in the responses and laser (duration,lag) tuples
for i in range(0, time - params[0] + params[1], params[1]):
    for j in range(num_units):
        for k in range(num_tastes):
            # If the lasers were used, get the appropriate durations and lags. Else assign zeros to both
            try:
                laser[int(i/params[1]), j, num_trials*k:num_trials*(k+1)] = np.vstack((trains_dig_in[k].laser_durations[:], trains_dig_in[k].laser_onset_lag[:])).T
            except:
                laser[int(i/params[1]), j, num_trials*k:num_trials*(k+1)] = np.zeros((num_trials, 2))


# First pull out the unique laser(duration,lag) combinations - these are the same irrespective of the unit and time
unique_lasers = np.vstack({tuple(row) for row in laser[0, 0, :, :]})
unique_lasers = unique_lasers[unique_lasers[:, 0].argsort(), :]
unique_lasers = unique_lasers[unique_lasers[:, 1].argsort(), :]
# Now get the sets of trials with these unique duration and lag combinations
trials = []
for i in range(len(unique_lasers)):
    this_trials = [j for j in range(laser.shape[2]) if np.array_equal(laser[0, 0, j, :], unique_lasers[i, :])]
    trials.append(this_trials)
    trials = np.array(trials)
laser_trials = trials.copy()
laser_conditions = unique_lasers.copy()

# #Get laser conditions from hdf5 file (after ancillary analysis.py)
# laser_conditions = hf5.root.ancillary_analysis.laser_combination_d_l[:]
# laser_trials = hf5.root.ancillary_analysis.trials[:]

# Run through the tastes and laser conditions, and build arrays with respective data
for x in range(laser_conditions.shape[0]):
    
    #Create identifier based on laser combination information and create group within "LFP_Lasers" node
    las_type = 'laser_combos_d_l_'+str(int(laser_conditions[x,0]))+'_'+str(int(laser_conditions[x,1]))         
    hf5.create_group('/LFP_Lasers', las_type)
    
    #Loop through taste arrays (dig_in files), identify trial number pertaining to laser condition, and built array with respective LFP data
    for y in range(num_tastes):
        taste = 'taste_' + str(y)
        
        #Collapse data across electrodes to obtain LFPS by tastes and laser conditions
        lfp_coll = np.mean(lfp_nodes[y][:],axis=0)
        # Pick the trials of taste y in laser condition x
        trial_group = np.where((laser_trials[x, :] >= y*num_trials)*(laser_trials[x, :] < (y+1)*num_trials) == True)[0] #Create array of trials to index based on the number of trials per taste and laser combinations (assumes equal trial numbers per taste)
        # The trials picked above are on the absolute scale of trial numbers (so taste 3, for instance, will have trials 90 to 119 if there were a total of 120 trials). We need to convert this into a 0-29 scale (for 30 trials of each taste) to be able to use these numbers on the LFP arrays
        trial_group = laser_trials[x, trial_group] - int(y*num_trials)
        laser_trial_LFPs = lfp_coll[trial_group]
               
        # Create arrays (based on tastants) to build under corresonding laser condition
        hf5.create_array('/LFP_Lasers/%s' % las_type, '%s' %taste, laser_trial_LFPs)
        hf5.flush()

print("If you want to compress the file to release disk space, run 'blech_hdf5_repack.py' upon completion.")        
hf5.close()
