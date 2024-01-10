# Run through all the raw electrode data, and subtract a common average reference from every electrode's recording
# The user specifies the electrodes to be used as a common average group 

# Import stuff!
import tables
import numpy as np
import pandas as pd
import os
import easygui
from tqdm import tqdm

# Ask for the directory where the hdf5 file sits, and change to that directory
dir_name = '/mnt/g/testing_hdf5s/env_Data_testing_blech_clust/JK14_20230916_Sacc_230916_104702' # easygui.diropenbox()
os.chdir(dir_name)

# Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
    if files[-2:] == 'h5':
        hdf5_name = files

# Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

#################################
#Potentially removable Below
#################################

# # Get the names of all files in this directory
# file_list = os.listdir('./')

# # Get the Intan amplifier ports used in the recordings
# ports = list(set(f[4] for f in file_list if f[:3] == 'amp'))
# # Sort the ports in alphabetical order
# ports.sort()
# if len(ports) == 1:
#     ports.append('additional option')

# # Get the electrode channels on each port
# num_electrodes = {}
# for port in ports:
#     num_electrodes[port] = list(set(int(f[6:9]) for f in file_list if f[:5] == 'amp-{}'.format(port)))

# # Count the number of electrodes on one of the ports (assume all ports have equal number of electrodes)
# #num_electrodes = [int(f[-7:-4]) for f in file_list if f[:3] == 'amp']
# #num_electrodes = np.max(num_electrodes) + 1

# # Ask the user how many common average groups there are in the data. Group = set of electrodes put in together in the same place in the brain (that work).
# num_groups = easygui.multenterbox(msg = "How many common average groups do you have in the dataset?", fields = ["Number of CAR groups"])
# num_groups = int(num_groups[0])

# # Ask the user to choose the port number and electrodes for each of the groups
# group_ports = []
# average_electrodes = []
# for i in range(num_groups):
#     group_ports.append(easygui.multchoicebox(msg = 'Choose the port for common average reference group {:d}'.format(i+1), 
#                                              choices = tuple(ports))[0])
#     average_electrodes.append(easygui.multchoicebox(msg = 'Choose the ELECTRODES TO AVERAGE ACROSS in the common average reference group {:d}. Remember to DESELECT the EMG electrodes'.format(i+1), 
#                                                     choices = num_electrodes[group_ports[-1]]))


# # Get the emg electrode ports and channel numbers from the user
# # If only one amplifier port was used in the experiment, that's the emg_port. Else ask the user to specify
# emg_port = ''
# if len(ports) == 1:
#     emg_port = list(ports[0])
# else:
#     emg_port = easygui.multchoicebox(msg = 'Which amplifier port were the EMG electrodes hooked up to? Just choose any amplifier port if you did not hook up an EMG at all.', 
#                                      choices = tuple(ports))
# # Now get the emg channel numbers, and convert them to integers
# emg_channels = easygui.multchoicebox(msg = 'Choose the CHANNEL NUMBERS FOR THE EMG ELECTRODES. Click clear all and ok if you did not use an EMG electrode', 
#                                      choices = num_electrodes[emg_port[0]])
# if emg_channels:
#     for i in range(len(emg_channels)):
#         emg_channels[i] = int(emg_channels[i])
# # set emg_channels to an empty list if no channels were chosen
# if emg_channels is None:
#     emg_channels = []
# emg_channels.sort()

# if len(emg_channels) > 0:
#     print('emg_port:', emg_port[0], '; emg_channels:', emg_channels)


# #Now convert the electrode numbers to be averaged across to the absolute scale (0-63 if there are 2 ports with 32 recordings each with no EMG)
# CAR_electrodes = []
# # Run through the common average groups
# e_nums = 0
# for group in range(num_groups):
#     group_p = group_ports[group] # define current ports (A, B, C, or D)
#     # Now run through the electrodes and port chosen for that group, and convert to the absolute scale
#     this_group_electrodes = []
#     print(e_nums)
     
#     for e_index, electrode in enumerate(average_electrodes[group]):
#         if len(emg_channels) == 0: # no EMG used
#             if group_ports[group] != group_p:
#                 this_group_electrodes.append(int(electrode) + e_nums)
#             else:
#                 this_group_electrodes.append(int(electrode))
#             #this_group_electrodes.append(int(electrode) + num_electrodes*ports.index(group_ports[group][0]))
#         else: # with EMG channel(s)
#             if group_ports[group] == emg_port[0] and int(electrode) < emg_channels[0]:
#                 if group_ports[group] != group_p:
#                     this_group_electrodes.append(int(electrode) + e_nums)
#                 else:
#                     this_group_electrodes.append(int(electrode))
#                 #this_group_electrodes.append(int(electrode) + num_electrodes*ports.index(group_ports[group][0]))
#             else:
#                 #this_group_electrodes.append(int(electrode) + num_electrodes*ports.index(group_ports[group][0]) - len(emg_channels))
#                 if group_ports[group] != group_p:
#                     this_group_electrodes.append(int(electrode) + e_nums - np.sum(np.array(emg_channels) < int(electrode)))
#                 else:
#                     this_group_electrodes.append(int(electrode) - np.sum(np.array(emg_channels) < int(electrode)))
                
#     e_nums = e_nums + len(average_electrodes[group])
#     CAR_electrodes.append(this_group_electrodes)

#################################
#Potentially removable Above
#################################
    

#################################
#Using csv file of channel map to determine groups
#################################
CAR_electrodes_csv = []
channel_map = pd.read_csv(os.path.join(dir_name, 'channel_map.csv'))
CAR_groups_csv = [i for i in np.unique(channel_map['area']) if i != 'Muscle']
num_groups = len(CAR_groups_csv)
print(f'Number of CAR groups: {num_groups}\n')
for g in CAR_groups_csv:
    this_group = np.array(channel_map.loc[(channel_map.area==g), 'electrode_num'])
    CAR_electrodes_csv.append(list(this_group))

    
#################################
#From here
#################################

# Pull out the raw electrode nodes of the HDF5 file
raw_electrodes = hf5.list_nodes('/raw')

# First get the common average references by averaging across the electrodes picked for each group
print("Calculating common average reference for {:d} groups".format(num_groups))
# common_average_reference = np.zeros((num_groups, hf5.root.raw.electrode0[:].shape[0]))
common_average_reference = np.zeros((num_groups, raw_electrodes[0].shape[0]))
for group in range(num_groups):
    # Stack up the voltage data from all the electrodes that need to be averaged across in this CAR group   
    # In hindsight, don't stack up all the data, it is a huge memory waste. Instead first add up the voltage values from each electrode to the same array, and divide by number of electrodes to get the average    
    for electrode in CAR_electrodes_csv[group]: #CAR_electrodes[group]:
        exec("common_average_reference[group, :] += hf5.root.raw.electrode{:d}[:]".format(electrode))

    # Average the voltage data across electrodes by dividing by the number of electrodes in this group
    common_average_reference[group, :] /= float(len(CAR_electrodes_csv[group])) #CAR_electrodes[group]))

print("Common average reference for {:d} groups calculated".format(num_groups))

# Now run through the raw electrode data and subtract the common average reference from each of them
for electrode in tqdm(raw_electrodes):
    electrode_num = int(str.split(electrode._v_pathname, 'electrode')[-1])
    # Get the common average group number that this electrode belongs to
    # We assume that each electrode belongs to only 1 common average reference group - IMPORTANT!
    group = int([i for i in range(num_groups) if electrode_num in CAR_electrodes_csv[i]][0])

    # Subtract the common average reference for that group from the voltage data of the electrode
    referenced_data = electrode[:] - common_average_reference[group]

    # First remove the node with this electrode's data
    hf5.remove_node("/raw/electrode{:d}".format(electrode_num))

    # Now make a new array replacing the node removed above with the referenced data
    hf5.create_array("/raw", "electrode{:d}".format(electrode_num), referenced_data)
    hf5.flush()

    del referenced_data

hf5.close()
print("Modified electrode arrays written to HDF5 file after subtracting the common average reference")

# Compress the file to clean up all the deleting and creating of arrays
print("Compressing the modified HDF5 file to save up on space")
# Use ptrepack to save a clean and fresh copy of the hdf5 file as tmp.hf5
os.system("ptrepack --chunkshape=auto --propindexes --complevel=9 --complib=blosc " + hdf5_name + " " +  "tmp.h5")

# Delete the old hdf5 file
os.remove(hdf5_name)

# And rename the new file with the same old name
os.rename("tmp.h5", hdf5_name)






