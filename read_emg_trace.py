"""
Save EMG trace data directly from dat file into a hdf5 file
default params: with port at 'B', and emg channel as '8', and '9'
"""

# Necessary python modules
import easygui
import os
import tables
import sys
import numpy as np
import multiprocessing

# Create EArrays in hdf5 file 
def create_hdf_arrays(file_name, emg_port, emg_channels):
    hf5 = tables.open_file(file_name, 'r+')

    atom = tables.IntAtom()
    
    # Create arrays for EMG electrodes
    for i in range(len(emg_channels)):
        try:
            hf5.remove_node('/raw_emg', 'emg%i' % i, recursive = True)
        except:
            pass
        
        el = hf5.create_earray('/raw_emg', 'emg%i' % i, atom, (0,))

    # Close the hdf5 file 
    hf5.close() 
    
# Read files into hdf5 arrays - the format should be 'one file per channel'
def read_files(hdf5_name, emg_port, emg_channels):
    hf5 = tables.open_file(hdf5_name, 'r+')

    # Read data from amplifier channels
    emg_counter = 0

    for channel in emg_channels:
        data = np.fromfile('amp-' + emg_port + '-%03d'%int(channel) + '.dat', dtype = np.dtype('int16'))
        exec("hf5.root.raw_emg.emg%i.append(data[:])" % emg_counter)
        emg_counter += 1

    hf5.flush()
    hf5.close()

# Get name of directory with the data files
# Ask for the directory where the hdf5 file sits, and change to that directory
dir_name = easygui.diropenbox()
os.chdir(dir_name)

# Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files

if len(hdf5_name) > 0:
    hf5 = tables.open_file(hdf5_name, 'r+')
    try:
        hf5.remove_node('/raw_emg', recursive = True)
    except:
        pass
else:
    # Grab directory name to create the name of the hdf5 file
    hdf5_name = str.split(dir_name, '/')[-1] + '.h5'

    # Create hdf5 file, and make groups for raw data, raw emgs, digital outputs and digital inputs, and close
    hf5 = tables.open_file(hdf5_name, 'w', title = hdf5_name[-1])
hf5.create_group('/', 'raw_emg')
hf5.close()

# Get the amplifier ports used
emg_port = ['B']
emg_channels = ['8', '9']

# Read the amplifier sampling rate from info.rhd - look at Intan's website for structure of header files
sampling_rate = np.fromfile('info.rhd', dtype = np.dtype('float32'))
sampling_rate = int(sampling_rate[2])

# Create arrays for each electrode
create_hdf_arrays(hdf5_name, emg_port[0], emg_channels)

# Read data files, and append to electrode arrays
read_files(hdf5_name, emg_port[0], emg_channels)

