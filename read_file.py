# Import stuff!
import tables
import os
import numpy as np

# Create EArrays in hdf5 file 
def create_hdf_arrays(file_name, ports, dig_in, e_channels, emg_port, emg_channels):
    hf5 = tables.open_file(file_name, 'r+')
    n_electrodes = 0
    for port in ports:
        n_electrodes = n_electrodes + len(e_channels[port])
    electrodes = np.concatenate(tuple(e_channels[k] for k in e_channels.keys()))
    #n_electrodes = [e_channelslen(ports)*32
    atom = tables.IntAtom()
    
    # Create arrays for digital inputs
    for i in dig_in:
        dig_inputs = hf5.create_earray('/digital_in', 'dig_in_%i' % i, atom, (0,))

    # Create arrays for neural electrodes, and make directories to store stuff coming out from blech_process
    for i in electrodes: #range(n_electrodes - len(emg_channels)):
        el = hf5.create_earray('/raw', 'electrode%i' % i, atom, (0,))
        
    # Create arrays for EMG electrodes
    for i in range(len(emg_channels)):
        el = hf5.create_earray('/raw_emg', 'emg%i' % i, atom, (0,))

    # Close the hdf5 file 
    hf5.close() 

# Read files into hdf5 arrays - the format should be 'one file per channel'
def read_files(hdf5_name, ports, dig_in, e_channels, emg_port, emg_channels):
    hf5 = tables.open_file(hdf5_name, 'r+')

    # Read digital inputs, and append to the respective hdf5 arrays
    for i in dig_in:
        inputs = np.fromfile('board-DIN-%02d'%i + '.dat', dtype = np.dtype('uint16'))
        exec("hf5.root.digital_in.dig_in_"+str(i)+".append(inputs[:])")

    # Read data from amplifier channels
    emg_counter = 0
    el_counter = 0
    for port in ports:
        for channel in e_channels[port]: 
            data = np.fromfile('amp-' + port + '-%03d'%channel + '.dat', dtype = np.dtype('int16'))
            exec("hf5.root.raw.electrode%i.append(data[:])" % channel)

        if port == emg_port:
            for i, channel in enumerate(emg_channels): 
                data = np.fromfile('amp-' + port + '-%03d'%channel + '.dat', dtype = np.dtype('int16'))
                exec("hf5.root.raw_emg.emg%i.append(data[:])" % i)
    
    # for port in ports:
    #     for channel in range(len(e_channels[port])):
    #         data = np.fromfile('amp-' + port + '-%03d'%channel + '.dat', dtype = np.dtype('int16'))
    #         if port == emg_port[0] and channel in emg_channels:
    #             exec("hf5.root.raw_emg.emg%i.append(data[:])" % emg_counter)
    #             emg_counter += 1
    #         else:
    #             exec("hf5.root.raw.electrode%i.append(data[:])" % el_counter)
    #             el_counter += 1
        hf5.flush()

    hf5.close()
                
# Create EArrays in hdf5 file 
def create_hdf_digin_arrays(file_name, dig_in):
    hf5 = tables.open_file(file_name, 'r+')
    atom = tables.IntAtom()
    
    # Create arrays for digital inputs
    for i in dig_in:
        dig_inputs = hf5.create_earray('/digital_in', 'dig_in_%i' % i, atom, (0,))

    # Close the hdf5 file 
    hf5.close()     

# Read files into hdf5 arrays - the format should be 'one file per channel'
def read_digin_files(hdf5_name, dig_in):
    hf5 = tables.open_file(hdf5_name, 'r+')

    # Read digital inputs, and append to the respective hdf5 arrays
    for i in dig_in:
        inputs = np.fromfile('board-DIN-%02d'%i + '.dat', dtype = np.dtype('uint16'))
        exec("hf5.root.digital_in.dig_in_"+str(i)+".append(inputs[:])")
        hf5.flush()
    
    # Close the hdf5 file
    hf5.close()    
