#Import necessary tools
import numpy as np
import tables
import easygui
import os, sys

# Get name of directory where the data files and hdf5 file sits, 
# and change to that directory for processing

try:
    dir_name = sys.argv[1]
    if dir_name == '-f':
        dir_name = easygui.diropenbox('Select the dir path where data are saved')
except:
    dir_name = easygui.diropenbox('Select the dir path where data are saved')

os.chdir(dir_name)

#Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
    if files[-2:] == 'h5':
        hdf5_name = files

#Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

#Create vector of electode numbers that have neurons on them (from unit_descriptor table)
electrodegroup = hf5.root.unit_descriptor[:]['electrode_number']

#Some electrodes may record from more than one neuron (shown as repeated number in unit_descriptor); Remove these duplicates within array
electrodegroup = np.unique(electrodegroup)

#Dictate whether EMG electrodes are present (based on experimental configuration) and allocate file names accordingly
noncell_channels = easygui.integerbox(msg='Number of channels', 
                                      title='Purposes other than cell recording (i.e. EMG)', 
                                      default='0',lowerbound=0,upperbound=64)

if noncell_channels == 0:		#If all channels are used for cell recording, move on.
    fieldValues = []
    
else:							#If there are channels used, otherwise: create variable array for channel specification
    Fields = []
    for channel in range(noncell_channels):
        chan_variable = "Channel " + "%01d" % (channel)
        Fields.append(chan_variable)

    fieldValues = easygui.multenterbox('Which Channels are these (i.e. EMG)?', 'Used Channels?', Fields) #Specify which channels are used for non-cell recording
    
    for i in range(len(Fields)):	#Check to make sure that user input has as many non-cell channel numbers as they indicated were used
        if fieldValues[i-1].strip() == "":
            errmsg = "You did not specify as many channels as you indicated you had!"
            print(errmsg)
            fieldValues = easygui.multenterbox('Your input did not match number of entries. Pay Attention!', 'Used Channels?', Fields)	#If user messed up, they will be given a chance to correct this
                
EMG_Channels = fieldValues

Raw_Electrodefiles = []			#Create and fill array with appropriate .dat file names according to occupied cell recording channels
for electrode in electrodegroup:
    if len(EMG_Channels) > 0:
        if electrode<int(EMG_Channels[0]):
            electrode = electrode
        else:
            electrode = electrode+len(EMG_Channels)
    if electrode > 31:
        electrode = electrode - 32
        ampletter = "-B-"
    else:
        ampletter = "-A-"

    filename = "amp" + ampletter + "%03d" % (electrode) + ".dat"
    Raw_Electrodefiles.append(filename)

#Import specific functions in order to filter the data file
from scipy.signal import butter
from scipy.signal import filtfilt

#Specify filtering parameters (linear finite impulse response filter) to define filter specificity across electrodes
frequency_request = 'Specify LFP bandpass filtering paramter (assumes 30kHz as sampling rate)'
freqparam = easygui.multenterbox(frequency_request, 
                                 'Low-Frequency Cut-off (Hz)', 
                                 ['low','high']) 
def get_filtered_electrode(data, freq = freqparam, sampling_rate = 30000.0):
    el = 0.195*(data)
    m, n = butter(2, [2.0*int(freqparam[0])/sampling_rate, 2.0*int(freqparam[1])/sampling_rate], btype = 'bandpass')
    filt_el = filtfilt(m, n, el)
    return filt_el

#Check if LFP data is already within file and remove node if so. Create new raw LFP group within H5 file. 
try:
    hf5.remove_node('/raw_LFP', recursive = True)
except:
    pass
hf5.create_group('/', 'raw_LFP')

#Loop through each neuron-recording electrode (from .dat files), filter data, and create array in new LFP node
for i in range(len(Raw_Electrodefiles)):

    #Read and filter data
    data = np.fromfile(Raw_Electrodefiles[i], dtype = np.dtype('int16'))
    filt_el = get_filtered_electrode(data)
    hf5.create_array('/raw_LFP','electrode%i' % electrodegroup[i], filt_el)
    hf5.flush()
    del filt_el, data

# Grab the names of the arrays containing digital inputs, and pull the data into a numpy array
dig_in_nodes = hf5.list_nodes('/digital_in')
dig_in = []
dig_in_pathname = []
for node in dig_in_nodes:
    dig_in_pathname.append(node._v_pathname)
    exec("dig_in.append(hf5.root.digital_in.%s[:])" % dig_in_pathname[-1].split('/')[-1])
dig_in = np.array(dig_in)

# Get the stimulus delivery times - take the end of the stimulus pulse as the time of delivery
dig_on = []
for i in range(len(dig_in)):
    dig_on.append(np.where(dig_in[i,:] == 1)[0])
change_points = []
for on_times in dig_on:
    changes = []
    for j in range(len(on_times) - 1):
        if np.abs(on_times[j] - on_times[j+1]) > 30:
            changes.append(on_times[j])
    try:
        changes.append(on_times[-1]) # append the last trial which will be missed by this method
    except:
        pass # Continue without appending anything if this port wasn't on at all
    change_points.append(changes)

# Show the user the number of trials on each digital input channel, and ask them to confirm
check = easygui.ynbox(msg = 'Digital input channels: ' + str(dig_in_pathname) + '\n' + 'No. of trials: ' + str([len(changes) for changes in change_points]), title = 'Check and confirm the number of trials detected on digital input channels')

# Go ahead only if the user approves by saying yes
if check:
    pass
else:
    print("Well, if you don't agree, blech_clust can't do much!")
    sys.exit()

# Ask the user which digital input channels should be used for slicing out LFP arrays, and convert the channel numbers into integers for pulling stuff out of change_points
dig_in_channels = easygui.multchoicebox(msg = 'Which digital input channels should be used to slice out LFP data trial-wise?', choices = ([path for path in dig_in_pathname]))
dig_in_channel_nums = []
for i in range(len(dig_in_pathname)):
    if dig_in_pathname[i] in dig_in_channels:
        dig_in_channel_nums.append(i)

# Ask the user for the pre and post stimulus durations to be pulled out, and convert to integers
durations = easygui.multenterbox(msg = 'What are the signal durations pre and post stimulus that you want to pull out', fields = ['Pre stimulus (ms)', 'Post stimulus (ms)'])
for i in range(len(durations)):
    durations[i] = int(durations[i])

# Grab the names of the arrays containing LFP recordings
lfp_nodes = hf5.list_nodes('/raw_LFP')

# Make the Parsed_LFP node in the hdf5 file if it doesn't exist, else move on
try:
    hf5.remove_node('/Parsed_LFP', recursive = True)
except:
    pass
hf5.create_group('/', 'Parsed_LFP')

# Ask if this analysis is looking at more than 1 trial and/or taste
msg   = "Do you want to slice LFPs based on trial initiations for each taste?"
trial_check = easygui.buttonbox(msg,choices = ["Yes","No"])

# Run through the tastes if user said there are more than 1 trial
if trial_check == "Yes":
    for i in range(len(dig_in_channels)):
        num_electrodes = len(lfp_nodes) 
        num_trials = len(change_points[dig_in_channel_nums[i]])
        this_taste_LFPs = np.zeros((num_electrodes, num_trials, durations[0] + durations[1]))
        for electrode in range(num_electrodes):
            for j in range(len(change_points[dig_in_channel_nums[i]])):
                this_taste_LFPs[electrode, j, :] = np.mean(lfp_nodes[electrode][change_points[dig_in_channel_nums[i]][j] - durations[0]*30:change_points[dig_in_channel_nums[i]][j] + durations[1]*30].reshape((-1, 30)), axis = 1)
        
        print (float(i)/len(dig_in_channels)) #Shows progress	
    
        # Put the LFP data for this taste in hdf5 file under /Parsed_LFP
        hf5.create_array('/Parsed_LFP', 'dig_in_%i_LFPs' % (dig_in_channel_nums[i]), this_taste_LFPs)
        hf5.flush()
        
if trial_check == "No":
    num_electrodes = len(lfp_nodes) 
    num_trials = len(change_points[dig_in_channel_nums[0]])-1
    this_taste_LFPs = np.zeros((num_electrodes, num_trials, durations[0] + durations[1]))
    for electrode in range(num_electrodes):
        this_taste_LFPs[electrode, 0, :] = np.mean(lfp_nodes[electrode][change_points[dig_in_channel_nums[0]][0] - durations[0]*30:change_points[dig_in_channel_nums[0]][0] + durations[1]*30].reshape((-1, 30)), axis = 1)
    
        # Put the LFP data for this session in hdf5 file under /Parsed_LFP
    hf5.create_array('/Parsed_LFP', 'dig_in_%i_LFPs' % (dig_in_channel_nums[0]), this_taste_LFPs)
    hf5.flush()
            
# Ask people if they want to delete rawLFPs or not, that way we offer the option to run analyses in many different ways. (ie. First half V back half)
msg   = "Do you want to delete the Raw LFP data?"
rawLFPdelete = easygui.buttonbox(msg,choices = ["Yes","No"])

if rawLFPdelete == "Yes":
    #Delete data
    hf5.remove_node('/raw_LFP', recursive = True)
hf5.flush()

print("If you want to compress the file to release disk space,"+"\nrun 'blech_hdf5_repack.py' upon completion.")
hf5.close()

