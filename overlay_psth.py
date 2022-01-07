# Import stuff!
import numpy as np
import tables
import pylab as plt
import easygui
import sys
import os

# Import stuff for datashadar
import datashader as ds
import datashader.transfer_functions as tf
from functools import partial
from datashader.utils import export_image
import pandas as pd
import matplotlib.pyplot as plt
from imageio import imread
import shutil

# A function that accepts a numpy array of waveforms and creates a datashader image from them
def convert_waveforms2img(waveforms, x_values, dir_name = "datashader_temp"):

    # Make a pandas dataframe with two columns, x and y, holding all the data. 
    # The individual waveforms are separated by a row of NaNs

    # First downsample the waveforms 10 times (to remove the effects of 10 times upsampling during de-jittering)
    waveforms = waveforms[:, ::10]
    # Then make a new array of waveforms - the last element of each waveform is a NaN
    new_waveforms = np.zeros((waveforms.shape[0], waveforms.shape[1] + 1))
    new_waveforms[:, -1] = np.nan
    new_waveforms[:, :-1] = waveforms
    
    # Now make an array of x's - the last element is a NaN
    x = np.zeros(x_values.shape[0] + 1)
    x[-1] = np.nan
    x[:-1] = x_values

    # Now make the dataframe
    df = pd.DataFrame({'x': np.tile(x, new_waveforms.shape[0]), 'y': new_waveforms.flatten()})	

    # Datashader function for exporting the temporary image with the waveforms
    export = partial(export_image, background = "white", export_path=dir_name)

    # Produce a datashader canvas
    canvas = ds.Canvas(x_range = (np.min(x_values), np.max(x_values)),
                       y_range = (df['y'].min() - 10, df['y'].max() + 10),
                       plot_height=1200, plot_width=1600)
    # Aggregate the data
    agg = canvas.line(df, 'x', 'y', ds.count())   
    # Transfer the aggregated data to image using log transform and export the temporary image file
    export(tf.shade(agg, how='eq_hist'), 'tempfile')
    del df, waveforms, new_waveforms


# Ask for the directory where the hdf5 file sits, and change to that directory
dir_name = easygui.diropenbox()
os.chdir(dir_name)

# Make directory to store the PSTH plots. Delete and remake the directory if it exists
try:
	os.system('rm -r '+'./overlay_PSTH')
except:
	pass
os.mkdir('./overlay_PSTH')

# Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files

# Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

# Now ask the user to put in the identities of the digital inputs
trains_dig_in = hf5.list_nodes('/spike_trains')
identities = easygui.multenterbox(msg = 'Put in the taste identities of the digital inputs', 
                                  fields = [train._v_name for train in trains_dig_in],
                                  values = ['Sucrose'])

# Ask what taste to plot
plot_tastes = easygui.multchoicebox(msg = 'Which tastes do you want to plot? ', choices = ([taste for taste in identities]))
plot_tastes_dig_in = []
for taste in plot_tastes:
    plot_tastes_dig_in.append(identities.index(taste))

# Ask the user for the pre stimulus duration used while making the spike arrays
pre_stim = easygui.multenterbox(msg = 'What was the pre-stimulus duration pulled into the spike arrays?', 
                                fields = ['Pre stimulus (ms)'],
                                values = [2000])
pre_stim = int(pre_stim[0])

# Get the psth paramaters from the user
params = easygui.multenterbox(msg = 'Enter the parameters for making the PSTHs',
                              fields = ['Window size (ms)', 'Step size (ms)'],
                              values = (250, 25))
for i in range(len(params)):
	params[i] = int(params[i])

# Ask the user about the type of units they want to do the calculations on (single or all units)
chosen_units = []
all_units = np.arange(trains_dig_in[0].spike_array.shape[1])
chosen_units = easygui.multchoicebox(msg = 'Which units do you want to choose? (*same as in setup*)', 
                                     choices = ([i for i in all_units]))
for i in range(len(chosen_units)):
	chosen_units[i] = int(chosen_units[i])
chosen_units = np.array(chosen_units)

# Open up the hdf5 file and extract variable values
def build_bin_resp_array(spike_array, pre_stim = 2000, window_size = 250, step_size = 25):
    """
    return:
        basline_resp [trials]; binned_taste_resp [trials, units, bins]
    """
    x = np.arange(0, spike_array.shape[-1], step_size)

    binned_taste_resps = [1000.0*np.mean(spike_array[:, :, s:s+window_size], axis = 2) for s in x]
    
    return np.moveaxis(np.array(binned_taste_resps), 0, -1)

spike_array = trains_dig_in[0].spike_array[:]
response = [build_bin_resp_array(trains_dig_in[i].spike_array[:], 
                                 pre_stim = 2000, 
                                 window_size = 250, 
                                 step_size = 25) \
            for i in range(len(trains_dig_in))]

# Extract neural response data from hdf5 file
num_units = len(chosen_units)
num_tastes = len(trains_dig_in)
dur = trains_dig_in[0].spike_array[:].shape[-1]
x = np.arange(0, dur-params[0]+1, params[1]) - pre_stim 
plot_places = np.where((x>=-1000)*(x<=4000))[0]
dir_name = "datashader_temp"

for i in chosen_units:
    fig, ax = plt.subplots(nrows = 1, ncols=2, squeeze=False, figsize = (18, 6))

	# First plot
    ax[0,0].set_title('Unit: %i, Window size: %i ms, Step size: %i ms' % (chosen_units[i], params[0], params[1]))
    for j, resp in enumerate(response): #j in plot_tastes_dig_in:
        if j in plot_tastes_dig_in:
            ax[0,0].plot(x[plot_places], np.mean(resp[:, i, plot_places], axis = 0), label = identities[j])
    ax[0,0].set_xlabel('Time from taste delivery (ms)')
    ax[0,0].set_ylabel('Firing rate (Hz)')
    ax[0,0].legend(loc='upper left', fontsize=10)

	# Second plot
    exec('waveforms = hf5.root.sorted_units.unit%03d.waveforms[:]' % (i))
    x1 = np.arange(waveforms.shape[1]/10) + 1
    convert_waveforms2img(waveforms, x1, dir_name = dir_name)
    img = imread(dir_name + "/tempfile.png")
    t = np.arange(waveforms.shape[1]/10)
    ax[0,1].imshow(img) 
    ax[0,1].set_xlabel('Time (samples (30 per ms))')
    ax[0,1].set_ylabel('Voltage (microvolts)')
    ax[0,1].set_title('Unit %i, total waveforms = %i' % (chosen_units[i], waveforms.shape[0]))
    plt.tight_layout()
    fig.savefig('./overlay_PSTH/' + '/Unit%i.png' % (chosen_units[i]), bbox_inches = 'tight')
    plt.close("all")

# Close hdf5 file
hf5.close()

# Also remove the directory with the temporary image files
shutil.rmtree(dir_name, ignore_errors = True)

