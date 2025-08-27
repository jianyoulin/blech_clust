 # Subtracts the two emg signals and filters and saves the results.

# Import stuff
import numpy as np
from scipy.signal import butter, filtfilt, periodogram
import easygui
import os
import matplotlib.pyplot as plt

# Ask for the directory where the data (emg_data.npy) sits
dir_name = easygui.diropenbox()
os.chdir(dir_name)

# Load the data
emg_data = np.load('emg_data.npy') # emg_data[emg#, n_tastes, n_trials, time]

# Ask the user for stimulus delivery time in each trial, and convert to an integer
pre_stim = easygui.multenterbox(msg = 'Enter the pre-stimulus time included in each trial', fields = ['Pre-stimulus time (ms)']) 
pre_stim = int(pre_stim[0])

# Get coefficients for Butterworth filters
m, n = butter(2, 2.0*300.0/1000.0, 'highpass')
c, d = butter(2, 2.0*15.0/1000.0, 'lowpass')

# check how many EMG channels used in this experiment
check = easygui.ynbox(msg = 'Did you have more than one EMG channel?', title = 'Check YES if you did')

# Bandpass filter the emg signals, and store them in a numpy array. Low pass filter the bandpassed signals, and store them in another array
emg_filt = np.zeros(emg_data.shape[1:])
env = np.zeros(emg_data.shape[1:])
for i in range(emg_data.shape[1]):
    for j in range(emg_data.shape[2]):
        if check:
            emg_filt[i, j, :] = filtfilt(m, n, emg_data[0, i, j, :] - emg_data[1, i, j, :])
        else:
            emg_filt[i, j, :] = filtfilt(m, n, emg_data[0, i, j, :])
        env[i, j, :] = filtfilt(c, d, np.abs(emg_filt[i, j, :]))    
            
# Get mean and std of baseline emg activity, and use it to select trials that have significant post stimulus activity
sig_trials = np.zeros((emg_data.shape[1], emg_data.shape[2]))
m = np.mean(np.abs(emg_filt[:, :, :pre_stim]))
s = np.std(np.abs(emg_filt[:, :, :pre_stim]))
for i in range(emg_data.shape[1]):
    for j in range(emg_data.shape[2]):
        if np.mean(np.abs(emg_filt[i, j, pre_stim:])) > m and np.max(np.abs(emg_filt[i, j, pre_stim:])) > m + 4.0*s:
            sig_trials[i, j] = 1	
    
# Save the highpass filtered signal, the envelope and the indicator of significant trials as a np array
np.save('emg_filt.npy', emg_filt)
np.save('env.npy', env)
np.save('sig_trials.npy', sig_trials)

def plot_emg_trials(emg_data: np.ndarray, trials: int = 10) -> None:
    """
    Plots multiple EMG trials in a single column of subplots.

    This function takes a 3D NumPy array of EMG data [taste, trials, time], where each row
    represents a single trial. It creates a figure with a subplot for each
    trial, arranged in a single column, making it easy to visualize and
    compare the data from all trials.

    Args:
        emg_data (np.ndarray): A 2D NumPy array where the rows correspond
                               to different trials and the columns are the
                               data points for each trial.
    """
    # Get the number of trials from the input data shape
    num_tastes = emg_data.shape[0]
    num_trials = emg_data.shape[1]

    # Create a figure and a set of subplots
    # The layout is (num_trials) rows and 1 column.
    # sharex=True ensures that all x-axes are aligned, which is helpful
    # for comparing data across trials.
    fig, axes = plt.subplots(
        nrows=trials,
        ncols=num_tastes,
        #figsize=(10, 2 * trials),  # Adjust figure size for better visibility
        sharex=True,
        sharey='row'
    )

    # Set the main title for the entire figure
    fig.suptitle(
        'Filtered EMG Data: first {} trials'.format(trials),
        fontsize=12,
        y=0.99  # Adjust title position
    )

    # Iterate through each trial and its corresponding subplot axis
    for i in range(trials):
        for t in range(num_tastes): 
            # Plot the data for the current trial on the current subplot
            # Using a simple line plot
            axes[i,t].plot(emg_data[t, i, :])

            # Set a title for each individual subplot
            if t == 0:
                axes[i,t].set_ylabel(f'Tr {i + 1}', fontsize=6)
            
            # set title for each taste
            if i == 0:
                axes[i,t].set_title(f'Taste {t + 1}', fontsize=10)
            
            # remove ytick labels for all trials
            axes[i,t].set_yticklabels('')
            # Remove the top and right spines for a cleaner look
            # ax.spines['top'].set_visible(False)
            # ax.spines['right'].set_visible(False)

            # # Add a horizontal line at y=0 for reference
            # ax[i,t].axhline(0, color='gray', linestyle='--', linewidth=0.8)
            
            # # Add grid lines for easier reading
            # ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # # Set labels for the last subplot only to avoid clutter
    # axes[-1].set_xlabel('Sample Number')
    # axes[num_trials // 2].set_ylabel('Amplitude (mV)')
    
    # Adjust the spacing between subplots to prevent titles from overlapping
    plt.tight_layout()

    # save the figure
    fig.savefig('filted_EMG_first_{}_trials.png'.format(trials), bbox_inches='tight')
    
plot_emg_trials(emg_filt, trials = 10)

    
                    




