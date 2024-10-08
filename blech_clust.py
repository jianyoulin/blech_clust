# Necessary python modules
import easygui
import os
import tables
import sys
import numpy as np
import pandas as pd
import multiprocessing
from pathlib import Path

# Necessary blech_clust modules
import read_file
from write_file import make_powershell_parallel_script

# get path for blech_clust folder
blech_clust_dir = os.getcwd()

# Get name of directory with the data files
try:
    dir_name = sys.argv[1]
    if dir_name == '-f':
        dir_name = easygui.diropenbox('Select the dir path where data are saved')
except:
    dir_name = easygui.diropenbox('Select the dir path where data are saved')

# Get the type of data files (.rhd or .dat)
file_type = easygui.multchoicebox(msg = 'What type of files am I dealing with?', 
                                  choices = ('one file per channel', '.dat', '.rhd'))

# Change to that directory
os.chdir(dir_name)

# Get the names of all files in this directory
file_list = os.listdir('./')

# # Grab directory name to create the name of the hdf5 file
# hdf5_name = str.split(dir_name, '/')

# Create hdf5 file, and make groups for raw data, raw emgs, digital outputs and digital inputs, and close
# hf5 = tables.open_file(hdf5_name[-1]+'.h5', 'w', title = hdf5_name[-1])
hf5 = tables.open_file(Path(dir_name).name +'.h5', 'w', title = Path(dir_name).name)
hf5.create_group('/', 'raw')
hf5.create_group('/', 'raw_emg')
hf5.create_group('/', 'digital_in')
hf5.create_group('/', 'digital_out')
hf5.close()

# Create directories to store waveforms, spike times, clustering results, and plots
os.mkdir('spike_waveforms')
os.mkdir('spike_times')
os.mkdir('clustering_results')
os.mkdir('Plots')

# Get the amplifier ports used
ports = list(set(f[4] for f in file_list if f[:3] == 'amp'))
# Sort the ports in alphabetical order
ports.sort()
print("Used Ports: {}".format(ports))

# Pull out the digital input channels used, and convert them to integers
dig_in = list(set(f[10:12] for f in file_list if f[:9] == 'board-DIN'))
for i in range(len(dig_in)):
    print(dig_in[i][:]) 
    dig_in[i] = int(dig_in[i][:])
dig_in.sort()
print("Used dig-ins: {}".format(dig_in))

# Read the amplifier sampling rate from info.rhd - look at Intan's website for structure of header files
sampling_rate = np.fromfile('info.rhd', dtype = np.dtype('float32'))
sampling_rate = int(sampling_rate[2])   

# Check with user to see if the right ports, inputs and sampling rate were identified. Screw user if something was wrong, and terminate blech_clust
check = easygui.ynbox(msg = 'Ports used: ' + str(ports) + '\n' + 'Sampling rate: ' + str(sampling_rate) + ' Hz' + '\n' + 'Digital inputs on Intan board: ' + str(dig_in), title = 'Check parameters from your recordings!')

# Go ahead only if the user approves by saying yes
if check:
    pass
else:
    print("Well, if you don't agree, blech_clust can't do much!")
    sys.exit()

# Get the electrode channels on each port
e_channels = {}
for port in ports:
    e_channels[port] = list(set(int(f[6:9]) for f in file_list if f[:5] == 'amp-{}'.format(port)))


# Get the emg electrode ports and channel numbers from the user
# If only one amplifier port was used in the experiment, that's the emg_port. Else ask the user to specify
emg_port = ''
if len(ports) == 1:
    emg_port = list(ports[0])
else:
    emg_port = easygui.multchoicebox(msg = """Which amplifier port were the EMG electrodes hooked up to? 
                                              Just choose any amplifier port 
                                              if you did not hook up an EMG at all...""", 
                                     choices = tuple(ports))
# Now get the emg channel numbers, and convert them to integers
emg_channels = easygui.multchoicebox(msg = 'Choose the channel numbers for the EMG electrodes. Click clear all and ok if you did not use an EMG electrode', 
                                     choices = tuple([i for i in range(len(e_channels[emg_port[0]]))]))
if emg_channels:
    for i in range(len(emg_channels)):
        emg_channels[i] = int(emg_channels[i])
# set emg_channels to an empty list if no channels were chosen
if emg_channels is None:
    emg_channels = []
emg_channels.sort()
print('EMG channels', emg_channels)

# assuming EMG channels from the same headstage
# adjusting e_channels by removing emg channels
for p in emg_port:
    for c in emg_channels:
        e_channels[p].remove(c)

# Create arrays for each electrode
read_file.create_hdf_arrays(Path(dir_name).name+'.h5', ports, dig_in, e_channels, emg_port, emg_channels)

# Read data files, and append to electrode arrays
if file_type[0] == 'one file per channel':
    read_file.read_files(Path(dir_name).name+'.h5', ports, dig_in, e_channels, emg_port, emg_channels)
else:
    print("Only files structured as one file per channel can be read at this time...")
    sys.exit() # Terminate blech_clust if something else has been used - to be changed later

# Read in clustering parameters
clustering_params = easygui.multenterbox(msg = 'Fill in the parameters for clustering (using a GMM)', 
                                         fields = ['Maximum number of clusters', 
                                                   'Maximum number of iterations (1000 is more than enough)', 
                                                   'Convergence criterion (usually 0.0001)', 
                                                   'Number of random restarts for GMM (10 is more than enough)'],
                                         values = [10, 1000, 0.0001, 10])
# Read in data cleaning parameters (to account for cases when the headstage fell off mid-experiment)
data_params = easygui.multenterbox(msg = 'Fill in the parameters for cleaning your data in case the head stage fell off', 
                                   fields = ['Voltage cutoff for disconnected headstage noise (in microV, usually 1500)', 
                                             'Maximum rate of cutoff breaches per sec (something like 0.2 is good if 1500 microV is the cutoff)', 
                                             'Maximum number of allowed seconds with at least 1 cutoff breach (10 is good for a 30-60 min recording)', 
                                             'Maximum allowed average number of cutoff breaches per sec (20 is a good number)', 
                                             'Intra-cluster waveform amplitude SD cutoff - larger waveforms will be thrown out (3 would be a good number)'],
                                    values = [3000, 2, 20, 40, 3])
# Ask the user for the bandpass filter frequencies for pulling out spikes
bandpass_params = easygui.multenterbox(msg = "Fill in the lower and upper frequencies for the bandpass filter for spike sorting", 
                                       fields = ['Lower frequency cutoff (Hz)', 'Upper frequency cutoff (Hz)'],
                                       values = [300, 3000])
# Ask the user for the size of the spike snapshot to be used for sorting
spike_snapshot = easygui.multenterbox(msg = "Fill in the size of the spike snapshot you want to use for sorting (use steps of 0.5ms - like 0.5, 1, 1.5, ..)", 
                                      fields = ['Time before spike minimum (ms)', 'Time after spike minimum (ms)'],
                                      values = [0.5, 1])

# And print them to a blech_params file
f = open(Path(dir_name).name+'.params', 'w')
for i in clustering_params:
    print(i, file=f)
for i in data_params:
    print(i, file=f)
for i in bandpass_params:
    print(i, file=f)
for i in spike_snapshot:
    print(i, file=f)
print(sampling_rate, file=f)
f.close()

# saving channel maps
temp_channels = {}
for port in ports:
    temp_channels[port] = list(set(int(f[6:9]) for f in file_list if f[:5] == 'amp-{}'.format(port)))

channel_map_list = []
for p in ports:
    for e, eds in enumerate(temp_channels[p]):
        if (p == emg_port[0]) and (eds in emg_channels): 
            channel_map_list.append(','.join([p, str(e), str(eds), 'EMG']))
        else:
            channel_map_list.append(','.join([p, str(e), str(eds), 'Electrode']))

areas = easygui.multenterbox(msg = 'Fill in areas the electrode is recording)',
                             fields = channel_map_list,
                             values = ['GC' if 'EMG' not in channel else 'Muscle' \
                                       for channel in channel_map_list])

channel_map_dict = {'headstage_port':[], 'amp_ch':[], 'electrode_type':[], 
                    'electrode_num': [], 'area':[]}
def fill_dict(map, p, e, eds, e_type, area):
    map['headstage_port'].append(p)
    map['amp_ch'].append(e)
    map['electrode_type'].append(e_type)
    map['electrode_num'].append(eds)
    map['area'].append(area)

for p in ports:
    for e, eds in enumerate(temp_channels[p]):
        if (p == emg_port[0]) and (eds in emg_channels):
            fill_dict(channel_map_dict, p, e, eds, 'EMG', areas[e])
        else:
            fill_dict(channel_map_dict, p, e, eds, 'Electrode', areas[e])
channel_map_df = pd.DataFrame(channel_map_dict)
# save to data dir
channel_map_df.to_csv(os.path.join(dir_name, 'channel_map.csv'))




# Make a directory for dumping files talking about memory usage in blech_process.py
os.mkdir('memory_monitor_clustering')

# # Ask for the HPC queue to use - was in previous version, now just use all.q
# clustering = easygui.multchoicebox(msg = 'Which method do you want to use for clustering waveforms?', choices = ('PCA', 'UMAP'))

# # Grab Brandeis unet username
# username = easygui.multenterbox(msg = 'Enter your Brandeis/Jetstream/personal computer id', fields = ['unet username'])

# Dump shell file for running array job on the user's blech_clust folder on the desktop
try:
    os.chdir(blech_clust_dir) #'/home/%s/Desktop/blech_clust' % username[0])
except:
    os.chdir('/mnt/c/Users/jiany/Desktop/blech_clust')

# Dump shell file(s) for running GNU parallel job on the user's blech_clust folder on the desktop
# First get number of CPUs - parallel be asked to run num_cpu-1 threads in parallel
num_cpu = multiprocessing.cpu_count()//2 # number of cpu cores to be used in clustering
# Then produce the file generating the parallel command

# n_electrodes = 0
# for port in ports:
#     n_electrodes = n_electrodes + len(e_channels[port])

# f = open('blech_clust_jetstream_parallel.sh', 'w')
# # print("parallel -k -j {:d} --noswap --load 100% --progress --memfree 4G --retry-failed --joblog {:s}/results.log bash blech_clust_jetstream_parallel1.sh ::: {{1..{:d}}}".format(int(num_cpu)-1, dir_name, int(n_electrodes-len(emg_channels))), file = f)

# print("parallel -k -j {:d} --noswap --load 100% --progress --memfree 4G --retry-failed --joblog {:s}{}results.log bash blech_clust_jetstream_parallel1.sh ::: {{1..{:d}}}".format(int(num_cpu), dir_name, os.path.sep, int(n_electrodes-len(emg_channels))), file = f)

# print("parallel -k -j {:d} --noswap --load 100% --progress --memfree 4G --retry-failed "\
#       "--joblog {:s}{}results.log bash blech_clust_jetstream_parallel1.sh ::: {{1..{:d}}}".\
#         format(int(num_cpu), dir_name, os.path.sep, int(n_electrodes-len(emg_channels))), file = f)
# f.close()
all_electrodes = np.concatenate(tuple(e_channels[k] for k in e_channels.keys()))

if 'win' in sys.platform:
    # make parallel script to clustering waveforms with multiple cores
    make_powershell_parallel_script(electrodes = all_electrodes, 
                                    num_cpu = num_cpu, 
                                    process_path = blech_clust_dir, 
                                    process_code = 'blech_process.py')
    # make parallel script to plot spike scatters using umap with multiple cores
    make_powershell_parallel_script(electrodes = all_electrodes, 
                                    num_cpu = num_cpu, 
                                    process_path = blech_clust_dir, 
                                    process_code = 'umap_spike_scatter.py')
    
else:
    f = open('blech_clust.sh', 'w')
    print("export OMP_NUM_THREADS=1", file = f)
    print("cd {}".format(os.getcwd()), file=f)
    #print("cd /home/%s/Desktop/blech_clust" % username[0], file=f)
    print("python blech_process.py", file=f)
    f.close()

    f = open('blech_clust_jetstream_parallel.sh', 'w')
    print("parallel -k -j {:d} --noswap --load 100% --progress --memfree 4G --retry-failed "\
            "--joblog {:s}{}results.log bash blech_clust_jetstream_parallel1.sh ::: {{{}}}"\
            .format(num_cpu, dir_name, os.path.sep, ",".join([str(x) for x in all_electrodes]))
            , file = f)
    f.close()

    # Then produce the file that runs blech_process.py
    f = open('blech_clust_jetstream_parallel1.sh', 'w')
    print("export OMP_NUM_THREADS=1", file = f)
    print("python blech_process.py $1 {}".format('PCA'), file = f)
    f.close()

    # produce bash file to run umap_spike_scatter and parallelize them
    f = open('bash_umap_spike_scatter.sh', 'w')
    print("parallel -k -j {:d} --load 100% --progress --memfree 4G --retry-failed "\
    "python umap_spike_scatter_parallel.py ::: {{{}}}"\
        .format(num_cpu, ",".join([str(x) for x in all_electrodes])),
     file = f)
    f.close()

# Dump the directory name where blech_process has to cd
f = open('blech.dir', 'w')
print(dir_name, file=f)
f.close()

# Dump the directory name where umap_scatter has to cd
f = open('umap_dir.txt', 'w')
print(dir_name, file=f)
f.close()

if 'win' in sys.platform:
    print('Since you are using Windows OS, please do the following for sorting processing')
    print('Next: run blech_common_avg_reference.py in Powershell 7.X terminal')
    print('Next: run blech_clust_PS_parallel.ps1 in Powershell 7.X terminal')
    print('Then: run blech_clust_spike_scatter_PS_parallel.ps1 in Powershell 7.X terminal')

else:
    print('cd to blech_clust directory')
    print('Next: bash blech_clust_jetstream_parallel.sh')
    print('Then: bash bash_umap_spike_scatter.sh')
    
# print("Now logout of the compute node and go back to the login node. Then go to the bkech_clust folder on your desktop and say: qsub -t 1-"+str(len(ports)*32-len(emg_channels))+" -q all.q -ckpt reloc -l mem_free=4G -l mem_token=4G blech_clust.sh")




# def make_powershell_parallel_script(electrodes = None, num_cpu = None, process_path = None, process_code = None):
#     """
#     electrodes:: a list of electrode numbers (int)
#     """
#     n_cores_to_be_used = num_cpu
#     path_ = os.path.join(process_path, process_code)
#     if 'umap' in process_code:
#         f = open('blech_clust_spike_scatter_PS_parallel.ps1', 'w')
#     else:
#         f = open('blech_clust_PS_parallel.ps1', 'w')
#     print("# Define the scripts and their arguments",  file = f)
#     print("$scripts = @(", file = f)
#     for i in electrodes:
#         print('\t@{Path = "%s"; Args = @("%i")}' % (path_, i), file=f)
#     print(")", file = f)
#     print('\n', file = f)
#     print("# Define the throttle limit",  file = f)
#     print('$throttleLimit = %i' % n_cores_to_be_used, file = f)
#     print('\n', file = f)
#     print('# Start the jobs with throttle limit', file = f)
#     print('$jobs = foreach ($script in $scripts) {', file=f)
#     print('\tStart-ThreadJob -ScriptBlock {', file=f)
#     print('\t\tparam ($scriptPath, $scriptArgs)', file=f)
#     print('\t\tpython $scriptPath $scriptArgs', file=f)
#     print('\t} -ArgumentList $script.Path, ($script.Args -join " ") -ThrottleLimit $throttleLimit', file=f)
#     print('}', file=f)
#     print('\n', file = f)
#     print('$jobs | ForEach-Object { $_ | Wait-Job }', file=f)
#     print('$jobs | ForEach-Object {', file=f)
#     print('\t$output = $_ | Receive-Job', file=f)
#     print('\tWrite-Output "Output from $($_.Name):"', file=f)
#     print('\tWrite-Output $output', file=f)
#     print('}', file=f)
#     print('\n', file = f)
#     print('# Clean up the jobs', file=f)
#     print('$jobs | ForEach-Object { $_ | Remove-Job }', file=f)
#     f.close()



