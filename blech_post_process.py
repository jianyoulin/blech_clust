# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:44:35 2019
2021/8/9
This code is modified to merge splited clusters or split a merged cluster 
also save a pickle file with details of unit description
2021/08/12 difference from V2 to V3: save units using a function

@author: jiany
"""

import os
import tables
import numpy as np
import easygui
import pickle
import ast
import pylab as plt
from sklearn.mixture import GaussianMixture
import blech_waveforms_datashader

# Get directory where the hdf5 file sits, and change to that directory
try: # Read root_data_dir.txt, and cd to that directory
    f = open('root_data_dir.dir', 'r')
    dir_name = []
    for line in f.readlines():
        dir_name.append(line)
    f.close()
    dir_name = easygui.diropenbox(msg='Select data folder', default = dir_name[0][:-1])
except:
    dir_name = easygui.diropenbox(msg='Select data folder') # "E:\testing_hdf5s\JY07_Clustering" #
#dir_name = easygui.diropenbox() # '/mnt/e/testing_hdf5s/for_clustering' #
os.chdir(dir_name)

# Clean up the memory monitor files, pass if clean up has been done already
if not os.path.exists('./memory_monitor_clustering/memory_usage.txt'):
    file_list = os.listdir('./memory_monitor_clustering')
    f = open('./memory_monitor_clustering/memory_usage.txt', 'w')
    for files in file_list:
        try:
            mem_usage = np.loadtxt('./memory_monitor_clustering/' + files)
            print('electrode'+files[:-4], '\t', str(mem_usage)+'MB', file=f)
            os.remove('./memory_monitor_clustering/{}'.format(files))
        except:
            pass    
    f.close()

try:
    with open('unit_details.pkl', 'rb') as pkl_file:
        unit_details = pickle.load(pkl_file)
except:
    unit_details = {'unit_number': [],
                    'electrode_number': [],
                    'fast_spiking': [],
                    'regular_spiking': [],
                    'single_unit': [],
                    'waveform_count': []}


# Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
    if files[-2:] == 'h5':
        hdf5_name = files

# Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

# Delete the raw node, if it exists in the hdf5 file, to cut down on file size
try:
    hf5.remove_node('/raw', recursive = 1)
    # And if successful, close the currently open hdf5 file and ptrepack the file
    hf5.close()
    print("Raw recordings removed")
    os.system("ptrepack --chunkshape=auto --propindexes --complevel=9 --complib=blosc " + hdf5_name + " " + hdf5_name[:-3] + "_repacked.h5")
    # Delete the old (raw and big) hdf5 file
    os.system("rm " + hdf5_name)
    # And open the new, repacked file
    hf5 = tables.open_file(hdf5_name[:-3] + "_repacked.h5", 'r+')
    print("File repacked")
except:
    print("Raw recordings have already been removed, so moving on ..")

# Make the sorted_units group in the hdf5 file if it doesn't already exist
try:
    hf5.create_group('/', 'sorted_units')
except:
    pass

# Define a unit_descriptor class to be used to add things (anything!) about the sorted units to a pytables table
class unit_descriptor(tables.IsDescription):
    unit_number = tables.Int32Col()
    electrode_number = tables.Int32Col()
    single_unit = tables.Int32Col()
    regular_spiking = tables.Int32Col()
    fast_spiking = tables.Int32Col()
    waveform_count = tables.Int32Col()

# Define a function to get unit description
def get_unit_info(unit_num, electrode_num, s_u, u_t, waveform_num):
    """
    s_u: single_unit; u_t: unit_type
    """
    description = easygui.multenterbox(msg = 'Description of this unit',
                                         fields = ['unit_number', 
                                                   'electrode_number',
                                                   'fast_spiking (Yes: 1; No: 0)',
                                                   'regular_spiking (Yes: 1; No: 0)',
                                                   'single_unit (Yes: 1; No: 0)',
                                                   'waveform_count'],
                                         values = [unit_num, electrode_num, 
                                                    1 if u_t=="fast_spiking" else 0,
                                                    1 if u_t=="regular_spiking" else 0,
                                                    1 if s_u=="True" else 0, waveform_num])
    return description
                                         

# Define a function to get unit description
def get_unit_info1(unit_num, electrode_num, s_u, u_t, waveform_num):
    """
    s_u: single_unit; u_t: unit_type
    """
    description = easygui.ynbox(msg = 'unit_number: {}\n\n'.format(unit_num)+\
                                'electrode_number: {}\n\n'.format(electrode_num)+\
                                'fast_spiking (Yes: 1; No: 0): {}\n\n'.format(1 if u_t=="fast_spiking" else 0)+\
                                'regular_spiking (Yes: 1; No: 0): {}\n\n'.format(1 if u_t=="regular_spiking" else 0)+\
                                'single_unit (Yes: 1; No: 0): {}\n\n'.format(1 if s_u=="True" else 0, waveform_num)+\
                                'waveform_count: {}'.format(waveform_num),
                          title = 'Check unit information from you selection')
    return description

def msg_text(v2=None, v1=None, lth=None):
    t = 'My merged cluster has {} percent (<2ms)'.format(round(v2,2))+\
        ' and {} percent (<1ms) ISI violations'.format(round(v1,2))+\
        ' out of {} total waveforms.'.format(lth)+\
        ' I want to still merge these clusters into one unit (True = Yes, False = No)'
    return t

def title_text(c=None, v1=None, v2=None, points=None):
    t = 'Split Cluster{:d},'.format(c)+\
        ' 2ms ISI violations={:.1f} percent\n'.format(v2)+\
        ' 1ms ISI violations={:.1f} percent\n'.format(v1)+\
        ' Number of waveforms={:d}'.format(points)
    return t

# Make a table under /sorted_units describing the sorted units. If unit_descriptor already exists, just open it up in the variable table
try:
    table = hf5.create_table('/', 'unit_descriptor', description = unit_descriptor)
except:
    table = hf5.root.unit_descriptor

def save_units(unit_description, max_unit, unit_name=None, electrode_num = None, 
               u_waveforms=None, u_times=None):
    """
    1) save unit waveforms and time to HDF5 file under sorted units node
    2) save unit description to descriptor table in HDF5 file
    3) save unit description to descriptor dictionary
    """
    hf5.create_group('/sorted_units', unit_name)
    #unit_waveforms = spike_waveforms[np.where(predictions == int(clusters[0]))[0], :]   # Waveforms of originally chosen cluster
    #unit_waveforms = u_waveforms[np.where(split_predictions == chosen_split[0])[0], :]   # Subsetting this set of waveforms to include only the chosen split
    #unit_times = spike_times[np.where(predictions == int(clusters[0]))[0]]          # Do the same thing for the spike times
    #unit_times = u_times[np.where(split_predictions == chosen_split[0])[0]]
    waveforms = hf5.create_array('/sorted_units/%s' % unit_name, 'waveforms', u_waveforms)
    times = hf5.create_array('/sorted_units/%s' % unit_name, 'times', u_times)
    unit_description['unit_number'] = int(unit_name[4:])
    unit_description['electrode_number'] = electrode_num
    single_unit = easygui.multchoicebox(msg = 'I am almost-SURE that this is a beautiful single unit (True = Yes, False = No)', choices = ('True', 'False'))
    unit_description['single_unit'] = int(ast.literal_eval(single_unit[0]))
    # If the user says that this is a single unit, ask them whether its regular or fast spiking
    unit_description['regular_spiking'] = 0
    unit_description['fast_spiking'] = 0
    unit_description['waveform_count'] = len(unit_times)
    if int(ast.literal_eval(single_unit[0])):
        unit_type = easygui.multchoicebox(msg = 'What type of unit is this (Regular spiking = Pyramidal cells, Fast spiking = PV+ interneurons)?', choices = ('regular_spiking', 'fast_spiking'))
        unit_description[unit_type[0]] = 1  
    else: unit_type = ['multi_unit']
    unit_description.append()
    table.flush()
    hf5.flush()
    
    # add info to unit_details dictionary
    this_unit = get_unit_info(unit_name, electrode_num, single_unit[0], unit_type[0], len(unit_times))
    print(this_unit)
    for i, j in enumerate(list(unit_details.keys())):
        unit_details[j].append(this_unit[i])
    # Finally increment max_unit and create a new unit name
    max_unit += 1
    unit_name = 'unit%03d' % int(max_unit + 1)
    return max_unit, unit_name

    # Get a new unit_descriptor table row for this new unit
    #unit_description = table.row

# Run an infinite loop as long as the user wants to pick clusters from the electrodes   
while True:
    # Get list of existing nodes/groups under /sorted_units
    node_list = hf5.list_nodes('/sorted_units')

    # If node_list is empty, start naming units from 000
    unit_name = ''
    max_unit = 0
    if len(node_list) == 0:     
        unit_name = 'unit%03d' % 0
    # Else name the new unit by incrementing the last unit by 1 
    else:
        unit_numbers = []
        for node in node_list:
            unit_numbers.append(node._v_pathname.split('/')[-1][-3:])
            unit_numbers[-1] = int(unit_numbers[-1])
        unit_numbers = np.array(unit_numbers)
        max_unit = np.max(unit_numbers)
        unit_name = 'unit%03d' % int(max_unit + 1)

    # Get a new unit_descriptor table row for this new unit
    unit_description = table.row
    
    # Get electrode number from user
    electrode_num = easygui.multenterbox(msg = 'Which electrode do you want to choose? Hit cancel to exit', 
                                         fields = ['Electrode #'])
    # Break if wrong input/cancel command was given
    try:
        electrode_num = int(electrode_num[0])
    except:
        break
    
    # Get the number of clusters in the chosen solution
    num_clusters = easygui.multenterbox(msg = 'Which solution do you want to choose for electrode %i?' % electrode_num, 
                                        fields = ['Number of clusters in the solution'])
    num_clusters = int(num_clusters[0])

    # Load data from the chosen electrode and solution
    spike_waveforms = np.load('./spike_waveforms/electrode%i/spike_waveforms.npy' % electrode_num) # [# of waveforms, 450 time points]
    spike_times = np.load('./spike_times/electrode%i/spike_times.npy' % electrode_num) # [# of time points]
    pca_slices = np.load('./spike_waveforms/electrode%i/pca_waveforms.npy' % electrode_num) # [# of waveforms, 3 pca components]
    energy = np.load('./spike_waveforms/electrode%i/energy.npy' % electrode_num)
    amplitudes = np.load('./spike_waveforms/electrode%i/spike_amplitudes.npy' % electrode_num)
    predictions = np.load('./clustering_results/electrode%i/clusters%i/predictions.npy' % (electrode_num, num_clusters))

    # Get cluster choices from the chosen solution
    clusters = easygui.multchoicebox(msg = 'Which clusters do you want to choose?', choices = tuple([str(i) for i in range(int(np.max(predictions) + 1))]))
    
    # if only one cluster chosen, to split or to be a unit
    if len(clusters) == 1:
        re_cluster = easygui.multchoicebox(msg = 'I want to split this cluster (True = Yes, False = No)', choices = ('True', 'False'))
        re_cluster = ast.literal_eval(re_cluster[0])
#################################################        
        # If the user asked to split/re-cluster, ask them for the clustering parameters and perform clustering
        split_predictions = []
        chosen_split = 0
        if re_cluster: #I want to split this cluster
            # Get clustering parameters from user
            clustering_params = easygui.multenterbox(msg = 'Fill in the parameters for re-clustering (using a GMM)',
                                                     fields = ['Number of clusters', 
                                                               'Maximum number of iterations (1000 is more than enough)',
                                                               'Convergence criterion (usually 0.0001)', 
                                                               'Number of random restarts for GMM (10 is more than enough)'],
                                                     values = [4, 1000, 0.000001, 10])
            n_clusters = int(clustering_params[0])
            n_iter = int(clustering_params[1])
            thresh = float(clustering_params[2])
            n_restarts = int(clustering_params[3]) 

            # Make data array to be put through the GMM - 5 components: 3 PCs, scaled energy, amplitude
            this_cluster = np.where(predictions == int(clusters[0]))[0] # clusters[0] since only one cluster is chosen in this if statement
            n_pc = 3
            data = np.zeros((len(this_cluster), n_pc + 2))  
            data[:,2:] = pca_slices[this_cluster,:n_pc]
            data[:,0] = energy[this_cluster]/np.max(energy[this_cluster])
            data[:,1] = np.abs(amplitudes[this_cluster])/np.max(np.abs(amplitudes[this_cluster]))

            # Cluster the data
            g = GaussianMixture(n_components = n_clusters, covariance_type = 'full', tol = thresh, max_iter = n_iter, n_init = n_restarts)
            g.fit(data)
        
            # Show the cluster plots if the solution converged
            if g.converged_:
                split_predictions = g.predict(data) # predictions of this selected cluster (from 0 to n_cluster)
                x = np.arange(len(spike_waveforms[0])/10) + 1
                for cluster in range(n_clusters):
                    split_points = np.where(split_predictions == cluster)[0] # obtain the indexes of each sub-clusters and plot them                
                    # plt.figure(cluster)
                    slices_dejittered = spike_waveforms[this_cluster, :]        # Waveforms and times from the chosen cluster
                    times_dejittered = spike_times[this_cluster]
                    times_dejittered = times_dejittered[split_points]       # Waveforms and times from the chosen split of the chosen cluster
                    ISIs = np.ediff1d(np.sort(times_dejittered))/30.0
                    violations1 = 100.0*float(np.sum(ISIs < 1.0)/split_points.shape[0])
                    violations2 = 100.0*float(np.sum(ISIs < 2.0)/split_points.shape[0])
                    fig, ax = blech_waveforms_datashader.waveforms_datashader(slices_dejittered[split_points, :], x)
                    # plt.plot(x-15, slices_dejittered[split_points, :].T, linewidth = 0.01, color = 'red')
                    ax.set_xlabel('Sample (30 samples per ms)')
                    ax.set_ylabel('Voltage (microvolts)')
                    ax.set_title(title_text(c=cluster, v1=violations1, v2=violations2, points=split_points.shape[0]))
                    #ax.set_title("Split Cluster{:d}, 2ms ISI violations={:.1f} percent".format(cluster, violations2) + "\n" + "1ms ISI violations={:.1f}%, Number of waveforms={:d}".format(violations1, split_points.shape[0]))
                    
            else:
                print("Solution did not converge - try again with higher number of iterations or lower convergence criterion")
                continue
            
            plt.show()

            
            # Ask the user for the split clusters they want to choose
            chosen_split = easygui.multchoicebox(msg = 'Which split clusters do you want to choose? Hit cancel to exit', 
                                                 choices = tuple([str(i) for i in range(n_clusters)]))
            try:
                chosen_split = [int(chosen_split[i]) for i in range(len(chosen_split))]
                split_merge = False
            except:
                continue


        # If the user re-clustered/split clusters, add the chosen clusters in split_clusters
            if len(chosen_split) == 1:
                unit_waveforms = spike_waveforms[np.where(predictions == int(clusters[0]))[0], :]   # Waveforms of originally chosen cluster
                unit_waveforms = unit_waveforms[np.where(split_predictions == chosen_split[0])[0], :]   # Subsetting this set of waveforms to include only the chosen split
                unit_times = spike_times[np.where(predictions == int(clusters[0]))[0]]          # Do the same thing for the spike times
                unit_times = unit_times[np.where(split_predictions == chosen_split[0])[0]]

                max_unit, unit_name = save_units(unit_description, max_unit, unit_name=unit_name, electrode_num = electrode_num, 
                           u_waveforms=unit_waveforms, u_times=unit_times)
                unit_description = table.row

                # hf5.create_group('/sorted_units', unit_name)
                # unit_waveforms = spike_waveforms[np.where(predictions == int(clusters[0]))[0], :]   # Waveforms of originally chosen cluster
                # unit_waveforms = unit_waveforms[np.where(split_predictions == chosen_split[0])[0], :]   # Subsetting this set of waveforms to include only the chosen split
                # unit_times = spike_times[np.where(predictions == int(clusters[0]))[0]]          # Do the same thing for the spike times
                # unit_times = unit_times[np.where(split_predictions == chosen_split[0])[0]]
                # waveforms = hf5.create_array('/sorted_units/%s' % unit_name, 'waveforms', unit_waveforms)
                # times = hf5.create_array('/sorted_units/%s' % unit_name, 'times', unit_times)
                # unit_description['unit_number'] = int(unit_name[4:-1])
                # unit_description['electrode_number'] = electrode_num
                # single_unit = easygui.multchoicebox(msg = 'I am almost-SURE that this is a beautiful single unit (True = Yes, False = No)', choices = ('True', 'False'))
                # unit_description['single_unit'] = int(ast.literal_eval(single_unit[0]))
                # # If the user says that this is a single unit, ask them whether its regular or fast spiking
                # unit_description['regular_spiking'] = 0
                # unit_description['fast_spiking'] = 0
                # unit_description['waveform_count'] = len(unit_times)
                # if int(ast.literal_eval(single_unit[0])):
                    # unit_type = easygui.multchoicebox(msg = 'What type of unit is this (Regular spiking = Pyramidal cells, Fast spiking = PV+ interneurons)?', choices = ('regular_spiking', 'fast_spiking'))
                    # unit_description[unit_type[0]] = 1      
                # unit_description.append()
                # table.flush()
                # hf5.flush()
                
                # # add info to unit_details dictionary
                # this_unit = get_unit_info(unit_name, electrode_num, len(unit_times))
                # print(this_unit)
                # for i, j in enumerate(list(unit_details.keys())):
                    # unit_details[j].append(this_unit[i])
                # # Finally increment max_unit and create a new unit name
                # max_unit += 1
                # unit_name = 'unit%03d' % int(max_unit + 1)

                # # Get a new unit_descriptor table row for this new unit
                # unit_description = table.row

            elif len(chosen_split) > 1:
                split_merge = easygui.multchoicebox(msg = 'I want to merge these splited-clusters into one unit (True = Yes, False = No)', choices = ('True', 'False'))
                split_merge = ast.literal_eval(split_merge[0])
                
                if split_merge: #I want to merge these splited-clusters
                    unit_waveforms = []
                    unit_times = []
                    cluster_unit_waveforms = spike_waveforms[np.where(predictions == int(clusters[0]))[0], :] # get the index of the original (level 1) chosen cluster
                    cluster_unit_times = spike_times[np.where(predictions == int(clusters[0]))[0]]
                    for split_cluster in chosen_split:
                        if len(unit_waveforms) == 0: # == []:
                            unit_waveforms = cluster_unit_waveforms[np.where(split_predictions == split_cluster)[0], :]
                            unit_times = cluster_unit_times[np.where(split_predictions == split_cluster)[0]]
                        else:
                            unit_waveforms = np.concatenate((unit_waveforms, cluster_unit_waveforms[np.where(split_predictions == split_cluster)[0], :]), axis=0)
                            unit_times = np.concatenate((unit_times, cluster_unit_times[np.where(split_predictions == split_cluster)[0]]))

                # Show the merged cluster to the user, and ask if they still want to merge
                    x = np.arange(len(unit_waveforms[0])/10) + 1
                    fig, ax = blech_waveforms_datashader.waveforms_datashader(unit_waveforms, x)
                    # plt.plot(x - 15, unit_waveforms[:, ::10].T, linewidth = 0.01, color = 'red')
                    ax.set_xlabel('Sample (30 samples per ms)')
                    ax.set_ylabel('Voltage (microvolts)')
                    ax.set_title('Merged cluster, No. of waveforms={:d}'.format(unit_waveforms.shape[0]))
                    plt.show()
     
                # Warn the user about the frequency of ISI violations in the merged unit
                    ISIs = np.ediff1d(np.sort(unit_times))/30.0
                    violations1 = 100.0*float(np.sum(ISIs < 1.0)/len(unit_times))
                    violations2 = 100.0*float(np.sum(ISIs < 2.0)/len(unit_times))
                    proceed = easygui.multchoicebox(msg = msg_text(v2=violations2, v1=violations1, lth=len(unit_times)), 
                                                    choices = ('True', 'False'))
                    #proceed = easygui.multchoicebox(msg = 'My merged cluster has %.1f percent (<2ms) and %.1f percent (<1ms) ISI violations out of %i total waveforms. I want to still merge these clusters into one unit (True = Yes, False = No)' % (violations2, violations1, len(unit_times)), choices = ('True', 'False'))
                    proceed = ast.literal_eval(proceed[0])

                    # Create unit if the user agrees to proceed, else include each split_cluster as a separate unit 
                    if proceed:
                        max_unit, unit_name = save_units(unit_description, max_unit, unit_name=unit_name, electrode_num = electrode_num, 
                                   u_waveforms=unit_waveforms, u_times=unit_times)
                        unit_description = table.row
                    
                        # hf5.create_group('/sorted_units', unit_name)
                        # waveforms = hf5.create_array('/sorted_units/%s' % unit_name, 'waveforms', unit_waveforms)
                        # times = hf5.create_array('/sorted_units/%s' % unit_name, 'times', unit_times)
                        # unit_description['unit_number'] = int(unit_name[4:-1])
                        # unit_description['electrode_number'] = electrode_num
                        # single_unit = easygui.multchoicebox(msg = 'I am almost-SURE that this is a beautiful single unit (True = Yes, False = No)', choices = ('True', 'False'))
                        # unit_description['single_unit'] = int(ast.literal_eval(single_unit[0]))
                        # # If the user says that this is a single unit, ask them whether its regular or fast spiking
                        # unit_description['regular_spiking'] = 0
                        # unit_description['fast_spiking'] = 0
                        # unit_description['waveform_count'] = len(unit_times)
                        # if int(ast.literal_eval(single_unit[0])):
                            # unit_type = easygui.multchoicebox(msg = 'What type of unit is this (Regular spiking = Pyramidal cells, Fast spiking = PV+ interneurons)?', choices = ('regular_spiking', 'fast_spiking'))
                            # unit_description[unit_type[0]] = 1
                        # unit_description.append()
                        # table.flush()
                        # hf5.flush()
                        
                        # # add info to unit_details dictionary
                        # this_unit = get_unit_info(unit_name, electrode_num, len(unit_times))
                        # for i, j in enumerate(list(unit_details.keys())):
                            # unit_details[j].append(this_unit[i])

                        # # Finally increment max_unit and create a new unit name
                        # max_unit += 1
                        # unit_name = 'unit%03d' % int(max_unit + 1)
                        # Get a new unit_descriptor table row for this new unit
                        # unit_description = table.row
                        
                else: # if not merge, then include each split_cluster as a separate unit
                    for split_cluster in split_merge:
                        
                        unit_waveforms = spike_waveforms[np.where(predictions == int(clusters[0]))[0], :]   # Waveforms of originally chosen cluster
                        unit_waveforms = unit_waveforms[np.where(split_predictions == split_cluster)[0], :] # Subsetting this set of waveforms to include only the chosen split
                        unit_times = spike_times[np.where(predictions == int(clusters[0]))[0]]          # Do the same thing for the spike times
                        unit_times = unit_times[np.where(split_predictions == split_cluster)[0]]
                        
                        max_unit, unit_name = save_units(unit_description, max_unit, unit_name=unit_name, electrode_num = electrode_num, 
                                   u_waveforms=unit_waveforms, u_times=unit_times)
                        unit_description = table.row
                         
                        # hf5.create_group('/sorted_units', unit_name)
                        # waveforms = hf5.create_array('/sorted_units/%s' % unit_name, 'waveforms', unit_waveforms)
                        # times = hf5.create_array('/sorted_units/%s' % unit_name, 'times', unit_times)
                        # unit_description['unit_number'] = int(unit_name[4:-1])
                        # unit_description['electrode_number'] = electrode_num
                        # single_unit = easygui.multchoicebox(msg = 'I am almost-SURE that electrode: %i cluster: %i split: %i is a beautiful single unit (True = Yes, False = No)' % (electrode_num, int(cluster), int(split_cluster)), choices = ('True', 'False'))
                        # unit_description['single_unit'] = int(ast.literal_eval(single_unit[0]))
                        # # If the user says that this is a single unit, ask them whether its regular or fast spiking
                        # unit_description['regular_spiking'] = 0
                        # unit_description['fast_spiking'] = 0
                        # unit_description['waveform_count'] = len(unit_times)
                        # if int(ast.literal_eval(single_unit[0])):
                            # unit_type = easygui.multchoicebox(msg = 'What type of unit is this (Regular spiking = Pyramidal cells, Fast spiking = PV+ interneurons)?', choices = ('regular_spiking', 'fast_spiking'))
                            # unit_description[unit_type[0]] = 1
                        # unit_description.append()
                        # table.flush()
                        # hf5.flush()             

                        # # add info to unit_details dictionary
                        # this_unit = get_unit_info(unit_name, electrode_num, len(unit_times))
                        # print(this_unit)
                        # for i, j in enumerate(list(unit_details.keys())):
                            # unit_details[j].append(this_unit[i])

                        # # Finally increment max_unit and create a new unit name
                        # max_unit += 1
                        # unit_name = 'unit%03d' % int(max_unit + 1)

                        # # Get a new unit_descriptor table row for this new unit
                        # unit_description = table.row
                #continue
            else: # if split=1 elif split>1 and then this else
                continue # go back to infinite while loop
        
        
        else: # If initially only 1 cluster was chosen (and it wasn't split), add that as a new unit in /sorted_units. 
              # Ask if the isolated unit is an almost-SURE single unit
            
            unit_waveforms = spike_waveforms[np.where(predictions == int(clusters[0]))[0], :]
            unit_times = spike_times[np.where(predictions == int(clusters[0]))[0]]

            max_unit, unit_name = save_units(unit_description, max_unit, unit_name=unit_name, electrode_num = electrode_num, 
                       u_waveforms=unit_waveforms, u_times=unit_times)
            unit_description = table.row
   
            
            # hf5.create_group('/sorted_units', unit_name)
            # waveforms = hf5.create_array('/sorted_units/%s' % unit_name, 'waveforms', unit_waveforms)
            # times = hf5.create_array('/sorted_units/%s' % unit_name, 'times', unit_times)
            # unit_description['unit_number'] = int(unit_name[4:-1])
            # unit_description['electrode_number'] = electrode_num
            # single_unit = easygui.multchoicebox(msg = 'I am almost-SURE that this is a beautiful single unit (True = Yes, False = No)', choices = ('True', 'False'))
            # unit_description['single_unit'] = int(ast.literal_eval(single_unit[0]))
            # # If the user says that this is a single unit, ask them whether its regular or fast spiking
            # unit_description['regular_spiking'] = 0
            # unit_description['fast_spiking'] = 0
            # unit_description['waveform_count'] = len(unit_times)
            # if int(ast.literal_eval(single_unit[0])):
                # unit_type = easygui.multchoicebox(msg = 'What type of unit is this (Regular spiking = Pyramidal cells, Fast spiking = PV+ interneurons)?', choices = ('regular_spiking', 'fast_spiking'))
                # unit_description[unit_type[0]] = 1
            # unit_description.append()
            # table.flush()
            # hf5.flush()

            # # add info to unit_details dictionary
            # this_unit = get_unit_info(unit_name, electrode_num, len(unit_times))
            # print(this_unit)
            # for i, j in enumerate(list(unit_details.keys())):
                # unit_details[j].append(this_unit[i])

            # # Finally increment max_unit and create a new unit name
            # max_unit += 1
            # unit_name = 'unit%03d' % int(max_unit + 1)

            # # Get a new unit_descriptor table row for this new unit
            # unit_description = table.row
            
            # #continue
        
    else: # when initially more then 1 cluster (2 or more) been chosen
        merge = easygui.multchoicebox(msg = 'I want to merge these clusters into one unit' +\
                                            '(True = Yes, False = No)', choices = ('True', 'False'))
        merge = ast.literal_eval(merge[0])
    
        # If the chosen clusters are going to be merged, merge them
        if merge:
            unit_waveforms = []
            unit_times = []
            merged_clusters_indexes = [] # for clecting indexes from different chosen clusters
            for cluster in clusters:
                if len(unit_waveforms) == 0: # == []:
                    unit_waveforms = spike_waveforms[np.where(predictions == int(cluster))[0], :]           
                    unit_times = spike_times[np.where(predictions == int(cluster))[0]]
                    merged_clusters_indexes = np.where(predictions == int(cluster))[0]
                else:
                    unit_waveforms = np.concatenate((unit_waveforms, spike_waveforms[np.where(predictions == int(cluster))[0], :]), axis=0)
                    unit_times = np.concatenate((unit_times, spike_times[np.where(predictions == int(cluster))[0]]))
                    merged_clusters_indexes = np.concatenate((np.where(predictions == int(cluster))[0], merged_clusters_indexes))

            # Show the merged cluster to the user, and ask if they still want to merge
            x = np.arange(len(unit_waveforms[0])/10) + 1
            fig, ax = blech_waveforms_datashader.waveforms_datashader(unit_waveforms, x)
            # plt.plot(x - 15, unit_waveforms[:, ::10].T, linewidth = 0.01, color = 'red')
            ax.set_xlabel('Sample (30 samples per ms)')
            ax.set_ylabel('Voltage (microvolts)')
            ax.set_title('Merged cluster, No. of waveforms={:d}'.format(unit_waveforms.shape[0]))
            plt.show()
 
            # Warn the user about the frequency of ISI violations in the merged unit
            ISIs = np.ediff1d(np.sort(unit_times))/30.0
            violations1 = 100.0*float(np.sum(ISIs < 1.0)/len(unit_times))
            violations2 = 100.0*float(np.sum(ISIs < 2.0)/len(unit_times))
            proceed = easygui.multchoicebox(msg = msg_text(v2=violations2, v1=violations1, lth=len(unit_times)), 
                                            choices = ('Save_the_Merged', 'Split_the_Merged')) #, 'Save_Individuals')) #('True', 'False'))
            #proceed = easygui.multchoicebox(msg = 'My merged cluster has %.1f percent (<2ms) and %.1f percent (<1ms) ISI violations out of %i total waveforms.'% (violations2, violations1, len(unit_times)) \
            #                                      'I want to still merge these clusters into one unit (True = Yes, False = No)' , choices = ('True', 'False'))
            #proceed = ast.literal_eval(proceed[0])

            # Create unit if the user agrees to proceed, else abort and go back to start of the loop 
            if proceed[0] == 'Save_the_Merged':

                max_unit, unit_name = save_units(unit_description, max_unit, unit_name=unit_name, electrode_num = electrode_num, 
                           u_waveforms=unit_waveforms, u_times=unit_times)
                unit_description = table.row

                # hf5.create_group('/sorted_units', unit_name)
                # waveforms = hf5.create_array('/sorted_units/%s' % unit_name, 'waveforms', unit_waveforms)
                # times = hf5.create_array('/sorted_units/%s' % unit_name, 'times', unit_times)
                # unit_description['unit_number'] = int(unit_name[4:-1])
                # unit_description['electrode_number'] = electrode_num
                # single_unit = easygui.multchoicebox(msg = 'I am almost-SURE that this is a beautiful single unit (True = Yes, False = No)', choices = ('True', 'False'))
                # unit_description['single_unit'] = int(ast.literal_eval(single_unit[0]))
                # # If the user says that this is a single unit, ask them whether its regular or fast spiking
                # unit_description['regular_spiking'] = 0
                # unit_description['fast_spiking'] = 0
                # unit_description['waveform_count'] = len(unit_times)
                # if int(ast.literal_eval(single_unit[0])):
                    # unit_type = easygui.multchoicebox(msg = 'What type of unit is this (Regular spiking = Pyramidal cells, Fast spiking = PV+ interneurons)?', choices = ('regular_spiking', 'fast_spiking'))
                    # unit_description[unit_type[0]] = 1
                # unit_description.append()
                # table.flush()
                # hf5.flush()

                # # add info to unit_details dictionary
                # this_unit = get_unit_info(unit_name, electrode_num, len(unit_times))
                # print(this_unit)
                # for i, j in enumerate(list(unit_details.keys())):
                    # unit_details[j].append(this_unit[i])
                
                # # Finally increment max_unit and create a new unit name
                # max_unit += 1
                # unit_name = 'unit%03d' % int(max_unit + 1)

                # # Get a new unit_descriptor table row for this new unit
                # unit_description = table.row
                
            elif proceed[0] == 'Split_the_Merged': #else:
                #continue
#######################
                # re_split the merged cluster
#                re_split = easygui.multchoicebox(msg = 'I want to split this merged cluster (True = Yes, False = No)', 
#                                                 choices = ('True', 'False'))
#                re_split = ast.literal_eval(re_split[0])
#                if re_split:
                #####
                print('Splitting the merged waveforms from each cluster')
                # Get clustering parameters from user
                clustering_params = easygui.multenterbox(msg = 'Fill in the parameters for re-clustering (using a GMM)',
                                                         fields = ['Number of clusters', 
                                                                   'Maximum number of iterations (1000 is more than enough)',
                                                                   'Convergence criterion (usually 0.0001)', 
                                                                   'Number of random restarts for GMM (10 is more than enough)'],
                                                         values = [4, 1000, 0.000001, 10])
                n_clusters = int(clustering_params[0])
                n_iter = int(clustering_params[1])
                thresh = float(clustering_params[2])
                n_restarts = int(clustering_params[3]) 

                # Make data array to be put through the GMM - 5 components: 3 PCs, scaled energy, amplitude
                this_cluster = merged_clusters_indexes #np.where(predictions == int(merged_clusters))[0]
                n_pc = 3
                data = np.zeros((len(this_cluster), n_pc + 2))  
                data[:,2:] = pca_slices[this_cluster,:n_pc]
                data[:,0] = energy[this_cluster]/np.max(energy[this_cluster])
                data[:,1] = np.abs(amplitudes[this_cluster])/np.max(np.abs(amplitudes[this_cluster]))

                # Cluster the data
                g = GaussianMixture(n_components = n_clusters, covariance_type = 'full', tol = thresh, max_iter = n_iter, n_init = n_restarts)
                g.fit(data)
                    
                # Show the cluster plots if the solution converged
                if g.converged_:
                    split_predictions = g.predict(data)
                    x = np.arange(len(spike_waveforms[0])/10) + 1
                    for cluster in range(n_clusters):
                        split_points = np.where(split_predictions == cluster)[0] # get indexes for this sub_merged cluster                
                        # plt.figure(cluster)
                        slices_dejittered = spike_waveforms[this_cluster, :]        # Waveforms and times from the chosen cluster
                        times_dejittered = spike_times[this_cluster]
                        times_dejittered = times_dejittered[split_points]       # Waveforms and times from the chosen split of the chosen cluster
                        ISIs = np.ediff1d(np.sort(times_dejittered))/30.0
                        violations1 = 100.0*float(np.sum(ISIs < 1.0)/split_points.shape[0])
                        violations2 = 100.0*float(np.sum(ISIs < 2.0)/split_points.shape[0])
                        fig, ax = blech_waveforms_datashader.waveforms_datashader(slices_dejittered[split_points, :], x)
                        # plt.plot(x-15, slices_dejittered[split_points, :].T, linewidth = 0.01, color = 'red')
                        ax.set_xlabel('Sample (30 samples per ms)')
                        ax.set_ylabel('Voltage (microvolts)')
                        ax.set_title(title_text(c=cluster, v1=violations1, v2=violations2, points=split_points.shape[0]))
                        #ax.set_title("Split Cluster{:d}, 2ms ISI violations={:.1f} percent".format(cluster, violations2) + "\n" + "1ms ISI violations={:.1f}%, Number of waveforms={:d}".format(violations1, split_points.shape[0]))
                        
                else:
                    print("Solution did not converge - try again with higher number of iterations or lower convergence criterion")
                    continue
                
                plt.show()

                #################################
                # Ask the user for the split clusters they want to choose
                chosen_merged_split = easygui.multchoicebox(msg = 'Which split clusters do you want to choose? Hit cancel to exit', 
                                                            choices = tuple([str(i) for i in range(n_clusters)]))
                if chosen_merged_split is None:
                    continue
                
                elif len(chosen_merged_split) == 1:
                    unit_waveforms = spike_waveforms[merged_clusters_indexes] #np.where(predictions == int(merged_clusters))[0], :]   # Waveforms of originally chosen cluster
                    unit_waveforms = unit_waveforms[np.where(split_predictions == int(chosen_merged_split[0]))[0], :]  # Subsetting this set of waveforms to include only the chosen split
                    unit_times = spike_times[merged_clusters_indexes] #np.where(predictions == int(merged_clusters))[0]]          # Do the same thing for the spike times
                    unit_times = unit_times[np.where(split_predictions == int(chosen_merged_split[0]))[0]]

                    max_unit, unit_name = save_units(unit_description, max_unit, unit_name=unit_name, electrode_num = electrode_num, 
                               u_waveforms=unit_waveforms, u_times=unit_times)
                    unit_description = table.row
                
####
                else: # if the select splits are greater than 1 cluster
                    merge_resplits = easygui.multchoicebox(msg = 'I want to merge these clusters into one unit (True = Yes, False = No)', choices = ('True', 'False'))
                    merge_resplits = ast.literal_eval(merge_resplits[0])
                
                    # If the chosen clusters are going to be merged, merge them
                    if merge_resplits:
                        unit_waveforms = []
                        unit_times = []
                        #merged_clusters_indexes = [] # for clecting indexes from different chosen clusters
                        cluster_waveforms = spike_waveforms[merged_clusters_indexes]
                        cluster_times = spike_times[merged_clusters_indexes] 
                        for c in chosen_merged_split:
                            if len(unit_waveforms) == 0: # == []:
                                unit_waveforms = cluster_waveforms[np.where(split_predictions == int(c))[0], :]           
                                unit_times = cluster_times[np.where(split_predictions == int(c))[0]]
                                #merged_clusters_indexes = np.where(predictions == int(cluster))[0]
                            else:
                                unit_waveforms = np.concatenate((unit_waveforms, cluster_waveforms[np.where(split_predictions == int(c))[0], :]), axis=0)
                                unit_times = np.concatenate((unit_times, cluster_times[np.where(split_predictions == int(c))[0]]))
                                #merged_clusters_indexes = np.concatenate((np.where(predictions == int(cluster))[0], merged_clusters_indexes))

                        # Show the merged cluster to the user, and ask if they still want to merge
                        x = np.arange(len(unit_waveforms[0])/10) + 1
                        fig, ax = blech_waveforms_datashader.waveforms_datashader(unit_waveforms, x)
                        # plt.plot(x - 15, unit_waveforms[:, ::10].T, linewidth = 0.01, color = 'red')
                        ax.set_xlabel('Sample (30 samples per ms)')
                        ax.set_ylabel('Voltage (microvolts)')
                        ax.set_title('Merged cluster, No. of waveforms={:d}'.format(unit_waveforms.shape[0]))
                        plt.show()
             
                        # Warn the user about the frequency of ISI violations in the merged unit
                        ISIs = np.ediff1d(np.sort(unit_times))/30.0
                        violations1 = 100.0*float(np.sum(ISIs < 1.0)/len(unit_times))
                        violations2 = 100.0*float(np.sum(ISIs < 2.0)/len(unit_times))
                        
                        proceed = easygui.multchoicebox(msg = msg_text(v2=violations2, v1=violations1, lth=len(unit_times)), choices = ('True', 'False'))
                        
                        #proceed = easygui.multchoicebox(msg = 'My merged cluster has %.1f percent (<2ms) and %.1f percent (<1ms) ISI violations out of %i total waveforms. I want to still merge these clusters into one unit (True = Yes, False = No)' % (violations2, violations1, len(unit_times)), choices = ('True', 'False'))
                        proceed = ast.literal_eval(proceed[0])

                        # Create unit if the user agrees to proceed, else abort and go back to start of the loop 
                        if proceed:
                            max_unit, unit_name = save_units(unit_description, max_unit, unit_name=unit_name, electrode_num = electrode_num, 
                                       u_waveforms=unit_waveforms, u_times=unit_times)
                            unit_description = table.row
                        else: continue
                            
                    else: # save each selected sub-split 
                        chosen_merged_splits = [int(i) for i in (chosen_merged_split)]
                        print('chosen_merged_split: ', chosen_merged_splits)
                        for c in chosen_merged_split:
                            unit_waveforms = spike_waveforms[merged_clusters_indexes] #np.where(predictions == int(merged_clusters))[0], :]   # Waveforms of originally chosen cluster
                            unit_waveforms = unit_waveforms[np.where(split_predictions == int(c))[0], :]  # Subsetting this set of waveforms to include only the chosen split
                            unit_times = spike_times[merged_clusters_indexes] #np.where(predictions == int(merged_clusters))[0]]          # Do the same thing for the spike times
                            unit_times = unit_times[np.where(split_predictions == int(c))[0]]

                            max_unit, unit_name = save_units(unit_description, max_unit, unit_name=unit_name, electrode_num = electrode_num, 
                                       u_waveforms=unit_waveforms, u_times=unit_times)
                            unit_description = table.row
                        
                            # hf5.create_group('/sorted_units', unit_name)
                            # waveforms = hf5.create_array('/sorted_units/%s' % unit_name, 'waveforms', unit_waveforms)
                            # times = hf5.create_array('/sorted_units/%s' % unit_name, 'times', unit_times)

                            # # fill in unit_ description first
                            # unit_description['unit_number'] = int(unit_name[4:-1])
                            # unit_description['electrode_number'] = electrode_num
                            # single_unit = easygui.multchoicebox(title = 'Electrode: %i cluster: %i split: %i ' % (electrode_num, int(num_clusters), int(split_merged_cluster)),
                                                                # msg = 'I am almost-SURE the selected sub_cluster is a beautiful single unit (True = Yes, False = No)', 
                                                                # choices = ('True', 'False'))
                            # unit_description['single_unit'] = int(ast.literal_eval(single_unit[0]))
                            # # If the user says that this is a single unit, ask them whether its regular or fast spiking
                            # unit_description['regular_spiking'] = 0
                            # unit_description['fast_spiking'] = 0
                            # unit_description['waveform_count'] = len(unit_times)
                            # if int(ast.literal_eval(single_unit[0])):
                                # unit_type = easygui.multchoicebox(msg = 'What type of unit is this (Regular spiking = Pyramidal cells, Fast spiking = PV+ interneurons)?', choices = ('regular_spiking', 'fast_spiking'))
                                # unit_description[unit_type[0]] = 1
                            # unit_description.append()
                            # table.flush()

                            # hf5.flush()             

                            # # add info to unit_details dictionary
                            # this_unit = get_unit_info(unit_name, electrode_num, len(unit_times))
                            # print(this_unit)
                            # for i, j in enumerate(list(unit_details.keys())):
                                # unit_details[j].append(this_unit[i])

                            # # Finally increment max_unit and create a new unit name
                            # max_unit += 1
                            # unit_name = 'unit%03d' % int(max_unit + 1)

                            # # Get a new unit_descriptor table row for this new unit
                            # unit_description = table.row
                            
                            # #split_merge = False
                    # #except:
                    # #    continue

            # else: # if not merge, then include each split_cluster as a separate unit
                # unit_waveforms = spike_waveforms[merged_clusters_indexes] #np.where(predictions == int(merged_clusters))[0], :]   # Waveforms of originally chosen cluster
                # unit_waveforms = unit_waveforms[np.where(split_predictions == int(chosen_merged_split[0]))[0], :]  # Subsetting this set of waveforms to include only the chosen split
                # unit_times = spike_times[merged_clusters_indexes] #np.where(predictions == int(merged_clusters))[0]]          # Do the same thing for the spike times
                # unit_times = unit_times[np.where(split_predictions == int(chosen_merged_split[0]))[0]]
                

                # save_units(unit_description, max_unit, unit_name=unit_name, electrode_num = electrode_num, 
                           # u_waveforms=unit_waveforms, u_times=unit_times)
                # unit_description = table.row
                #continue

        else: # Otherwise include each cluster as a separate unit
            for cluster in clusters:

                unit_waveforms = spike_waveforms[np.where(predictions == int(cluster))[0], :]
                unit_times = spike_times[np.where(predictions == int(cluster))[0]]

                max_unit, unit_name = save_units(unit_description, max_unit, unit_name=unit_name, electrode_num = electrode_num, 
                           u_waveforms=unit_waveforms, u_times=unit_times)
                unit_description = table.row

                # hf5.create_group('/sorted_units', unit_name)
                # waveforms = hf5.create_array('/sorted_units/%s' % unit_name, 'waveforms', unit_waveforms)
                # times = hf5.create_array('/sorted_units/%s' % unit_name, 'times', unit_times)
                # unit_description['unit_number'] = int(unit_name[4:-1])
                # unit_description['electrode_number'] = electrode_num
                # single_unit = easygui.multchoicebox(msg = 'I am almost-SURE that electrode: %i cluster: %i is a beautiful single unit (True = Yes, False = No)' % (electrode_num, int(cluster)), choices = ('True', 'False'))
                # unit_description['single_unit'] = int(ast.literal_eval(single_unit[0]))
                # # If the user says that this is a single unit, ask them whether its regular or fast spiking
                # unit_description['regular_spiking'] = 0
                # unit_description['fast_spiking'] = 0
                # unit_description['waveform_count'] = len(unit_times)
                # if int(ast.literal_eval(single_unit[0])):
                    # unit_type = easygui.multchoicebox(msg = 'What type of unit is this (Regular spiking = Pyramidal cells, Fast spiking = PV+ interneurons)?', choices = ('regular_spiking', 'fast_spiking'))
                    # unit_description[unit_type[0]] = 1
                # unit_description.append()
                # table.flush()
                # hf5.flush()             

                # # add info to unit_details dictionary
                # this_unit = get_unit_info(unit_name, electrode_num, len(unit_times))
                # for i, j in enumerate(list(unit_details.keys())):
                    # unit_details[j].append(this_unit[i])

                # # Finally increment max_unit and create a new unit name
                # max_unit += 1
                # unit_name = 'unit%03d' % int(max_unit + 1)

                # # Get a new unit_descriptor table row for this new unit
                # unit_description = table.row

# save unit info pickle file 
with open('unit_details.pkl', 'wb') as f:
    pickle.dump(unit_details, f)

# Close the hdf5 file
hf5.close()
     



    




