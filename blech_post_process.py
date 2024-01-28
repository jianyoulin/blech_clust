# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:44:35 2019
2021/8/9
This code is modified to merge splited clusters or split a merged cluster 
also save a pickle file with details of unit description
2021/08/12 difference from V2 to V3: save units using a function
2023/12/25 adding more functions to modulating the code
           so that it is easier to modify later
2024/01//01 adding functions to do sorting with repeated merging and spliting


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

# Get name of directory with the data files
try:
    dir_name = sys.argv[1]
except:
    dir_name = easygui.diropenbox('Select the dir path where data are saved')
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

# Create a dictionary to save unit information
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
# hdf_name: 'JK14_20230916_Sacc_230916_104702.h5'

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
    # os.system("rm " + hdf5_name)
    os.remove(hdf5_name)
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
    s_u: single_unit (boolean); u_t: unit_type (str)
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
                                                    1 if s_u=="True" else 0, 
                                                    waveform_num])
    return description
                                         

# Make a table under /sorted_units describing the sorted units. 
# If unit_descriptor already exists, just open it up in the variable table
try:
    table = hf5.create_table('/', 'unit_descriptor', description = unit_descriptor)
except:
    table = hf5.root.unit_descriptor

def save_units(hf5_file, table, max_unit, unit_name=None, electrode_num = None, 
               u_waveforms=None, u_times=None):
    """
    1) save unit waveforms and time to HDF5 file under sorted units node
    2) save unit description to descriptor table in HDF5 file
    3) save unit description to descriptor dictionary
    u_waveforms: Unit waveforms
    u_times = Unit times
    """
    unit_name = 'unit%03d' % int(max_unit + 1)

    unit_type = easygui.multchoicebox(msg = 'What is the type of the slected unit?',
                                      choices = ('Regular_spiking', 'Fast_spiking', 'Multi_unit'))
    single_unit = 1 if 'M' not in unit_type[0] else 0
    unit_description = table.row

    # single_unit = easygui.multchoicebox(msg = 'I am almost-SURE that this is ' +\
    #                                           'a beautiful single unit (True = Yes, False = No)', 
    #                                     choices = ('True', 'False'))
    unit_description['unit_number'] = int(unit_name[4:])
    unit_description['electrode_number'] = electrode_num
    unit_description['single_unit'] = single_unit #int(ast.literal_eval(single_unit[0]))
    unit_description['regular_spiking'] = 1 if 'R' in unit_type[0] else 0
    unit_description['fast_spiking'] = 1 if 'F' in unit_type[0] else 0
    unit_description['waveform_count'] = len(u_times)

    # create a group in hf5 to save waveforms and spike times
    hf5_file.create_group('/sorted_units', unit_name)
    waveforms = hf5_file.create_array('/sorted_units/%s' % unit_name, 'waveforms', u_waveforms)
    times = hf5_file.create_array('/sorted_units/%s' % unit_name, 'times', u_times)

    # add to unit description table
    unit_description.append()
    table.flush()
    hf5_file.flush()
    # add info to unit_details dictionary
    
    this_unit = get_unit_info(unit_name, electrode_num, single_unit, 
                              unit_type[0], len(u_times))
    print(this_unit)
    for i, j in enumerate(list(unit_details.keys())):
        unit_details[j].append(this_unit[i])
    # Finally increment max_unit and create a new unit name
    max_unit += 1
#        unit_name = 'unit%03d' % int(max_unit + 1)
    return max_unit


# Functions to confirm the acceptance of the re-clustered solution
def plot_waveforms(spike_waveforms, ISI1, ISI2, cluster = -1): #, fig_title=None):
    x = np.arange(len(spike_waveforms[0,:])/10) + 1
    fig, ax = blech_waveforms_datashader.waveforms_datashader(spike_waveforms, x)
    ax.set_xlabel('Sample (30 samples per ms)')
    ax.set_ylabel('Voltage (microvolts)')
    ax.set_title(f"Split Cluster {cluster}: total waveforms = {spike_waveforms.shape[0]} "+ \
            f"\n2ms ISI violations={round(ISI2,2)}% percent "+ \
            f"\n1ms ISI violations={round(ISI1,2)}%")
    plt.show()
    return ax

def ISI(spike_times):
    """
    calculate all interspike intervals
    return: % of violations for 1 (violation1) and 2 (violation2) ms 
     """
    ISIs = np.ediff1d(np.sort(spike_times))/30.0
    violations1 = 100.0*float(np.sum(ISIs < 1.0)/len(spike_times))
    violations2 = 100.0*float(np.sum(ISIs < 2.0)/len(spike_times))
    return violations1, violations2

def confirm_unit(spike_waveforms, spike_times, cluster=-1, check=False):
    """ Show the merged cluster to the user, 
    and ask if they still want to merge """

    violations1, violations2 = ISI(spike_times)
    ax = plot_waveforms(spike_waveforms, violations1, violations2, cluster = cluster)

    # Warn the user about the frequency of ISI violations in the merged unit
    if check:
        # prompt_title = 'My merged cluster has %.1f percent (<2ms)' % (violations2)+ \
        #             ' and %.1f percent (<1ms) ISI violations out of %i total waveforms.'% (violations1, len(spike_times))+ \
        #             ' I want to still merge these clusters into one unit (True = Yes, False = No)'
        proceed = easygui.multchoicebox(msg = 'I want to still merge these clusters into one unit (True = Yes, False = No)',
                                        choices = ('True', 'False'))
        return ast.literal_eval(proceed[0])

def gmm_wf_clustering(pca_slices, energy, amplitudes,
                      clustering_params=[4, 1000, 0.000001, 10]):
    """
    using Gaussian mixture model to cluster waveforms
    chosen_cluster: Int
    return: trained clustering results, and composed data
    """
    # Get clustering parameters from user
    if not clustering_params:
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
    else:
        n_clusters, n_iter, thresh, n_restarts = clustering_params

    # Make data array to be put through the GMM - 5 components: 3 PCs, scaled energy, amplitude
    # this_cluster = np.where(predictions == chosen_cluster)[0] # clusters[0] since only one cluster is chosen in this if statement
    #this_cluster = np.where(predictions == int(clusters[0]))[0]
    n_pc = 3
    data = np.zeros((len(energy), n_pc + 2))  
    data[:,2:] = pca_slices[:, :n_pc]
    data[:,0] = energy/np.max(energy)
    data[:,1] = np.abs(amplitudes)/np.max(np.abs(amplitudes))

    # Cluster the data
    g = GaussianMixture(n_components = n_clusters, covariance_type = 'full', 
                        tol = thresh, max_iter = n_iter, n_init = n_restarts)
    g.fit(data)
    return g, data

def collect_spike_info(chosen_clusters, preds, wfs, sts, pcs, en, amps):
    """ preds: redictions
    wfs: waveforms
    sts: spike_times
    pcs: pca_slices
    en: waveform energe
    amps: waveform amplitudes
    """

    spike_waveforms = []
    spike_times = []
    pca_slices = []
    energy = []
    amplitudes = []
    spike_indexes = []
    for split_cluster in chosen_clusters:
        this_cluster = np.where(preds == split_cluster)[0]
        print(len(this_cluster))
        if len(spike_waveforms) == 0: # == []:
            spike_waveforms = wfs[this_cluster, :]
            spike_times = sts[this_cluster]
            pca_slices = pcs[this_cluster, :]
            energy = en[this_cluster]
            amplitudes = amps[this_cluster]
            spike_indexes = this_cluster
        else:
            spike_waveforms = np.concatenate((spike_waveforms, wfs[this_cluster, :]), axis=0)
            spike_times = np.concatenate((spike_times, sts[this_cluster]))
            pca_slices = np.concatenate((pca_slices, pcs[this_cluster, :]), axis=0)
            energy = np.concatenate((energy, en[this_cluster]))
            amplitudes = np.concatenate((amplitudes, amps[this_cluster]))
            spike_indexes = np.concatenate((spike_indexes, this_cluster))
        print(spike_waveforms.shape, spike_indexes.shape)

    return spike_waveforms, spike_times, pca_slices, energy, amplitudes


def sorting(spike_waveforms, spike_times, pca_slices, energy, amplitudes, max_unit, table):
    """
    method: 'confirm' or 'split'
    """
    
    while True:
        g, data = gmm_wf_clustering(pca_slices, energy, 
                                    amplitudes, clustering_params=None) #[4, 1000, 0.000001, 10]):

        # Show the cluster plots if the solution converged
        if g.converged_:
            split_predictions = g.predict(data) # predictions of this selected cluster (from 0 to n_cluster)
            n_clusters = list(np.unique(split_predictions))
            n_clusters = sum(np.array(n_clusters) >= 0)

            for cluster in range(n_clusters):
                split_points = np.where(split_predictions == cluster)[0] # obtain the indexes of each sub-clusters and plot them                
                # plt.figure(cluster)
                slices_dejittered = spike_waveforms[split_points, :]        # Waveforms and times from the chosen cluster
                times_dejittered = spike_times[split_points]       # Waveforms and times from the chosen split of the chosen cluster
                
                violations1, violations2 = ISI(times_dejittered)

                ax = plot_waveforms(slices_dejittered, violations1, violations2, cluster=cluster) #, fig_title=None)

        else:
            print("Solution did not converge - try again with higher number of iterations or "+ \
                  "lower convergence criterion")
            continue

        # Ask the user for the split clusters they want to choose
        chosen_split = easygui.multchoicebox(msg = 'Which split clusters do you want to choose? '+ \
                                                   'Hit cancel to exit', 
                                             choices = tuple([str(i) for i in range(n_clusters)]))
        if len(chosen_split) >= 1:
            chosen_split = [int(chosen_split[i]) for i in range(len(chosen_split))]
            unit_waveforms, unit_times, _, _, _ = collect_spike_info(chosen_split, split_predictions, 
                    spike_waveforms, spike_times, 
                    pca_slices, energy, amplitudes)
            confirm_unit(unit_waveforms, unit_times, cluster=-1, check=False)

            next_step = easygui.multchoicebox(msg = 'What is the type of the slected unit?',
                                              choices = ('save each (s)', 'save the merged[sm]', 'split the merged[sp]'))
            print(next_step[0])

            if next_step[0] == 'save each (s)':
                for cluster in chosen_split:
                    unit_waveforms = spike_waveforms[np.where(split_predictions == cluster)[0], :]   # Subsetting this set of waveforms to include only the chosen split
                    unit_times = spike_times[np.where(split_predictions == cluster)[0]]
                    #unit_name = 'unit%03d' % int(max_unit + 1)
                    max_unit = save_units(hf5, table, max_unit, #unit_name=unit_name, 
                                          electrode_num = electrode_num,
                                          u_waveforms=unit_waveforms, u_times=unit_times)
                    # unit_description = table.row
                    print(f'unit{max_unit:03d} is saved')
                break                    
            elif next_step[0] == 'save the merged[sm]':
                unit_waveforms, unit_times, _, _, _ = collect_spike_info(chosen_split, split_predictions, 
                                    spike_waveforms, spike_times, 
                                    pca_slices, energy, amplitudes)
                confirm = confirm_unit(unit_waveforms, unit_times, cluster=-1, check=True)
                if confirm:
#                        unit_name = 'unit%03d' % int(max_unit + 1)
                    max_unit = save_units(hf5, table, max_unit, #unit_name=unit_name, 
                            electrode_num = electrode_num,
                            u_waveforms=unit_waveforms, u_times=unit_times)
                    # unit_description = table.row
                    print(f'unit{max_unit:03d} is saved')
                    break # if save a unit, break sorting on this channel
                else:
                    print('re-split the the merged cluster, then sort')
                    continue # keep sorting wile loop going

            elif next_step[0] == 'split the merged[sp]':
                spike_waveforms, spike_times, pca_slices, energy, amplitudes = \
                                collect_spike_info(chosen_split, split_predictions, 
                                                spike_waveforms, spike_times, 
                                                pca_slices, energy, amplitudes)
                confirm = confirm_unit(spike_waveforms, spike_times)
                sorting(spike_waveforms, spike_times, pca_slices, 
                        energy, amplitudes, max_unit, table)
                print('Re-clustering the selected splitted-clusters')
                break # break out to the first layer of sorting
        else:
            print('no cluster is chosen, re-clustering the previously chosen sub-clusters')
            continue

# Run an infinite loop as long as the user wants to pick clusters from the electrodes   
while True:
    # Get list of existing nodes/groups under /sorted_units
    node_list = hf5.list_nodes('/sorted_units')

    # If node_list is empty, start naming units from 000
    # max_unit as the current max
    unit_name = ''
#    max_unit = 0
    if len(node_list) == 0:     
#        unit_name = 'unit%03d' % 0
        max_unit = -1
    # Else name the new unit by incrementing the last unit by 1 
    else:
        unit_numbers = []
        for node in node_list:
            unit_numbers.append(node._v_pathname.split('/')[-1][-3:])
            unit_numbers[-1] = int(unit_numbers[-1])
        unit_numbers = np.array(unit_numbers)
        max_unit = np.max(unit_numbers)
#        unit_name = 'unit%03d' % int(max_unit + 1)

    # Get a new unit_descriptor table row for this new unit
#    ###
     # should I add the below line into the save unit function
#    ###
        
    # unit_description = table.row # add an empty row in the unit description table 
    
    # Get electrode number from user
    electrode_num = easygui.multenterbox(msg = 'Which electrode do you want to choose? ' \
                                               'Hit cancel to exit', 
                                         fields = ['Electrode #'])
    # Break if wrong input/cancel command was given
    try:
        electrode_num = int(electrode_num[0])
        print(f'electrode num: {electrode_num}')
    except:
        print('no electrodes selected')
        break
    
    # Get the number of clusters in the chosen solution
    num_clusters = easygui.multenterbox(msg = 'Which solution do you want to choose for electrode %i?' % electrode_num, 
                                        fields = ['Number of clusters in the solution'])
    num_clusters = int(num_clusters[0])
    print(f'Selected Cluster Solution: {num_clusters}')

    # Load data from the chosen electrode and solution
    spike_waveforms = np.load('./spike_waveforms/electrode%i/spike_waveforms.npy' \
                              % electrode_num) # [# of waveforms, 450 time points]
    spike_times = np.load('./spike_times/electrode%i/spike_times.npy' \
                          % electrode_num) # [# of time points]
    pca_slices = np.load('./spike_waveforms/electrode%i/pca_waveforms.npy' \
                         % electrode_num) # [# of waveforms, 3 pca components]
    energy = np.load('./spike_waveforms/electrode%i/energy.npy' % electrode_num)
    amplitudes = np.load('./spike_waveforms/electrode%i/spike_amplitudes.npy' % electrode_num)
    predictions = np.load('./clustering_results/electrode%i/clusters%i/predictions.npy' \
                          % (electrode_num, num_clusters))

    # Get cluster choices from the chosen solution
    clusters = easygui.multchoicebox(msg = 'Which clusters do you want to choose?', 
                                     choices = tuple([str(i) for i in range(int(np.max(predictions) + 1))]))
    clusters = [int(i) for i in clusters]
    print(f'Selected Cluster(s): {clusters}')

    next_step = easygui.multchoicebox(msg = 'Do you want to save each selected unit (Yes/Y/No/N)?\n'+ \
                                      '\nIf NO, merge selected clusters (if more than 1 selected cluster) and start sorting.',
                                      choices = ('Yes', 'No'))[0]
    if next_step[0].lower() == 'y':
        for cluster in clusters:
            unit_waveforms = spike_waveforms[np.where(predictions == int(cluster))[0], :]
            unit_times = spike_times[np.where(predictions == int(cluster))[0]]
            confirm_unit(unit_waveforms, unit_times)

            max_unit = save_units(hf5, table, max_unit, #unit_name=unit_name, 
                                  electrode_num = electrode_num, 
                                  u_waveforms=unit_waveforms, u_times=unit_times)
            # unit_description = table.row
        continue
    
    else: # merge slected cluster and then do sorting (merge/split repeatedly)
        spike_waveforms, spike_times, pca_slices, energy, amplitudes = \
                        collect_spike_info(clusters, predictions, 
                                        spike_waveforms, spike_times, pca_slices,
                                        energy, amplitudes)
        confirm = confirm_unit(spike_waveforms, spike_times)
        sorting(spike_waveforms, spike_times, pca_slices, energy, 
                    amplitudes, max_unit, table)

# save unit info pickle file 
with open('unit_details.pkl', 'wb') as f:
    pickle.dump(unit_details, f)

# Close the hdf5 file
hf5.close()

"""
    # if only one cluster chosen, to split or to be a unit
    if len(clusters) == 1:
        re_cluster = easygui.multchoicebox(msg = 'I want to split this cluster (True = Yes, False = No)', choices = ('True', 'False'))
        re_cluster = ast.literal_eval(re_cluster[0])
       
        # If the user asked to split/re-cluster, ask them for the clustering parameters and perform clustering
        split_predictions = []
        chosen_split = 0
        if re_cluster: #I want to split this cluster
            # using cluster_function
            g1 = gmm_wf_clustering(int(clusters[0]), predictions, pca_slices, energy, amplitudes,
                                   clustering_params=[4, 1000, 0.000001, 10])
                
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
            g = GaussianMixture(n_components = n_clusters, covariance_type = 'full', 
                                tol = thresh, max_iter = n_iter, n_init = n_restarts)
            g.fit(data)
        
            # Show the cluster plots if the solution converged
            if g1.converged_:
                split_predictions = g1.predict(data) # predictions of this selected cluster (from 0 to n_cluster)
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
                    ax.set_title("Split Cluster{:d}, 2ms ISI violations={:.1f} percent".format(cluster, violations2) + "\n" + "1ms ISI violations={:.1f}%, Number of waveforms={:d}".format(violations1, split_points.shape[0]))
                    
            else:
                print("Solution did not converge - try again with higher number of iterations or lower convergence criterion")
                continue
            
            #plt.show()

            
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

                save_units(hf5, unit_description, max_unit, unit_name=unit_name, electrode_num = electrode_num, 
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
                    for split_cluster in [0]: #chosen_split:
                        if len(unit_waveforms) == 0: # == []:
                            unit_waveforms = cluster_unit_waveforms[np.where(split_predictions == split_cluster)[0], :]
                            unit_times = cluster_unit_times[np.where(split_predictions == split_cluster)[0]]
                        else:
                            unit_waveforms = np.concatenate((unit_waveforms, cluster_unit_waveforms[np.where(split_predictions == split_cluster)[0], :]), axis=0)
                            unit_times = np.concatenate((unit_times, cluster_unit_times[np.where(split_predictions == split_cluster)[0]]))
                        print(unit_waveforms.shape)
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
                    proceed = easygui.multchoicebox(msg = 'My merged cluster has %.1f percent (<2ms) and %.1f percent (<1ms) ISI violations out of %i total waveforms. I want to still merge these clusters into one unit (True = Yes, False = No)' % (violations2, violations1, len(unit_times)), choices = ('True', 'False'))
                    proceed = ast.literal_eval(proceed[0])

                    # Create unit if the user agrees to proceed, else include each split_cluster as a separate unit 
                    if proceed:
                        save_units(hf5, unit_description, max_unit, unit_name=unit_name, electrode_num = electrode_num, 
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
                        
                    else:
                        continue
                else: # if not merge, then include each split_cluster as a separate unit
                    for split_cluster in chosen_split:
                        
                        unit_waveforms = spike_waveforms[np.where(predictions == int(clusters[0]))[0], :]   # Waveforms of originally chosen cluster
                        unit_waveforms = unit_waveforms[np.where(split_predictions == split_cluster)[0], :] # Subsetting this set of waveforms to include only the chosen split
                        unit_times = spike_times[np.where(predictions == int(clusters[0]))[0]]          # Do the same thing for the spike times
                        unit_times = unit_times[np.where(split_predictions == split_cluster)[0]]
                        
                        save_units(hf5, unit_description, max_unit, unit_name=unit_name, electrode_num = electrode_num, 
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
            else:
                continue
        
        
        else: # If initially only 1 cluster was chosen (and it wasn't split), 
              # add that as a new unit in /sorted_units. 
              # Ask if the isolated unit is an almost-SURE single unit
            
            unit_waveforms = spike_waveforms[np.where(predictions == int(clusters[0]))[0], :]
            unit_times = spike_times[np.where(predictions == int(clusters[0]))[0]]

            save_units(hf5, unit_description, max_unit, unit_name=unit_name, electrode_num = electrode_num, 
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
        merge = easygui.multchoicebox(msg = 'I want to merge these clusters into one unit (True = Yes, False = No)', 
                                      choices = ('True', 'False'))
        merge = ast.literal_eval(merge[0])
    
        # If the chosen clusters are going to be merged, merge them
        if merge:
            unit_waveforms = []
            unit_times = []
            merged_clusters_indexes = [] # for clecting indexes from different chosen clusters
            for cluster in clusters:
                if len(unit_waveforms) == 0: # empty list == []:
                    unit_waveforms = spike_waveforms[np.where(predictions == int(cluster))[0], :]           
                    unit_times = spike_times[np.where(predictions == int(cluster))[0]]
                    merged_clusters_indexes = np.where(predictions == int(cluster))[0]
                else:
                    unit_waveforms = np.concatenate((unit_waveforms, spike_waveforms[np.where(predictions == int(cluster))[0], :]), axis=0)
                    unit_times = np.concatenate((unit_times, spike_times[np.where(predictions == int(cluster))[0]]))
                    merged_clusters_indexes = np.concatenate((np.where(predictions == int(cluster))[0], 
                                                              merged_clusters_indexes))

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
            proceed = easygui.multchoicebox(msg = 'My merged cluster has %.1f percent (<2ms) and %.1f percent (<1ms) ISI violations out of %i total waveforms. I want to still merge these clusters into one unit (True = Yes, False = No)' % (violations2, violations1, len(unit_times)), choices = ('True', 'False'))
            proceed = ast.literal_eval(proceed[0])

            # Create unit if the user agrees to proceed, else abort and go back to start of the loop 
            if proceed:
                # ask if re_split the merged cluster
                re_split = easygui.multchoicebox(msg = 'I want to split this merged cluster (True = Yes, False = No)', 
                                                 choices = ('True', 'False'))
                re_split = ast.literal_eval(re_split[0])
                if not re_split:
                    # ask if re_split the merged cluster
                    save_merge = easygui.multchoicebox(msg = 'I want to save this merged cluster (True = Yes, False = No)', 
                                                       choices = ('True', 'False'))
                    save_merge = ast.literal_eval(save_merge[0])

                    if save_merge:
                        save_units(hf5, unit_description, max_unit, unit_name=unit_name, electrode_num = electrode_num, 
                                   u_waveforms=unit_waveforms, u_times=unit_times)
                        unit_description = table.row
                    else:
                        continue
               
                else: # if I want to split the merged cluster
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
                    g = GaussianMixture(n_components = n_clusters, covariance_type = 'full', 
                                        tol = thresh, max_iter = n_iter, n_init = n_restarts)
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
                            ax.set_title("Split Cluster{:d}, 2ms ISI violations={:.1f} percent".format(cluster, violations2) + "\n" + "1ms ISI violations={:.1f}%, Number of waveforms={:d}".format(violations1, split_points.shape[0]))
                            
                    else:
                        print("Solution did not converge - try again with higher number of iterations or lower convergence criterion")
                        continue
                    
                    plt.show()

                    # Ask the user for the split clusters they want to choose
                    chosen_merged_split = easygui.multchoicebox(msg = 'Which split clusters do you want to choose? Hit cancel to exit', 
                                                                choices = tuple([str(i) for i in range(n_clusters)]))
                    if chosen_merged_split is None:
                        continue
                    else:
                        chosen_merged_split = [int(chosen_merged_split[i]) for i in range(len(chosen_merged_split))]
                        print('chosen_merged_split: ', chosen_merged_split)
                        for split_merged_cluster in chosen_merged_split:

                            unit_waveforms = spike_waveforms[merged_clusters_indexes] #np.where(predictions == int(merged_clusters))[0], :]   # Waveforms of originally chosen cluster
                            unit_waveforms = unit_waveforms[np.where(split_predictions == split_merged_cluster)[0], :]  # Subsetting this set of waveforms to include only the chosen split
                            unit_times = spike_times[merged_clusters_indexes] #np.where(predictions == int(merged_clusters))[0]]          # Do the same thing for the spike times
                            unit_times = unit_times[np.where(split_predictions == split_merged_cluster)[0]]

                            save_units(hf5, unit_description, max_unit, unit_name=unit_name, electrode_num = electrode_num, 
                                       u_waveforms=unit_waveforms, u_times=unit_times)
                            unit_description = table.row

        else: # initially choose more than 1 cluster but not merge
              # include each cluster as a separate unit
            for cluster in clusters:

                unit_waveforms = spike_waveforms[np.where(predictions == int(cluster))[0], :]
                unit_times = spike_times[np.where(predictions == int(cluster))[0]]

                save_units(hf5, unit_description, max_unit, unit_name=unit_name, electrode_num = electrode_num, 
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
     
"""


    




