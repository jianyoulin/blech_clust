# blech_clust

Python and R based code for clustering and sorting electrophysiology data recorded using the Intan RHD2132 chips. 
Originally written for cortical multi-electrode recordings in Don Katz's lab at Brandeis. 


Steps (sequency of script to run in the Ubuntu terminal) to analyze EMG data recorded from Intan System

Here's the set of codes you run from blech_clust to process your data now (need to run it after activating your R-equipped conda environment):
1. DIR=Path to your data directory    :::define dir_path
2. Python plot_raw_trace.py $DIR    :::Detect where EMG channels are
3. python blech_clust.py $DIR
4. python emg_make_arrays.py $DIR
5. python filter_emg.py $DIR
6. python emg_local_BSA.py $DIR
7. bash blech_emg_jetstream_parallel.sh 
8. python emg_local_BSA_post_process.py $DIR ::: save results of bash comments into hdf5 file

After this, the results will be saved in the HDF5 file. The data will be as follows:
Node: hf5.root.emg_BSA_results
Arrays:

omega: The frequencies that were tested on the envelope of the EMG signal (shape = 20)
taste_0_p to taste_3_p (assuming we have 4 tastes): The probabilities of each of the frequencies on each trial of each taste. Shape of each taste-specific array is num_trials x 7000 (pre_time+post_time) x 20 (shape of omega).

9. python emg_BSA_segmentation.py $DIR ::: run this on each data file separately
10. python emg_BSA_segmentation_plot.py ::: run the plot code to string all the files together.
