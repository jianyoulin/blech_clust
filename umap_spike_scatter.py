import numpy as np
import os
import umap
import pylab as plt
import glob
from tqdm import trange
from joblib import Parallel, delayed
import multiprocessing as mp
import warnings
import sys
import easygui
warnings.filterwarnings('ignore')

# # Get name of directory with the data files
# if sys.argv[1] != '':
#     data_dir = os.path.abspath(sys.argv[1])
#     if data_dir[-1] != '/':
#         data_dir += '/'
# else:
#     data_dir = easygui.diropenbox(msg = 'Please select data directory')

# Read blech.dir, and cd to that directory
with open('umap_dir.txt','r') as f:
    data_dir = f.readline()[:-1]

# # temp for data_dir
# data_dir = '/mnt/g/testing_hdf5s/env_Data_testing_blech_clust/JK14_20230916_Sacc_230916_104702'
# print(f'Data dir : {data_dir}')

# Read the clustering params for the file
with open(glob.glob(data_dir + '/*params*')[0],'r') as param_file:
    params = [float(line) for line in param_file.readlines()[:-1]]
cluster_num = int(params[0])

# obtain electrode number
electrode_num = int(sys.argv[-1])

# Get PCA waveforms from spike_waveforms
# Get cluster predictions from clustering_results
# Plot output in Plots


def umap_plots(data_dir, electrode_num):
    # If processing has happened, the file will exist
    pca_file = os.path.join(data_dir,
            f'spike_waveforms/electrode{electrode_num}/pca_waveforms.npy')

    if os.path.isfile(pca_file):
        
        try:
            pca_waveforms = np.load(os.path.join(data_dir,
                    f'spike_waveforms/electrode{electrode_num}/pca_waveforms.npy'))

            umap_waveforms = umap.UMAP(n_components = 2).\
                    fit_transform(pca_waveforms[:,:])
            
            clustering_results = [np.load(os.path.join(data_dir,
                    f'clustering_results/electrode{electrode_num}/'\
                    f'clusters{cluster}/predictions.npy')) for cluster in \
                    range(2,cluster_num+1)] 
            
            spike_times = np.load(os.path.join(data_dir,
                    f'spike_times/electrode{electrode_num}/spike_times.npy'))

            print(f'Processing for Electrode {electrode_num} complete')

            for cluster in range(2,cluster_num+1):

                figs_path = f'Plots/{electrode_num}/Plots/'\
                        f'{cluster}_clusters_waveforms_ISIs/'

                fig1, ax1 = plt.subplots()
                scatter = ax1.scatter(umap_waveforms[:,0],umap_waveforms[:,1],\
                        c = clustering_results[cluster-2], s = 2, cmap = 'brg')
                legend = ax1.legend(*scatter.legend_elements())
                ax1.add_artist(legend)
                fig1_name = f'cluster{cluster}_umap.png' 
                fig1_path = os.path.join(data_dir,figs_path, fig1_name)
                print(fig1_path)
                fig1.savefig(fig1_path, dpi = 300)
                plt.close(fig1)

                nbins = np.min([100,int(umap_waveforms.shape[0]/100)])
                fig2, ax2 = plt.subplots()
                ax2.hexbin(umap_waveforms[:,0],umap_waveforms[:,1], gridsize = nbins)
                fig2_name = f'cluster{cluster}_umap_hist.png'
                
                fig2_path = os.path.join(data_dir,figs_path, fig2_name)
                fig2.savefig(fig2_path, dpi = 300)
                plt.close(fig2)

                fig3, ax3 = plt.subplots(2,2,figsize=(20,10))
                nbins = np.min([100,int(umap_waveforms.shape[0]/100)])
                ax3[0,0].hexbin(spike_times, umap_waveforms[:,0], gridsize = nbins)
                scatter = ax3[1,0].scatter(spike_times, umap_waveforms[:,0],\
                        c = clustering_results[cluster-2], s = 2, cmap = 'brg')
                legend = ax3[1,0].legend(*scatter.legend_elements())
                ax3[1,0].add_artist(legend)

                nbins = np.min([100,int(umap_waveforms.shape[0]/100)])
                ax3[0,1].hexbin(spike_times, umap_waveforms[:,1], gridsize = nbins)
                scatter = ax3[1,1].scatter(spike_times, umap_waveforms[:,1],\
                        c = clustering_results[cluster-2], s = 2, cmap = 'brg')
                legend = ax3[1,1].legend(*scatter.legend_elements())
                ax3[1,1].add_artist(legend)

                plt.tight_layout()

                fig3_name = f'cluster{cluster}_umap_timeseries.png'
                fig3_path = os.path.join(data_dir, figs_path, fig3_name)
                fig3.savefig(fig3_path, dpi = 300)
                plt.close(fig3)

                fig4, ax4 = plt.subplots(2,2,figsize=(20,10))
                nbins = np.min([100,int(pca_waveforms.shape[0]/100)])
                ax4[0,0].hexbin(spike_times, pca_waveforms[:,0], gridsize = nbins)
                scatter = ax4[1,0].scatter(spike_times, pca_waveforms[:,0],\
                        c = clustering_results[cluster-2], s = 2, cmap = 'brg')
                legend = ax4[1,0].legend(*scatter.legend_elements())
                ax4[1,0].add_artist(legend)

                nbins = np.min([100,int(pca_waveforms.shape[0]/100)])
                ax4[0,1].hexbin(spike_times, pca_waveforms[:,1], gridsize = nbins)
                scatter = ax4[1,1].scatter(spike_times, pca_waveforms[:,1],\
                        c = clustering_results[cluster-2], s = 2, cmap = 'brg')
                legend = ax4[1,1].legend(*scatter.legend_elements())
                ax4[1,1].add_artist(legend)

                plt.tight_layout()

                fig4_name = f'cluster{cluster}_pca_timeseries.png'
                fig4_path = os.path.join(data_dir, figs_path, fig4_name)
                fig4.savefig(fig4_path, dpi = 300)
                plt.close(fig4)
                print(f'Saved images for Clustering Solution {cluster}')

        except:
            # In other words, I'm too lazy to actually debug shit
            raise Exception('Something went wrong :(')

    else:
        print(f'No electrode{electrode_num} found')


umap_plots(data_dir, electrode_num)


# for electrode_num in trange(len(os.listdir(data_dir + '/clustering_results'))):
#     electrode_num = int(electrode_num[9:])
#     print(electrode_num)
#     #umap_plots(data_dir, electrode_num)

# for electrode_num in os.listdir(data_dir + '/clustering_results'):
#     electrode_num = int(electrode_num[9:])
#     umap_plots(data_dir, electrode_num)
