from ..features import build_features
from ..visualization import visualize

import numpy as np
import scipy.stats as stats
import scipy.signal
import numpy.matlib

import pdb
import matplotlib.pyplot as plt
from sklearn.covariance import GraphLasso
import sys

def detect_changes(data_retina, params):

    res=params['res']
    block_width=params['block_width']
    gauss_width=params['gauss_width']
    file=data_retina['file']
    method_corr=params['methods']

    stim=data_retina['stimTimes'][0,0]
    data_categories=build_features.correlation_categories(data_retina)
    spikes=data_retina['spikeTimes']
    num_neurons=spikes.shape[1]
    last_spike=np.array([spikes[0,i][0].max() for i in range(num_neurons) if spikes[0,i].size >0])
    duration=last_spike.max() + 200
    last_stim=np.array([stim[i].max() for i in range(len(stim))])
    

    spikes=data_retina['spikeTimes']
    num_neurons=spikes.shape[1]
    plt.figure() #figsize=(18,5
    for i in range(num_neurons):
        plt.plot(spikes[0,i][0].T, i*np.ones(spikes[0,i][0].size), '.k')
    plt.xticks(rotation='vertical')

    #visualize.plot_stim(stim, data_categories, file)
    sum_diff_corr={}
    for method in method_corr:
        sum_diff_corr[method]=block_corr_change_retina(data_retina, duration, gauss_width, block_width, res, case=method)
    return (sum_diff_corr, stim)

def detect_brian_spikes_changes(brian_data, params):

    res=params['res']
    block_width=params['block_width']
    gauss_width=params['gauss_width']
    #file=data_retina['file']
    method_corr=params['methods']
    sum_diff_corr={}
    for method in method_corr:
        sum_diff_corr[method]=summary_corr_matrices(brian_data, gauss_width, block_width, case=method)

    return (sum_diff_corr)





def summary_corr_matrices(discrete_spikes, gauss_width, block_width, case):
    """
    calculate sum of chnages in correlation matrices calculated by block
    """
    
    import sklearn.covariance
    #from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance, \
    #log_likelihood, empirical_covariance
    gauss_filter=stats.norm.pdf(range(-gauss_width,gauss_width+1))
    gauss_circuit_array=scipy.signal.convolve(discrete_spikes,numpy.matlib.repmat(gauss_filter,1,1))
    temp_circuit_array=gauss_circuit_array.T
    #sklearn_cov_matrices= np.array([sklearn.covariance.ledoit_wolf(temp_circuit_array[i: i+block_width,:])[0] for i in range(0,np.shape(gauss_circuit_array)[1]- block_width, block_width)])
    ##sklearn_cov_matrices= np.array([sklearn.covariance.oas(temp_circuit_array[i: i+block_width,:])[0] for i in range(0,np.shape(gauss_circuit_array)[1]- block_width, block_width)])
    sklearn_cov_matrices= np.array([sklearn.covariance.ledoit_wolf(sklearn.preprocessing.scale(temp_circuit_array[i: i+block_width,:]))[0] for i in range(0,np.shape(gauss_circuit_array)[1]- block_width, block_width)])
    
    #sklearn.covariance.ledoit_wolf()
    #corr_matrices=np.array([np.corrcoef(gauss_circuit_array[:,i: i+block_width]) for i in range(0,np.shape(gauss_circuit_array)[1]- block_width, block_width)])
    corr_matrices=sklearn_cov_matrices  
    corr_matrices[np.isnan(corr_matrices)]=0
    print(corr_matrices.shape)
    #sqrt_corr_matrices=np.sqrt(corr_matrices)
    #case='base'
    if case=='diff_abs':
        diff_corr_matrices=np.abs(np.diff(corr_matrices, axis=0))
        trans_diff_corr=np.nansum(diff_corr_matrices, axis=(1,2))
    elif case=='diff_base':
        diff_corr_matrices=np.diff(corr_matrices, axis=0)
        trans_diff_corr=np.power(np.nansum(diff_corr_matrices, axis=(1,2)),2)
    elif case=='diff_sqrt':
        diff_corr_matrices=np.diff(sqrt_corr_matrices, axis=0) 
        diff_corr_matrices=np.power(diff_corr_matrices,2)
        trans_diff_corr=np.nansum(diff_corr_matrices, axis=(1,2))
    elif case=='diff_variation':
        diff_corr_matrices=np.diff(corr_matrices, axis=0)
        trans_diff_corr=np.nanstd(diff_corr_matrices, axis=(1,2))/np.nanmean(diff_corr_matrices, axis=(1,2))
    elif case=='variation':
        trans_diff_corr=np.nanstd(corr_matrices, axis=(1,2))/np.nanmean(corr_matrices, axis=(1,2))
    elif case=='stdev':
        trans_diff_corr=np.nanstd(corr_matrices, axis=(1,2))#/np.nanmean(corr_matrices, axis=(1,2))
    elif case=='mean':
        trans_diff_corr=np.nanmean(corr_matrices, axis=(1,2))#/np.nanmean(corr_matrices, axis=(1,2))
    elif case=='pop_sum':
        pop_sum=np.sum(gauss_circuit_array, axis=0)
        trans_diff_corr=convolve_series(pop_sum)
    elif case=='frobenius':
        trans_diff_corr=np.linalg.norm(corr_matrices, axis=(1,2))
    elif case=='diff_frobenius':
        diff_corr_matrices=np.diff(corr_matrices, axis=0)
        trans_diff_corr=np.linalg.norm(diff_corr_matrices, axis=(1,2))
    
    
    return trans_diff_corr

def block_corr_change_retina(data_retina, duration, gauss_width, block_width, res, case):
    #stim=data_retina['stimTimes'][0,0]
    spikes=data_retina['spikeTimes']
    num_neurons=spikes.shape[1]
    last_spike=np.array([spikes[0,i][0].max() for i in range(num_neurons) if spikes[0,i].size >0])
    duration=last_spike.max() + 200
    discrete_spikes=np.array([build_features.spikes_to_discrete(spikes[0,i][0], duration,res) for i in range(num_neurons) ])
    #print(discrete_spikes.shape)
    sum_diff_corr=summary_corr_matrices(discrete_spikes, gauss_width, block_width, case)
    return sum_diff_corr

def convolve_series(series):
    conv_filters=[5,10,20,50,100]
    filtered_coactivation=[]
    
    for i,filter in enumerate(conv_filters):
        print(i,filter)
        temp_coactivation=np.convolve(series,np.ones(filter))
        filtered_coactivation.append(temp_coactivation)
    return filtered_coactivation
        #print len(filtered_coactivation)

