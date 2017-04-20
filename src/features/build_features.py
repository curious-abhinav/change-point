import numpy as np

def correlation_categories(data_retina):
    """ assign numerical values to stimuli based on correlation category as per variable stim_categories"""

    stim_categories={'white': 3, 'fullfield' :0, 'multiscale':1, 'spatexp':2, 'spattempexp':2,'tempexp':4, 'natmov':1, 'scramnat':3, 'lowcont_white':3, 'lowcont_multiscale':1 }

    stim=data_retina['stimTimes'][0,0]
    categories_stim=stim.dtype
    stims=[descr[0] for descr in categories_stim.descr]
    corr_categories=[stim_categories[i] for j in stims for i in list(stim_categories.keys())if i in j]
    return corr_categories



def spikes_to_discrete(spikes,duration,res):
    """
    convert spike times to discrete vectors
    
    INPUT
    duration and res in same units
    """  
    n_bins=np.floor(duration/res)
    discrete_spikes=np.zeros(n_bins)
    spike_times=list(np.floor(spikes/res))
    discrete_spikes[spike_times]=1

    return discrete_spikes

