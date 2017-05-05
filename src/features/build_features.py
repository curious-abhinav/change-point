import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def correlation_categories(data_retina):
    """ assign numerical values to stimuli based on correlation category as per variable stim_categories"""

    stim_categories = {'white': 3, 'fullfield': 0, 'multiscale': 1, 'spatexp': 2, 'spattempexp': 2,
                       'tempexp': 4, 'natmov': 1, 'scramnat': 3, 'lowcont_white': 3, 'lowcont_multiscale': 1}

    stim = data_retina['stimTimes'][0, 0]
    categories_stim = stim.dtype
    stims = [descr[0] for descr in categories_stim.descr]
    corr_categories = [stim_categories[i]
                       for j in stims for i in list(stim_categories.keys())if i in j]
    return corr_categories


def spikes_to_discrete(spikes, duration, res):
    """
    convert spike times to discrete vectors

    INPUT
    duration and res in same units
    """
    n_bins = np.floor(duration / res)
    discrete_spikes = np.zeros(n_bins)
    spike_times = list(np.floor(spikes / res))
    discrete_spikes[spike_times] = 1

    return discrete_spikes


def sim_spikes(brian_params, discrete_params):
    """ Use brian to simulate spiking output of neural circuits"""

    epoch_sizes = brian_params['epoch_sizes']
    # neurons in each cluster, in each epoch
    cluster_neurons = brian_params['cluster_neurons']
    jitter_neurons = brian_params['sigma']
    time_constants = brian_params['time_constant']
    num_epochs = len(epoch_sizes)
    spike_trains = []
    df_spikes = pd.DataFrame(columns=['Epoch', 'Spikes', 'Discrete_spikes'])

    for _i in brian_params:
        (epoch_number, epoch_size, num_neurons, jitter, time_constant) = brian_params.loc[
            _i, ['epoch_number', 'epoch_size', 'cluster_neurons', 'jitter', 'time_constant']]
        spikes = brian_spikes(N=num_neurons, tau=time_constant,
                              duration=epoch_size, sigma=jitter)
        res = discrete_params['res']
        discrete_spikes = brian_spikes_to_discrete_array(
            spikes, epoch_size, res)
        df_spikes.append([epoch_number, spikes, discrete_spikes])

        for i, epoch in enumerate(epoch_sizes):
            for epoch_neurons in cluster_neurons:
                for neurons in epoch_neurons:
                    pass


def brian_spikes(N=15, tau=20, v0_max=3.0, duration=7000, sigma=0.01):
    """
    generate spikes based on biophysical parameters for a neural population
    """

    #import binary_array_networks

    import brian2
    from brian2 import start_scope, ms, NeuronGroup, SpikeMonitor

    start_scope()

    #N = 15
    tau = tau * ms
    #v0_max = 3.
    duration = duration * ms
    #sigma = 0.01

    # case='1_3'
    # res=2*ms
    eqs = '''
    dv/dt = (v0-v)/tau+sigma*xi*tau**-0.5 : 1
    v0 : 1
    '''
    # dv/dt = (2-v)/tau : 1
    G = NeuronGroup(N, eqs, threshold='v>1', reset='v=0')
    G.v = 'rand()'

    spikemon = SpikeMonitor(G)
    G.v0 = 1.1
    brian2.run(duration)

    return spikemon


def brian_spikes_to_discrete_array(spikemon, duration, res):
    """
    convert brian2 spikemonitor object into binary array

    """
    import brian2
    from brian2 import start_scope, ms, NeuronGroup, SpikeMonitor

    res = res * ms
    duration = duration * ms
    n_bins = np.floor(duration / res)
    n_neurons = spikemon.source.N
    array_bins = np.zeros([n_neurons, n_bins], dtype=np.int8)
    neurons_spiketimes = list(spikemon.i)
    bins_with_spikes = list(np.floor(spikemon.t / res))
    array_bins[neurons_spiketimes, bins_with_spikes] = 1

    return array_bins


def sim_brian_spikes(n_epoch1=np.array([30, 30, 30]), n_epoch2=np.array([44, 45, 1]), duration_epochs=5400, num_tot_neurons=90, tau_epoch1=np.array([15, 30, 45]), tau_epoch2=np.array([15, 53, 75]), res=2):

    #import brian2
    #from brian2 import start_scope, ms, NeuronGroup, SpikeMonitor

    # res unit: ms
    # unit: ms

    # thresh=0.05
    # window_bins=19
    # conv_filters=[5,10,20,50,100]

    # tau_epoch1=np.array([15,37,53])
    # tau_epoch1=np.array([20,33,47])

    # tau_epoch1=np.array([15,37,53])
    # tau_epoch1=np.array([20,33,47])

    # smooth_bin_width=11
    # block_width=37
    #gauss_width=10 *res

    import brian2
    from brian2 import start_scope, ms, NeuronGroup, SpikeMonitor

    sigma_equal = 1
    sigma_all = 0.1
    if sigma_equal == 1:
        sigma_epoch1 = sigma_all * np.ones(n_epoch1.shape)
        sigma_epoch2 = sigma_all * np.ones(n_epoch2.shape)
    else:
        sigma_epoch1 = 0.01 * np.ones(3)
        sigma_epoch2 = 0.08 * np.ones(3)

        def params_3_2_const_coactivations():
            res = 2  # unit: ms
            duration_epochs = 1800  # unit: ms
            num_tot_neurons = 90
            thresh = 0.05
            window_bins = 19
            n_epoch1 = np.array([30, 30, 30])
            n_epoch2 = np.array([44, 45, 1])
            smooth_bin_width = 11
            block_width = 50
            gauss_width = 10 * res

            sigma_equal = 1
            sigma_all = 0.1
            if sigma_equal == 1:
                sigma_epoch1 = sigma_all * np.ones(n_epoch1.shape)
                sigma_epoch2 = sigma_all * np.ones(n_epoch2.shape)
            else:
                sigma_epoch1 = 0.01 * np.ones(3)
                sigma_epoch2 = 0.08 * np.ones(3)

    # vectorise the compression distance function
    # vec_ncd=np.vectorize(cp_utils.ncd)

    #%%
    # generate spikes
    # spikes=[]

    #spikes_epoch1=cp_utils.brian_spikes(N=num_tot_neurons, duration=duration_epochs)
    spikes_epoch1_cluster1 = brian_spikes(N=n_epoch1[0], tau=tau_epoch1[
                                          0], duration=duration_epochs, sigma=sigma_epoch1[0])
    spikes_epoch1_cluster2 = brian_spikes(N=n_epoch1[1], tau=tau_epoch1[
                                          1], duration=duration_epochs, sigma=sigma_epoch1[1])
    spikes_epoch1_cluster3 = brian_spikes(N=n_epoch1[2], tau=tau_epoch1[
                                          2], duration=duration_epochs, sigma=sigma_epoch1[2])

    # spikes.extend([[spikes_epoch1]])

    spikes_epoch2_cluster1 = brian_spikes(N=n_epoch2[0], tau=tau_epoch2[
                                          0], duration=duration_epochs, sigma=sigma_epoch2[0])
    spikes_epoch2_cluster2 = brian_spikes(N=n_epoch2[1], tau=tau_epoch2[
                                          1], duration=duration_epochs, sigma=sigma_epoch2[1])
    spikes_epoch2_cluster3 = brian_spikes(N=n_epoch2[2], tau=tau_epoch2[
                                          2], duration=duration_epochs, sigma=sigma_epoch2[2])

    duration = duration_epochs
    #discrete_epoch1=cp_utils.brian_spikes_to_discrete_array(spikes_epoch1, duration, res)

    def plot_spikes():
        plt.figure(figsize=(10, 5))
        plt.plot((spikes_epoch1_cluster1.t), spikes_epoch1_cluster1.i, '.k')
        plt.plot((spikes_epoch1_cluster2.t),
                 spikes_epoch1_cluster2.i + sum(n_epoch1[:1]), '.k')
        plt.plot((spikes_epoch1_cluster3.t),
                 spikes_epoch1_cluster3.i + sum(n_epoch1[:2]), '.k')

        plt.plot((spikes_epoch2_cluster1.t + duration_epochs * ms),
                 spikes_epoch2_cluster1.i, '.k')
        plt.plot((spikes_epoch2_cluster2.t + duration_epochs * ms),
                 spikes_epoch2_cluster2.i + sum(n_epoch2[:1]), '.k')
        plt.plot((spikes_epoch2_cluster3.t + duration_epochs * ms),
                 spikes_epoch2_cluster3.i + sum(n_epoch2[:2]), '.k')
        plt.xlabel('Time')
        plt.ylabel('Neuron')

    plot_spikes()

    discrete_epoch1_cluster1 = brian_spikes_to_discrete_array(
        spikes_epoch1_cluster1, duration, res)
    discrete_epoch1_cluster2 = brian_spikes_to_discrete_array(
        spikes_epoch1_cluster2, duration, res)
    discrete_epoch1_cluster3 = brian_spikes_to_discrete_array(
        spikes_epoch1_cluster3, duration, res)
    discrete_epoch1 = np.concatenate(
        (discrete_epoch1_cluster1, discrete_epoch1_cluster2, discrete_epoch1_cluster3), axis=0)
    discrete_epoch1.shape

    discrete_epoch2_cluster1 = brian_spikes_to_discrete_array(
        spikes_epoch2_cluster1, duration, res)
    discrete_epoch2_cluster2 = brian_spikes_to_discrete_array(
        spikes_epoch2_cluster2, duration, res)
    discrete_epoch2_cluster3 = brian_spikes_to_discrete_array(
        spikes_epoch2_cluster3, duration, res)
    discrete_epoch2 = np.concatenate(
        (discrete_epoch2_cluster1, discrete_epoch2_cluster2, discrete_epoch2_cluster3), axis=0)

    circuit_binary_array = np.concatenate(
        (discrete_epoch1, discrete_epoch2), axis=1)

    return circuit_binary_array
