from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import scipy.io
import numpy as np

from ..features import build_features

firebrick=mcolors.CSS4_COLORS['firebrick']
red=mcolors.CSS4_COLORS['red']
coral=mcolors.CSS4_COLORS['coral']
seagreen=mcolors.CSS4_COLORS['seagreen']
grey=mcolors.CSS4_COLORS['grey']
royalblue=mcolors.CSS4_COLORS['royalblue']
color_ctgry=(grey, royalblue, red, coral, seagreen)
spat_corr={'grey': 'None', 'royalblue': 'Delta---', 'red':'Delta--', 'coral':'Delta-', 'seagreen':'None+' }


def plot_stim(stim, data_categories, file):
    """ Plot retinal stimulus with color encodings for correlation categories """
    NUM_COLORS = len(data_categories)

    #cm = plt.get_cmap('gist_rainbow')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in data_categories])
    for i in range(NUM_COLORS):
        ax.plot(stim[i][0,:], 150*np.ones(stim[i].shape[1]), '.', color=color_ctgry[data_categories[i]])#(np.arange(10)*(i+1))
    plt.xticks(rotation='vertical')
    plt.title(file)
    return None

def plot_file_stim(file):
    """ Plot retinal stimulus data and assign encodings to correlation categories"""
    data_retina=scipy.io.loadmat(dir+file)
    stim=data_retina['stimTimes'][0,0]
    data_categories=build_features.correlation_categories(data_retina)
    print(file, data_categories)
    try: 
        plot_stim(stim, data_categories, file)
    except:
        pass

def plot_cp_results(sum_diff_corr,stim, data_retina, params):
    """ Plot dynamics of summary statistic"""
    stim=data_retina['stimTimes'][0,0]
    data_categories=build_features.correlation_categories(data_retina)

    
    res=params['res']
    block_width=params['block_width']
    gauss_width=params['gauss_width']
    file=data_retina['file']
    method_corr=params['methods']


    for method in method_corr:
        #color=iter(cm.Dark2(np.linspace(0,1,len(stim.dtype.names))))
        NUM_COLORS=len(stim.dtype.names)

        cm = plt.get_cmap('Dark2')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
        if method=='pop_sum':
            for count,i in enumerate(stim.dtype.names):
                #subc=next(color)
                plt.plot(stim[i][0,:], 150*np.ones(stim[i].shape[1]), '.', color=color_ctgry[data_categories[count]])
                
            for i in sum_diff_corr[method]:
                time_pt=np.arange(0,res*i.size,res)
                plt.plot(time_pt,i)
                plt.xticks(rotation='vertical')

        else:

            time_pt=np.arange(0,res*block_width*sum_diff_corr[method].size,res*block_width)
            plt.plot(time_pt, sum_diff_corr[method])
            for count,i in enumerate(stim.dtype.names):
                #subc=next(color)
                #print(i)  printing stim names
                plt.plot(stim[i][0,:], np.nanmax(sum_diff_corr[method][np.isfinite(sum_diff_corr[method])])*np.ones(stim[i].shape[1]), '.', color=color_ctgry[data_categories[count]])
                plt.xticks(rotation='vertical')
        plt.title(method + ",  " + file)