# importing the libraries
from cobaya import run, load_samples
from cobaya.yaml import yaml_load_file
import pathlib
path = str(pathlib.Path(__file__).parent.resolve())
from cobaya.samplers.mcmc import plot_progress
import matplotlib.pyplot as plt
from getdist.mcsamples import loadMCSamples, MCSamplesFromCobaya
import getdist.plots as gdplt
import scipy.stats as st
import numpy as np
import pandas as pd
import datetime

def run_from_file(path_yaml,output=None):
    info = yaml_load_file(path+'/'+path_yaml)
    info['output'] = output
    updated_info, sampler = run(info)
    return updated_info, sampler


def give_sample(updated_info, sampler):
    gd_sample = MCSamplesFromCobaya(updated_info, sampler.products()["sample"],settings={'ignore_rows':0.3})
    return gd_sample


def give_sample_from_file(chain_path):
    gd_samples = []
    for single_path in chain_path:
        gd_samples.append(loadMCSamples(path+'/'+single_path, settings={'ignore_rows':0.3}))
    return gd_samples

# ploting the results
def tri_plot(gd_samples, gd_sample_names, params=['H0','omegam'], color='blue', filled = True, wid=7):
    gdplot = gdplt.get_subplot_plotter(width_inch=wid)
    gdplot.triangle_plot(gd_samples, params, legend_labels=gd_sample_names, contour_colors=color, filled=filled, title_limit=1)
    #gdplot.add_legend(gd_sample_names,bbox_to_anchor=(1.0,1.0))
    return gdplot


def save_plot(gdplot, filename, save_path):
    gdplot.export(filename, path + '/' + save_path)
    pass

def show_progress(sampler, save_path, do_save=False):
    plot_progress(sampler.progress, figure_kwargs={"figsize": (6,4)})
    plt.tight_layout()
    plt.show()
    if do_save:
        plt.savefig(path + '/' +save_path+'/progress.png')
    return

def give_chi2_MCMC(sampler, is_path=False):
    if is_path:
        full_chain = load_samples(sampler, skip=0.3, combined=True)
        return full_chain.bestfit()['chi2']
    return np.min(sampler.products()['sample']["chi2"])

def give_chi2_bf(path, num_l):
    a = pd.read_csv(path+'.minimum',on_bad_lines='skip')
    chi2 = np.zeros(num_l)
    for i in range(num_l):
        chi2[i] =  float(a.iloc[-i-1,0].split()[1])
    return np.sum(chi2)

def delta_chi2(sampler_1,sampler_2):
    return give_chi2_MCMC(sampler_1)-give_chi2_MCMC(sampler_2)

def give_cred_int(chi2,k):
    prob = st.chi2.cdf(chi2,k)
    X = st.chi2(1)
    return np.sqrt(X.ppf(prob))

#####################################

def fact_run_with_mpi(yaml_path, yaml_name, name):
    t = datetime.datetime.now()
    dt_string = t.strftime("%Y-%m-%d_%H-%M-%S")


    info = yaml_load_file(yaml_path+yaml_name)
    info['output'] = 'chains/'+name+'_'+dt_string+'/chain'


    ###################################################
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    from cobaya.log import LoggedError
    success = False
    try:
        updated_info, sampler = run(info, debug=40)
        success = True
    except LoggedError as err:
        pass

    # Did it work? (e.g. did not get stuck)
    success = all(comm.allgather(success))

    if not success and rank == 0:
        print("Sampling failed!")
    ####################################################
    pass

# Things I need to add:
# 1- 
