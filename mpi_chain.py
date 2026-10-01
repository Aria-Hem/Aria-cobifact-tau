# importing the libraries
from cobaya import run
from cobaya.yaml import yaml_load_file

import datetime
t = datetime.datetime.now()
dt_string = t.strftime("%Y-%m-%d_%H-%M-%S")

#from cobifact import *

from mpi4py import MPI

def run_with_mpi(yaml_path, yaml_name, name, sequence_num, tau=0.05,  bao=True):
    info = yaml_load_file(yaml_path+yaml_name)
    if bao:
        info['output'] = 'chains/'+name+'_'+sequence_num+'/chain'
        info['likelihood']['bao.generic']['measurements_file']='/home/aria-hemmatian/Desktop/test2/bao_fake_v2/myBAO_'+sequence_num+'_wN.dat'
        info['params']['tau']['prior']['loc'] = tau
    else:
        info['output'] = 'chains/'+name+'_'+dt_string+'/chain'



    ###################################################
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    from cobaya.log import LoggedError
    success = False
    try:
        updated_info, sampler = run(info)
        success = True
    except LoggedError as err:
        pass

    # Did it work? (e.g. did not get stuck)
    success = all(comm.allgather(success))
    print(sampler)
    if not success and rank == 0:
        print("Sampling failed!")
    ####################################################
    if rank == 0:
        print('Results saved at: %s' %(info['output']))
    pass


def fake_bao_comparison(sequence_num,tau):
    yaml_path = '/home/aria-hemmatian/Desktop/test2/yamls/'
    yaml_name = "test_cmb_min.yaml"
    name = 'fakes/fake_bao_'+str(tau)

    run_with_mpi(yaml_path, yaml_name, name, sequence_num, tau=tau)

    yaml_name_w = "wtest_cmb_min.yaml"
    name_w = 'fakes/fake_bao_w0wa_'+str(tau)

    run_with_mpi(yaml_path, yaml_name_w, name_w, sequence_num, tau=tau)
    pass

def run_mcmc(sequence_num,tau,cpl=True):
    yaml_path = '/home/aria-hemmatian/Desktop/test2/yamls/'
    yaml_name = "test_cmb_myBAO.yaml"
    name = 'fakes/mcmc/fake_bao_'+str(tau)
    if cpl:
        yaml_name = "wtest_cmb_myBAO.yaml"
        name = 'fakes/mcmc/fake_bao_w0wa_'+str(tau)
    run_with_mpi(yaml_path, yaml_name, name, sequence_num, tau=tau)

num = 22
t = 9
fake_bao_comparison(str(num).zfill(3),t/100)
#run_mcmc(str(num).zfill(3),t/100,True)

