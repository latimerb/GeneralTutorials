import LFPy
import scipy
import numpy 
numpy.random.seed(12512)
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import pdb
import neuron

cellParameters = {
    'morphology': 'Henckens_AM-C1A.swc',
    'templatefile': ['Henckens_AM-C1A_LFPytemplate.hoc'],
    'templatename': 'Henckens_AMC1A',
    'passive_parameters':{'g_pas':1./30000,'e_pas':-70.},
    'tstart' : 0,
    'tstop' : 70,
    'dt' : 2**-4,
    'passive': False,
    'nsegs_method': 'fixed_length',
    'max_nsegs_length':5,
    'cm':0.5,
    'Ra':150,
    'v_init':-70.,
    'celsius':31.,
    'pt3d':True,
    'verbose':True
}

SynapseParameters = {
    'syntype' : 'Exp2Syn',
    'e' : -75,
    'tau1' : 0.83,
    'tau2' : 4.2,
    'weight' : 0.009,
    'record_current' : True,
}


# Define electrode parameters
# Create a grid of measurement locations, in (mum)
X, Z = np.mgrid[-350:501:100, -350:501:100]
Y = np.zeros(X.shape)

electrodeParameters = {
    'sigma' : 0.3,      # extracellular conductivity
    'x' : X.flatten(),  # electrode requires 1d vector of positions
    'y' : Y.flatten(),
    'z' : Z.flatten()
}


# Delete old sections
LFPy.cell.neuron.h("forall delete_section()")

# Create cell and print the variables belonging to it.
cell = LFPy.TemplateCell(**cellParameters)
cell.set_pos(x=0, y=0, z=0)
cell.set_rotation(z=np.pi,x=0)

#from pprint import pprint
#print(pprint(vars(cell)))


num_syns = 1

insert_synapses_AMPA_args = {
    'section' : 'dend[0]',
    'n' : num_syns,
    'spTimesFun' : LFPy.inputgenerators.get_activation_times_from_distribution,
    'args' : dict(n=1, tstart=0, tstop=cellParameters['tstop'],
                  distribution=scipy.stats.poisson,
                  rvs_args=dict(mu=500.0)
                  )
}

wgts = []

def insert_synapses(synparams, section, n, spTimesFun, args):
    '''find n compartments to insert synapses onto'''
    spk_times_save = []
    idx = cell.get_rand_idx_area_norm(section=section, nidx=n)
    
    mu_wgt = synparams['weight']
    #Insert synapses in an iterative fashion
    for i in idx:
  
        # make synaptic weights vary around the mean
        rand_wgt = numpy.random.normal(mu_wgt,mu_wgt/10)
        synparams['weight'] = rand_wgt
        wgts.append(synparams['weight'])
        synparams.update({'idx' : int(i)})
        
        # Some input spike train using the function call
        [spiketimes] = spTimesFun(**args)

        # Create synapse(s) and setting times using the Synapse class in LFPy
        s = LFPy.Synapse(cell, **synparams)
        s.set_spike_times(spiketimes)

        spk_times_save.append(spiketimes)
        
    return spk_times_save

# Synapse
synapse = LFPy.Synapse(cell,
                       idx = cell.get_closest_idx(x=30,y=0,z=0),
                       **SynapseParameters)

# Synapse will be applied at 50 ms
synapse.set_spike_times(array([50])) 

#syn_spike_times = insert_synapses(SynapseParameters,**insert_synapses_AMPA_args)

# Set electrode
electrode = LFPy.RecExtElectrode(**electrodeParameters)

import neuron

cell.simulate(electrode=electrode,
              rec_imem=True,
              rec_ipas=True,
              rec_icap=True,
              rec_vmem=True,
              rec_current_dipole_moment=True)

from utils import contour3d,plot_elec_grid_stick,zerosurf,plot_ex1

#plot_elec_grid_stick(cell,electrode)
plot_ex1(cell,electrode,X,Y,Z,time_show=[50,53,56,59,62],space_lim=[np.min(X[:,0]),np.max(X[:,0]),np.min(Z[0,:]),np.max(Z[0,:])])
#contour3d(cell,electrode,X,Y,Z,tshow=53)
#zerosurf(cell,electrode,X,Y,Z,tshow=53)
plt.show()
