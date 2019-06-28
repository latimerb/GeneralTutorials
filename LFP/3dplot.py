import LFPy
import scipy
import numpy 
numpy.random.seed(12512)
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import pdb
import neuron
import pandas as pd

cellParameters = {
    'morphology': 'Henckens_AM-C1A.swc',
    'templatefile': ['stylized_pyrtypeC_LFPytemplate.hoc'],
    'templatename': 'stylized_pyrtypeC',
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
X, Z = np.mgrid[-650:801:20, -650:901:20]
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

####################################################################


from matplotlib import cm
import matplotlib.colors as colors

t_show = [50,53,56,59,62]

fig = plt.figure(figsize=(10,8))
fig.subplots_adjust(left=0.005,bottom=0.005,right=0.83,top=0.76,wspace=0.1,hspace=0.01)
ex_ct = []
inh_ct = []
rat_ct = []
for i in np.arange(0,len(t_show)):

    exc = pd.read_csv('./data/stick/zoomoutexcitatory_LFP_at_{}.csv'.format(t_show[i]),header=None)
    inh = pd.read_csv('./data/stick/zoomoutinhibitory_LFP_at_{}.csv'.format(t_show[i]),header=None)

    ratio = np.divide(np.abs(exc.values),np.abs(inh.values))

    plt.subplot(3,len(t_show),i+1)
    A = exc.values.T[::-1]
    hm1=plt.matshow(np.abs(A),norm=colors.LogNorm(vmin=10e-5,vmax=1),cmap=cm.jet,fignum=0)
    ex_ct.append(hm1)
    if i==0:
        plt.plot([61,66],[74,74],'k-')
        plt.plot([61,61],[74,69],'k-')
    plt.axis('off')

    plt.subplot(3,len(t_show),i+len(t_show)+1)
    B = inh.values.T[::-1]
    hm2=plt.matshow(np.abs(B),norm=colors.LogNorm(vmin=10e-5,vmax=1),cmap=cm.jet,fignum=0)
    inh_ct.append(hm2)
    if i==0:
        plt.plot([61,66],[74,74],'k-')
        plt.plot([61,61],[74,69],'k-')
    plt.axis('off')

    plt.subplot(3,len(t_show),i+2*len(t_show)+1)
    C = np.divide(np.abs(A),np.abs(B))
    bounds = np.linspace(0,10,11)
    hm=plt.matshow(C,cmap=cm.OrRd,norm=colors.BoundaryNorm(boundaries=bounds,ncolors=256),fignum=0)
    rat_ct.append(hm)
    if i==0:
        plt.plot([61,66],[74,74],'k-')
        plt.plot([61,61],[74,69],'k-')
    hm.set_clim(0,10)
    plt.axis('off')

cbaxes1 = fig.add_axes([0.85,0.52,0.03,0.23])
cbar = plt.colorbar(ex_ct[0],cax=cbaxes1)
cbaxes2 = fig.add_axes([0.85,0.27,0.03,0.23])
cbar = plt.colorbar(inh_ct[0],cax=cbaxes2)
cbaxes3 = fig.add_axes([0.85,0.02,0.03,0.23])
cbar = plt.colorbar(rat_ct[0],cax=cbaxes3)
fig.savefig('heatmaps_zoomout.svg',bbox_inches='tight',pad_inches=0)
#from utils import contour3d,plot_elec_grid_stick,zerosurf,plot_ex1

#plot_ex1(cell,electrode,X,Y,Z,time_show=[50,53,56,59,62],space_lim=[np.min(X[:,0]),np.max(X[:,0]),np.min(Z[0,:]),np.max(Z[0,:])])
plt.show()
