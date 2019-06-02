import matplotlib.pyplot as plt
import numpy as np
import LFPy 

def Plot_geo_currs_volt(cell,electrode,synapse):

	plt.figure(figsize=(15, 10))
	plt.subplot(1,3,1)
	# Plot geometry of the cell
	for sec in LFPy.cell.neuron.h.allsec():
	    idx = cell.get_idx(sec.name())
	    plt.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
		    np.r_[cell.zstart[idx], cell.zend[idx][-1]],
		    'k',linewidth=8)
	    plt.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
		    np.r_[cell.zstart[idx], cell.zend[idx][-1]],
		    'r_')
	    for i in np.arange(0,cell.xstart[idx].shape[0]):
		segs = [5,4,3,2,1]
		plt.text(cell.xstart[idx][i]-75,cell.zend[idx][i]-50,'Seg. {}'.format(segs[i]))
	# Plot synapse as red dot
	plt.plot([cell.synapses[0].x], [cell.synapses[0].z], \
	     color='r', marker='o', markersize=5,label='synapse')
	# Plot electrodes as green dots
	plt.plot(electrode.x, electrode.z, '.', marker='o', color='g',label='electrodes')
	for i in np.arange(0,9):
	    plt.text(electrode.x[i]+10,electrode.z[i]-5,'{}'.format(i))

	# Plot distances used in LFP calculation
	#elec_idx = 2
	#seg_idx = 1
	#idx = cell.get_idx('soma[0]')
	#xr = np.arange(cell.xstart[idx][0],0)
	#yr = electrode.z[elec_idx]*np.ones((xr.shape[0],))
	#plot(xr,yr,color=[0.5,0.5,0.5]),text(cell.xstart[idx][0]/2,electrode.z[elec_idx]+5,'r')
	#h = -250*np.ones((86,))
	#hy = np.arange(14,100)
	#plot(h,hy,color=[0.5,0.5,0.5]),text(-240,50,'h')
	#l = -200*np.ones((113,))
	#ly = np.arange(-12.5,100)
	#plot(l,ly,color=[0.5,0.5,0.5]), text(-180,40,'l')
	plt.xlabel('distance (um)')
	plt.ylabel('distance (um)')

	plt.ylim(-450,450)
	plt.xlim(-350,100)
	plt.legend()

	sp = [14,11,8,5,2]
	segs = [5,4,3,2,1]
	for i in np.arange(1,6):
	    ax=plt.subplot(5,3,sp[i-1])
	    plt.plot(cell.tvec,cell.imem[i-1,:],'blue',label='imembrane')
	    plt.plot(cell.tvec,cell.ipas[i-1,:],'red',label='i_pas')
	    plt.plot(cell.tvec,cell.icap[i-1,:],'green',label='i_cap')
	    
	    if i!=5:
		plt.text(45,0.06,'segment {}'.format(segs[i-1]))
		plt.ylim(-0.02,0.11)
		
	    else:
		plt.text(45,-0.1,'segment {}'.format(segs[i-1]))
		ax.plot(cell.tvec, synapse.i,'m',label='i_syn')
		plt.legend(loc='lower right')
		plt.title('currents (nA)')
		plt.ylim(-0.5,0.11)
	    if i==1:
		plt.xlabel('time (ms)')
	    plt.xlim(42.5,62.5)
	    
		
	#### Calculate extracellular potential by hand

	# electrode 5 is at 0,100
	#h = np.zeros((5,1))
	#l = np.zeros((5,1))
	#r = np.zeros((5,1))

	#for i in np.arange(0,5):
	#    h[i,0] = electrode.z[5]-cell.zend[i]
	#    l[i,0] = electrode.z[5]-cell.zstart[i]
	#    r[i,0] = electrode.x[5]-cell.xstart[i]

	#dist = np.log((np.sqrt(h**2+r**2)-h)/(np.sqrt(l**2+r**2)-l))
	#v_ext=1000*(1/(4*np.pi*300*5))*np.dot(np.transpose(cell.imem),dist)

	elecs_to_plot = [8,5,0]
	for i in np.arange(1,4):
	    ax=plt.subplot(3,3,3*i)
	    plt.plot(cell.tvec,electrode.LFP[elecs_to_plot[i-1],:],'k')
	    plt.xlim(42.5,62.5)
	    plt.text(45,0.000002,'electrode {}'.format(elecs_to_plot[i-1]))
	    ax.yaxis.tick_right()
	    if i==1:
		#plot(cell.tvec,v_ext,'k')
		plt.title('extracellular voltage (mV)')
	    if i==3:
		plt.xlabel('time(ms)')
	    plt.ylim(-0.0001,0.0001)	


def Plot_active_currs_volt(cell,electrode,synapse):

	plt.figure(figsize=(15, 10))
	plt.subplot(1,3,1)
	# Plot geometry of the cell
	for sec in LFPy.cell.neuron.h.allsec():
	    idx = cell.get_idx(sec.name())
	    plt.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
		    np.r_[cell.zstart[idx], cell.zend[idx][-1]],
		    'k',linewidth=8)
	    plt.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
		    np.r_[cell.zstart[idx], cell.zend[idx][-1]],
		    'r_')
	    for i in np.arange(0,cell.xstart[idx].shape[0]):
		segs = [5,4,3,2,1]
		plt.text(cell.xstart[idx][i]-75,cell.zend[idx][i]-50,'Seg. {}'.format(segs[i]))
	# Plot synapse as red dot
	plt.plot([cell.synapses[0].x], [cell.synapses[0].z], \
	     color='r', marker='o', markersize=5,label='synapse')
	# Plot electrodes as green dots
	plt.plot(electrode.x, electrode.z, '.', marker='o', color='g',label='electrodes')
	for i in np.arange(0,9):
	    plt.text(electrode.x[i]+10,electrode.z[i]-5,'{}'.format(i))

	# Plot distances used in LFP calculation
	#elec_idx = 2
	#seg_idx = 1
	#idx = cell.get_idx('soma[0]')
	#xr = np.arange(cell.xstart[idx][0],0)
	#yr = electrode.z[elec_idx]*np.ones((xr.shape[0],))
	#plot(xr,yr,color=[0.5,0.5,0.5]),text(cell.xstart[idx][0]/2,electrode.z[elec_idx]+5,'r')
	#h = -250*np.ones((86,))
	#hy = np.arange(14,100)
	#plot(h,hy,color=[0.5,0.5,0.5]),text(-240,50,'h')
	#l = -200*np.ones((113,))
	#ly = np.arange(-12.5,100)
	#plot(l,ly,color=[0.5,0.5,0.5]), text(-180,40,'l')
	plt.xlabel('distance (um)')
	plt.ylabel('distance (um)')

	plt.ylim(-450,1200)
	plt.xlim(-350,100)
	plt.legend()

	sp = [14,11,8,5,2]
	segs = [5,4,3,2,1]
	for i in np.arange(1,6):
	    ax=plt.subplot(5,3,sp[i-1])
	    plt.plot(cell.tvec,cell.imem[i-1,:],'blue',label='imembrane')
	    plt.plot(cell.tvec,cell.ipas[i-1,:],'red',label='i_pas')
	    #plt.plot(cell.tvec,cell.icap[i-1,:],'green',label='i_cap')
            plt.plot(cell.tvec,cell.rec_variables['ina'][i-1,:],'black',label='i_na')
            plt.plot(cell.tvec,cell.rec_variables['ik'][i-1,:],'cyan',label='i_k')
	    
	    if i!=5:
		plt.text(45,0.06,'segment {}'.format(segs[i-1]))
		#plt.ylim(-0.02,0.11)
		
	    else:
		plt.text(45,-0.1,'segment {}'.format(segs[i-1]))
		ax.plot(cell.tvec, synapse.i,'m',label='i_syn')
		plt.legend(loc='lower right')
		plt.title('currents (nA)')
		#plt.ylim(-0.5,0.11)
	    if i==1:
		plt.xlabel('time (ms)')
	    plt.xlim(42.5,62.5)
	    
		
	#### Calculate extracellular potential by hand

	# electrode 5 is at 0,100
	#h = np.zeros((5,1))
	#l = np.zeros((5,1))
	#r = np.zeros((5,1))

	#for i in np.arange(0,5):
	#    h[i,0] = electrode.z[5]-cell.zend[i]
	#    l[i,0] = electrode.z[5]-cell.zstart[i]
	#    r[i,0] = electrode.x[5]-cell.xstart[i]

	#dist = np.log((np.sqrt(h**2+r**2)-h)/(np.sqrt(l**2+r**2)-l))
	#v_ext=1000*(1/(4*np.pi*300*5))*np.dot(np.transpose(cell.imem),dist)

	elecs_to_plot = [12,5,0]
	for i in np.arange(1,4):
	    ax=plt.subplot(3,3,3*i)
	    plt.plot(cell.tvec,electrode.LFP[elecs_to_plot[i-1],:],'k')
	    plt.xlim(42.5,62.5)
	    plt.text(45,0.000002,'electrode {}'.format(elecs_to_plot[i-1]))
	    ax.yaxis.tick_right()
	    if i==1:
		#plot(cell.tvec,v_ext,'k')
		plt.title('extracellular voltage (mV)')
	    if i==3:
		plt.xlabel('time(ms)')
	    plt.ylim(-0.001,0.001)	
