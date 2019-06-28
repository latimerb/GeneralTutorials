import matplotlib.pyplot as plt
import numpy as np
import LFPy 
import neuron
import scipy.signal as ss
import scipy.stats as st
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as colors

def Plot_geo_currs_volt(cell,electrode,synapse):

    plt.figure(figsize=(15, 10))
    plt.subplot(1,2,1)
    # Plot geometry of the cell
    for sec in LFPy.cell.neuron.h.allsec():
        idx = cell.get_idx(sec.name())
        plt.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
            np.r_[cell.zstart[idx], cell.zend[idx][-1]],
            'k',linewidth=8)
        #plt.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
            #np.r_[cell.zstart[idx], cell.zend[idx][-1]],
            #'r_')
        for i in np.arange(0,cell.xstart[idx].shape[0]):
            segs = [5,4,3,2,1]
            #plt.text(cell.xstart[idx][i]-75,cell.zend[idx][i]-50,'Seg. {}'.format(segs[i]))
            # Plot synapse as red dot
            plt.plot([cell.synapses[0].x], [cell.synapses[0].z], \
                 color='r', marker='o', markersize=5,label='synapse')
            # Plot electrodes as green dots
            plt.plot(electrode.x, electrode.z, '.', marker='o', color='g',label='electrodes')
        
            plt.xlabel('distance (um)')
            plt.ylabel('distance (um)')
        
            plt.ylim(-750,750)
            plt.xlim(-750,750)

        i=0
        for LFP in electrode.LFP:
            tvec = cell.tvec[cell.tvec>40]*4 + electrode.x[i] - 150
            #zscore = (LFP[cell.tvec>40]-np.mean(LFP[cell.tvec>40]))/np.std(LFP[cell.tvec>40])
            trace = 50000*LFP[cell.tvec>40]+electrode.z[i]

            plt.plot(tvec,trace,'k')
            plt.text(electrode.x[i],electrode.z[i],'{}'.format(i))
            i+=1
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
            
            #plt.legend()
        
            sp = [10,8,6,4,2]
            segs = [5,4,3,2,1]
        for i in np.arange(1,6):
            ax=plt.subplot(5,2,sp[i-1])
            plt.plot(cell.tvec,cell.imem[i-1,:],'blue',label='imembrane')
            plt.plot(cell.tvec,cell.ipas[i-1,:],'red',label='i_pas')
            plt.plot(cell.tvec,cell.icap[i-1,:],'green',label='i_cap')
            plt.xlim(45,75)

            if i!=5:
                plt.text(45,0.06,'segment {}'.format(segs[i-1]))
                plt.ylim(-0.5,0.7)
                
            else:
                plt.text(45,-0.1,'segment {}'.format(segs[i-1]))
                ax.plot(cell.tvec, synapse.i,'m',label='i_syn')
                plt.legend(loc='lower right')
                plt.title('currents (nA)')
                #plt.ylim(-0.6,0.2)
            if i==1:
                plt.xlabel('time (ms)')
                
            
            
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
    
        


def Plot_ball_and_stick(cell,electrode,synapse):
    plt.figure(figsize=(15, 10))
    plt.subplot(1,2,1)
    # Plot geometry of the cell
    for sec in LFPy.cell.neuron.h.allsec():
        idx = cell.get_idx(sec.name())
        print(sec.name())
        if sec.name()=="axon[0]":
            lw = 2
        if sec.name()=="soma[0]":
            lw = 20
        if sec.name()=="dend[0]":
            lw = 7
        plt.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
            np.r_[cell.zstart[idx], cell.zend[idx][-1]],
            'k',linewidth=lw)
       
      
        # Plot synapse as red dot
        plt.plot([cell.synapses[0].x], [cell.synapses[0].z], \
                 color='r', marker='o', markersize=5,label='synapse')
        # Plot electrodes as green dots
        plt.plot(electrode.x, electrode.z, '.', marker='o', color='g',label='electrodes')
        
        # Plot LFP
        i=0
        for LFP in electrode.LFP:
            tvec = cell.tvec[cell.tvec>40]*4 + electrode.x[i] - 150
            #zscore = (LFP[cell.tvec>40]-np.mean(LFP[cell.tvec>40]))/np.std(LFP[cell.tvec>40])
            trace = 50000*LFP[cell.tvec>40]+electrode.z[i]

            plt.plot(tvec,trace,'k')
            i+=1

        # Labels and lims
        plt.xlabel('distance (um)')
        plt.ylabel('distance (um)')
        
        plt.ylim(-750,750)
        plt.xlim(-750,750)

        # Plot currents
        #sp = [10,8,6,4,2]
        #segs = [5,4,3,2,1]
        #for i in np.arange(1,6):
            #ax=plt.subplot(5,2,sp[i-1])
            #plt.plot(cell.tvec,cell.imem[i-1,:],'blue',label='imembrane')
            #plt.plot(cell.tvec,cell.ipas[i-1,:],'red',label='i_pas')
            #plt.plot(cell.tvec,cell.icap[i-1,:],'green',label='i_cap')
            #plt.xlim(45,75)

            #if i!=5:
                #plt.text(45,0.06,'segment {}'.format(segs[i-1]))
                #plt.ylim(-0.5,0.7)
                
            #else:
                #plt.text(45,-0.1,'segment {}'.format(segs[i-1]))
                #ax.plot(cell.tvec, synapse.i,'m',label='i_syn')
                #plt.legend(loc='lower right')
                #plt.title('currents (nA)')
                #plt.ylim(-0.6,0.2)
            #if i==1:
                #plt.xlabel('time (ms)')



def plot_ex1(cell, electrode, X, Y, Z, time_show, space_lim):
    '''
    plot the morphology and LFP contours, synaptic current and soma trace
    '''
    #figure object
    fig = plt.figure(figsize=(2*len(time_show), 5))
    fig.subplots_adjust(left=None, bottom=None, right=0.8, top=None, 
            wspace=0.1, hspace=0.1)
    ax0 = fig.add_subplot(2,1,1)
    ax0.plot(cell.tvec,cell.synapses[0].i)
    ax0.plot([time_show[0]+1,time_show[0]+1],[cell.synapses[0].i[0],cell.synapses[0].i[0]+0.01],'k-')
    ax0.text(time_show[0]+1+0.01,cell.synapses[0].i[0]+0.004,'10 pA')

    ax0.set_xlim(49-0.4,63+0.4)
    ax0.set_ylim(np.min(cell.synapses[0].i)-0.01,np.max(cell.synapses[0].i)+0.01)
    ax0.set_xticks([])
    ax0.set_xticklabels([])
    ax0.set_yticks([])
    ax0.set_yticklabels([])

    ct = []
    for i in np.arange(0,len(time_show)):
        ax0.plot([time_show[i],time_show[i]],[-1,1],'k--')
        #some plot parameters
        t_show = time_show[i] #time point to show LFP
        tidx = np.where(cell.tvec == t_show)
        #contour lines:
        n_contours = 200
        n_contours_black = 0
        
        #This is the extracellular potential, reshaped to the X, Z mesh
        LFP = np.arcsinh(electrode.LFP[:, tidx]).reshape(X.shape)
        
        # Plot LFP around the cell with in color and with equipotential lines
        ax1 = fig.add_subplot(2,len(time_show),i+len(time_show)+1)
        
        #plot_morphology(plot_synapses=True)
        for sec in LFPy.cell.neuron.h.allsec():
           
            idx = cell.get_idx(sec.name())
            if sec.name()=="stylized_pyrtypeC[0].soma[0]":
                ax1.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
                    np.r_[cell.zstart[idx], cell.zend[idx][-1]],
                    color=[0.7,0.7,0.7],lw=6)
            else:
                ax1.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
                    np.r_[cell.zstart[idx], cell.zend[idx][-1]],
                    color=[0.7,0.7,0.7],lw=2)
        for j in range(len(cell.synapses)):
            ax1.plot([cell.synapses[j].x], [cell.synapses[j].z], '.',
                #color=cell.synapses[j].color, marker=cell.synapses[j].marker,
                markersize=10)
        
        #contour lines
        ct.append(ax1.contourf(X, Z, LFP*1000, n_contours,vmin=-0.00007*1000,vmax=0.00002*1000))
        #np.savetxt('./data/excitatory_LFP_at_{}.csv'.format(t_show),LFP*1000,delimiter=',')
        ct[i].set_clim((-0.00007*1000, 0.00002*1000))
        ct2 = ax1.contour(X, Z, LFP*1000, n_contours_black, colors='k')

        # Figure formatting and labels

        #ax1.set_title('LFP at t=' + str(t_show) + ' ms', fontsize=12)
        ax1.set_xticks([])
        ax1.set_xticklabels([])
        ax1.set_yticks([])
        ax1.set_yticklabels([])
        x1 = space_lim[0]; x2 = space_lim[1]
        y1 = space_lim[2]; y2 = space_lim[3]
        ax1.set_xlim(x1,x2)
        ax1.set_ylim(y1,y2)
        #ax1.set_ylim(-500,100)
        if i==0:
            ax1.plot([x1+70, x1+70], [y2-200, y2-100], color='k', lw=2)
            #ax1.text(x1+1, y2-150, '100 um')
            ax1.plot([x1+70, x1+170], [y2-200,y2-200], color='k', lw=2)
            #ax1.text(x1+80, y2-230, '100 um')
    
    #[right/left,top/bottom,width,height]
    cbaxes = fig.add_axes([0.82,0.11,0.03,0.36])
    cbar = plt.colorbar(ct[1],cax=cbaxes)
    #cbar.set_label('Potential (uV)',rotation=270)
    #fig2 = plt.figure(figsize=(15, 6))
    # Plot synaptic input current
    #ax2 = fig2.add_subplot(121)
    #ax2.plot(cell.tvec, cell.synapses[0].i)
        
    # Plot soma potential
    #ax3 = fig2.add_subplot(122)
    #ax3.plot(cell.tvec, cell.somav)

    #ax2.set_title('synaptic input current', fontsize=12)
    #ax2.set_ylabel('(nA)')
    #ax2.set_xlabel('time (ms)')
    #ax2.set_xlim(40,65)

    #ax3.set_title('somatic membrane potential', fontsize=12)
    #ax3.set_ylabel('(mV)')
    #ax3.set_xlabel('time (ms)')
    #ax3.set_xlim(40,65)
    
    return fig

def plot_ex2(cell, electrode, X, Y, Z, time_show):
    '''
    plot the morphology and LFP contours, synaptic current and soma trace
    '''
    #figure object
    fig = plt.figure(figsize=(15, 7))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, 
            wspace=0.2, hspace=0.2)
    for i in np.arange(0,len(time_show)):
        #some plot parameters
        t_show = time_show[i] #time point to show LFP
        tidx = np.where(cell.tvec == t_show)
        #contour lines:
        n_contours = 200
        n_contours_black = 0
        
        #This is the extracellular potential, reshaped to the X, Z mesh
        LFP = np.arcsinh(electrode.LFP[:, tidx]).reshape(X.shape)
        
        # Plot LFP around the cell with in color and with equipotential lines
        ax1 = fig.add_subplot(1,len(time_show),i+1)
        
        #plot_morphology(plot_synapses=True)
        for sec in LFPy.cell.neuron.h.allsec():
            idx = cell.get_idx(sec.name())
            ax1.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
                    np.r_[cell.zstart[idx], cell.zend[idx][-1]],
                    color='k')
        for j in range(len(cell.synapses)):
            ax1.plot([cell.synapses[j].x], [cell.synapses[j].z], '.',
                #color=cell.synapses[j].color, marker=cell.synapses[j].marker, 
                markersize=10)
        
        #contour lines
        ct1 = ax1.contourf(X, Z, LFP, n_contours)
        ct1.set_clim((-0.00007, 0.00002))
        ct2 = ax1.contour(X, Z, LFP, n_contours_black, colors='k')
        

        # Figure formatting and labels
        
        ax1.set_title('LFP at t=' + str(t_show) + ' ms', fontsize=12)
        ax1.set_xticks([])
        ax1.set_xticklabels([])
        ax1.set_yticks([])
        ax1.set_yticklabels([])
        ax1.set_xlim(-900,400)
        #ax1.set_ylim(-400,300)
        if i==0:
            ax1.plot([-600, -600], [-600, -400], color='k', lw=2)
            ax1.text(-580, -500, '200 um')

        
        
    fig2 = plt.figure(figsize=(15, 6))
    # Plot synaptic input current
    ax2 = fig2.add_subplot(223)
    ax2.plot(cell.tvec, cell.synapses[0].i)
        
    # Plot soma potential
    ax3 = fig2.add_subplot(224)
    ax3.plot(cell.tvec, cell.somav)

    ax2.set_title('synaptic input current', fontsize=12)
    ax2.set_ylabel('(nA)')
    ax2.set_xlabel('time (ms)')

    ax3.set_title('somatic membrane potential', fontsize=12)
    ax3.set_ylabel('(mV)')
    ax3.set_xlabel('time (ms)')
    
    return fig


def plot_elec_grid(cell, electrode):
    '''example2.py plotting function'''
    fig = plt.figure(figsize=[15, 15])
    ax = fig.add_axes([0.1, 0.1, 0.533334, 0.8], frameon=False)
    # Plot geometry of the cell
    for sec in LFPy.cell.neuron.h.allsec():
        idx = cell.get_idx(sec.name())
        if sec.name()=="stylized_pyrtypeC[0].soma[0]":
            ax.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
                np.r_[cell.zstart[idx], cell.zend[idx][-1]],
                'k',linewidth=8)
        else:
            ax.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
                np.r_[cell.zstart[idx], cell.zend[idx][-1]],
                'k',linewidth=2)


    # Plot synapses as red dots

    ax.plot([cell.synapses[0].x], [cell.synapses[0].z], color=[1,0,0], marker='o', markersize=5)
    #creating array of points and corresponding diameters along structure
    #for i in range(cell.xend.size):
    #    if i == 0:
    #        xcoords = np.array([cell.xmid[i]])
    #       ycoords = np.array([cell.ymid[i]])
    #        zcoords = np.array([cell.zmid[i]])
    #        diams = np.array([cell.diam[i]])    
    #    else:
    #       # if cell.zmid[i] < 100 and cell.zmid[i] > -100 and \
    #               # cell.xmid[i] < 100 and cell.xmid[i] > -100:
    #        xcoords = np.r_[xcoords, np.linspace(cell.xstart[i],
    #                                        cell.xend[i], cell.length[i]*3)]   
    #        ycoords = np.r_[ycoords, np.linspace(cell.ystart[i],
    #                                        cell.yend[i], cell.length[i]*3)]   
    #        zcoords = np.r_[zcoords, np.linspace(cell.zstart[i],
    #                                        cell.zend[i], cell.length[i]*3)]   
    #        diams = np.r_[diams, np.linspace(cell.diam[i], cell.diam[i],
    #                                        cell.length[i]*3)]
    
    #sort along depth-axis
    #argsort = np.argsort(ycoords)
    #plotting
    
    #ax.scatter(xcoords[argsort], zcoords[argsort], s=diams[argsort]**2*20,
     #          c=ycoords[argsort], edgecolors='none', cmap='gray')
    ax.plot(electrode.x, electrode.z, '.', marker='o', markersize=5, color='k')
    
    i = 0
    limLFP = abs(electrode.LFP).max()
    #print("absolute max:",limLFP)
    for LFP in electrode.LFP:
        tvec = cell.tvec*0.6 + electrode.x[i] + 2
        #print("current max:",np.max(np.abs(LFP)))
        if np.max(np.abs(LFP))>0.25*limLFP:
            factor = 400
            color = 'r'
        elif np.max(np.abs(LFP))>0.04*limLFP:
            factor = 3000
            color = 'b'
        else:
            factor = 10000
            color = 'g'

        trace = LFP*factor + electrode.z[i]

        ax.plot(tvec,trace, color=color, lw = 2)
        i += 1
    
    #ax.plot([22, 28], [-60, -60], color='k', lw = 3)
    #ax.text(22, -65, '10 ms')
    
    #ax.plot([40, 50], [-60, -60], color='k', lw = 3)
    #ax.text(42, -65, '10 $\mu$m')
    
    #ax.plot([60, 60], [20, 30], color='r', lw=2)
    #ax.text(62, 20, '5 mV')
    
    #ax.plot([60, 60], [0, 10], color='g', lw=2)
    #ax.text(62, 0, '1 mV')
    
    #ax.plot([60, 60], [-20, -10], color='b', lw=2)
    #ax.text(62, -20, '0.1 mV')
    
    
    ax.axis([-500, 150, -350, 650])

    # scale bar - horizontal
    ax.plot([-425,-375],[300,300],'k')
    ax.text(-420,275,'50 um')

    # scale bar - vertical
    ax.plot([-425,-425],[300,350],'k')
    ax.text(-475,325,'50 um')

    # scale bar - voltage vertical
    ax.plot([-100,-100],[300,350],'b')
    ax.text(-135,320,'20 $\mu$V')

    # scale bar - voltage horizontal
    ax.plot([-100,-50],[300,300],'b')
    ax.text(-95,280,'5 ms')

    ax.set_title('Location-dependent extracellular spike shapes')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return fig

def plot_elec_grid2(cell, electrode):
    '''plotting'''
    fig = plt.figure(dpi=160)
    
    ax1 = fig.add_axes([0.05, 0.1, 0.55, 0.9], frameon=False)
    cax = fig.add_axes([0.05, 0.115, 0.55, 0.015])
    
    ax1.plot(electrode.x, electrode.z, '.', marker='o', markersize=1, color='k',
             zorder=0)
    
    #normalize to min peak
    LFPmin = electrode.LFP.min(axis=1)
    LFPnorm = -(electrode.LFP.T / LFPmin).T
    
    i = 0
    zips = []
    for x in LFPnorm:
        zips.append(list(zip(cell.tvec*1.6 + electrode.x[i] + 2,
                        x*12 + electrode.z[i])))
        i += 1
    
    line_segments = LineCollection(zips,
                                    linewidths = (1),
                                    linestyles = 'solid',
                                    cmap='nipy_spectral',
                                    zorder=1,
                                    rasterized=False)
    line_segments.set_array(np.log10(-LFPmin))
    ax1.add_collection(line_segments)
    
    axcb = fig.colorbar(line_segments, cax=cax, orientation='horizontal')
    axcb.outline.set_visible(False)
    xticklabels = np.array([-0.1  , -0.05 , -0.02 , -0.01 , -0.005, -0.002])
    xticks = np.log10(-xticklabels)
    axcb.set_ticks(xticks)
    axcb.set_ticklabels(np.round(-10**xticks, decimals=3))  
    axcb.set_label('spike amplitude (mV)', va='center')
    
    ax1.plot([22, 38], [100, 100], color='k', lw = 1)
    ax1.text(22, 102, '10 ms')
    
    ax1.plot([60, 80], [100, 100], color='k', lw = 1)
    ax1.text(60, 102, '20 $\mu$m')
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    axis = ax1.axis(ax1.axis('equal'))
    ax1.set_xlim(axis[0]*1.02, axis[1]*1.02)
    
    # plot morphology
    zips = []
    for x, z in cell.get_pt3d_polygons():
        zips.append(list(zip(x, z)))
    from matplotlib.collections import PolyCollection
    polycol = PolyCollection(zips, edgecolors='none',
                             facecolors='gray', zorder=-1, rasterized=False)
    ax1.add_collection(polycol)

    ax1.text(-0.05, 0.95, 'a',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='demibold',
        transform=ax1.transAxes)
    

    # plot extracellular spike in detail
    ind = np.where(electrode.LFP == electrode.LFP.min())[0][0]
    timeind = (cell.tvec >= 0) & (cell.tvec <= 10)
    xticks = np.arange(10)
    xticklabels = xticks
    LFPtrace = electrode.LFP[ind, ]
    vline0 = cell.tvec[cell.somav==cell.somav.max()]
    vline1 = cell.tvec[LFPtrace == LFPtrace.min()]
    vline2 = cell.tvec[LFPtrace == LFPtrace.max()]
    
    # plot asterix to link trace in (a) and (c)
    ax1.plot(electrode.x[ind], electrode.z[ind], '*', markersize=5, 
             markeredgecolor='none', markerfacecolor='k')
    
    ax2 = fig.add_axes([0.75, 0.6, 0.2, 0.35], frameon=True)
    ax2.plot(cell.tvec[timeind], cell.somav[timeind], lw=1, color='k', clip_on=False)
    
    ax2.vlines(vline0, cell.somav.min(), cell.somav.max(), 'k', 'dashed', lw=0.25)
    ax2.vlines(vline1, cell.somav.min(), cell.somav.max(), 'k', 'dashdot', lw=0.25)
    ax2.vlines(vline2, cell.somav.min(), cell.somav.max(), 'k', 'dotted', lw=0.25)
    
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticks)
    ax2.axis(ax2.axis('tight'))
    ax2.set_ylabel(r'$V_\mathrm{soma}(t)$ (mV)')
    
    for loc, spine in ax2.spines.items():
        if loc in ['right', 'top']:
            spine.set_color('none')            
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    
    ax2.set_title('somatic potential', va='center')

    ax2.text(-0.3, 1.0, 'b',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='demibold',
        transform=ax2.transAxes)

    ax3 = fig.add_axes([0.75, 0.1, 0.2, 0.35], frameon=True)
    ax3.plot(cell.tvec[timeind], LFPtrace[timeind], lw=1, color='k', clip_on=False)
    ax3.plot(0.5, 0, '*', markersize=5, markeredgecolor='none', markerfacecolor='k')

    ax3.vlines(vline0, LFPtrace.min(), LFPtrace.max(), 'k', 'dashed', lw=0.25)
    ax3.vlines(vline1, LFPtrace.min(), LFPtrace.max(), 'k', 'dashdot', lw=0.25)
    ax3.vlines(vline2, LFPtrace.min(), LFPtrace.max(), 'k', 'dotted', lw=0.25)

    ax3.set_xticks(xticks)
    ax3.set_xticklabels(xticks)
    ax3.axis(ax3.axis('tight'))
    
    for loc, spine in ax3.spines.items():
        if loc in ['right', 'top']:
            spine.set_color('none')            
    ax3.xaxis.set_ticks_position('bottom')
    ax3.yaxis.set_ticks_position('left')

    ax3.set_xlabel(r'$t$ (ms)', va='center')
    ax3.set_ylabel(r'$\Phi(\mathbf{r},t)$ (mV)')
                   
    ax3.set_title('extracellular spike', va='center')

    ax3.text(-0.3, 1.0, 'c',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='demibold',
        transform=ax3.transAxes)

    return fig

def plot_vmem_heat(cell,time_show):
    tidx = np.where(cell.tvec == time_show)[0][0]
    print("tidx:",tidx)
    fig = plt.figure(figsize=(10,10))
    i = 0
    for sec in LFPy.cell.neuron.h.allsec():
        idx = cell.get_idx(sec.name())
        for seg in sec:
            c = cell.vmem[i,tidx] - np.mean(cell.vmem[i,:])
            # Plot each segment in color corresponding to voltage
            plt.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
                np.r_[cell.zstart[idx], cell.zend[idx][-1]],
                color=[0,0,c/1.8],linewidth=8)
            i+=1
    #cell.vmem[i,800]
    return fig

def plot_elec_grid_stick(cell, electrode):
    '''example2.py plotting function'''
    fig = plt.figure(figsize=[15, 15])
    ax = fig.add_axes([0.1, 0.1, 0.533334, 0.8], frameon=False)
    # Plot geometry of the cell
    for sec in LFPy.cell.neuron.h.allsec():
        idx = cell.get_idx(sec.name())
        if sec.name()=="stylized_pyrtypeC[0].soma[0]":
            ax.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
                np.r_[cell.zstart[idx], cell.zend[idx][-1]],
                color=[0.7,0.7,0.7],linewidth=8)
        else:
            ax.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
                np.r_[cell.zstart[idx], cell.zend[idx][-1]],
                color=[0.7,0.7,0.7],linewidth=2)


    # Plot synapses as red dots

    ax.plot([cell.synapses[0].x], [cell.synapses[0].z], color=[1,0,0], marker='o', markersize=5)
    #creating array of points and corresponding diameters along structure
    #for i in range(cell.xend.size):
    #    if i == 0:
    #        xcoords = np.array([cell.xmid[i]])
    #       ycoords = np.array([cell.ymid[i]])
    #        zcoords = np.array([cell.zmid[i]])
    #        diams = np.array([cell.diam[i]])    
    #    else:
    #       # if cell.zmid[i] < 100 and cell.zmid[i] > -100 and \
    #               # cell.xmid[i] < 100 and cell.xmid[i] > -100:
    #        xcoords = np.r_[xcoords, np.linspace(cell.xstart[i],
    #                                        cell.xend[i], cell.length[i]*3)]   
    #        ycoords = np.r_[ycoords, np.linspace(cell.ystart[i],
    #                                        cell.yend[i], cell.length[i]*3)]   
    #        zcoords = np.r_[zcoords, np.linspace(cell.zstart[i],
    #                                        cell.zend[i], cell.length[i]*3)]   
    #        diams = np.r_[diams, np.linspace(cell.diam[i], cell.diam[i],
    #                                        cell.length[i]*3)]
    
    #sort along depth-axis
    #argsort = np.argsort(ycoords)
    #plotting
    
    #ax.scatter(xcoords[argsort], zcoords[argsort], s=diams[argsort]**2*20,
     #          c=ycoords[argsort], edgecolors='none', cmap='gray')
    #ax.plot(electrode.x, electrode.z, '.', marker='o', markersize=5, color='k')
    
    
    limLFP =  abs(electrode.LFP).max() # 0.010196
    max_factor = 60 # converts to scale of microns
    # Define spectrum of colors
    colors = np.zeros((120,3))
    for i in np.arange(0,120):
        if i<20 and i>=0:
            colors[i,:] = [150 + 5.5*i,0,0] #Dark red to red
        elif i<40 and i>=20:
            colors[i,:] = [255,(i-19)*6,0] # Red to orange
        elif i<60 and i>=40:
            colors[i,:] = [255,120+(i-39)*6.75,0] # Orange to yellow
        elif i<80 and i>=60:
            colors[i,:] = [255-(i-59)*12.5,255,0] # yellow to green
        elif i<100 and i>=80:
            colors[i,:] = [0,255,(i-79)*12.5] # green to blue-green
        else:
            colors[i,:] = [0,255-(i-99)*12.5,255] # blue-green to blue
        # Plot colorbar
        rectangle = plt.Rectangle((6.5*i-343, 720), 10, 100, fc=colors[i,:]/255)
        plt.gca().add_patch(rectangle)        
        #ax.plot([i,i,i+10,i+10],[720,720,770,770],color=colors[i,:]/255,lw = 10)
    #print("colors: ",colors)
    plt.text(-350,845,"10")
    plt.text(30,845,"1")
    plt.text(410,845,"0.1")
    # time scale bar
    plt.plot([-280,-320],[100,100],'k')
    plt.text(-320,80,'5 ms')
    i = 0
    smallest = 4.85#4.08 #4.85
    biggest = 9.67#7.87 # 9.67
    #for LFP in electrode.LFP:
        #print("!!!",-np.log(np.max(LFP[cell.tvec>50])-np.min(LFP[cell.tvec>50])))
    #    if -np.log(np.max(LFP[cell.tvec>50])-np.min(LFP[cell.tvec>50]))<smallest:
    #        smallest = -np.log(np.max(LFP[cell.tvec>50])-np.min(LFP[cell.tvec>50]))
    #    if -np.log(np.max(LFP[cell.tvec>50])-np.min(LFP[cell.tvec>50]))>biggest:
    #        biggest = -np.log(np.max(LFP[cell.tvec>50])-np.min(LFP[cell.tvec>50]))
    #print("SMALLEST: ",smallest)
    #print("BIGGEST: ",biggest)

    for LFP in electrode.LFP:
        tvec = cell.tvec[cell.tvec>50]*9 + electrode.x[i] + -450
        
        #print("amp. of peak:",np.max(np.abs(LFP)))
        # assign color        
        x = (-np.log(np.max(LFP[cell.tvec>50])-np.min(LFP[cell.tvec>50])))
        factor = max_factor*(x-smallest)/(biggest-smallest)
        #print("FACTOR:",factor)
        #print("electrode number {}".format(i))
        #print("colors.shape[0]*factor/max_factor", int(np.floor(colors.shape[0]*factor/max_factor)))
        if int(np.floor((colors.shape[0]-1)*factor/max_factor))<=0:
            color = colors[0,:]
        else:
            if int(np.floor((colors.shape[0]-1)*factor/max_factor))>=colors.shape[0]:
                color = colors[-1,:]
            else:
                color = colors[int(np.floor((colors.shape[0]-1)*factor/max_factor)),:]

        #print("electrode {} color: {}".format(i,color/255))
        
        # Take zscore and adjust y axis        
        zscore = LFP[cell.tvec>50]/(np.max(LFP[cell.tvec>50])-np.min(LFP[cell.tvec>50]))
        trace = zscore*max_factor + electrode.z[i]

        ax.plot(tvec,trace, color=color/255, lw = 2)
        #ax.text(tvec[0],trace[0],i)
        i += 1
    #print("smallest: ",smallest)
    
    #ax.plot([22, 28], [-60, -60], color='k', lw = 3)
    #ax.text(22, -65, '10 ms')
    
    #ax.plot([40, 50], [-60, -60], color='k', lw = 3)
    #ax.text(42, -65, '10 $\mu$m')
    
    #ax.plot([60, 60], [20, 30], color='r', lw=2)
    #ax.text(62, 20, '5 mV')
    
    #ax.plot([60, 60], [0, 10], color='g', lw=2)
    #ax.text(62, 0, '1 mV')
    
    #ax.plot([60, 60], [-20, -10], color='b', lw=2)
    #ax.text(62, -20, '0.1 mV')
    
    
    #ax.axis([-500, 150, -350, 650])

    # scale bar - horizontal
    #ax.plot([-425,-375],[300,300],'k')
    #ax.text(-420,275,'50 um')

    # scale bar - vertical
    #ax.plot([-425,-425],[300,350],'k')
    #ax.text(-475,325,'50 um')

    # scale bar - voltage vertical
    #ax.plot([-100,-100],[300,350],'b')
    #ax.text(-135,320,'20 $\mu$V')

    # scale bar - voltage horizontal
    #ax.plot([-100,-50],[300,300],'b')
    #ax.text(-95,280,'5 ms')

    #ax.set_title('Location-dependent extracellular spike shapes')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title('uV')

    return fig

def decimate(x, q=10, n=4, k=0.8, filterfun=ss.cheby1):
    """
    scipy.signal.decimate like downsampling using filtfilt instead of lfilter,
    and filter coeffs from butterworth or chebyshev type 1.

    Parameters
    ----------
    x : ndarray
        Array to be downsampled along last axis.
    q : int
        Downsampling factor.
    n : int
        Filter order.
    k : float
        Aliasing filter critical frequency Wn will be set as Wn=k/q.
    filterfun : function
        `scipy.signal.filter_design.cheby1` or
        `scipy.signal.filter_design.butter` function

    Returns
    -------
    ndarray
        Downsampled signal.

    """
    if not isinstance(q, int):
        raise TypeError("q must be an integer")

    if n is None:
        n = 1

    if filterfun == ss.butter:
        b, a = filterfun(n, k / q)
    elif filterfun == ss.cheby1:
        b, a = filterfun(n, 0.05, k / q)
    else:
        raise Exception('only ss.butter or ss.cheby1 supported')

    try:
        y = ss.filtfilt(b, a, x)
    except: # Multidim array can only be processed at once for scipy >= 0.9.0
        y = []
        for data in x:
            y.append(ss.filtfilt(b, a, data))
        y = np.array(y)

    try:
        return y[:, ::q]
    except:
        return y[::q]

def remove_axis_junk(ax, lines=['right', 'top']):
    """remove chosen lines from plotting axis"""
    for loc, spine in ax.spines.items():
        if loc in lines:
            spine.set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def draw_lineplot(
        ax, data, dt=0.1,
        T=(0, 200),
        scaling_factor=1.,
        vlimround=None,
        label='local',
        scalebar=True,
        unit='mV',
        ylabels=True,
        color='r',
        ztransform=True,
        filter=False,
        filterargs=dict(N=2, Wn=0.02, btype='lowpass')):
    """helper function to draw line plots"""
    tvec = np.arange(data.shape[1])*dt
    tinds = (tvec >= T[0]) & (tvec <= T[1])

    # apply temporal filter
    if filter:
        b, a = ss.butter(**filterargs)
        data = ss.filtfilt(b, a, data, axis=-1)

    #subtract mean in each channel
    if ztransform:
        dataT = data.T - data.mean(axis=1)
        data = dataT.T

    zvec = -np.arange(data.shape[0])
    vlim = abs(data[:, tinds]).max()
    if vlimround is None:
        vlimround = 2.**np.round(np.log2(vlim)) / scaling_factor
    else:
        pass

    yticklabels=[]
    yticks = []

    for i, z in enumerate(zvec):
        if i == 0:
            ax.plot(tvec[tinds], data[i][tinds] / vlimround + z, lw=1,
                    rasterized=False, label=label, clip_on=False,
                    color=color)
        else:
            ax.plot(tvec[tinds], data[i][tinds] / vlimround + z, lw=1,
                    rasterized=False, clip_on=False,
                    color=color)
        yticklabels.append('ch. %i' % (i+1))
        yticks.append(z)

    if scalebar:
        ax.plot([tvec[-1], tvec[-1]],
                [-1, -2], lw=2, color='k', clip_on=False)
        ax.text(tvec[-1]+np.diff(T)*0.02, -1.5,
                '$2^{' + '{}'.format(np.log2(vlimround)) + '}$ ' + '{0}'.format(unit),
                color='k', rotation='vertical',
                va='center')

    ax.axis(ax.axis('tight'))
    ax.yaxis.set_ticks(yticks)
    if ylabels:
        ax.yaxis.set_ticklabels(yticklabels)
        ax.set_ylabel('channel', labelpad=0.1)
    else:
        ax.yaxis.set_ticklabels([])
    remove_axis_junk(ax, lines=['right', 'top'])
    ax.set_xlabel(r't (ms)', labelpad=0.1)

    return vlimround

import pdb

def contour3d(cell,electrode,X,Y,Z,tshow):
    
    tidx = cell.tvec==tshow

#    for sec in LFPy.cell.neuron.h.allsec():
#        idx = cell.get_idx(sec.name())
#        if sec.name()=="stylized_pyrtypeC[0].soma[0]":
#            ax.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
#                np.r_[cell.ystart[idx], cell.yend[idx][-1]],
#                np.r_[cell.zstart[idx], cell.zend[idx][-1]],
#                color=[0.7,0.7,0.7],linewidth=8)
#        else:
#            ax.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
#                np.r_[cell.ystart[idx], cell.yend[idx][-1]],
#                np.r_[cell.zstart[idx], cell.zend[idx][-1]],
#                color=[0.7,0.7,0.7],linewidth=2)
    #ax = fig.add_subplot(111)
    #data = np.concatenate((electrode.x.reshape(-1,1),
    #                        electrode.y.reshape(-1,1),
    #                        electrode.z.reshape(-1,1),
    #                        electrode.LFP[:,tidx]),axis=1)

    x_mesh, z_mesh = np.meshgrid(electrode.x,electrode.z)
    LFP_mesh = np.zeros((Z[0,0,:].shape[0],x_mesh.shape[0],x_mesh.shape[1]))
    pdb.set_trace()
    for k in np.arange(0,Z[0,0,:].shape[0]):
        for i in np.arange(0,x_mesh.shape[0]):
            for j in np.arange(0,x_mesh.shape[1]):
                LFP_mesh[k,i,j] = electrode.LFP[np.nonzero((electrode.x==x_mesh[i,j]) 
                                                & (electrode.z==z_mesh[i,j]) 
                                                & (electrode.y==Y[0,0,k]))[0][0],tidx]
    pdb.set_trace() 
    for i in np.arange(0,Z[0,0,:].shape[0]):
        fig = plt.figure(figsize=[15, 15])
        ax = fig.gca(projection='3d')
        ax.contourf(x_mesh,z_mesh,LFP_mesh[:,0,:],
                    offset=Y[0,0,i],
                    zdir='y',
                    cmap = cm.coolwarm)
        #sp[i].set_clim(np.min(LFP_mesh),np.max(LFP_mesh))
        ax.set_zlim(np.min(Z[0,0,:]),np.max(Z[0,0,:]))
         
    #sp = ax.scatter(data[:,0],
    #                data[:,1],
    #                data[:,2],
    #                s=20, c=data[:,3], 
    #                norm=MidpointNormalize(midpoint=0.),
    #                cmap='RdBu_r')
    #plt.colorbar(sp[0])
    #plt.show()
    return

def zerosurf(cell,electrode,X,Y,Z,tshow):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    tidx=cell.tvec==tshow
    LFP = electrode.LFP[:,tidx]

    # Make 2D array for LFP for plotting surfaces and contours
    x_mesh, y_mesh = np.meshgrid(electrode.x,electrode.y)
    LFP_mesh = np.zeros((Z[0,0,:].shape[0],x_mesh.shape[0],x_mesh.shape[1]))

    for k in np.arange(0,Z[0,0,:].shape[0]):
        for i in np.arange(0,x_mesh.shape[0]):
            for j in np.arange(0,x_mesh.shape[1]):
                LFP_mesh[k,i,j] = electrode.LFP[np.nonzero((electrode.x==x_mesh[i,j]) 
                                                & (electrode.y==y_mesh[i,j]) 
                                                & (electrode.z==Z[0,0,k]))[0][0],tidx]
    
    # Plot zero line
        pdb.set_trace()
        ax.contour(x_mesh,y_mesh,LFP_mesh[k,:,:],offset=Z[0,0,k],levels=10,zdir='z')
    ax.set_zlim(np.min(Z[0,0,:]),np.max(Z[0,0,:]))
    return



