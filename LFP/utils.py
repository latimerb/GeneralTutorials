import matplotlib.pyplot as plt
import numpy as np
import LFPy 
import neuron
from matplotlib.collections import LineCollection

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

def plot_ex1(cell, electrode, X, Y, Z, time_show, space_lim):
    '''
    plot the morphology and LFP contours, synaptic current and soma trace
    '''
    #figure object
    fig = plt.figure(figsize=(5, 5))
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
            if sec.name()=="soma[0]":
                ax1.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
                    np.r_[cell.zstart[idx], cell.zend[idx][-1]],
                    color=[0.7,0.7,0.7],lw=6)
            else:
                ax1.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
                    np.r_[cell.zstart[idx], cell.zend[idx][-1]],
                    color=[0.7,0.7,0.7],lw=2)
        for i in range(len(cell.synapses)):
            ax1.plot([cell.synapses[i].x], [cell.synapses[i].z], '.',
                #color=cell.synapses[i].color, marker=cell.synapses[i].marker,
                markersize=10)
        
        #contour lines
        ct1 = ax1.contourf(X, Z, LFP, n_contours)
        ct1.set_clim((-0.00007, 0.00002))
        ct2 = ax1.contour(X, Z, LFP, n_contours_black, colors='k')

        # Figure formatting and labels

        #cbar = fig.colorbar(ct1)
        #cbar.set_label('Potential (uV)',rotation=270)
        #cbar.ax1.set_yticklabels([])
        ax1.set_title('LFP at t=' + str(t_show) + ' ms', fontsize=12)
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
        if sec.name()=="soma[0]":
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
    for LFP in electrode.LFP:
        tvec = cell.tvec*0.2 + electrode.x[i] + 2
        #print(np.max(LFP))    
        #if np.max(LFP)>0.018:
        #    factor = 1000
        #    color = 'r'
        #else:
        factor = 500000
        color = 'b'
 
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
