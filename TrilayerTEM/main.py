
# I used this to access the source code from pyInteferometry. 
# Can download it and put in path like this (hacky) or instead install as module, etc. 
import sys
sys.path.insert(1, '../../pyInt-reorg/src')  

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
from scipy.ndimage import gaussian_filter
from visualization import overlay_vdf
from diskset import DiskSet
from matplotlib.patches import RegularPolygon
from matplotlib.path import Path
from masking import make_contour_mask

def make_vdf_from_mask_and_dataset(filepath, maskpath, vdfpath):

	with open(filepath, 'rb') as f: avgdp = pickle.load(f)
	with open(maskpath, 'rb') as f: mask  = pickle.load(f)

	nqx, nqy = avgdp.shape[0], avgdp.shape[1]
	nx, ny   = 1, 1
	data = np.zeros((nx, ny, nqx, nqy))
	data[0,0,:,:] = avgdp[:,:]
	nregions = np.max(mask.flatten())+1 
	vdfs = np.zeros((nregions, nx, ny))

	for i in range(0, nregions):
		region_mask = (mask == i)
		vdfs[i,:,:] = np.sum(data*region_mask, axis=(2,3))/np.sum(region_mask)
		checkme = False
		if checkme:
			f, ax = plt.subplots(2,2)
			ax[0,0].imshow(region_mask)
			ax[1,0].imshow(vdfs[i,:,:])
			ax[1,1].imshow((data*region_mask)[0,0,:,:])
			ax[0,1].imshow(data[0,0,:,:])
			plt.show(); exit()

	N = nregions + 1
	Nx, Ny = N//3 + 1, 3
	f, ax = plt.subplots(Nx, Ny)
	ax = ax.flatten()
	ax[0].imshow(mask)
	for i in range(nregions):
		ax[i+1].imshow(vdfs[i,:,:], vmin=0, vmax=10)
		ax[i+1].set_title(i)
	plt.show()

	with open(vdfpath, 'wb') as f: pickle.dump(vdfs, f)

def make_linecuts_coloredmaps():

	# changes between the three color map conventions
	#rgbbasisvector1 = [255, 127.5, 0] 
	rgbbasisvector1 = [155, 55, 255] 
	#rgbbasisvector1 = [255, 50, 150] 

    filepath = '/Users/isaaccraig/Desktop/TLGproj/stem-data/{}/dat_ds{}.pkl'.format('ABt-nd1', 8) #c7, 2
    rgbbasisvector1 = [155, 55, 255]
    rgbbasisvector2 = [255-el for el in rgbbasisvector1]
    avgring1, avgring2 = load_3disk_vdf(filepath)
    mat = np.zeros((avgring1.shape[0], avgring1.shape[1], 3))
    for n in range(3):
        mat[:,:,n] = rgbbasisvector1[n]/255 * avgring1 + rgbbasisvector2[n]/255 * avgring2
    ax[1].imshow(mat, origin='lower')
    ax[1].set_xticks([]) 
    ax[1].set_yticks([]) 

    f,ax = plt.subplots(2,2)
    ax = ax.flatten()
    rgbbasisvector1 = [155, 55, 255]
    rgbbasisvector2 = [255-el for el in rgbbasisvector1]
    make_legend(ax[1], inc3layer=False, abt_offset=True)
    make_legend_sq(ax[3], xcoord=np.array([1,1]), ycoord=np.array([0,1]))
    plt.show()

    f,ax = plt.subplots(3,2)
    ax = ax.flatten()
    make_legend(ax[4], inc3layer=False, abt_offset=True)
    make_legend(ax[5], inc3layer=False, abt_offset=False)
    xrange = np.arange(0, 3, 0.01)
    U, V = np.meshgrid(xrange, xrange)
    halfway = int(U.shape[0]/2)
    displacement_lineplot(ax[0], ax[2], U[:,halfway], V[:,halfway], abt_offset=True, f=1)
    displacement_lineplot(ax[1], ax[3], U[:,halfway], V[:,halfway], abt_offset=False, f=1)
    plt.show()

def make_hisogram_coloredmaps():

    # make rigid figure plot
    colors = rigid_map(inc3layer=False, abt_offset=True, f=1)
    avgring1, avgring2 = colors[:,:,0], colors[:,:,2]
    fig, ax = plt.subplots(1,1)
    ax.imshow(colors)
    plt.savefig("rigid-abt.svg", dpi=300)
    fig, ax = plt.subplots(1,1)
    make_histoscatter(avgring1, avgring2, fig, ax)
    plt.savefig("histo-rigid-abt.svg", dpi=300)
    
    # make ABt experimental
    filepath = '../data/{}/dat_ds{}.pkl'.format('ABt-nd1', 8) 
    avgring1, avgring2 = load_3disk_vdf(filepath)
    fig, ax = plt.subplots(1,1)
    make_histoscatter(avgring1, avgring2, fig, ax)
    plt.savefig("histo-abt.svg", dpi=300)

    # make AtA experimental
    filepath = '../data/{}/dat_ds{}.pkl'.format('c7', 2) 
    avgring1, avgring2 = load_3disk_vdf(filepath)   
    fig, ax = plt.subplots(1,1)
    make_histoscatter(avgring1[40:,:], avgring2[40:,:], fig, ax)
    plt.savefig("histo-ata.svg", dpi=300)

def manual_define_2pt(img):
    plt.close('all')
    fig, ax = plt.subplots()
    vertices = []
    def click_event(click):
        x,y = click.xdata, click.ydata
        vertices.append([x,y])
        ax.scatter(x,y,color='k')
        fig.canvas.draw()
        if len(vertices) == 2:
            fig.canvas.mpl_disconnect(cid)
            plt.close('all')
    print("please click point")
    ax.imshow(img, cmap='gray')
    cid = fig.canvas.mpl_connect('button_press_event', click_event)
    plt.show()
    return vertices

def make_linecut(img, ptA, ptB, nm_per_pix, ax):
    ax[0].imshow(img)
    ax[0].scatter(ptA[0], ptA[1], c='r')
    ax[0].scatter(ptB[0], ptB[1], c='r')
    ax[0].plot([ptA[0], ptB[0]], [ptA[1], ptB[1]], c='r')
    N = 200
    x, y = np.linspace(ptA[0], ptB[0], N), np.linspace(ptA[1], ptB[1], N)
    path_len = nm_per_pix * ((ptA[0] - ptB[0]) ** 2 + (ptA[1] - ptB[1]) ** 2 ) ** 0.5
    zi = scipy.ndimage.map_coordinates(np.transpose(img), np.vstack((x,y))) # extract values along line w/ cubic interpolation
    d = np.linspace(0, path_len, N)
    ax[1].plot(d, zi, 'k')
    ax[1].set_title('nm')
    return x,y,d,zi
    
def make_legend_sq(ax, N=100, xcoord=None, ycoord=None):
    mat = np.zeros((N, N, 3))
    for i in range(N):
        for j in range(N):
            r1 = (i/N)
            r2 = (j/N)     
            for n in range(3):
                mat[i,j,n] = rgbbasisvector1[n]/255 * r1 + rgbbasisvector2[n]/255 * r2
    if xcoord is not None: ax.scatter(xcoord*N, ycoord*N, c='k')
    ax.imshow(mat, origin='lower')
    ax.set_xticks([]) 
    ax.set_yticks([]) 

def make_legend(ax, inc3layer=False, abt_offset=False, f=1):
    xrange = np.arange(-0.50, 0.51, 0.005)
    nx = len(xrange)
    U, V = np.meshgrid(xrange, xrange)
    displacement_colorplot(ax, U, V, inc3layer, abt_offset, f)
    ax.axis('off')
    ax.set_xlim([-15, nx+15])
    ax.set_ylim([-15, nx+15])

def plot_hexagon(ax, nx, ny, data, orientation=0, radius=1/2):
    hex = RegularPolygon((nx/2, ny/2), numVertices=6, radius=radius*nx, fc='none', edgecolor='k', lw=2, orientation=orientation)
    verts = hex.get_path().vertices
    trans = hex.get_patch_transform()
    points = trans.transform(verts)
    for i in range(len(points)):
        old_pt = points[i]
        points[i] = [old_pt[1], old_pt[0]]
    mask = make_contour_mask(nx, ny, points)
    if len(data.shape) == 3:   data[mask <= 0,:] = [1.0, 1.0, 1.0]
    elif len(data.shape) == 2: data[mask <= 0] = 0.0
    ax.imshow(data)
    ax.add_patch(hex)
    return data

def displacement_colorplot(ax, Ux, Uy, inc3layer, abt_offset, f):
    nx, ny = Ux.shape
    g1 = np.array([ 0, 2/np.sqrt(3)])
    g2 = np.array([-1, 1/np.sqrt(3)])
    gvecs1 = [ g1, g2, g1-g2 ]
    gvecs2 = [ g1+g2, 2*g2-g1, 2*g1-g2 ]
    A1, A2, B1, B2 = 4/9, 4/9, -1/3, -1/3
    colors1 = np.zeros((nx, ny, 3))
    maxr1, maxr2 = 0,0
    minr1, minr2 = np.inf, np.inf
    for i in range(nx):
        for j in range(ny):
            u = [Ux[i,j], Uy[i,j]]
            if abt_offset:
                u = np.array(u) + np.array([0, 1/np.sqrt(3)]) 
            r1, r2 = 0, 0
            if inc3layer:
                for n in range(len(gvecs1)): 
                    r1 += ((np.cos(np.pi * np.dot(gvecs1[n], u))))**2 * 3/4
                    r1 += ((np.cos(np.pi * np.dot(gvecs1[n], np.array(u)*f))))**2 * 1/4
                for n in range(len(gvecs2)):  
                    r2 += ((np.cos(np.pi * np.dot(gvecs2[n], u))))**2 * 3/4
                    r2 += ((np.cos(np.pi * np.dot(gvecs2[n], np.array(u)*f))))**2 * 1/4
            else:
                for n in range(len(gvecs1)): r1 += ((np.cos(np.pi * np.dot(gvecs1[n], u))))**2 
                for n in range(len(gvecs2)): r2 += ((np.cos(np.pi * np.dot(gvecs2[n], u))))**2 
            r1, r2 = A1*r1 + B1, A2*r2 + B2
            if abt_offset: r1 = 1 - r1        
            for n in range(3):
                colors1[i,j,n] = rgbbasisvector1[n]/255 * r1 + rgbbasisvector2[n]/255 * r2
    f = 2 * np.max(Ux) * g2[0]
    colors1 = plot_hexagon(ax, nx, ny, colors1, radius=1/(2*f), orientation=0) 
    for axis in ['top','bottom','left','right']: ax.spines[axis].set_linewidth(1)

def displacement_lineplot(ax1, ax2, Ux, Uy, abt_offset, f):
    g1 = np.array([ 0, 2/np.sqrt(3)])
    g2 = np.array([-1, 1/np.sqrt(3)])
    nx = len(Ux)
    ring1 = np.zeros((nx))
    ring2 = np.zeros((nx))
    color = np.zeros((nx, nx, 3))
    gvecs1 = [ g1, g2, g1-g2 ]
    gvecs2 = [ g1+g2, 2*g2-g1, 2*g1-g2 ]
    for i in range(nx):
        u = [Ux[i], Uy[i]]
        if abt_offset: u = np.array(u) + np.array([0, 1/np.sqrt(3)]) 
        r1, r2 = 0, 0
        for n in range(len(gvecs1)): r1 += ((np.cos(np.pi * np.dot(gvecs1[n], u))))**2 
        for n in range(len(gvecs2)): r2 += ((np.cos(np.pi * np.dot(gvecs2[n], u))))**2 
        r1, r2 = r1/3, r2/3
        if abt_offset: r1 = 1 - r1
        ring1[i] = r1 
        ring2[i] = r2 
    ring1 = (ring1 - np.min(ring1))/(np.max(ring1)-np.min(ring1)) 
    ring2 = (ring2 - np.min(ring2))/(np.max(ring2)-np.min(ring2)) 
    for i in range(nx):
        color[:,i, 0] = ring1[i] 
        color[:,i, 1] = ring1[i]/2 + ring2[i]/2 
        color[:,i, 2] = ring2[i] 
    ax1.plot(nx * ring1, c='r')
    ax1.plot(nx * ring1, c='r')
    ax1.plot(nx * ring2, c='k')
    ax1.imshow(color, origin='lower')
    ax2.plot(ring1, c='r')
    ax2.plot(ring2, c='k')
    for i in range(nx): print('{} {} {}'.format(i, ring1[i], ring2[i]))

def vdf_partition(avgring1_m, avgring2_m, thresh1=0.75, thresh2=0.5):
    Ntot, NAB, NAA, NAASP, NABSP = 0,0,0,0,0
    stack_assign = np.zeros((avgring1_m.shape[0], avgring1_m.shape[1], 3))
    for i in range(stack_assign.shape[0]):
        for j in range(stack_assign.shape[1]):
            if (avgring1_m[i,j]) > thresh1: 
                Ntot+=1
                if (avgring2_m[i,j]) > thresh2: 
                    stack_assign[i,j,:] = [1,1,1]
                    NAA+=1
                else:
                    stack_assign[i,j,:] = [1, 165/255, 0]
                    NAASP+=1
            else: 
                Ntot+=1
                if (avgring2_m[i,j]) > thresh2: 
                    stack_assign[i,j,:] = [0,0,1]
                    NAB+=1
                else: 
                    stack_assign[i,j,:] = [0,0,0]
                    NABSP+=1
    return stack_assign

def load_3disk_vdf(filepath):
    with open(filepath, 'rb') as f: diskset = pickle.load(f)
    vdf = overlay_vdf(diskset, plotflag= False)
    dfs = diskset.df_set()
    g = diskset.d_set() 
    ringnos = diskset.determine_rings()
    avgring1 = np.zeros((vdf.shape[0],vdf.shape[1]))  
    avgring2 = np.zeros((avgring1.shape[0], avgring1.shape[1]))
    if dfs.shape[0] != 12: print('warning! not 12 disks for {}'.format(filepath))
    for i in range(dfs.shape[0]):
        if ringnos[i] == 1:
            avgring1 += dfs[i,:avgring1.shape[0],:avgring1.shape[1]]
        elif ringnos[i] == 2:
            avgring2 += dfs[i,:avgring1.shape[0],:avgring1.shape[1]]
    avgring1 = gaussian_filter(avgring1,1)
    avgring1 = avgring1 - np.min(avgring1.flatten())
    avgring1 = avgring1/np.max(avgring1.flatten())
    avgring2 = gaussian_filter(avgring2,1)
    avgring2 = avgring2 - np.min(avgring2.flatten())
    avgring2 = avgring2/np.max(avgring2.flatten())
    return avgring1, avgring2

def make_histoscatter(I1, I2, fig, ax):
    N = 80
    counts = np.zeros((N+1,N+1))    
    for x in range(I1.shape[0]):
        for y in range(I1.shape[1]):
            if not np.isnan(I1[x,y]):
                I1_ind = int(np.round(I1[x,y] * N, 1))
                I2_ind = int(np.round(I2[x,y] * N, 1))
                counts[I1_ind, I2_ind] += 1

    im = ax.imshow(counts, cmap='inferno', origin='lower')
    ax.set_xlabel('I1')
    ax.set_ylabel('I2')
    fig.colorbar(im, ax=ax)

def rigid_map(inc3layer=False, abt_offset=True, f=1):
    xrange = np.arange(-3.5, 3.55, 0.01)
    nx = len(xrange)
    Ux, Uy= np.meshgrid(xrange, xrange)
    nx, ny = Ux.shape
    g1 = np.array([ 0, 2/np.sqrt(3)])
    g2 = np.array([-1, 1/np.sqrt(3)])
    gvecs1 = [ g1, g2, g1-g2 ]
    gvecs2 = [ g1+g2, 2*g2-g1, 2*g1-g2 ]
    colors1 = np.zeros((nx, ny, 3))
    for i in range(nx):
        for j in range(ny):
            u = [Ux[i,j], Uy[i,j]]
            if abt_offset:
                u = np.array(u) + np.array([0, 1/np.sqrt(3)]) 
            r1, r2 = 0, 0
            if inc3layer:
                for n in range(len(gvecs1)): 
                    r1 += ((np.cos(np.pi * np.dot(gvecs1[n], u))))**2 * 3/4
                    r1 += ((np.cos(np.pi * np.dot(gvecs1[n], np.array(u)*f))))**2 * 1/4
                for n in range(len(gvecs2)):  
                    r2 += ((np.cos(np.pi * np.dot(gvecs2[n], u))))**2 * 3/4
                    r2 += ((np.cos(np.pi * np.dot(gvecs2[n], np.array(u)*f))))**2 * 1/4
            else:
                for n in range(len(gvecs1)): r1 += ((np.cos(np.pi * np.dot(gvecs1[n], u))))**2 
                for n in range(len(gvecs2)): r2 += ((np.cos(np.pi * np.dot(gvecs2[n], u))))**2 
            
            r1, r2 = (r1-3/4)/(3-3/4), (r2-3/4)/(3-3/4)
            if abt_offset: r1 = 1 - r1    
            colors1[i,j,0] = r1 
            colors1[i,j,1] = r1/2 + r2/2 
            colors1[i,j,2] = r2 

    f = 2 * np.max(Ux) * g2[0]
    #colors1 = crop_hexagon(nx, ny, colors1, radius=( 1/(2*f) ), orientation=0) 
    return colors1

def crop_hexagon(nx, ny, data, orientation=0, radius=1/2):
    hex = RegularPolygon((nx/2, ny/2), numVertices=6, radius=radius*nx, fc='none', edgecolor='k', lw=2, orientation=orientation)
    verts = hex.get_path().vertices
    trans = hex.get_patch_transform()
    points = trans.transform(verts)
    for i in range(len(points)):
        old_pt = points[i]
        points[i] = [old_pt[1], old_pt[0]]
    mask = make_contour_mask(nx, ny, points)
    data[mask <= 0,:] = np.nan
    return data

if __name__ == "__main__": # command line call to main driver 
	
	# various plot generation and post-processing (linecuts, scatter plots, colored maps)
	#make_hisogram_coloredmaps()
	make_linecuts_coloredmaps()
	
	# to extract the VDFs: (see mask generation file for creation of mask_ds1.pkl)
	#make_vdf_from_mask_and_dataset('avgdp_ds1.pkl', 'mask_ds1.pkl','vdf_ds1.pkl')


