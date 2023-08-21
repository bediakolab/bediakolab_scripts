
import pickle
import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt
from skimage import measure
from scipy.spatial import ConvexHull
from matplotlib.colors import LogNorm
from scipy import ndimage as ndi
import matplotlib


## A LOT WAS HARD CODED FOR THESE DUE TO DIFFICULTY OBTAINING REGIONS AUTOMATICALLY. SEE SAVED MASK OBJECTS.

just_make_inset_plt = False
auto_locate = False
dsnum  =  8
sample =  '6c'
r2_dist    = 241
r1_dist    = 140
hbn_thresh = 10 
T0, T1, T2, T3 = 2, 10, 3, 10
r2_thresh  = 15 
x_0guess   = 407
y_0guess   = 400 
r1guess    = 175
r2guess    = 310
pix_overlap_disk_width_ring2 = 20
pix_overlap_disk_width_ring1 = 20
aberated_disk_thresh = [0.3, 0.55]
thresholds_r2    = [0.3, 0.55]  
exclude_scale_r2 =  1.15 
elip_thresh      =  0.05 
thresholds_r1    = [0.20, 0.83]  
exclude_scale_r1 =  1.1 
circ_thresh      =  0.5
xlocs = [298+10,539+3,641,500,260,159,274,386,514,530,415,288]
ylocs = [176-5,196-5,414-3,612-1,590-8,370-8,449-5,531-5,474-5,335-5,254-5,310-5]

filepath   = '{}/avgdp_ds{}.pkl'.format(sample, dsnum)
maskpath   = '{}/mask_ds{}-rings12.pkl'.format(sample, dsnum)

def thresh2ellipse(thresh, data, scale):
	contours = measure.find_contours(data, thresh)
	if len(contours) > 0:
		numpix = []
		for contour in contours: 
			numpix.append(np.sum(make_contour_mask(nx, ny, contour).flatten()))
	else:
		return [], [], None, np.ones(data.shape)
	contour = contours[np.argmax(numpix)]
	x0, y0, ap, bp, e, phi, xfit, yfit = fit_ellipse(contour[:,1], contour[:,0])
	t = np.linspace(0, 2*np.pi, 100)
	xfit = x0 + ap*scale * np.cos(t) * np.cos(phi) - bp*scale * np.sin(t) * np.sin(phi)
	yfit = y0 + ap*scale * np.cos(t) * np.sin(phi) + bp*scale * np.sin(t) * np.cos(phi)
	contour = np.zeros((len(t), 2))
	contour[:,1], contour[:,0] = xfit, yfit
	mask = make_contour_mask(data.shape[0], data.shape[1], contour)
	return xfit, yfit, contour, mask

def extract_overlaps_disks(i, dp_slice, thresholds, exclude_scale, regions=3, threshavg=5, hbnblock=False):
	avg = np.mean(dp_slice.flatten())
	if hbnblock:
		xfit, yfit, contour, hbn_mask = thresh2ellipse(threshavg, np.abs(dp_slice - avg), 1.25)
		dp_slice = dp_slice * (1-hbn_mask)
	dp_slice = dp_slice - np.min(dp_slice.flatten())
	dp_slice = dp_slice / np.max(dp_slice.flatten())
	nx, ny = dp_slice.shape[0], dp_slice.shape[1]
	f, ax = plt.subplots(3,3)
	ax = ax.flatten()
	ax[0].imshow(dp_slice, origin='lower')
	# get inner overlap
	xfit, yfit, contour, inner_mask = thresh2ellipse(thresholds[1], dp_slice, 0.85)
	ax[0].plot(xfit, yfit, 'r')
	ax[0].set_title('disk {}'.format(i))
	ax[3].plot(xfit, yfit, 'r')
	ax[3].imshow(inner_mask, origin='lower')
	ax[3].set_title('1/2/3')
	xfit, yfit, contour, inner_exclude = thresh2ellipse(thresholds[1], dp_slice, exclude_scale)
	ax[0].plot(xfit, yfit, 'g')
	ax[2].plot(xfit, yfit, 'g')
	xfit, yfit, contour, outermask_raw = thresh2ellipse(thresholds[0], dp_slice, 0.95)
	ax[0].plot(xfit, yfit, 'c')
	outermask = np.logical_and(outermask_raw , (1-inner_exclude))
	ax[2].imshow(np.logical_and(outermask , (1-inner_exclude)), origin='lower')
	ax[1].imshow(outermask_raw, origin='lower')
	ax[2].set_title('1/2 remove 1/2/3')
	ax[1].set_title('1/2')
	ax[1].plot(xfit, yfit, 'c')
	ax[2].plot(xfit, yfit, 'c')
	ax[4].imshow(outermask+inner_mask, origin='lower')
	labeled, nregions = ndi.label(outermask+inner_mask)
	if nregions >= regions:
		sizes = []
		masks = []
		for n in range(1,nregions+1):
			masks.append((labeled == n))
			sizes.append(np.sum((labeled == n).flatten()))
	else:
		print('failed to find {} regions...'.format(regions))
		return labeled
	labeled_three  = np.zeros((nx, ny))
	if regions >= 1:
		mask = masks.pop(np.argmax(sizes)); sizes.remove(np.max(sizes))
		labeled_three += mask
	if regions >= 2:
		mask = masks.pop(np.argmax(sizes)); sizes.remove(np.max(sizes))
		labeled_three += 2 * mask
	if regions >= 3:
		mask = masks.pop(np.argmax(sizes)); sizes.remove(np.max(sizes))
		labeled_three += 3 * mask
	ax[5].imshow(labeled, origin='lower')
	ax[6].imshow(labeled_three, origin='lower')
	plt.show()
	return labeled_three

def fit_ellipse(x, y):
	D1 = np.vstack([x**2, x*y, y**2]).T
	D2 = np.vstack([x, y, np.ones(len(x))]).T
	S1 = D1.T @ D1
	S2 = D1.T @ D2
	S3 = D2.T @ D2
	T = -np.linalg.inv(S3) @ S2.T
	M = S1 + S2 @ T
	C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
	M = np.linalg.inv(C) @ M
	eigval, eigvec = np.linalg.eig(M)
	con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
	ak = eigvec[:, np.nonzero(con > 0)[0]]
	coeffs = np.concatenate((ak, T @ ak)).ravel() # a, b, c, d, e, f
	# We use the formulas from https://mathworld.wolfram.com/Ellipse.html
	a,b,c,d,f,g = coeffs[0], coeffs[1] / 2, coeffs[2], coeffs[3] / 2, coeffs[4] / 2, coeffs[5]
	den = b**2 - a*c
	x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den # The location of the ellipse centre.
	num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
	fac = np.sqrt((a - c)**2 + 4*b**2)
	ap = np.sqrt(num / den / (fac - a - c)) # The semi-major and semi-minor axis lengths (these are not sorted).
	bp = np.sqrt(num / den / (-fac - a - c))
	width_gt_height = True
	if ap < bp: 
		width_gt_height = False
		ap, bp = bp, ap
	r = (bp/ap)**2 
	if r > 1: r = 1/r
	e = np.sqrt(1 - r) 
	if b == 0: phi = 0 if a < c else np.pi/2
	else:
		phi = np.arctan((2.*b) / (a - c)) / 2
		if a > c: phi += np.pi/2
	if not width_gt_height: phi += np.pi/2
	phi = phi % np.pi
	# A grid of the parametric variable, t.
	t = np.linspace(0, 2*np.pi, 100)
	xfit = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
	yfit = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
	return x0, y0, ap, bp, e, phi, xfit, yfit

def ring2_mask(avgdp):
	nx, ny = avgdp.shape[0], avgdp.shape[1]
	x, y = np.meshgrid(np.arange(nx), np.arange(ny))
	mask1 = np.sqrt((x-x_0guess)**2+(y-y_0guess)**2) > r1guess
	mask2 = np.sqrt((x-x_0guess)**2+(y-y_0guess)**2) < r2guess
	return avgdp*mask1*mask2

def manual_define_pt(img):
    plt.close('all')
    fig, ax = plt.subplots()
    vertices = []
    def click_event(click):
        x,y = click.xdata, click.ydata
        vertices.append([x,y])
        ax.scatter(x,y,color='k')
        fig.canvas.draw()
        fig.canvas.mpl_disconnect(cid)
        plt.close('all')
    print("please click point")
    ax.imshow(img, cmap='Blues',norm=matplotlib.colors.LogNorm())
    cid = fig.canvas.mpl_connect('button_press_event', click_event)
    plt.show()
    x, y = vertices[0][0], vertices[0][1]
    return int(round(x)),int(round(y))

def circ_mask(img, x_0, y_0, r):
	nx, ny = img.shape[0], img.shape[1]
	x, y = np.meshgrid(np.arange(nx), np.arange(ny))
	mask = np.sqrt((x-x_0)**2+(y-y_0)**2) < r
	return mask*img

def is_mostly_filled_in(mask, contour):
	nx, ny = mask.shape[0], mask.shape[1]
	m = make_contour_mask(nx, ny, contour) 
	return (np.mean((m*mask).flatten())/np.mean(m.flatten())) > 0.5

def make_contour_mask(nx, ny, contour, transpose=False):
	p = Path(contour)
	x, y = np.meshgrid(np.arange(nx), np.arange(ny)) # make a canvas with coordinates
	x, y = x.flatten(), y.flatten()
	if transpose: points = np.vstack((y,x)).T
	else: points = np.vstack((x,y)).T
	# identifies if each coordinate is contained in the path of points, generating a mask
	grid = p.contains_points(points)
	return grid.reshape(nx,ny).T # reshape into a matrix

def find_disks(mask, ax, xc=None, yc=None, oval_only=False, circle_only=False, maxN=None, dist=None):
	nx, ny = mask.shape[0], mask.shape[1]
	contours = measure.find_contours(mask, 0.5)
	diskmasktot = np.zeros((nx, ny))
	xlist, ylist, numpix = [], [], []
	for contour in contours: 
		if len(contour[:,1]) < 8: continue
		hull = ConvexHull(contour)
		xpts, ypts = contour[hull.vertices,1], contour[hull.vertices,0]
		contour = np.zeros((len(xpts), 2))
		contour[:,1], contour[:,0] = xpts, ypts
		if not is_mostly_filled_in(mask, contour): continue
		try: x0, y0, ap, bp, e, phi, xfit, yfit = fit_ellipse(contour[:,1], contour[:,0]) # fit to ellipse 
		except: continue
		if oval_only:
			if np.abs(ap - bp)/ap < elip_thresh: continue
		elif circle_only:
			if np.abs(ap - bp)/ap > circ_thresh: continue
		if xc is not None:
			if not ( np.abs(( ((x0-xc)**2 + (y0-yc)**2)**0.5 ) - dist) < r2_thresh ): 
				continue
		diskmask = make_contour_mask(nx, ny, contour)
		numpix.append(np.sum(diskmask.flatten()))
		diskmasktot += diskmask
		ax.plot(xfit, yfit, 'grey')
		xlist.append(x0)
		ylist.append(y0)
	if maxN is not None:
		xlisttrim, ylisttrim = [], []
		for i in range(maxN):
			try:
				xlisttrim.append(xlist.pop(np.argmax(numpix)))
				ylisttrim.append(ylist.pop(np.argmax(numpix)))
				numpix.remove(np.max(numpix))
			except:
				continue # too few
		return xlisttrim, ylisttrim, diskmasktot
	return xlist, ylist, diskmasktot
   
with open(filepath, 'rb') as f: avgdp = pickle.load(f)
if just_make_inset_plt:
	dsnums  =  [10]
	vmaxes = [4.0]
	vmins = [0.5]
	sample =  '6c'
	f, ax = plt.subplots(2,2)
	ax = ax.flatten()
	for i in range(len(dsnums)):
		filepath = '{}/avgdp_ds{}.pkl'.format(sample, dsnums[i])
		with open(filepath, 'rb') as f: avgdp = pickle.load(f)
		ax[i].imshow(avgdp[575:675,475:575], origin='lower',cmap='Blues', vmax=vmaxes[i], vmin=vmins[i])
	plt.savefig('diskavgs.svg')
	plt.show(); exit()

f, ax = plt.subplots(1,1)
ax.imshow(avgdp, origin='lower',cmap='Blues',norm=matplotlib.colors.LogNorm())
plt.show(); exit()

avgdp_clean = avgdp.copy()
if auto_locate:
	f, ax = plt.subplots(2,2)
	avgdp = ring2_mask(avgdp)
	ax[0,0].imshow(avgdp_clean, origin='lower',cmap='inferno', vmax=3.75, vmin=0)
	#plt.savefig('avgdp_ds16.png', dpi=600)
	ax[0,1].imshow(avgdp > hbn_thresh, origin='lower',cmap='inferno')
	xl, yl, _ = find_disks(avgdp > hbn_thresh, ax[0,1], maxN=6, circle_only=True)
	avgdp = avgdp_clean
	xc, yc = np.mean(xl), np.mean(yl)
	ax[0,1].scatter(xc, yc, c='r')
	ax[1,0].imshow(np.logical_and(avgdp < T1 , avgdp > T0), origin='lower',cmap='inferno')
	plt.show()
	xl, yl, diskmaskr2 = find_disks(np.logical_and(avgdp < T1 , avgdp > T0), ax[1,0], xc=xc, yc=yc, maxN=6, oval_only=False, dist=r2_dist)
	ax[1,1].imshow(np.logical_and(avgdp > T2 , avgdp < T3), origin='lower',cmap='inferno')
	xl2, yl2, diskmaskr1 = find_disks(np.logical_and(avgdp > T2 , avgdp < T3), ax[1,1], xc=xc, yc=yc, maxN=6, oval_only=False, dist=r1_dist)
	diskmask = np.logical_or(diskmaskr2, diskmaskr1)
	avgdp = avgdp*diskmask
	for i in range(len(xl2)):
		xl.append(xl2[i])
		yl.append(yl2[i])
	plt.show()

	slicedps = []
	xlocs = []
	ylocs = []
	for i in range(len(xl)):
		x = int(xl[i])
		y = int(yl[i])
		if i < 6:
			w = pix_overlap_disk_width_ring2
		if i >= 6:
			w = pix_overlap_disk_width_ring1
		slicedp = avgdp[y-w:y+w, x-w:x+w]
		slicedps.append(slicedp)
		xlocs.append(xl[i])
		ylocs.append(yl[i])

	f, ax = plt.subplots(len(slicedps)//3 + 1, 4)
	ax = ax.flatten()
	ax[-1].imshow(avgdp, origin='lower')
	for i in range(len(slicedps)):
		ax[i].imshow(slicedps[i], origin='lower')#, norm=LogNorm(vmin=np.min(slicedp.flatten()), vmax=np.max(slicedp.flatten())))
		ax[i].set_title(i)
		ax[-1].scatter(xlocs[i], ylocs[i], s=2, c='r')
		ax[-1].text(xlocs[i], ylocs[i], i, c='r')
	plt.show()

elif xlocs != None:

	slicedps = []
	for i in range(12):
		x, y = xlocs[i], ylocs[i]
		if i < 6:
			w = pix_overlap_disk_width_ring2
		if i >= 6:
			w = pix_overlap_disk_width_ring1
		slicedp = avgdp[y-w:y+w, x-w:x+w]
		slicedps.append(slicedp)

	f, ax = plt.subplots(len(slicedps)//3 + 1, 4)
	ax = ax.flatten()
	ax[-1].imshow(avgdp, origin='lower')
	for i in range(len(slicedps)):
		ax[i].imshow(slicedps[i], origin='lower')#, norm=LogNorm(vmin=np.min(slicedp.flatten()), vmax=np.max(slicedp.flatten())))
		ax[i].set_title(i)
		ax[-1].scatter(xlocs[i], ylocs[i], s=2, c='r')
		ax[-1].text(xlocs[i], ylocs[i], i, c='r')
	plt.show()

else:
	slicedps = []
	xlocs = []
	ylocs = []
	for i in range(12):
		x, y = manual_define_pt(avgdp_clean)
		if i < 6:
			w = pix_overlap_disk_width_ring2
		if i >= 6:
			w = pix_overlap_disk_width_ring1
		slicedp = avgdp[y-w:y+w, x-w:x+w]
		slicedps.append(slicedp)
		xlocs.append(x)
		ylocs.append(y)
		print("{} --- ({},{})".format(i,x,y))

	f, ax = plt.subplots(len(slicedps)//3 + 1, 4)
	ax = ax.flatten()
	ax[-1].imshow(avgdp, origin='lower')
	for i in range(len(slicedps)):
		ax[i].imshow(slicedps[i], origin='lower')#, norm=LogNorm(vmin=np.min(slicedp.flatten()), vmax=np.max(slicedp.flatten())))
		ax[i].set_title(i)
		ax[-1].scatter(xlocs[i], ylocs[i], s=2, c='r')
		ax[-1].text(xlocs[i], ylocs[i], i, c='r')
	plt.show()

nx, ny = avgdp.shape[0], avgdp.shape[1]
overlap_mask_sum = np.zeros((nx, ny))
for i in range(len(slicedps)):
	if i in [0,1,2,3,4,5]:
		overlap_mask = extract_overlaps_disks(i, slicedps[i], [0.3,0.8], exclude_scale_r2, regions=3, hbnblock=False)
		w = pix_overlap_disk_width_ring2
	elif i == 8:
		slicedps[i][17,33] = 0 
		overlap_mask = extract_overlaps_disks(i, slicedps[i], [0.3,0.65], exclude_scale_r1, regions=3, hbnblock=True, threshavg=1.5)
		w = pix_overlap_disk_width_ring1
	elif i in [10,11]:
		overlap_mask = extract_overlaps_disks(i, slicedps[i], [0.3,0.6], exclude_scale_r1, regions=3, hbnblock=True)
		w = pix_overlap_disk_width_ring1
	elif i in [6,7]:
		overlap_mask = extract_overlaps_disks(i, slicedps[i], [0.25,0.55], exclude_scale_r1, regions=3, hbnblock=True, threshavg=4)
		w = pix_overlap_disk_width_ring1
	elif i in [9]:
		overlap_mask = extract_overlaps_disks(i, slicedps[i], [0.35,0.75], 1.25, regions=3, hbnblock=True, threshavg=2)
		w = pix_overlap_disk_width_ring1
	overlap_mask_sum[int(ylocs[i])-w:int(ylocs[i])+w, int(xlocs[i])-w:int(xlocs[i])+w] = overlap_mask
f, ax = plt.subplots(1,2)
ax[0].imshow(overlap_mask_sum)
regions, _ = ndi.label(overlap_mask_sum > 0)
ax[1].imshow(regions)
plt.show()

with open(maskpath, 'wb') as f: pickle.dump(regions, f)



