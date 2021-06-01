##############################################################################
## calculate the heterostrain and twist angle for a provived STM dataset.   ##
## this is accomplished by fitting the tunneling intensities to bivariate   ##
## gaussians, then preforming Deluanay triangulation on the resultant       ##
## gasssian centers.                                                        ##
## WARNING: this is highly dependent on the hyper-parameters chosen, which  ##
## are printed to an output file upon execution for book keeping. Some      ##
## example datasets and their corresponding parameters are provided in the  ##
## folder 'examples' for reference.                                         ##
##                                                                          ##
## usage : python3 STMtriangulate.py folder/dataset.txt                     ##
## hyperparameters are set in the dictionary at the bottom of the file.     ##
## must be called in a directory containing a folder holding the STM data   ##
## STM data must be formatted in accordance with the provided example data. ## 
##############################################################################

import numpy as np
from scipy.optimize import curve_fit, least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from skimage import measure
from scipy.spatial import Delaunay
import re
import sys
import seaborn as sns

# 2d gaussian
def gaussian(x, y, x0, y0, alpha, A):
    return A * np.exp( -((x-x0)/alpha)**2 -((y-y0)/alpha)**2)

# unwrapped lc of gaussians for lsq fit
def _gaussian(M, *args):
    x, y = M
    arr = np.zeros(x.shape)
    for i in range(len(args)//4):
       arr += gaussian(x, y, *args[i*4:i*4+4])
    return arr

# given three triangle lengths return angles
def get_angles(l1, l2, l3):
    a12 = np.arccos((l1**2 + l2**2 - l3**2)/(2*l1*l2))
    a23 = np.arccos((l2**2 + l3**2 - l1**2)/(2*l3*l2))
    a31 = np.arccos((l1**2 + l3**2 - l2**2)/(2*l1*l3))
    return a12, a23, a31

# given three vertices give lengths of triangle
def get_lengths(v1, v2, v3):
    l1 = ((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)**0.5
    l2 = ((v1[0] - v3[0])**2 + (v1[1] - v3[1])**2)**0.5
    l3 = ((v3[0] - v2[0])**2 + (v3[1] - v2[1])**2)**0.5
    return l1, l2, l3

# distance between two vertices
def get_dist(v1, v2):
    l = ((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)**0.5
    return l

# determine heterostrain and twist angle given lengths of moire triangle
def fit_heterostrain(l1, l2, l3, params):

    delta = 0.16 # graphene Poisson ratio

    # returns residuals
    def cost_func(L, theta_t, theta_s, eps):
        k = 4*np.pi/(np.sqrt(3)*params['a_graphene'])
        K = np.array( [[k, 0], [k*0.5, k*0.86602540378], [-k*0.5, k*0.86602540378]] )
        R_t = np.array([[np.cos(theta_t), -np.sin(theta_t)], [np.sin(theta_t), np.cos(theta_t)]])
        R_s = np.array([[np.cos(theta_s), -np.sin(theta_s)], [np.sin(theta_s), np.cos(theta_s)]])
        R_ns = np.array([[np.cos(-theta_s), -np.sin(-theta_s)], [np.sin(-theta_s), np.cos(-theta_s)]])
        E = np.array([[1/(1+eps), 0],[0, 1/(1-delta*eps)]])
        M = R_t - np.matmul(R_ns, np.matmul(E, R_s))
        Y = [0,0,0]
        for i in range(3):
            v = np.matmul(M, K[i,:])
            l = (np.dot(v,v))**0.5
            Y[i] = 4*np.pi/(np.sqrt(3)*l)
        return [y - l for y,l in zip(Y,L)]

    # wrapped cost function
    def _cost_func(vars):
        L = np.array([l1,l2,l3])
        return cost_func(L, vars[0], vars[1], vars[2])

    guess_prms = [params['guess_theta_t'] * np.pi/180, params['guess_theta_s'] * np.pi/180, params['guess_hs']/100]
    opt = least_squares(_cost_func, guess_prms)
    return opt.x # theta_t, theta_s, eps

# normalizes dataset, filtering out low intensity outliers if requested
def normalize(dat, lowerbound_filter, upperbound_filter, truncation):
    # normalize
    (nx, ny) = dat.shape
    if ( truncation ):
        d_avg = np.mean(dat[0:nx-50,0:ny])
        d_std = np.std(dat[0:nx-50,0:ny])
        dat[nx-50:nx,0:ny] = d_avg - 0.5*d_std

    if ( lowerbound_filter > 0): # if -1 then off
        d_avg = np.mean(dat)
        d_std = np.std(dat)
        for i in range(nx):
            for j in range(ny):
                el = dat[i,j]
                if (el - d_avg)/d_std < -lowerbound_filter:
                    dat[i,j] = d_avg - lowerbound_filter*d_std
    if ( upperbound_filter > 0): # if -1 then off
        d_avg = np.mean(dat)
        d_std = np.std(dat)
        for i in range(nx):
            for j in range(ny):
                el = dat[i,j]
                if (el - d_avg)/d_std > upperbound_filter:
                    dat[i,j] = d_avg + upperbound_filter*d_std
    d_max = np.max(dat)
    d_min = np.min(dat)
    d_range = d_max - d_min
    dat = (dat - d_min)/d_range
    return dat

# filter points based on a z-score criterion of valuevec (sigmas, intensities, etc)
def filter(valuevec, points, criterion, filter_name):
    m = np.nanmean(valuevec)
    sd = np.nanstd(valuevec)
    for i in range(len(valuevec)):
        if np.abs(m - valuevec[i])/sd > criterion:
            print("removing {},{} from {} filter".format(points[i, 0], points[i, 1], filter_name))
            points[i, :] = np.nan
            valuevec[i] = np.nan

# points within a distance criterion are combined
def combine_nearby_spots(spots, combine_criterion):
    n = len(spots)
    distance_table = 100 * np.ones((n,n))
    bool_arr = np.ones((n,1))
    for i in range(n):
        for j in range(i+1, n):
            distance_table[i,j] = get_dist([spots[i][0], spots[i][1]], [spots[j][0], spots[j][1]])
    for i in range(n):
        d = np.min(distance_table[i,:])
        if d < combine_criterion:
            j = np.argmin(distance_table[i,:])
            spot_i = spots[i]
            spot_j = spots[j]
            #print('combining points at {:.2f},{:.2f} and {:.2f},{:.2f} at d={:.2f}'.format(spot_i[0], spot_i[1], spot_j[0], spot_j[1], d))
            spots[i] = [ (spot_i[0]*spot_i[2]+spot_j[0]*spot_j[2])*1/(spot_i[2]+spot_j[2]),
                         (spot_i[1]*spot_i[2]+spot_j[1]*spot_j[2])*1/(spot_i[2]+spot_j[2]),
                          spot_i[2]+spot_j[2], (spot_i[3]+spot_j[3])*0.5 ]
            spots[j] = spots[i]
            bool_arr[i] = 0 # remove point i
    new_spots = []
    for i in range(len(spots)):
        if bool_arr[i]:
            new_spots.append(spots[i])
    spots = new_spots
    return spots

# obtain heterostrain and twist angle through fitting data to a series of guassians,
# triangulating, and extracting these from the resultant mesh
#   plots to the provided mpl axis
#   returns vectors of heterostrain and twist angle
#   will plot data, gassian fit, Delaunay mesh, and fit heterostrain/twist to provided ax
def mesh_process(dat, ax, params):

    # get domain and normalize
    (nx, ny) = dat.shape
    x, y = np.linspace(0, 1, nx), np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    dat = normalize(dat, params['lowerbound_filter'], params['upperbound_filter'], params['truncation'])

    # determine countours
    print('finding average spots')
    contours = measure.find_contours(dat, params['contour_boundary'])
    spots = []
    rads = []
    if (params['plot_avg']):
        ax.imshow(dat)

    # plot contours and averages
    for n, contour in enumerate(contours):
        xcent = np.mean(contour[:, 1])
        ycent = np.mean(contour[:, 0])
        avg_y_rad = np.mean([np.abs(v - xcent) for v in contour[:, 1]])
        avg_x_rad = np.mean([np.abs(v - ycent) for v in contour[:, 0]])
        rad = (avg_x_rad + avg_y_rad) * 0.5
        rads.append(rad)
        I = dat[int(ycent),int(xcent)]
        spots.append([xcent, ycent, rad, I])

    # combine spots close to eachother
    print('filtering average spots')
    for i in range(params['times_to_combine']):
        spots = combine_nearby_spots(spots, params['combine_criterion'])

    # remove spots with small radii
    avg_rad = np.mean(rads)
    sd_rad = np.std(rads)
    new_spots = []
    for i in range(len(spots)):
        if (avg_rad - spots[i][2])/sd_rad < params['guess_radius_criterion'] :
            new_spots.append(spots[i])
    spots = new_spots

    # manual removal
    if (params["manual_removal_before"]):
        # remove points
        bool_arr = np.ones((n,1))
        if (len(params["removed_before_keys"]) == 0):
            # print points
            i = 0
            for spot in spots:
                print("{}\t\t\t\t({:.4f},{:.4f})".format(i, spots[i][0], spots[i][1]))
                i += 1
            remove_i = input("Enter number of removed point: , -1 to terminate\n")
            while (int(remove_i) is not -1):
                i = int(remove_i.strip())
                params["removed_before_keys"].append(i)
                bool_arr[i] = 0 # remove point i
                remove_i = input("Enter number of removed point: , -1 to terminate\n")
        else:
            for i in params["removed_before_keys"]:
                print("requested removal of point {}".format(i))
                bool_arr[i] = 0 # remove point i
        new_spots = []
        for i in range(len(spots)):
            if bool_arr[i]:
                new_spots.append(spots[i])
        spots = new_spots

    # plot averaged spots
    if (params['plot_avg']):
        ax.axis('image')
        for i in range(len(spots)):
            circle = plt.Circle((spots[i][0],spots[i][1]), color='r', radius=spots[i][2], linewidth=2.5, fill=False)
            ax.add_patch(circle)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
        print("exiting after plot_avg, set this false to continue")
        exit()

    # use averaged data as input to guassian fit, guess_prms holds initial guesses
    guess_prms = []
    for spot in spots:
        t = [spot[0]/nx, spot[1]/ny, spot[2]/nx, spot[3]]
        guess_prms.extend(t)

    # plot as 2D image and the fit as overlaid contours
    guess = np.zeros(dat.shape)
    for i in range(len(spots)):
        g = gaussian(X, Y, *guess_prms[i*4:i*4+4])
        guess += g

    # least squares opt
    print('fitting guassians')
    xdata = np.vstack((X.ravel(), Y.ravel()))
    popt, pcov = curve_fit(_gaussian, xdata, dat.ravel(), guess_prms, xtol=params['xtol'])

    # extract fit function and toss out erroneous points
    print('filtering guassians')
    points = np.zeros((len(spots), 2))
    sigs   = np.zeros((len(spots), 1))
    intens = np.zeros((len(spots), 1))
    fit    = np.zeros(dat.shape)
    for i in range(len(spots)):
        g = gaussian(X, Y, *popt[i*4:i*4+4])
        if (popt[i*4] > 1.5 or popt[i*4+1] > 1.5):
            print("removing {},{} from FOV filter".format(popt[i*4], popt[i*4+1]))
            points[i, :] = np.nan
            sigs[i] = np.nan
            intens[i] = np.nan
        elif (np.abs(popt[i*4+3]) > params['upperbound_sigma']):
            print("removing {},{} from upperbound sigma filter".format(popt[i*4], popt[i*4+1]))
            points[i, :] = np.nan
            sigs[i] = np.nan
            intens[i] = np.nan
        else:
            points[i, :] = popt[i*4:i*4+2]
            sigs[i] = popt[i*4+3]
            try:
                intens[i] = dat[int(points[i, 1]*ny),int(points[i, 0]*nx)]
            except:
                intens[i] = np.nan
        fit += g

    # filter out points with abnorbally high or low sigma
    filter(sigs, points, params['sigma_criterion_1'], "initial sigma")
    # filter out points with abnorbally high or low intensity
    filter(intens, points, params['inten_criterion'], "intensity")
    # filter out points with abnorbal sigma more strictly
    filter(sigs, points, params['sigma_criterion_2'], "second sigma")

    # manual removal
    if (params["manual_removal"]):

        # show points for user to pick
        if (params['manual_removal_plot']):
            ax.imshow(dat, origin='bottom', extent=(0, 1, 0, 1))
            ax.contour(X, Y, fit, colors='w')
            ax.axis('image')
            plt.plot(points[:,0], points[:,1], 'ro')
            plt.show()

        # remove points
        if (len(params["removed_pt_keys"]) == 0):
            # print points
            i = 0
            for point in points:
                print("{}\t\t\t\t({:.4f},{:.4f})".format(i, points[i,0], points[i,1]))
                i += 1
            remove_i = input("Enter number of removed point: , -1 to terminate\n")
            while (int(remove_i) is not -1):
                i = int(remove_i.strip())
                params["removed_pts"].append((points[i,0], points[i,1]))
                params["removed_pt_keys"].append(i)
                points[i, :] = np.nan
                remove_i = input("Enter number of removed point: , -1 to terminate\n")
        else:
            for i in params["removed_pt_keys"]:
                print("requested removal of point {}".format(i))
                points[i, :] = np.nan

    # fit mesh
    print('meshing')
    points = points[~np.isnan(points)]
    points = np.reshape(points, (len(points)//2,2))
    try:
        tri = Delaunay(points)
    except:
        print("Failed to identify enough points for triangularization after filtering")
        print("Please increase num_slices or change filering criteria")
        exit(0)

    # plot fit guassians and Delaunay mesh
    ax.imshow(dat, origin='bottom', extent=(0, 1, 0, 1))
    ax.contour(X, Y, fit, colors='w')
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.plot(points[:,0], points[:,1], 'ro')
    if (params["plot_full_mesh"]):
        plt.triplot(points[:,0], points[:,1], tri.simplices, color='b')

    # obtain heterostrains and angles from mesh
    print('calculating heterostrain and twist')
    het_strains = []
    thetas = []
    lens = []

    # get traingles within angle criterion for further filtering by length
    for i in range(len(tri.simplices)):
        v1, v2, v3 = points[tri.simplices[i,:]]
        l1, l2, l3 = get_lengths(v1, v2, v3)
        m_l = np.mean([l1, l2, l3])
        a12, a23, a31 = get_angles(l1,l2,l3) # want roughly pi/3 for hexagonal
        if (np.abs(a12 - np.pi/3) < params['angle_criterion'] and
            np.abs(a23 - np.pi/3) < params['angle_criterion'] and
            np.abs(a31 - np.pi/3) < params['angle_criterion'] ):
            lens.append(m_l)

    for i in range(len(tri.simplices)):
        v1, v2, v3 = points[tri.simplices[i,:]]
        l1, l2, l3 = get_lengths(v1, v2, v3)
        m_l = np.mean([l1, l2, l3])
        a12, a23, a31 = get_angles(l1,l2,l3) # want roughly pi/3 for hexagonal
        if (np.abs(a12 - np.pi/3) < params['angle_criterion'] and
            np.abs(a23 - np.pi/3) < params['angle_criterion'] and
            np.abs(a31 - np.pi/3) < params['angle_criterion'] and
            ( m_l - np.mean(lens))/np.std(lens) < params['ml_criterion'] ):

            plt.plot([v2[0], v3[0]], [v2[1], v3[1]], color="r")
            plt.plot([v2[0], v1[0]], [v2[1], v1[1]], color="r")
            plt.plot([v3[0], v1[0]], [v3[1], v1[1]], color="r")
            center_x = np.mean([v1[0], v2[0], v3[0]])
            center_y = np.mean([v1[1], v2[1], v3[1]])
            theta_t, theta_s, eps = fit_heterostrain(l1, l2, l3, params)
            thetas.append(np.abs(theta_t) * 180/np.pi)
            het_strains.append(np.abs(eps*100))

    plt.title('angle = {:.2f} deg, het strain = {:.2f}%'.format(np.mean(thetas), np.mean(het_strains)))
    return thetas, het_strains

# reads in excel file and returns partitioned data
def read_excel(filenm, num_slices):
    dat_pd = pd.read_excel(filenm, index_col=0)
    dat = dat_pd.to_numpy()
    (nx, ny) = dat.shape
    increment = nx//num_slices
    if (nx%num_slices > 0):
        print("Cannot cleanly slice into {}x{} chunks".format(num_slices,num_slices))
        exit(0)
    chunks = []
    for i in range(num_slices):
        for j in range(num_slices):
            chunks.append( dat[i*increment:i*increment+increment, j*increment:j*increment+increment] )
    return chunks, np.nan

# reads the text file of formatted STM data
def read_txt(filenm, num_slices):

    # read the POSCAR file
    lines = []
    with open(filenm) as f:
       line = f.readline()
       lines.append(line)
       while line:
           line = f.readline()
           lines.append(line)

    # parse
    for line in lines:
        if re.match("Data Size:*", line):
            l = line.split(":")
            domain_size = [ int(e.strip()) for e in l[1].split("x") ]
            if (domain_size[0] != domain_size[1]):
                print("expect domain sizes equal, cannot have {}x{} atm".format(domain_size[0], domain_size[1]))
                exit(0)
            dat = np.zeros((domain_size[0], domain_size[1]))
            i = 0
        elif re.match("Surface Size:*", line):
            l = line.split(":")
            domain_len = [ float(e.strip()) for e in l[1].split("x") ]
            if (domain_len[0] != domain_len[1]):
                print("expect domain sizes equal, cannot have {}x{} atm".format(domain_len[0], domain_len[1]))
                exit(0)
        elif re.match("X Unit:*", line) or re.match("Y Unit:*", line):
            l = line.split(":")
            unit = l[1].strip()
            if (unit.lower() != "nanometer"):
                print("unexpected unit in STM data, expect domain to be in nm")
                exit(0)
        elif (line.strip() != ""):
            l = line.split("\t")
            if re.match("-?\d+.\d+", l[0]):
                dat[i, :] = [float(e) for e in l[0:-1]]
                i += 1

    nx = domain_size[0]
    increment = nx//num_slices
    if (nx%num_slices > 0):
        print("Cannot cleanly slice into {}x{} chunks".format(num_slices,num_slices))
        exit(0)
    chunks = []
    for i in range(num_slices):
        for j in range(num_slices):
            chunks.append( dat[i*increment:i*increment+increment, j*increment:j*increment+increment] )
    return chunks, domain_len[0]/num_slices

# write a file
def write(filenm, data_name, data):
    with open(filenm, 'w') as f:
        f.write('{}\n'.format(data_name))
        if isinstance(data,dict):
            for k, v in data.items():
                f.write('{} : {}\n'.format(k, v))
        else:
            for el in data:
                f.write('{}\n'.format(el))

# read the data, wrapper for all file types
def read(filenm, num_slices):
    f = filenm.split(".")
    if (f[1] == "txt"):
        chunks, FOV_len = read_txt(filenm, num_slices)
    elif (f[1] == "xlsx"):
        chunks, FOV_len = read_excel(filenm, num_slices)
    else:
        print("STM data has unrecongnized file format")
        exit(0)
    return chunks, FOV_len

if __name__ == '__main__':

    params = {
        "plot_avg"          : False,           # plot the average spots used as a guess for guassian fit
        "plot_full_mesh"    : False,           # plot full Delaunay mesh including regions discarded from angle filtering
        "num_slices"        : 2,               # slice data into chunks, will be faster but omits some data
        "contour_boundary"  : 0.55,            # initial boundaries will be at this percent intentsity
        "sigma_criterion_1" : 5.0,             # first sigma filter removes points this many std away
        "inten_criterion"   : 5.0,             # first intens filter removes points this many std away
        "sigma_criterion_2" : 5.0,             # second sigma filter removes points this many std away
        "angle_criterion"   : 0.35,            # will ignore regions where moire triangle angles are greater than angle_criterion rad from pi/3
        "upperbound_sigma"  : 1e3,             # throw away points with sigmas > this upperbound before filtering
        "guess_theta_t"     : 0.2,             # degrees of guess angle for twist
        "guess_theta_s"     : 25,              # guess angle of heterostrain application
        "guess_hs"          : 0.05,            # guessed percent heterostrain
        "xtol"              : 1e-1,            # tolerance for gaussian fit, decrease for noisy data where average peaks are ok
        "removed_pts"       : [],              # for manual remove of points of known indeces
        "manual_removal"    : False,           # for manual remove of points, will query for labels (set manual_removal_plot=True)
        "manual_removal_before" : False,       # same as above but for point removal before fit
        "removed_before_keys" : [],            # to print removed points to output
        "removed_pt_keys"   : [],              # to print removed points to output
        'manual_removal_plot' : False,         # bool to plot peaks before manual removal selection
        'guess_radius_criterion' : -0.1,       # decrease me if there are a lot of erroneous mall peaks
        'combine_criterion' : 4.0,             # increase me if plot_avg breaks peaks into two nearby circles
        'lowerbound_filter' : 2,               # increase me if it looks like the background intensity of plot_avg is too > 0.0, -1 if off
        'upperbound_filter' : -1,              # increase me to truncate data above a threshold, -1 if off
        'ml_criterion' : 0.75,                 # remove points that have mean delaunay lengths to nearby points greater than a given criterion
                                               # due to common issue with delaunay algorithm - will include erroneous connections
        'times_to_combine' : 3,                # times to run through the nearby point combination proecedure 
        'truncation' : False,                  # trucate field of view
    }

    # get inputs
    params['filename'] = sys.argv[1]
    filedir = params['filename'].split('/')
    filedir = '/'.join(filedir[0:1])

    # parse
    chunks, FOV_len = read(params['filename'], params['num_slices'])
    params['FOV_length'] = FOV_len
    params['a_graphene'] = 0.246/params['FOV_length'] # 0.246 nm, normalized in units of FOV
    nchunks = params['num_slices']**2

    # process chunks
    for i in range(nchunks):
        # process chunk i
        f, ax = plt.subplots()
        print("processing chunk {} of {} ".format(i+1,nchunks))
        (nx, ny) = chunks[i].shape
        thetas, het_strains = mesh_process(chunks[i], ax, params)
        plt.savefig("{}/mesh_chunk{}of{}.png".format(filedir, i+1,nchunks), dpi=600)

        # plot histograms
        f, (ax1, ax2) = plt.subplots(1,2)
        ax1.hist(thetas, bins='auto')
        ax1.set_title('local twist angle')
        ax2.hist(het_strains, bins='auto')
        ax2.set_title('local heterostrain')
        plt.savefig("{}/histo_chunk{}of{}.png".format(filedir, i+1,nchunks))

        # write data
        write("{}/heterostrains_chunk{}of{}.txt".format(filedir, i+1,nchunks), "percent heterostrain obtained", het_strains)
        write("{}/angles_chunk{}of{}.txt".format(filedir, i+1,nchunks), "twist angles (degrees)", thetas)
        write("{}/output_chunk{}of{}.txt".format(filedir, i+1,nchunks), "parameters used:", params)
