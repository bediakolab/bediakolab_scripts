import numpy as np
from scipy.optimize import curve_fit, least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from skimage import measure
from scipy.spatial import Delaunay
from scipy.fft import fft, fftfreq, fftshift
from scipy import fftpack
import re
import sys
import seaborn as sns

# distance between two vertices
def get_dist(v1, v2):
    l = ((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)**0.5
    return l

def normalize(dat, lowerbound_filter, upperbound_filter):
    # normalize
    (nx, ny) = dat.shape
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

def get_pt_range(slope, j):
    # get data values along the line of given slope
    if (slope > 1):
        x0, y0 = 0, 0
        x1, y1 = (nx-1)/slope, nx-1
        x0_pt = x0 + j
        x1_pt = min( x1 + j, nx - 1)
        y0_pt = y0
        y1_pt = (x1_pt - x0_pt) * slope
    elif (slope > 0):
        x0, y0 = 0, 0
        x1, y1 = nx-1,(nx-1)*slope
        y0_pt = y0 + j
        y1_pt = min( y1 + j, nx - 1)
        x0_pt = x0
        x1_pt = (y1_pt - y0_pt)/slope
    elif (slope > -1):
        print("I got lazy this hasn't been implemented for slope < 0, let me know and I'll add")
        exit()
        x0, y0 = nx-1, 0
        x1, y1 = 0, (nx-1)*(-slope)
    else:
        print("I got lazy this hasn't been implemented for slope < 0, let me know and I'll add")
        exit()
        x0, y0 = nx-1, 0
        x1, y1 = (nx-1)/(-slope), nx-1
    return  x0_pt, x1_pt, y0_pt, y1_pt

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

# will extract moire wavelengths along different axes using FT of various
# dataset slices to obtain heterostrain and twist angle, use for noiser/small angle datasets
def ft_process(dat, ax, params):
    return diagonal_slices(dat, params, params['slope'])

# returns slices of a matrix along the direction given by the vector d
def diagonal_slices(dat, params, slope):

    (nx, ny) = dat.shape
    offset = params['offset']
    interval = int(np.floor((ny-offset) / params['n_images']))
    palette = sns.color_palette("bright", params['n_images'] + 1)
    dat_trace = []
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.imshow(dat, origin='bottom')
    if (nx != ny):
        print("must have square domain")
        exit(0)

    i = 0
    for j in range(offset, nx, interval):
        n = int(nx*np.sqrt(3))
        dat_trace_int = np.zeros((n, 1))
        for jj in range(j, min(j+interval,nx-1)):
            x0_pt, x1_pt, y0_pt, y1_pt = get_pt_range(slope, jj)
            if (jj == int(j + interval/2)):
                ax1.plot([x0_pt, x1_pt], [y0_pt, y1_pt], 'o-', color=palette[i])
            l = int(get_dist([x0_pt, y0_pt], [x1_pt, y1_pt]))
            x, y = np.linspace(x0_pt, x1_pt, l), np.linspace(y0_pt, y1_pt, l)
            slice = dat[x.astype(np.int), y.astype(np.int)]
            len_diff = n - len(slice) #1.5
            dat_trace_int[ int(len_diff/2) : int(n - len_diff/2), 0 ] = (dat_trace_int[ int(len_diff/2) : int(n - len_diff/2), 0 ] + slice) * 0.5

        # plot data with superposed line
        # also plot data trace and its FT
        if (i == params['to_plot']):
            d = [e/max(dat_trace_int) for e in dat_trace_int]
            d = d[80:n-80]
            N = len(d)
            T = params['FOV_length']/n
            xf = fftfreq(N, T)
            yf = fft(d)
            ax1.axis('image')
            ax2.plot( [i*params['FOV_length']/nx for i in range(N)] , d, color=palette[i])
            ax3.plot(fftshift(xf), np.abs(fftshift(yf))/max(fftshift(yf)), color=palette[i])
            ax1.set_xlabel('pixels')
            ax1.set_ylabel('pixels')
            ax2.set_title('trace {}'.format(params['to_plot'] + 1))
            ax3.set_title('FT(trace {})'.format(params['to_plot'] + 1))
            ax2.set_xlabel('nm')
            ax3.set_xlabel('1/nm')
        i += 1

    plt.show()
    exit()

    return True

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
        "num_slices"        : 1,               # slice data into chunks, will be faster but omits some data
        "guess_theta_t"     : 0.3,             # degrees of guess angle for twist
        "guess_theta_s"     : 25,              # guess angle of heterostrain application
        "guess_hs"          : 0.2,             # guessed percent heterostrain
        "slope"             : 16,              # slope of slices taken
        "offset"            : 14,              # offset (along x or y axis depending on slope) for the first trace drawn
        "n_images"          : 9,               # number of slices to draw through the data
        "to_plot"           : 6                # plots the trace and FT of this slice (0 based indexing)
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
        ft_process(chunks[i], ax, params)
