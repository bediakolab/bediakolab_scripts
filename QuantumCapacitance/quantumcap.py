
##############################################################################
## calculate quantum capacitance associated with a given twist angle of     ##
## bilayer graphene at a given applied voltage                              ##
##                                                                          ##
## must be called from a folder containing the relevant dos data in an      ##
## excel sheet titled calc_DOS.xlsx                                         ##
##                                                                          ##
## excel sheet must be formatted with energies (eV) in leftmost column,     ##
## dos values in the following columns, and twist angle labels in the first ##
## row.                                                                     ##
##                                                                          ##
## usage : python3 quantumcap.py [applied voltage in V] [twist in degrees]  ##
## ie when Vapp=0.1 and twist=1.1, type: python3 quantumCap.py 0.1 1.1      ##
##############################################################################

# note: dos.png printed after each function call to see a plot of the dos
# and the selected charge neutrality point

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
# will use simpson's rule to integrate
from scipy.integrate import simps
from scipy.optimize import curve_fit, least_squares
mpl.rcParams['lines.markersize'] = 2.5

##################################################################
################    global variables to change    ################
##################################################################

# Vapp stepsize
g_stepsize = 0.01

# if 0.5, then calculate for v_applied in range [-0.5, 0.5]
g_vrange = 0.5

# hemholtz capacitance
g_Ch = 10 # uF/cm2

# 1/kBT at RT, in eV-1
g_beta = 38.6

# controls how the charge neutrality point is selected from the dos
# for twists >= 1.11, 5 was ok. For 0.42 and 0.77 needed 100.
g_cnp_steepness = 100

# guess that Vh = Vapp * g_guess_vh_f to start, 1/2 seems to work well
# change if getting convergence issues
g_guess_vh_f = 0.5

# where the dos data is
g_dos_file = "calc_DOS.xlsx"

##################################################################
## read excel file with dos data                                ##
## @params                                                      ##
##   dos_file: string, name of excel sheet to read              ##
##   twist_angle: float, desired twist angle in degrees         ##
## @returns                                                     ##
##   dos: numpy array of dos, normalized                        ##
##   erange: numpy array of sampled energies in eV              ##
##################################################################
def import_dos(dos_file, twist_angle):
    print("reading dos from {}".format(dos_file))
    data = pd.read_excel(dos_file, index_col=None, header=None)
    data_np = data.to_numpy()
    headers = data_np[0,1:]
    erange = data_np[1:,0] #eV
    found_data = False
    for i in range(len(headers)):
        angle = float(headers[i].strip('°'))
        if np.abs(twist_angle - angle) < 0.01:
            print("Using {} in provided data".format(headers[i]))
            dos = data_np[1:,i+1]
            found_data = True
            break
    if not found_data:
        print("Error: twist angle of {} not in provided dataset".format(twist_angle))
        exit(0)
    dos, erange = center_dos(dos, erange, twist_angle)
    return dos, erange

##################################################################
## normalize w/ simpson integration                             ##
## @params                                                      ##
##   x: iterable of sampled points of y                         ##
##   y: iterable of data to normalize                           ##
## @returns                                                     ##
##   y_norm: numpy array of normalized y                        ##
##################################################################
def normalize(x,y):
    I = np.abs(simps(y, x))
    y_norm = np.array([yel/I for yel in y])
    return y_norm

##################################################################
## shifts so that the charge neutrality point is at 0           ##
## @params                                                      ##
##   dos: numpy array of dos, normalized                        ##
##   erange: numpy array of sampled energies in eV              ##
##   trunc: float, will truncate data to be in 0 +/- trunc eV   ##
## @returns                                                     ##
##   dos: np array of dos, normalized so in 1/eV, centered      ##
##   erange: np array of energies, eV, trucated & centered      ##
##################################################################
def center_dos(dos, erange, twist_angle):
    indx = np.argmin([d + (g_cnp_steepness*e)**2 for d,e in zip(dos, erange)])
    erange = np.array([e - erange[indx] for e in erange])
    f, ax = plt.subplots()
    ax.plot(erange,dos,'k')
    ax.set_xlabel('E (eV), origin at charge neutrality point')
    ax.set_ylabel('Normalized Density of States')
    ax.scatter(erange[indx],dos[indx],c='r')
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    #plt.show(); exit()
    plt.savefig("dos_twist_{}.png".format(twist_angle), dpi=600)
    return dos, erange

##################################################################
## get vh self consistently s.t. Vhemholtz+Vquantum=Vapplied    ##
## uses non-linear least squares and simpson integration        ##
## @params                                                      ##
##   Vapp: float, applied voltage in farads                     ##
##   twist_angle: float, twist angle                            ##
## @returns                                                     ##
##   Vh: float, hemholtz voltage in farads                      ##
##################################################################
def compute_Vh(Vapp, dos, erange):
    # fermi func and d(fermi function)/d(mu) for a given E, mu, beta(global)
    def fermif(mu, E):
        return 1/(1 + np.exp(g_beta * (E - mu)))
    # result has units of 1/E, in units of provided beta (here eV)
    def dfermi_dmu(mu, E):
        return g_beta * np.exp( g_beta * (E - mu))/( (1 + np.exp(g_beta * (E - mu)))**2 )

    # cost function to get Vh self consistently
    def _costfunc(Vh):
        Vq = Vapp - Vh[0] # Vapp = Vq + Vh, in Volts
        mu = Vq # mu in eV
        # fake_linear_dos = normalize(erange, [ abs(e) for e in erange ])
        integrand = [ dfermi_dmu(mu, erange[i]) * dos[i] for i in range(len(erange)) ] # 1/(eV)^2
        # Cq = (qe)^2 * integral of ( dos(E) * dfermi/du(mu=eVq, E) dE )
        integral = np.abs(simps(integrand, erange)) # in 1/(eV*nm2)
        integral = 16.022 * integral # in uF/cm2
        #print("**************** iterating ******************")
        #print("**** Ch*Vh is {:.12f}             ****".format(g_Ch*Vh[0]))
        #print("**** Cq*Vq is {:.12f}             ****".format(integral*Vq))
        #print("*********************************************\n")
        return (g_Ch*Vh[0] - integral*Vq)/abs(g_Ch*Vh[0])

    # nonlinear least squares
    guess_prms = g_guess_vh_f*Vapp
    opt = least_squares(_costfunc, guess_prms)
    Vh = opt.x
    return Vh[0]

def calculateVq(Vapp, dos, erange):
    if np.abs(Vapp) < 1e-5:
        return np.nan, np.nan, np.nan
    Vh = compute_Vh(Vapp, dos, erange) # volts
    Vq = Vapp - Vh # volts
    Cq = (g_Ch * Vh)/Vq
    print("**************** Finished! ******************")
    print("**** given Vapp = {:.4f} volts           ****".format(Vapp))
    print("**** given twist = {:.4f} degrees        ****".format(twist_angle))
    print("**** Vh = {:.8f} volts               ****".format(Vh))
    print("**** Vq = {:.8f} volts               ****".format(Vq))
    print("**** Cq = {:.8f} uF/cm^2             ****".format(Cq))
    print("*********************************************")
    return Vq, Vh, Cq

# write a file
def write(filenm, data):
    with open(filenm, 'w') as f:
        n = len(data.items())
        keys = list(data.keys())
        f.write('\t\t\t\t'.join(keys))
        f.write('\n')
        lsts = data.values()
        for i in range(len(data[keys[0]])):
            f.write('\t'.join([str(data[k][i]) for k in keys]))
            f.write('\n')

##################################################################
## command line interface                                       ##
## script inputs are Vapp and twist                             ##
##################################################################
if __name__ == "__main__":
    try:
        if len(sys.argv) == 3:
            Vapp = [float(sys.argv[1])]
            twist_angle = float(sys.argv[2])
            print("Calculating for Vapp of {} and twist angle of {} degrees ... ".format(sys.argv[1], sys.argv[2]))
        elif len(sys.argv) < 3:
            twist_angle = float(sys.argv[1])
            Vapp = np.arange(-g_vrange, g_vrange, g_stepsize)
            print("Calculating for twist angle of {} degrees in Vapp range [-{}V, {}V] ... ".format(sys.argv[1],g_vrange,g_vrange))
    except:
        print("You must give numeric values for the twist angle")
        print("For example when twist=1.1 to preform a series calculations using Vapp in [-0.5V,0.5V], type: \n\t\t python3 quantumCap.py 1.1")
        print("Alternatively you can specify a single Vapp, such as 0.1V, using \n\t\t python3 quantumCap.py 0.1 1.1")
        exit(0)

    # import
    dos, erange  = import_dos(g_dos_file, twist_angle) # erange in eV

    # preform calculation for each voltage and save results
    n_V = len(Vapp)
    vq_vec = np.zeros(n_V)
    vh_vec = np.zeros(n_V)
    cq_vec = np.zeros(n_V)
    for i in range(n_V):
        vq, vh, cq = calculateVq(Vapp[i], dos, erange)
        cq_vec[i] = cq
        vq_vec[i] = vq
        vh_vec[i] = vh

    # remove the nan I returned for V=0 since can't handle
    cq_vec = cq_vec[~np.isnan(cq_vec)]
    vq_vec = vq_vec[~np.isnan(vq_vec)]
    vh_vec = vh_vec[~np.isnan(vh_vec)]

    # write vq and cq
    data_dict = dict()
    data_dict['V_Q'] = vq_vec
    data_dict['C_Q'] = cq_vec
    data_dict['V_H'] = vh_vec
    write('data_twist_{}.txt'.format(twist_angle), data_dict)

    # plot vq vs cq
    f, ax = plt.subplots()
    ax.scatter(vq_vec, cq_vec, c='k')
    ax.set_xlabel('$V_Q (V)$')
    ax.set_ylabel('$C_Q (\mu F cm^{-2})$')
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.savefig("vq_vs_cq_twist_{}.png".format(twist_angle), dpi=600)

    # plot vq vs vh
    f, ax = plt.subplots()
    ax.scatter(vq_vec, [vh + vq for vh,vq in zip(vq_vec, vh_vec)], c='k')
    ax.set_xlabel('$V_Q (V)$')
    ax.set_ylabel('$V_{applied} (V)$')
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.savefig("vq_vs_vapp_twist_{}.png".format(twist_angle), dpi=600)
