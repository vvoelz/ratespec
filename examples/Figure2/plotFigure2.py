#!/usr/bin/env python

import os, sys, glob
sys.path.append('/Users/vincentvoelz/scripts/ratespec')

import scipy
from scipy.linalg import pinv

import numpy as np

import matplotlib
from pylab import *

from RateSpecClass import *
from RateSpecTools import *
from PlottingTools import *

# For nice plots :)
matplotlib.use('Agg')
import matplotlib.pyplot as plt




def addAxes(pos):
    """Add axes according the pos list, and return  the axes handle."""
    rect = pos[0]+margins[0], pos[1]+margins[2], pos[2]-margins[0]-margins[1], pos[3]-margins[2]-margins[3]
    return fig.add_axes(rect)


# Default plotting parameters
if (1):
    plt.rc('figure', figsize=(4.0, 3.5))  # in inches
    plt.rc('figure.subplot', left=0.125, right=0.9, bottom=0.1, top=0.90)
    plt.rc('lines', linewidth=1.5, markersize=5)
    plt.rc('font', size=8.0)
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')
    plt.rc('legend', fontsize='medium')

# Define workspace for all panels
panelpos = []
# top two panels
panelpos.append( [0,   0.5, 0.5, 0.5] ) # bottom left corner x, y; width, height
panelpos.append( [0.5, 0.5, 0.5, 0.5] ) # bottom left corner x, y; width, height

# bottom two panels
panelpos.append( [0,   0, 0.5, 0.5] ) # bottom left corner x, y; width, height
panelpos.append( [0.5, 0, 0.5, 0.5] ) # bottom left corner x, y; width, height


# define margins within each panel 
margins = [0.1, 0.05, 0.1, 0.02 ]  # for each panel (left, right, bottom, top)

fig = plt.figure(1)
plt.clf()

#### Filenames ###

outpdffile = 'Figure2.pdf'

tags = ['Lridge', 'Llasso', 'Lenet']
ratespec_files = ['data/triexp.sigma0.050.nRates100.%s.ratespec.dat'%tag for tag in tags]
mctraj_files = ['data/triexp.sigma0.050.nRates100.%s.mctraj.dat'%tag for tag in tags]
dat_files = ['data/triexp.sigma0.050.dat' for tag in tags]

for i in range(len(ratespec_files)):

    ratespec_filename = ratespec_files[i]
    mctraj_filename = mctraj_files[i]

    data = scipy.loadtxt(dat_files[i])
    Times = data[:,0]*1.0e-6  # convert from us to seconds
    Values  = data[:,1]

    mctraj_data = scipy.loadtxt(mctraj_filename)  # step     w       sigma   tau     neglogP
    ratespec_data = scipy.loadtxt(ratespec_filename)

    Timescales = ratespec_data[:,0]
    maxLikA = ratespec_data[:,1]
    meanA = ratespec_data[:,2]
    stdA = ratespec_data[:,3]
    ci_5pc = ratespec_data[:,4]
    ci_95pc = ratespec_data[:,5]

    # plot mean +/- std spectrum
    ax = addAxes(panelpos[i])
    #matplotlib.pyplot.errorbar(Timescales, meanA, yerr=stdA)
    PlotStd = False
    plot(Timescales, meanA, 'k-', linewidth=2)
    hold(True)
    if PlotStd:
        plot(Timescales, meanA+stdA, 'k-', linewidth=1)
        hold(True)
        plot(Timescales, meanA-stdA, 'k-', linewidth=1)
    else:
        plot(Timescales, ci_5pc, 'k-', linewidth=1)
        hold(True)
        plot(Timescales, ci_95pc, 'k-', linewidth=1)
    ax.set_xscale('log')
    xlabel('timescale (s)')
    ax.set_xticks([1.0e-9, 1.0e-6, 1.0e-3, 1.])


if (1):
    # plot distribution of sigma 
    ax = addAxes(panelpos[3])
    nskip = 1000
    rhocounts, rhobins = np.histogram(mctraj_data[nskip:,4], bins=np.arange(0, 1.025, 0.025))
    rhocounts = rhocounts/float(rhocounts.max())
    plt.plot(rhobins, [0]+rhocounts.tolist(), 'b-', linestyle='steps', linewidth=1)
    hold(True)
    plt.axis([0, 1, -0.01, 1.2])
    xlabel('$\\rho$')
    ylabel('$P(\\rho)$')
    ax.set_yticks([])



plt.savefig(outpdffile, format='pdf')

#show()



