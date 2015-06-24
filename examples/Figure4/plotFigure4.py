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
from CurveFitting import *

# For nice plots :)
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def addAxes(pos, NixTopMargin=False, NixBottomMargin=False):
    """Add axes according the pos list, and return  the axes handle.
    # margin indices 0,1,2,3 for each panel (left, right, bottom, top)
    """
    rect = pos[0]+margins[0], pos[1]+margins[2], pos[2]-margins[0]-margins[1], pos[3]-margins[2]-margins[3]
    return fig.add_axes(rect)


# Default plotting parameters
if (1):
    plt.rc('figure', figsize=(3.0, 5.0))  # in inches
    plt.rc('figure.subplot', left=0.125, right=0.9, bottom=0.1, top=0.90)
    plt.rc('lines', linewidth=1, markersize=2)
    plt.rc('font', size=8.0)
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')
    plt.rc('legend', fontsize='medium')

# Define workspace for all panels
panelpos = []
# top three panels
panelpos.append( [0,   0.6, 0.5, 0.4] ) # bottom left corner x, y; width, height
panelpos.append( [0.5, 0.6, 0.5, 0.4] ) # bottom left corner x, y; width, height

# middle top three panels
panelpos.append( [0,   0.4, 0.5, 0.2] ) # bottom left corner x, y; width, height
panelpos.append( [0.5, 0.4, 0.5, 0.2] ) # bottom left corner x, y; width, height

# middle bottom three panels
panelpos.append( [0,   0.25, 0.5, 0.15] ) # bottom left corner x, y; width, height
panelpos.append( [0.5, 0.25, 0.5, 0.15] ) # bottom left corner x, y; width, height

# middle bottom three panels
panelpos.append( [0,   0, 0.5, 0.25] ) # bottom left corner x, y; width, height
panelpos.append( [0.5, 0, 0.5, 0.25] ) # bottom left corner x, y; width, height



# define margins within each panel 
base_margins = [0.08, 0.02, 0.08, 0.01 ]  # for each panel (left, right, bottom, top)
alt_margins = [0.08, 0.02, 0.0, 0.0 ]
alt2_margins = [0.08, 0.02, 0.08, 0.0 ]


margins = base_margins

fig = plt.figure(1)
plt.clf()

#### Filenames ###

outpdffile = 'Figure4.pdf'

priors = ['ridge', 'lasso', 'enet']
setnames = ['upper', 'lower']
for j in range(len(priors)):

  prior = priors[j] 

  if j==0:
      ratespec_files = ['data/Fig3%sWW.ridge.nRates100.L%s.ratespec.dat'%(s,prior) for s in setnames]
      mctraj_files = ['data/Fig3%sWW.ridge.nRates100.L%s.mctraj.dat'%(s,prior) for s in setnames]
  else:
      ratespec_files = ['data/Fig3%sWW.%s.noscale.nRates100.L%s.ratespec.dat'%(s,prior,prior) for s in setnames]
      mctraj_files = ['data/Fig3%sWW.%s.noscale.nRates100.L%s.mctraj.dat'%(s, prior, prior) for s in setnames]

  dat_files = ['data/Fig3%sWW.dat'%s for s in setnames]

  for i in range(len(ratespec_files)):

   try:

    margins = base_margins

    ratespec_filename = ratespec_files[i]
    mctraj_filename = mctraj_files[i]

    data = scipy.loadtxt(dat_files[i])
    Times = data[:,0]*1.0e-6  # convert from us to seconds
    Values, Scaleby, Shiftby  = scaleValuesWithInfo(data[:,1])

    mctraj_data = scipy.loadtxt(mctraj_filename)  # step     w       sigma   tau     neglogP
    ratespec_data = scipy.loadtxt(ratespec_filename)

    Timescales = ratespec_data[:,0]
    maxLikA = ratespec_data[:,1]
    meanA = ratespec_data[:,2]
    stdA = ratespec_data[:,3]
    ci_5pc = ratespec_data[:,4]
    ci_95pc = ratespec_data[:,5]


    # plot timecourse + with noise
    if j == 0:
        ax = addAxes(panelpos[i])
        plt.plot(Times, Values*Scaleby+Shiftby, 'b.', markersize=0.5)
        hold(True)
        Rates = 1./Timescales
        FitData = SumSpectra(meanA, Rates, Times)
        plt.plot(Times, FitData*Scaleby+Shiftby, 'k-', markersize=0.5)
        hold(True)

        # plot a residual above the curve fits
        v0 = [0.0, 1.0, 1.0]  # time in us here
        # omit the first nskip data points
        nskip = 0
        v, yFit = singleExpFit(data[nskip:,0:2], v0)  # v0 is initial guess [A, B, tau] for A + B*exp(-t/tau)
        plt.plot(Times[nskip:], yFit, 'r-')
        hold(True)

        # plot a residual above the curve fits
        residuals = data[nskip:,1] - yFit
        h = 1.5
        mag = 2.  # magnify residuals by this factor
        plt.plot(Times[nskip:], h+mag*residuals, 'r-', linewidth=0.5)
        hold(True)
        plt.plot([0.2*1e-6, 100*1e-6], [h,h], 'k-', linewidth=0.5)
        hold(True)


        ax.set_xscale('log')
        plt.axis([1e-7, 1e-3, -0.2, 2.0])
        xlabel('time (s)')
        ax.set_xticks([1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

        

    # plot mean +/- std spectrum
    if j < 2:
        margins = alt_margins
    else:
        margins = alt2_margins
    ax = addAxes(panelpos[i+(j+1)*len(ratespec_files)])
    #matplotlib.pyplot.errorbar(Timescales, meanA, yerr=stdA)
    plot(Timescales, meanA*Scaleby, 'k-', linewidth=1)
    hold(True)
    plot(Timescales, ci_5pc*Scaleby, 'k-', linewidth=0.5)
    hold(True)
    plot(Timescales, ci_95pc*Scaleby, 'k-', linewidth=0.5)
    ax.set_xscale('log')
    if j==0:
        plt.axis([1e-8, 1e-2, -0.035, 0.15])
    elif j==1:
        plt.axis([1e-8, 1e-2, -0.2, 1.2])
    else:
        plt.axis([1e-8, 1e-2, -0.02, 0.15])
    xlabel('timescale (s)')
    if j==1:
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    if j==2:
        ax.set_yticks([0, 0.04, 0.08, 0.12])
    ax.set_xticks([1.0e-7, 1.0e-6, 1.0e-5, 1e-4, 1e-3])

   except:
       print 'Problems with file', ratespec_files[i]

  

plt.savefig(outpdffile, format='pdf')

#show()



