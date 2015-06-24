#!/usr/bin/env python

import os, sys, glob
sys.path.append('../')

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



def analyticSpectrum(Rates, beta, tau=1.0, du = 0.01, maxval=100.0):
    """Returns an "analytic" solution of the rate spectrum as outlined in Berberan-Santos et al. (ChemPhys 2005):
    
    H(k) = tau0/\pi * \int_0^\infty exp(-k tau0 u) * exp(-u^beta cos(beta*pi)) * sin( u^beta sin(beta*pi) )
    """

    u = np.arange(0., maxval, du)

    print 'Summing over', len(u), 'values for beta =', beta, '...'
    print 'du', du

    A = []

    for k in Rates:
        if k >= 1.0:
            H_k = tau/np.pi * du * np.sum( np.exp( -1.0*k*tau*u)*np.exp( -1.0*(u**beta)*np.cos(beta*np.pi))*np.sin((u**beta) * np.sin(beta*np.pi)) )
        else:
            H_k = tau/np.pi * du * np.sum( np.exp( -1.0*(u**beta)*np.cos(beta*np.pi/2.0))*np.cos((u**beta) * np.sin(beta*np.pi/2.0) - k*tau*u) )

        A.append(H_k)

    return np.array(A)




def addAxes(pos):
    """Add axes according the pos list, and return  the axes handle."""
    rect = pos[0]+margins[0], pos[1]+margins[2], pos[2]-margins[0]-margins[1], pos[3]-margins[2]-margins[3]
    return fig.add_axes(rect)


# Default plotting parameters
if (1):
    plt.rc('figure', figsize=(6.5, 6.5))  # in inches
    plt.rc('figure.subplot', left=0.125, right=0.9, bottom=0.1, top=0.90)
    plt.rc('lines', linewidth=0.5, markersize=2)
    plt.rc('font', size=8.0)
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')
    plt.rc('legend', fontsize='medium')

# Define workspace for all panels
panelpos = []
# top four panels
panelpos.append( [0,    0.75, 0.25, 0.25] ) # bottom left corner x, y; width, height
panelpos.append( [0.25, 0.75, 0.25, 0.25] ) # bottom left corner x, y; width, height
panelpos.append( [0.5,  0.75, 0.25, 0.25] ) # bottom left corner x, y; width, height
panelpos.append( [0.75, 0.75, 0.25, 0.25] ) # bottom left corner x, y; width, height

# top four panels
panelpos.append( [0,    0.5, 0.25, 0.25] ) # bottom left corner x, y; width, height
panelpos.append( [0.25, 0.5, 0.25, 0.25] ) # bottom left corner x, y; width, height
panelpos.append( [0.5,  0.5, 0.25, 0.25] ) # bottom left corner x, y; width, height
panelpos.append( [0.75, 0.5, 0.25, 0.25] ) # bottom left corner x, y; width, height
# top four panels
panelpos.append( [0,    0.25, 0.25, 0.25] ) # bottom left corner x, y; width, height
panelpos.append( [0.25, 0.25, 0.25, 0.25] ) # bottom left corner x, y; width, height
panelpos.append( [0.5,  0.25, 0.25, 0.25] ) # bottom left corner x, y; width, height
panelpos.append( [0.75, 0.25, 0.25, 0.25] ) # bottom left corner x, y; width, height

# bottom four panels
panelpos.append( [0,    0.0, 0.25, 0.25] ) # bottom left corner x, y; width, height
panelpos.append( [0.25, 0.0, 0.25, 0.25] ) # bottom left corner x, y; width, height
panelpos.append( [0.5,  0.0, 0.25, 0.25] ) # bottom left corner x, y; width, height
panelpos.append( [0.75, 0.0, 0.25, 0.25] ) # bottom left corner x, y; width, height


# define margins within each panel 
margins = [0.06, 0.02, 0.07, 0.02]  # for each panel (left, right, bottom, top)

fig = plt.figure(1)
plt.clf()

#### Filenames ###

outpdffile = 'Figure3.pdf'


sigma = 0.050
betas = [0.3, 0.5, 0.7, 0.9]
priors = ['Lridge', 'Llasso', 'Lenet']

for j in range(len(priors)):

  prior = priors[j]

  ratespec_files = ['data/stretched.sigma0.0500.beta%3.2f.dat.nRates100.%s.ratespec.dat'%(beta,prior) for beta in betas]
  mctraj_files = ['data/stretched.sigma0.0500.beta%3.2f.dat.nRates100.%s.mctraj.dat'%(beta,prior) for beta in betas]
  dat_files = ['data/stretched.sigma0.0500.beta%3.2f.dat'%beta for beta in betas]

  for i in range(len(ratespec_files)):

   
   ratespec_filename = ratespec_files[i]
   mctraj_filename = mctraj_files[i]

   if (prior == 'Lenet') and (betas[i] == 0.5):
       ratespec_filename = ratespec_filename.replace('0.30', '0.30.noscale').replace('0.50', '0.50.noscale')
       mctraj_filename = mctraj_filename.replace('0.30', '0.30.noscale').replace('0.50', '0.50.noscale')
       
   try:

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


    # plot timecourse + with noise
    if j == 0:
      ax = addAxes(panelpos[i])
      plt.plot(Times, Values, 'b.', markersize=2)
      hold(True)
      Rates = rangeLog(1.0e-3, 1.e3, 100)
      FitData = SumSpectra(meanA, Rates, Times)
      plt.plot(Times, FitData, 'k-')
      hold(True)
      ax.set_xscale('log')
      plt.axis([min(Times), max(Times), -0.2, 1.2])
      xlabel('time (s)')
      #ax.set_xticks([1.0e-3, 1.0, 1.0e-3])


    # plot mean +/- std spectrum
    ax = addAxes(panelpos[i+(j+1)*len(betas)])
    #matplotlib.pyplot.errorbar(Timescales, meanA, yerr=stdA)
    PlotStd = False
    plot(Timescales, meanA, 'k-', linewidth=1)
    hold(True)
    if PlotStd:
        plot(Timescales, meanA+stdA, 'k-', linewidth=0.5)
        hold(True)
        plot(Timescales, meanA-stdA, 'k-', linewidth=0.5)
    else:
        plot(Timescales, ci_5pc, 'k-', linewidth=0.5)
        hold(True)
        plot(Timescales, ci_95pc, 'k-', linewidth=0.5)
    ax.set_xscale('log')
    xlabel('timescale (s)')
    ax.set_xticks([1.0e-9, 1.0e-6, 1.0e-3, 1.])

    hold(True)
    Rates = 1./Timescales
    A_anal = analyticSpectrum(Rates, betas[i], tau=1.0, du=0.01, maxval=1000.0)
    dRates = Rates[1:] - Rates[0:-1]
    A_anal = A_anal[0:-1]*dRates

    TimescaleSpectrum(Timescales[0:-1], A_anal, timeunit='s', linestyle='r-')  # plot as a function of rates
    #ax.set_xscale('linear')
    #axis([0, 1, A_anal.min(), A_anal.max()])

   except: 
      print 'Problems with prior', prior, 'ratespec_filename', ratespec_filename




plt.savefig(outpdffile, format='pdf')

#show()



