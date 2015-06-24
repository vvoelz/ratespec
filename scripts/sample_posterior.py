#!/usr/bin/env python

import os, sys, glob
sys.path.append('../')

import scipy
from scipy.linalg import pinv

import numpy as np

#import matplotlib
#from pylab import *

from RateSpecClass import *
from RateSpecTools import *

UsePlotting = True
try:
    from PlottingTools import *
except:
    UsePlotting = False


usage = """Usage: sample_posterior.py datfile nsteps nRates timeUnit minrate maxrate outname

    Will sample the posterior distribution of rate spectra (exp mixture) models
 
    datfile -   time series   
    nsteps -    number of steps to perform MC sampling
    nRates -    number of rates (exp-spaced) to use in the spectrum
    timeUnit -  the unit (in sec) of the time axis of the data
    minrate -  the minimum rate (in 1/s) to consider
    maxrate -  the maximum rate to consider
    outname -   Will write files:

      'outname.nRates*.L*.ratespec.dat'
      'outname.nRates*.L*.mctraj.dat'
      'outname.nRates*.L*.Atraj.dat'

    FLAGS 
    -s          Standarize the data
    -t          Scale the data
    -lasso      Use L1-norm regularization (the lasso)
    -enet [rho] Use a combination L1- an L2-norm regularization (the ElasticNet), with mixture 0 < rho < 1.


    Example: >> ./sample_posterior.py data/triexp.sigma0.025.dat 100 100 1.0e-6 1.0 1.e9 triexp.sigma0.025 -t
    Example: >> ./sample_posterior.py data/triexp.sigma0.050.dat 100 100 1.0e-6 1.0 1.e9 triexp.sigma0.050 -t -enet 0.5
""" 
    
if len(sys.argv) < 8:
    print usage
    sys.exit(1)

infile = sys.argv[1]
MCsteps = int(sys.argv[2])
nRates = int(sys.argv[3])
timeUnit = float(sys.argv[4])
minrate = float(sys.argv[5])
maxrate = float(sys.argv[6])
outname = sys.argv[7]

UseLnorm = 'ridge'
if sys.argv.count('-lasso') > 0:
    UseLnorm = 'lasso'
if sys.argv.count('-enet') > 0:
    UseLnorm = 'enet'
    UseRho = float(sys.argv[ sys.argv.index('-enet') + 1] )
else:
    UseRho = 0.5

Standardize = False
if sys.argv.count('-s') > 0:
    Standardize = True

Scale = False
if sys.argv.count('-t') > 0:
    Scale = True

# instantiate a RateSpecClass object
if UseLnorm == 'lasso':
    # a good starting sigma and tau are: sigma=noise, tau = 10*noise, such that (sigma/tau)^2 = 0.01
    r = RateSpecClass(dataFile=infile, timeUnit=timeUnit, nRates=nRates, RateRange=(minrate, maxrate), Lnorm=UseLnorm, scaleData=Scale, standardizeData=Standardize, init_sigma=0.1, init_tau=1.0, dsigma=1.0e-2, dtau=1.0e-2) 
if UseLnorm == 'enet':
    # a good starting sigma and tau are: sigma=noise, tau = 10*noise, such that (sigma/tau)^2 = 0.01
    r = RateSpecClass(dataFile=infile, timeUnit=timeUnit, nRates=nRates, RateRange=(minrate, maxrate), Lnorm=UseLnorm, scaleData=Scale, standardizeData=Standardize, init_sigma=0.1, init_tau=1.0, init_rho=UseRho, dsigma=5.0e-2, dtau=5.0e-2)
else:
    r = RateSpecClass(dataFile=infile, timeUnit=timeUnit, nRates=nRates, RateRange=(minrate, maxrate), Lnorm=UseLnorm, scaleData=Scale, standardizeData=Standardize, init_sigma=0.03, init_tau=0.05)

# sample the posterior distribution of possible rate spectra
r.sampleSpectrum(nsteps=MCsteps)  

ratespec_filename = outname + '.nRates%d.L%s.ratespec.dat'%(nRates, UseLnorm) 
mctraj_filename = outname + '.nRates%d.L%s.mctraj.dat'%(nRates, UseLnorm)
Atraj_filename = outname + '.nRates%d.L%s.Atraj.dat'%(nRates, UseLnorm)

r.writeSpectrum(ratespec_filename)
r.writeMCResults(mctraj_filename, Asample_filename=Atraj_filename)


# Calculate the spectrum with the current value of lambda (w)
w = (r.sigma/r.tau)**2
if UseLnorm == 'enet':
    A, rss, ndof = fitRateSpectrum(r.Times, r.Data, r.Rates, w, Lnorm=r.Lnorm, standardizeData=Standardize, rho=r.rho)
else:
    A, rss, ndof = fitRateSpectrum(r.Times, r.Data, r.Rates, w, Lnorm=r.Lnorm, standardizeData=Standardize)

print 'A, rss, ndof', A, rss, ndof

# Calculate the fitted time trace
if Standardize:
    print 'r.offset', r.offset
    FitData = SumSpectra(A, r.Rates, r.Times, offset = r.offset)
else:
    FitData = SumSpectra(A, r.Rates, r.Times)
print 'FitData', FitData

# make a plot
if (0):
  figure()

  subplot(2,1,1)
  TimeSeriesWithFit(r.Times, r.Data, FitData)

  subplot(2,1,2)
  TimescaleSpectrum(r.Timescales, A, timeunit='s')

  show()


if UsePlotting:
    outpdf = outname + '.nRates%d.L%s.pdf'%(nRates, UseLnorm)
    plotResults(r, outpdffile=outpdf, showPlot=True)



