#!/usr/bin/env python

import os, sys, glob

import scipy
from scipy.linalg import pinv

import numpy as np

import matplotlib
from pylab import *

from RateSpecClass import *
from RateSpecTools import *

UsePlotting = False
try:
    from PlottingTools import *
    UsePlotting = True
except:
    print 'Cannot load the plotting libraries.   Continuing without graphical output...'


usage = """Usage:  ./compute_spectrum.py datafile nRates lambda timeunit [minrate maxrate]

    Will compute the rate spectrum of the data specified in the input file, using the supplied
    parameters.  The spectrum rates are exponentially-spaced (linear on a log-scale) by default.

    The input data is standardized by defult as well.

    If possible, a graph of the resulting spectrum will be displayed.

    Example: >> ./compute_spectrum.py data/triexp.sigma0.050.dat 100 2.0 1.0e-6 1.0 1.e9 -t 

    REQUIRED
    datafile - a two-column data file (time
    nRates  - number of rates (log-spaced)
    lambda  - the regularization parameter
    timeunits   - the units of the time value in the datafile (in seconds: i.e. microseconds -> 1.0e-6)

    OPTIONAL
    minrate, maxrate  - the min and max rates (in 1/s) (log-spaced).  Default: 1.0 1.e9

    FLAGS 
    -t          Scale the data
    -s          Standarize the data
    -lasso      Use L1-norm regularization (the Lasso) 
    -enet [rho] Use a combination L1- an L2-norm regularization (the ElasticNet), with mixture 0 < rho < 1.

    OUTPUT
    Will write output files:  [datafilename].ratespec.dat  [datafilename].timetrace.dat
    
"""

if len(sys.argv) >= 5:
    datafile = sys.argv[1]
    nRates = int(sys.argv[2])
    w = float(sys.argv[3])
    timeunit = float(sys.argv[4])
else:
    print usage
    sys.exit(1)

if len(sys.argv) >= 7:
    minrate, maxrate = float(sys.argv[5]), float(sys.argv[6])
else:
    minrate, maxrate = 1.0, 1.e9

Standardize = False
if sys.argv.count('-s') > 0:
    Standardize = True 

Scale = False
if sys.argv.count('-t') > 0:
    Scale = True

UseLnorm = 'ridge'
if sys.argv.count('-lasso') > 0:
    UseLnorm = 'lasso'
if sys.argv.count('-enet') > 0:
    UseLnorm = 'enet'
    UseRho = float(sys.argv[ sys.argv.index('-enet') + 1] )
 

# Create a rate spectrum class object
r = RateSpecClass(dataFile=datafile, timeUnit=timeunit, nRates=nRates, RateRange=(minrate, maxrate), standardizeData=Standardize, scaleData=Scale, Lnorm=UseLnorm )

# Calculate the spectrum
if UseLnorm == 'enet':
    A, rss, ndof = fitRateSpectrum(r.Times, r.Data, r.Rates, w, Lnorm=r.Lnorm, standardizeData=Standardize, rho=UseRho)
else:
    A, rss, ndof = fitRateSpectrum(r.Times, r.Data, r.Rates, w, Lnorm=r.Lnorm, standardizeData=Standardize)
print 'A, rss, ndof', A, rss, ndof

# Write the spectrum to file
outspecfile = datafile.replace('.dat','.ratespec.dat')
fout = open(outspecfile,'w')
fout.write('#timescale(s)\tamplitude\n')
for i in range( len(r.Timescales)):
    fout.write('%e\t%e\n'%(r.Timescales[i], A[i]))
fout.close()

# Calculate the fitted time trace
if Standardize:
    print 'r.offset', r.offset
    FitData = SumSpectra(A, r.Rates, r.Times, offset = r.offset)
else:
    FitData = SumSpectra(A, r.Rates, r.Times)
print 'FitData', FitData


# Write the predicted time trace to file
outtracefile = datafile.replace('.dat','.timetrace.dat')
fout = open(outtracefile,'w')
fout.write('#time(s)\tvalue\n')
for i in range( len(r.Times)):
    fout.write('%e\t%e\n'%(r.Times[i], FitData[i]))
fout.close()


# make a plot
if UsePlotting:
  figure()

  subplot(2,1,1)
  TimeSeriesWithFit(r.Times, r.Data, FitData)

  subplot(2,1,2)
  TimescaleSpectrum(r.Timescales, A, timeunit='s', UseStems=True)

  show()

