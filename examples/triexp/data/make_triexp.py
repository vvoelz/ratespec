#!/usr/bin/env python

import os, sys, glob
sys.path.append('../../')

import scipy
from scipy.linalg import pinv

import numpy as np

import matplotlib
from pylab import *

from RateSpecTools import *

sigmas = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2]

LinearSpacing = True  # use a linear-spaced time axis, closer to what might obtained in an experiment

for s in sigmas:

    # make a tri-exponential test data set
  
    Times, Data = testData(ntimes=10000, sigma = s, linear=LinearSpacing)

    # write to file
    if LinearSpacing:
        Times, Data = testData(ntimes=10000, sigma = s, linear=LinearSpacing)
        outfile = 'trilinear.sigma%3.4f.dat'%s
    else:
        Times, Data = testData(sigma = s, linear=LinearSpacing)
        outfile = 'triexp.sigma%3.4f.dat'%s

    print 'Writing to', outfile, '...'
    fout = open(outfile,'w')
    # write header
    fout.write('#time(us)\tvalue\n')
    for t in range(Times.shape[0]):
        fout.write('%e\t%e\n'%(Times[t]*1.0e6,Data[t]))
    fout.close()
 

