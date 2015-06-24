#!/usr/bin/env python

import os, sys, glob, string

import scipy
from scipy.optimize import leastsq

import numpy as np

from RateSpecTools import *

def evalY_singleExp(v,x):
    yFit = v[0] + v[1]*np.exp(-1.0*x/v[2])
    return yFit

def evalY_doubleExp(v,x):
    yFit = v[0] + v[1]*np.exp(-1.0*x/v[3]) + v[2]*np.exp(-1.0*x/v[4]) 
    return yFit

def evalY_stretchedExp(v,x):
    yFit = v[0] + v[1]*np.exp(-1.0*(x/v[2])**min(1.0, v[3]))
    return yFit


def myCost_singleExp(v, x, y):
    yFit = evalY_singleExp(v,x)
    return y-yFit

def myCost_doubleExp(v, x, y):
    yFit = evalY_doubleExp(v,x)
    return y-yFit

def myCost_stretchedExp(v, x, y):
    yFit = evalY_stretchedExp(v,x)
    return y-yFit



def singleExpFit(data, v0):
    """Performs a single-exponential fit:  A + B*exp(-t/tau)).
    INPUT
    data - an N x 2 array containing time and values as columns
    v0   - initial guess [A, B, tau]

    RETURNS 
    v, yFit - best-fit parameters v and the yFit values 
    """

    #data = loadtxt(infile)

    nPoints = data.shape[0]

    x = data[:,0]
    y = data[:,1]

    v, success = leastsq(myCost_singleExp, v0, args=(x,y), maxfev=100000)
    yFit = evalY_singleExp(v,x)

    return v, yFit

def doubleExpFit(data, v0):
    """Performs a double-exponential fit:  A + B*exp(-t/tau1) + C*exp(-t/tau1)
    INPUT
    data - an N x 2 array containing time and values as columns
    v0   - initial guess [A, B, C, tau1, tau2]

    RETURNS 
    v, yFit - best-fit parameters v and the yFit values 
    """

    #data = loadtxt(infile)

    nPoints = data.shape[0]

    x = data[:,0]
    y = data[:,1]

    v, success = leastsq(myCost_doubleExp, v0, args=(x,y), maxfev=100000)
    yFit = evalY_doubleExp(v,x)

    return v, yFit

def stretchedExpFit(data, v0):
    """Performs a stretched-exponential fit:  A + B*exp(-(t/tau)**beta ) 
    INPUT
    data - an N x 2 array containing time and values as columns
    v0   - initial guess [A, B, tau, beta]

    RETURNS 
    v, yFit - best-fit parameters v and the yFit values 
    """

    #data = loadtxt(infile)

    nPoints = data.shape[0]

    x = data[:,0]
    y = data[:,1]

    v, success = leastsq(myCost_stretchedExp, v0, args=(x,y), maxfev=100000)
    yFit = evalY_stretchedExp(v,x)

    return v, yFit




def singleExpFit_Jackknife(data, v0, nSubsets):
    """Performs a single-exponential fit:  A + B*exp(-t/tau)), with
    error estimates from a jackknife with nSubsets

    INPUT
    data - an N x 2 array containing time and values as columns
    v0   - initial guess [A, B, tau]
    nSubsets - the number of subsets to use in the jackknife

    RETURNS 
    vmean, vstd, yFitMean - best-fit parameters v and the yFit values 
    """


    nPoints = data.shape[0]
    # print 'nPoints', nPoints

    # Make data subsets - fit to each
    dataSubsets = []
    Ind = [] 
    for c in range(nSubsets):
        i = c
        while i < nPoints:
          Ind.append(i)
          i += nSubsets 
        dataSubsets.append( data[Ind,:] )

    v_trials = []
    yFit_trials = []
    for trial in range(len(dataSubsets)):
        subdata = dataSubsets[trial]
        v, yFit = singleExpFit(subdata, v0)
        v_trials.append(v)
        yFit_trials.append(yFit)

    vmean = np.array(v_trials).mean(axis=0)
    vstd = np.array(v_trials).std(axis=0)
    yFitMean = evalY_singleExp(vmean, data[:,0])

    return vmean, vstd, yFitMean


def doubleExpFit_Jackknife(data, v0, nSubsets):
    """Performs a double-exponential fit:  A + B*exp(-t/tau1) + C*exp(-t/tau1), with
    error estimates from a jackknife with nSubsets

    INPUT
    data - an N x 2 array containing time and values as columns
    v0   - initial guess [A, B, C, tau1, tau1]
    nSubsets - the number of subsets to use in the jackknife

    RETURNS 
    vmean, vstd, yFitMean, yFitStd - best-fit parameters v and the yFit values 
    """


    nPoints = data.shape[0]
    # print 'nPoints', nPoints

    # Make data subsets - fit to each
    dataSubsets = []
    Ind = []
    for c in range(nSubsets):
        i = c
        while i < nPoints:
          Ind.append(i)
          i += nSubsets
        dataSubsets.append( data[Ind,:] )

    v_trials = []
    yFit_trials = []
    for trial in range(len(dataSubsets)):
        subdata = dataSubsets[trial]
        v, yFit = doubleExpFit(subdata, v0)
        v_trials.append(v)
        yFit_trials.append(yFit)

    vmean = np.array(v_trials).mean(axis=0)
    vstd = np.array(v_trials).std(axis=0)
    yFitMean = evalY_doubleExp(vmean, data[:,0])

    return vmean, vstd, yFitMean

