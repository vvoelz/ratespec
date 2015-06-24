#!/usr/bin/env python

import os, sys, glob

import scipy
from scipy.linalg import pinv

import numpy as np

import matplotlib
from pylab import *


def rangeLog(min, max, n):
    """Return an array of n log-spaces values from min to max.
    NOTE: All values must be greater than 0"""

    logmin, logmax = log(min), log(max)
    return np.exp( np.arange( logmin, logmax, (logmax-logmin)/float(n) ) )

def rangeLin(min, max, n):
    """Return an array of n linear-spaced values from min to max."""

    return  np.arange( min, max, (max-min)/n )


def Xmatrix(k,t,w, standardizeData=True):
    """Return X matrix Xij = [ exp(-k_j*t_i); I*w^(1/2) ] from arrays k and t. """
   
    K, N = len(k), len(t)
    X = np.zeros( (N+K,K) )
    for i in range(N):
      for j in range(K):
        X[i,j] = np.exp(-1.0*k[j]*t[i])
    for i in range(K):
        X[i+len(t),i] = w**0.5

    Xmean = (X[0:N,:]).mean(axis=0) 
    if standardizeData:
      for j in range(K):
        X[0:N,j] = X[0:N,j] - Xmean[j]

    return X, Xmean


def Xsubmatrix(k,t, standardizeData=True):
    """Return X matrix Xij = [ exp(-k_j*t_i)] from arrays k and t. """

    K, N = len(k), len(t)
    X = np.zeros( (N,K) )
    for i in range(N):
      for j in range(K):
        X[i,j] = np.exp(-1.0*k[j]*t[i])

    Xmean = X.mean(axis=0)
    if standardizeData:
      for j in range(K):
        X[:,j] = X[:,j] - Xmean[j]

    return X, Xmean



def SumSpectra(A, Rates, Times, offset=0.0):
    """Return the sum of the exponential relaxations.
    If the data is standardized, an offset constant zero rate must be provided."""

    #print '*** in SumSpectra: ***'
    result = np.zeros( Times.shape )
    for i in range(len(Rates)):
        #print '***', Rates[i]
        result += A[i]*np.exp( -1.0*Rates[i]*Times) 
    return result 



def testData(nTimes = 1000, taus = [1.0e-6, 1.0e-4, 5.0e-3], amps = [0.3, 0.3, 0.4], sigma = 0.05, linear=False):
    """
    nTimes = 727 # same number as W55F dataset
    taus = [1.0e-6, 1.0e-4, 5.0e-3] # timescales for a test data curve
    amps = [0.3, 0.3, 0.4]  # ampltudes of each relaxation
    sigma = 0.05   # add artifical noise to the data
    """

    if linear:
        Times = rangeLin(1.0e-7, 1.0e-3, nTimes) # in seconds
    else:
        Times = rangeLog(1.0e-9, 1.0, nTimes) # in seconds

    Data = np.zeros( Times.shape )
    for i in range(len(taus)):
        Data += amps[i]*np.exp(-1.*Times/taus[i])
    Data += sigma*np.random.randn( len(Data) )

    return Times, Data

def testStretchedExpData(nTimes = 1000, beta = 0.5, sigma = 0.05, linear=False):
    """
    nTimes = 727 # same number as W55F dataset
    beta = the stretching exponenent (should be between 0 and 1)
    sigma = 0.05   # add artifical noise to the data
    """

    if linear:
        Times = rangeLin(1.0e-3, 1.0e+3, nTimes) # in seconds
    else:
        Times = rangeLog(1.0e-3, 1.0e+3, nTimes) # in seconds

    Data = np.exp(-1.*(Times**beta))
    Data += sigma*np.random.randn( len(Data) )

    return Times, Data


def fitRateSpectrum(Times, Data, Rates, w, Lnorm='ridge', standardizeData=True, CalcNdof=False, rho=0.5):
    """Using pseudo-inverse, with Tikhonov regularization (w parameter) to solve the inverse lapace tranform.
    Returns coefficients A_k, residual sum of squares (rss), and number of degrees of freedom, for each relaxation rate.
    """

    
    if Lnorm == 'lasso':
        # Use L1-norm Lasso regression
        try:
            from scikits.learn.linear_model import Lasso 
        except:
            print 'Error: could NOT import Lasso from scikits.learn.linear_model.  Using L2 norm (ridge).'
            Lnorm = 'ridge'

    if Lnorm == 'enet':
        # Use L1-L2-mixture norm Lasso regression
        try:
            from scikits.learn.linear_model import ElasticNet
        except:
            print 'Error: could NOT import ElasticNet from scikits.learn.linear_model.  Using L2 norm (ridge).'
            Lnorm = 'ridge'


    if Lnorm == 'lasso':

        lasso = Lasso(alpha = w, fit_intercept=False) # assume the data is already "centered" -- i.e. no zero rate
        X, Xmean = Xsubmatrix(Rates, Times, standardizeData=standardizeData)
        #print 'X.shape', X.shape, 'Data.shape', Data.shape
        lasso.fit(X, Data, max_iter=1e6, tol=1e-7)
        A = lasso.coef_

        # Compute "residual sum of squares" (note loss function is different for L1-norm)
        y_pred_lasso = lasso.predict(X)
        diff = y_pred_lasso - Data


    elif Lnorm == 'enet':

        # NOTE: The convention for rho is backwards in scikits.learn, instead of rho we must send (1-rho)
        enet = ElasticNet(alpha = w, rho=(1.-rho), fit_intercept=False) # assume the data is already "centered" -- i.e. no zero rate
        X, Xmean = Xsubmatrix(Rates, Times, standardizeData=standardizeData)
        #print 'X.shape', X.shape, 'Data.shape', Data.shape
        #enet.fit(X, Data, max_iter=1e6, tol=1e-7)
        enet.fit(X, Data, max_iter=1e6, tol=1e-3)  # for testing
        A = enet.coef_

        # Compute "residual sum of squares" (note loss function is different for L1-norm)
        y_pred_enet = enet.predict(X)
        diff = y_pred_enet - Data


    elif Lnorm == 'ridge':
        X, Xmean = Xmatrix(Rates, Times, w, standardizeData=standardizeData )
        Xinv = linalg.pinv(X)

        y = np.array( Data.tolist() + [0. for k in Rates] )
        if standardizeData:
            y - y.mean()
        A = np.dot(Xinv, y)

        # Compute "residual sum of squares" (note loss function is different for L1-norm)
        diff = SumSpectra(A, Rates, Times) - Data

    rss = np.dot(diff,diff)  # Residual sum of squares

    if CalcNdof:
        Xsub, Xmean = Xsubmatrix(Rates, Times, standardizeData=standardizeData)
        XT = np.transpose(Xsub)
        I_XT = np.eye(XT.shape[0])
        I_X = np.eye(Xsub.shape[0])
        Xtemp = np.dot(Xsub, np.linalg.inv(np.dot(XT,Xsub) + w*I_XT))
        ndof = np.trace(I_X - np.dot(Xtemp,XT))
    else:
        ndof = None

    return A, rss, ndof



def scaleValues(values):
    """Scale a numpy array of values so that (min, max) = (0,1)."""

    values = values - values.min()
    return values/values.max()

def scaleValuesWithInfo(values):
    """Scale a numpy array of values so that (min, max) = (0,1).
    Returns:
    scaled values
    Scaleby
    Shiftby   -- i.e. to get back the original values, first multiply by Scaleby, then add Shiftby
    """

    Shiftby = values.min()
    values = values - Shiftby
    Scaleby = values.max()
    return values/Scaleby, Scaleby, Shiftby








