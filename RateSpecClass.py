#!/usr/bin/env python

import os, sys, glob, string

import scipy
from scipy.linalg import pinv

import numpy as np
import math

from RateSpecTools import *


class RateSpecClass(object):

    def __init__(self, dataFile=None, timeUnit=1.0e-6,  nRates=100, RateRange=(1.0, 1.e9),
                       linearRates=False, Lnorm='ridge', standardizeData=True, scaleData=False,
                       init_sigma=0.002, init_tau=0.002, init_rho=0.5, dsigma=5.0e-4, dtau=5.0e-4, drho=0.01): 
        """Initialize the RateSpecClass.

        OPTIONS
        dataFile      - A text file containing two columns: time and values (headers in the file are okay)
        timeUnit      - The unit (in seconds) of time axis in the file
        nRates        - the number of rates K to use in the spectrum calculation
        RateRange     - a duple (minrate, maxrate) specifying the range of rate spectrum values (in units 1/sec)
        linearRates   - Default is to use exponentially-spaced rates (i.e. linear on a semilogx plot).  If set to True,
                        the rate range wil be linearly-spaced.  Warning: NOT RECOMMENDED!
        Lnorm         - Specifies the norm used in the penalty function.
			Must be either: 'ridge' (L2 norm), 'lasso' (L1 norm), or 'enet' (elastic net - rho*L1 + (1-rho)*L2 penalty)
        standardizeData -  If set to True, the spectra will be computed for data that is
                        *centered* i.e. (1/N * sum_i(X_ij) is subtracted from each column of the design matrix,
                        and *shifted*, i.e. a constant offset beta0 = (1/N * sum_i(yi)) is subtracted from the y-values (and added back later) 
                        Default: True.  HIGHLY RECOMMENDED.  
        scaleData     - Alternatively, scale the input data y-values to be the interval [0,1]. Default: False

        OPTIONS for posterior sampling of the regularization parameter w = sigma^2/tau^2 
        init_sigma    - The initial value of sigma in the Monte Carlo algorithm. Default: 0.002
        init_tau      - The initial value of tau in the Monte Carlo algorithm. Default: 0.002
        init_rho      - The initial value of the L1/L2 mixing parameter rho in elastic net (only used for this)
        dsigma        - The step size of sigma (drawn from N(0,dsigma^2) at each step) in the Monte Carlo algorithm.   Default: 5.0e-4 
        dtau          - The step size of tau (drawn from N(0,dtau^2) at each step) in the Monte Carlo algorithm.   Default: 5.0e-3
        drho          - The step size of rho (in practice, several fixed bins are used for sample in rho).
           
        """

        # The filename containing the time series data
        self.dataFile = dataFile   
        expdata = scipy.loadtxt(self.dataFile)
        self.Times = expdata[:,0]*timeUnit  # convert to units in sec (default is microsec)
        self.nTimes = len(self.Times)
        self.Data = expdata[:,1]
        self.standardizeData = standardizeData
        self.offset = 0.0
        self.scaleData = scaleData

        if self.scaleData:
            self.Data = scaleValues(self.Data)

        if self.standardizeData:
            self.offset = self.Data.mean()

        # sort the times in order, in case they're mixed up
        SortInd = self.Times.argsort()
        self.Times = self.Times[SortInd]
        self.Data = self.Data[SortInd]

        # Define range and number of discrete rates to use in the fit
        self.nRates = nRates
        self.RateRange = RateRange   # (minrate, maxrate) in 1/sec
        self.linearRates = linearRates
        if self.standardizeData:
            self.offset = self.Data.mean() 
            if self.linearRates:
                self.Rates = np.array(rangeLin(RateRange[0], RateRange[1], nRates).tolist() )  # in 1/s 
            else:
                self.Rates = np.array(rangeLog(RateRange[0], RateRange[1], nRates).tolist() )  # in 1/s 
        else:
            if self.linearRates: 
                self.Rates = np.array([0.] + rangeLin(RateRange[0], RateRange[1], nRates).tolist() )  # in 1/s  (add a k=0 constant rate too.)
            else:
                self.Rates = np.array([0.] + rangeLog(RateRange[0], RateRange[1], nRates).tolist() )  # in 1/s  (add a k=0 constant rate too.)

        self.Timescales = 1./self.Rates

        # Define the norm ('ridge', 'lasso', 'enet') used for the regularization
        self.Lnorm = Lnorm 

        if (self.Lnorm == 'lasso') or (self.Lnorm == 'enet')  :
          try:
            from scikits.learn import linear_model
          except:
            print 'Error: could NOT import scikits.learn, need for L1 Lasso regression.  Using L2 instead.'
            self.Lnorm = 'ridge' 

        # Initial guesses and step sizes for sigma and tau (and rho, for elastic net)
        self.sigma = init_sigma
        self.tau = init_tau 
        self.rho = init_rho

        self.dsigma = dsigma
        self.dtau = dtau
        self.drho = drho

        self.neglogP = 1.e99
  




    def sampleSpectrum(self, nsteps=100, Verbose=True):
        """Perform Monte Carlo sampling of the posterior."""

        # initialize Monte Carlo parameter for sampling of the posterior 
        neglogP = self.neglogP
        sigma = self.sigma 
        tau = self.tau
        rho = self.rho
        dsigma, dtau, drho = self.dsigma, self.dtau, self.drho

        rho_bins = np.arange(0.1, 1.0+drho, drho)  # values rho < 0.01  are *not* reliable (see scikits.learn docs)
        rho_index = abs(rho_bins - rho).argmin()  # find the nearest rho bin to start given rho, do sampling in indices

        #print 'rho_bins', rho_bins
        #print 'initial rho_index', rho_index

        # instantiate a class to store the MC results
        self.results = MCResults() 

        for step in range(nsteps):

            # Gibbs sampling - randomly tweak either sigma or tau 
            SigmaTweaked = (np.random.rand() < 0.5)  # 1 for sigma, 0 for tau
            if SigmaTweaked:
                new_sigma = sigma + dsigma*np.random.randn()
                new_tau = tau
            else:
                new_sigma = sigma 
                new_tau = tau + dtau*np.random.randn()

            # tweak the rho bin, allowing circular sampling
            new_rho_index = (rho_index + (np.random.randint(3)-1))%len(rho_bins)  # tweak rho (only used in elastic net) -1, same, or +1
            new_rho = rho_bins[new_rho_index]

            #print 'new_rho_index', new_rho_index
            #print 'new_rho', new_rho
               

            # calculate best-fit Laplace Transform given regularization parameter w = sigma^2/tau^2
            w = (new_sigma/new_tau)**2
            if self.Lnorm == 'enet':
                A, rss, ndof = fitRateSpectrum(self.Times, self.Data, self.Rates, w, Lnorm=self.Lnorm, standardizeData=self.standardizeData, rho=new_rho)
            else:
                A, rss, ndof = fitRateSpectrum(self.Times, self.Data, self.Rates, w, Lnorm=self.Lnorm, standardizeData=self.standardizeData)

            # Calculate posterior log-likelihood, 
            if self.Lnorm == 'lasso':
                rss_A = (np.abs(A)).sum()  # Residual sum of squares of the coefficients
                neglogP_Nterms  = (self.nTimes + 1.)*np.log(new_sigma) +  rss/(2.*new_sigma**2.)
                neglogP_Kterms = (2.*self.nRates + 1.)*np.log(new_tau) + rss_A/(new_tau**2.)
            elif self.Lnorm == 'ridge':
                rss_A = np.dot(A, A)  # Residual sum of squares of the coefficients
                neglogP_Nterms  = (self.nTimes + 1.)*np.log(new_sigma) +  rss/(2.*new_sigma**2.)
                neglogP_Kterms = (self.nRates + 1.)*np.log(new_tau) + rss_A/(2.*new_tau**2.)
            else: # i.e. elastic net
                rss_A = new_rho*np.dot(A, A) + 2.*(1. - new_rho)*(np.abs(A)).sum() 
                neglogP_Nterms  = (self.nTimes + 1.)*np.log(new_sigma) +  rss/(2.*new_sigma**2.)
                neglogP_Kterms = (self.nRates + 1.)*np.log(new_tau) + self.nRates*self.enetNormCorrection(new_rho,new_tau) + rss_A/(2.*new_tau**2.)

            new_neglogP = neglogP_Nterms + neglogP_Kterms


            # accept using Metropolis criterion
            temp = 1.
            accept_this = False
            if (new_neglogP != -np.inf):
                if (new_neglogP < neglogP):
                    accept_this = True
                elif np.random.rand() < np.exp( (neglogP - new_neglogP)/temp ) :
                    accept_this = True

            if accept_this:
                neglogP = new_neglogP
                sigma = new_sigma
                tau = new_tau
                rho = new_rho
                rho_index = new_rho_index

                # tally acceptance
                self.results.accepted += 1
                if SigmaTweaked:
                    self.results.accepted_sigma += 1
                else:
                    self.results.accepted_tau += 1

                # record the sample 
                if Verbose:
                    print 'step %d of %d:'%(step, nsteps), 
                self.results.recordSample(w, sigma, tau, rho, new_neglogP, A, neglogP_Nterms, neglogP_Kterms)

            else:
                if Verbose:
                    print 'step %d of %d: not accepted'%(step, nsteps)


        # store current state, in case we want to do more sampling
        self.neglogP = neglogP 
        self.sigma = sigma
        self.tau = tau 
        self.rho = rho


    def enetNormCorrection(self, rho, tau):
        """Return the correction factor X for enet normalization factor (log[tau] + X) ."""

        gamma = (rho - 1.0)/(tau*(2.*rho)**0.5)
        return -0.5 * np.log(rho) + gamma**2 + np.log( 1 + math.erf(gamma))



    def writeSpectrum(self, filename):
        """Write rate spectrum to file."""

        meanA, stdA, ci_5pc, ci_95pc = self.results.expectationSpectrum()
        bestA = self.results.maxLikelihoodSpectrum()

        fout = open(filename,'w')
        fout.write('#rate\tA_maxlik\t<A>\tstdA\tCI_5pc\tCI_95pc\n')
        for i in range( len(self.Timescales)):
            fout.write('%e\t%e\t%e\t%e\t%e\t%e\n'%(self.Timescales[i], bestA[i], meanA[i], stdA[i], ci_5pc[i], ci_95pc[i]) )
        fout.close()


    def writeMCResults(self, filename, Asample_filename=None):
        """Write the Monte Carlo sampling results to file.  If Asample_filename is provided,
        write the individual samples of the A amplitudes to file."""

        fout = open(filename,'w')
        fout.write('#step\tw\tsigma\ttau\trho\tneglogP\tneglogP_Nterms\tneglogP_Kterms\n')
        for i in range( len(self.results.wtraj) ):
            fout.write('%d\t'%i + '%e\t%e\t%e\t%e\t%e\t%e\t%e\n'%(tuple(self.results.wtraj[i])) )
        fout.close()

        if Asample_filename != None:
            fout = open(Asample_filename,'w')
            fout.write('#timescales(s)\n')
            fout.write( string.joinfields( ['%e'%j for j in self.Timescales], '\t') + '\n' )
            fout.write('#Asamples\n')
            for i in range( len(self.results.Asamples) ):
                fout.write( string.joinfields( ['%e'%j for j in self.results.Asamples[i]], '\t') + '\n' )
            fout.close()



class MCResults(object):

    def __init__(self):
        """Initialize a class to store, update, print, write results of MonteCarlo sampling."""

        # Keep track of acceptance counts for w, sigma and tau separately 
        self.accepted = 0
        self.total = 0
        self.accepted_sigma = 0
        self.total_sigma = 0
        self.accepted_tau = 0
        self.total_tau = 0

        # Keep track of samples obtained
        self.wtraj = [] # keep track of accepted samples (w, sigma, tau, neglogP) 
        self.Asamples = []  # keep track of the sampled A coefficeints

    def recordSample(self, w, sigma, tau, rho, neglogP, A, neglogP_Nterms, neglogP_Kterms, Verbose=True):
        """Record the (accepted) sample in the results the (accepted) sample in the results."""

        self.wtraj.append( [w, sigma, tau, rho, neglogP, neglogP_Nterms, neglogP_Kterms]  )
        self.Asamples.append( A )

        if Verbose:
            print 'w=%3.6f sigma=%3.6f tau=%3.6f rho=%3.6f -logP=%e'%(w, sigma, tau, rho, neglogP),
            print 'accepted sigmas/taus = %d/%d'%(self. accepted_sigma, self.accepted_tau)



    def expectationSpectrum(self):
        """Compute the mean, std, and 5% and 95% confidence intervals from the posterior samples.
        Returns: meanA, stdA, ci_5pc, ci_95pc"""

        # convert wtraj to array
        wtraj = np.array(self.wtraj)

        # calculate statistical weights for all snapshots of A amplitudes 
        neglogPs = wtraj[:,3] - wtraj[:,3].min()  # make the smallest value zero, so weights (P) don't blow up
        Aweights = np.exp(-1.*neglogPs)
        Aweights = Aweights/Aweights.sum()

        print 'Aweights', Aweights

        # get rid of NaNs in the weights
        for i in range(len(Aweights)):
            if np.isnan(Aweights[i]):
                Aweights[i] = 0

        # Calculate the mean A amplitudes
        Asamples = np.array(self.Asamples)
        meanA = np.dot(Aweights, Asamples)

        # Calculate the sample variance in each A amplitude
        Adiffs = np.zeros(Asamples.shape)
        for row in range(Adiffs.shape[0]):
            Adiffs[row,:] = Asamples[row,:] - meanA
        varA = np.dot( np.ones(Adiffs.shape[0],), Adiffs**2 )
        stdA = varA**0.5

        # Calculate 95% confidence interval
        ci_5pc, ci_95pc = np.zeros( meanA.shape ), np.zeros( meanA.shape )
        (M, K) = Asamples.shape
        for col in range(K):
            sortInd = Asamples[:,col].argsort()
            # compute cdf and mark where 0.05 is crossed
            cdf = 0.0
            for m in range(M):
                cdf += Aweights[sortInd[m]]
                if cdf > 0.05:
                    ci_5pc[col] = Asamples[sortInd[m],col]
                    break
            # compute cdf and mark where 0.95 is crossed
            cdf = 0.0
            for m in range(M):
                cdf += Aweights[sortInd][m]
                if cdf > 0.95:
                    ci_95pc[col] = Asamples[sortInd[m],col]
                    break

        return meanA, stdA, ci_5pc, ci_95pc

    def maxLikelihoodSpectrum(self):
        """Return the model with the largest posterior likelihood."""

        # convert wtraj to array
        wtraj = np.array(self.wtraj)

        # Find the wtraj index with the smallest (-log P)
        MaxLikInd = wtraj[:,3].argmin() # 

        return self.Asamples[MaxLikInd]


