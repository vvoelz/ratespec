#!/usr/bin/env python

import os, sys, glob, string
import numpy as np

import scipy

import matplotlib
from pylab import *

from RateSpecTools import *
from RateSpecClass import *


def plotResults(r, outpdffile=None, showPlot=False, plotMaxLikModel=False):

    """Plot the results of the rate spectra calculation stored in the RateSpecClass object r.
    If plotMaxLikModel = True, overlay the maximum Likelihood model on the data; otherwise use expectation model."""

    if plotMaxLikModel:
        bestfitA = r.results.maxLikelihoodSpectrum()
    else:
        bestfitA, stdA, ci_5pc, ci_95pc = r.results.expectationSpectrum()
        print 'bestfitA, stdA, ci_5pc, ci_95pc', bestfitA, stdA, ci_5pc, ci_95pc

    bestfitData = SumSpectra(bestfitA, r.Rates, r.Times)

    wtraj = np.array(r.results.wtraj)

    figure()
    # title(filename)

    # plot data with best fit
    subplot(3,2,1)
    TimeSeriesWithFit(r.Times, r.Data, bestfitData) 

    # plot neglogP versus param
    subplot(3,2,2)
    neglogPs = wtraj[:,4] - wtraj[:,4].min() + 1.
    loglog(wtraj[:,0], neglogPs, 'k.')
    axis([min(wtraj[:,0]), max(wtraj[:,0]), 0.9*min(neglogPs), 1.1*max(neglogPs)])
    xlabel('$\lambda$')
    ylabel('$-\log P/P_0$')

    # plot mean +/- std spectrum
    ax = subplot(3,2,3)
    #matplotlib.pyplot.errorbar(Timescales, meanA, yerr=stdA)
    plot(r.Timescales, bestfitA, 'k-', linewidth=2)
    hold(True)
    plot(r.Timescales, ci_5pc, 'k-', linewidth=1)
    hold(True)
    plot(r.Timescales, ci_95pc, 'k-', linewidth=1)
    ax.set_xscale('log')
    xlabel('timescale (s)')

    # plot w trajectory 
    subplot(3,2,4)
    plot(range(len(r.results.wtraj)), wtraj[:,0], 'r-')
    xlabel('accepted steps')
    ylabel('$\lambda$')

    # plot tau trajectory 
    subplot(3,2,6)
    plot(range(len(r.results.wtraj)), wtraj[:,3], 'r-')
    xlabel('accepted steps')
    ylabel('$\tau$')


    if showPlot:
        show()

    if outpdffile != None:
        savefig(outpdffile, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1)



def TimeSeriesWithFit(Times, Data, FitData, timeunit='s', markersize=2, linestyle='k-', markerstyle='b.'):
    """Make a (panel) plot of the time series data with the best fit."""
    semilogx(Times, Data, markerstyle, markersize=markersize)
    hold(True)
    semilogx(Times, FitData, linestyle, linewidth=1)
    xlabel('time (%s)'%timeunit)

def TimescaleSpectrum(Timescales, A, timeunit='s', linestyle='k-', UseStems=False):
    """Make a (panel) plot of the spectrum vs timescales, using standard error for errorbars."""
    ax = gca()
    if not UseStems:
        plot(Timescales, A, linestyle, linewidth=2)
    else:
        for i in range(len(Timescales)):
            plot([Timescales[i], Timescales[i]], [0, A[i]], 'k.-', linewidth=1, markersize=5)
            hold(True) 
    ax.set_xscale('log')
    xlabel('timescale (%s)'%timeunit)


def TimescaleSpectrumStd(Timescales, A, stdA, timeunit='s'):
    """Make a (panel) plot of the spectrum vs timescales, using standard error for errorbars."""
    ax = gca()
    matplotlib.pyplot.errorbar(Timescales, meanA, yerr=stdA)
    ax.set_xscale('log')
    xlabel('timescale (%s)'%timeunit)




def plotMCTrajData(filename, nskip=0):
    """Read in the MC trajectory data and make a plot of it.
    Skip the first nskip points (as these may be far from the mean)"""

    data = scipy.loadtxt(filename)  # #step	w	sigma	tau	neglogP

    figure()

    # plot the lambda trajectory
    subplot(2,3,1)
    plot(data[nskip:,0], data[nskip:,1])
    xlabel('$\lambda$')
    ylabel('accepted steps')

    # plot the sigma trajectory
    subplot(2,3,2)
    plot(data[nskip:,0], data[nskip:,2])
    xlabel('$\sigma$')
    ylabel('accepted steps')

    # plot the tau trajectory
    subplot(2,3,3)
    plot(data[nskip:,0], data[nskip:,3])
    xlabel('$\tau$')
    ylabel('accepted steps')

    # try a contour plot of sigma and tau
    subplot(2,3,4)
    myhist, myextent = histBin( data[nskip:,2], data[nskip:,3], 20)
    # convert to log scale
    myhist = np.log(np.array(myhist) + 1.)
    #contour(myhist, extent = myextent, interpolation = 'nearest')
    contourf(myhist, extent = myextent, interpolation = 'nearest')


    #import matplotlib.cm as cm
    #hexbin(data[nskip:,2], data[nskip:,3], gridsize=10, bins='log',cmap=cm.jet,linewidths=0,edgecolors=None)

    show()

def histBin(xdata, ydata, xbins,ybins=None):
    if (ybins == None): ybins = xbins
    xmin,xmax = min(xdata),max(xdata)
    xwidth = xmax-xmin
    ymin,ymax = min(ydata),max(ydata)
    ywidth = ymax-ymin
    def xbin(xval):
        xbin = int(xbins*(xval-xmin)/xwidth)
        return max(min(xbin,xbins-1),0)
    def ybin(yval):
        ybin = int(ybins*(yval-ymin)/ywidth)
        return max(min(ybin,ybins-1),0)
    hist = [[0 for x in xrange(xbins)] for y in xrange(ybins)]
    for i in range(xdata.shape[0]):
        x,y = xdata[i], ydata[i]
        hist[ybin(y)][xbin(x)] += 1
    extent = (xmin,xmax,ymin,ymax)
    return hist,extent

def plotSummary(mctraj_filename, ratespec_filename, nskip=0):
    """Read in the MC trajectory data and make a plot of it.
    Skip the first nskip points (as these may be far from the mean)"""

    data = scipy.loadtxt(mctraj_filename)  # step     w       sigma   tau     neglogP
    ratespec_data = scipy.loadtxt(ratespec_filename)

    figure()

    # plot the lambda trajectory
    subplot(2,2,1)
    plot(data[nskip:,0], data[nskip:,1])
    xlabel('accepted steps')
    ylabel('$\lambda$')
    #title(mctraj_filename)

    # try a contour plot of sigma and tau
    subplot(2,2,2)
    myhist, myextent = histBin( data[nskip:,2], data[nskip:,3], 20)
    # convert to log scale
    myhist = np.log(np.array(myhist) + 1.)
    #contour(myhist, extent = myextent, interpolation = 'nearest')
    contourf(myhist, extent = myextent, interpolation = 'nearest')

    # plot mean +/- std spectrum
    ax = subplot(2,2,3)
    Timescales = ratespec_data[:,0]
    maxLikA = ratespec_data[:,1]
    meanA = ratespec_data[:,2]
    stdA = ratespec_data[:,3]
    ci_5pc = ratespec_data[:,4]
    ci_95pc = ratespec_data[:,5]
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

    # plot mean +/- std spectrum
    subplot(2,2,4)
    wcounts, wbins = np.histogram(data[nskip:,1], bins=30)
    plot(wbins[0:-1], wcounts, linestyle='steps', linewidth=2)
    xlabel('$\lambda$')

    show()

