#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:54:12 2018

@author: kamranbinaee
"""



from __future__ import division

import sys
import PerformParser as pp
import pandas as pd
import numpy as np
import scipy
from scipy import signal as sig
import performFun as pF

import cv2
import os
import scipy.io as sio
import matplotlib

#%matplotlib notebook
import Quaternion as qu
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
from scipy.fftpack import fft
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import norm

gazeX = np.array([])
gazeY = np.array([])
handX = np.array([])
handY = np.array([])
handZ = np.array([])
myColor = np.array([])


def myPolyfit(x, y, degree):
    # Polynomial Regression
    x = np.array(x, dtype = float)
    y = np.array(y, dtype = float)
    results = {}

    coeffs = np.polyfit(x, y, degree)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results

#================================= To read One Subject Data =============================================
fileTimeList = ['2016-4-19-14-4', '2016-4-22-11-57', '2016-4-27-13-28', '2016-4-28-10-57', '2016-4-29-11-56',
 '2016-5-3-12-52', '2016-5-4-13-3', '2016-5-5-13-7', '2016-5-6-11-2', '2016-5-6-13-4']
 
#fileTimeList = ['2016-4-19-14-4', '2016-4-22-11-57', '2016-4-27-13-28', '2016-4-28-10-57', '2016-4-29-11-56',
# '2016-5-4-13-3', '2016-5-5-13-7', '2016-5-6-11-2', '2016-5-6-13-4']

# Best Sbjects
#fileTimeList = [ '2016-5-3-12-52', '2016-5-4-13-3', '2016-5-6-13-4' ]

#fileTimeList = ['2016-4-19-14-4']
# Worst Sbjects
#fileTimeList = ['2016-4-19-14-4', '2016-4-27-13-28', '2016-5-5-13-7']

#fileTime = '2016-5-3-12-52'

successCounter = np.zeros((10,135), dtype = float)
cdfMatrix = np.zeros((10,135), dtype = float)
i = 0
for fileTime in fileTimeList:
    expCfgName = "gd_pilot.cfg"
    sysCfgName = "PERFORMVR.cfg"
     
    filePath = "../Data/" + fileTime + "/"
    fileName = "exp_data-" + fileTime
 
    print('Subject #', i)
     
    sessionDict = pF.loadSessionDict(filePath,fileName,expCfgName,sysCfgName,startFresh=False)
     
    rawDataFrame = sessionDict['raw']
    processedDataFrame = sessionDict['processed']
    calibDataFrame = sessionDict['calibration']
    trialInfoDataFrame = sessionDict['trialInfo']
    print('Done!')
    sum = 0
    for j in range(len(trialInfoDataFrame.ballCaughtQ.values)):
        #print(trialInfoDataFrame.ballCaughtQ.values[j])
        if (trialInfoDataFrame.ballCaughtQ.values[j]):
            successCounter[i,j] = 1.0
            sum = sum + 1
            #print('HI')
        else:
            successCounter[i,j] = 0.0
        cdfMatrix[i,j] = sum
    i = i+1
#print(successCounter[1,0:10])
myMatrix = np.sum(successCounter, axis = 0)
#print(myMatrix)
print(cdfMatrix)
#print(np.mean(myMatrix, axis= 0))
plt.figure(figsize=(12, 8), dpi=200)
#plt.plot(range(len(myMatrix)), myMatrix, '-r', linewidth = 3)
#plt.errorbar(range(len(myMatrix)), (np.mean(cdfMatrix,axis=0)), yerr = (np.std(cdfMatrix,axis=0)),
#                   fmt = '-ob', linewidth = 3, markersize = 10, label = 'Success Rate')

num_plots = 10

# Have a look at the colormaps here and decide which one you'd like:
# http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])

# Plot several different functions...
x = np.arange(10)
labels = []
for i in range(0, num_plots):
    plt.plot(np.arange(135)/1.35, cdfMatrix[i,:]/1.35, linewidth = 3)
    labels.append(r' Subject$%i$' % (i+1))

# I'm basically just demonstrating several different legend options here...
plt.legend(labels, ncol=2, loc=[0.05, 0.75], 
           #bbox_to_anchor=[0.5, 1.1], 
           columnspacing=1.0, labelspacing=1.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=True, shadow=True)


#plt.plot((finalMeanX_F[:,-focusFrame]), (finalMeanY_F[:,-2]), '.r')
#plt.fill_between(t, y1 = meanHandX_S - stdHandX_S , y2 = meanHandX_S + stdHandX_S ,color = 'b', label='Success', alpha = 0.6)
#plt.fill_between(t, y1 = meanHandX_F - stdHandX_F , y2 = meanHandX_F + stdHandX_F ,color = 'r', label='Fail', alpha = 0.6)
#plt.axvline(x=-postBDList[0], color='k', linestyle='--')
#plt.axvline(x=-postBDList[0]-0.5, color='k', linestyle='--')
#plt.ylim(0,10.1)
plt.xlim(-0.1,135.1/1.35)
plt.grid(True)
#plt.legend(loc=[0.85,1.01])
#plt.title("Paddle Y Velocity Vs. Ball-Gaze Angle"+"\nfor "+fileName+" at "+str(int((22-focusFrame)*(13.33)))+ " ms after Reappearance")
plt.title("Success Rate for All subjects", fontsize = 25)
plt.xlabel('% of Total Trials', fontsize = 25)
plt.ylabel('Success %', fontsize = 25)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
currentFile = os.getcwd()
#print(currentFile)
#plt.savefig(currentFile+'/Outputs/GeneralOutputs/learningRate_All.png',dpi=600)
plt.savefig(currentFile+'/Outputs/GeneralOutputs/learningRate_CDF_All.png',dpi=600)
#plt.savefig(currentFile+'/Outputs/HandVelocityFigures/'+fileName +'_X.svg', format='svg', dpi=1000)
#plt.show()
#plt.close()


