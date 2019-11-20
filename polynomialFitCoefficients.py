#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed June 6th 02:28:53 2019

@author: kamranbinaee
This Python code is developed to find the record the coefficient of fits to gaze
and ball data. Since the reviewer was asking about the quality of the fit
"""

from __future__ import division

import sys
import PerformParser as pp
import pandas as pd
import numpy as np
import scipy
from scipy import signal as sig
import performFun as pF
import pickle


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
import matplotlib.ticker as plticker


#================================= To read One Subject Data =============================================
fileTimeList = ['2016-4-19-14-4', '2016-4-22-11-57', '2016-4-27-13-28', '2016-4-28-10-57', '2016-4-29-11-56',
  '2016-5-4-13-3', '2016-5-5-13-7', '2016-5-6-11-2', '2016-5-6-13-4']
badFiles = ['2016-5-3-12-52']
#fileTimeList = ['2016-4-19-14-4', '2016-4-22-11-57', '2016-4-27-13-28', '2016-4-28-10-57', '2016-4-29-11-56']
#fileTimeList = ['2016-5-3-12-52']#, '2016-4-19-14-4']
#SAPfileTimeList = ['2016-5-6-13-4']
def dotproduct( v1, v2):
    r = sum((a*b) for a, b in zip(v1, v2))
    return r


def length(v):
    return np.sqrt(dotproduct(v, v))


def vectorAngle( v1, v2):
    r = (180.0/np.pi)*np.arccos((dotproduct(v1, v2)) / (length(v1) * length(v2)))#np.arccos((np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))))#
    return r

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

def myFunc(preBD, postBD):
    gazeCoeff = np.zeros((1,3))
    ballCoeff = np.zeros((1,3))
    blankSampleNumber = 38
    error_gazeX = np.zeros((blankSampleNumber))
    error_gazeY = np.zeros((blankSampleNumber))
    error_ballX = np.zeros((blankSampleNumber))
    error_ballY = np.zeros((blankSampleNumber))
    eyeToScreenDistance = 0.0725
    #myWindow = 35
    numberOfRows = len(fileTimeList)*15
    myDataFrame = pd.DataFrame(data = [],#np.array([['dummy', 0.6, 0.4, -1,-1,-1,-1,-1,-1,-1,-1]]),
                               index = np.arange(0, numberOfRows),
                               columns = ['subjectID','PreBD', 'PostBD', 'gaze_1', 'gaze_2', 'gaze_3', 'gaze_4','ball_1', 'ball_2', 'ball_3', 'ball_4'])
    mainCounter = 0
    #print('df:', myDataFrame)
    img = np.zeros((4,4))
    for fileTime in fileTimeList: 
        print('.... File Time : ', fileTime)
        expCfgName = "gd_pilot.cfg"
        sysCfgName = "PERFORMVR.cfg"
         
        filePath = "../Data/" + fileTime + "/"
        fileName = "exp_data-" + fileTime
         
        sessionDict = pF.loadSessionDict(filePath,fileName,expCfgName,sysCfgName,startFresh=False)
         
        rawDataFrame = sessionDict['raw']
        processedDataFrame = sessionDict['processed']
        calibDataFrame = sessionDict['calibration']
        trialInfoDataFrame = sessionDict['trialInfo']

        gb = trialInfoDataFrame.groupby([trialInfoDataFrame.preBlankDur, trialInfoDataFrame.postBlankDur])
        #gb = trialInfoDataFrame.groupby(trialInfoDataFrame.postBlankDur)
        preBDList = [0.6, 0.8, 1.0]
        postBDList = [0.3, 0.4, 0.5]
        #preBD = 0.6
        #postBD = 0.3
        fileName = 'PostBD = '+str(postBD)
        #slicedDF = gb.get_group(postBD)
        slicedDF = gb.get_group((preBD, postBD))
        
        #slicedDF = trialInfoDataFrame
        #gb = trialInfoDataFrame.groupby([trialInfoDataFrame.postBlankDur])
        #slicedDF = gb.get_group((0.3))
        #crIndex = slicedDF.ballCrossingIndex.values
        #stIndex = slicedDF.trialStartIdx.values
        ballOnIndex = slicedDF.ballOnIdx.values
        ballOffIndex = slicedDF.ballOffIdx.values

        for trialID in range(15):
        #trialID = 0
            myDataFrame.PreBD.loc[mainCounter]= preBD
            myDataFrame.PostBD.loc[mainCounter]= postBD
            myDataFrame.subjectID.loc[mainCounter]= fileTime
            x = processedDataFrame.gazePoint.X.values
            x = np.array(x, dtype = float)
            gazeX = (180/np.pi)*np.arctan(x/eyeToScreenDistance)
            y = processedDataFrame.gazePoint.Y.values
            y = np.array(y, dtype = float)
            gazeY = (180/np.pi)*np.arctan(y/eyeToScreenDistance)
    
            x = processedDataFrame.ballOnScreen.X.values
            x = np.array(x, dtype = float)
            ballX = (180/np.pi)*np.arctan(x/eyeToScreenDistance)
            #y = processedDataFrame.rotatedBallOnScreen.Y.values[stIndex[trialID]:crIndex[trialID]]
            y = processedDataFrame.ballOnScreen.Y.values
            y = np.array(y, dtype = float)
            ballY = (180/np.pi)*np.arctan(y/eyeToScreenDistance)
            #print('Success?', slicedDF.ballCaughtQ.values[trialID])

            #print('\n')
            # Pick the data during the blank to use for extrapolation
            idx = np.arange(slicedDF.ballOffIdx.values[trialID], slicedDF.ballOnIdx.values[trialID])    
            

            for degree in [2]:# [1,2,3,4]:

                dataX = gazeX[idx]
                dataY = gazeY[idx]
                
                coeffs = np.polyfit(dataX, dataY, degree)

                # r-squared
                pX = np.poly1d(coeffs)
                # fit values, and mean
                yhat = pX(dataX)
                errorY = np.abs(yhat - dataY)
                if (np.max(errorY)<2):
                    gazeCoeff = np.vstack((gazeCoeff,[coeffs]))
                    #error_gazeY = np.vstack((error_gazeY, errorY))
                else:
                    print('Exception Gaze', np.max(errorY))
                
                # Pick the data during the blank to use for extrapolation
                dataX = ballX[idx]
                dataY = ballY[idx]
                coeffs = np.polyfit(dataX, dataY, degree)            
                        
                # r-squared
                pX = np.poly1d(coeffs)
                # fit values, and mean
                yhat = pX(dataX)
                errorY = np.abs(yhat - dataY)
                if (np.max(errorY)<2):
                    ballCoeff = np.vstack((ballCoeff,[coeffs]))
                    #error_ballY = np.vstack((error_ballY, errorY))
                else:
                    print('Exception Ball', np.max(errorY))
            
            mainCounter = mainCounter + 1
            #print('Gaze:', degG, ' Ball:', degB)
    ballCoeff = np.delete(ballCoeff, 0, 0)
    gazeCoeff = np.delete(gazeCoeff, 0, 0)
    return error_gazeY, error_ballY, ballCoeff, gazeCoeff
#myDegree = 4
#postBD = 0.5
myResult = []
for postBD in [0.3, 0.4, 0.5]:
    for preBD in [0.6, 0.8, 1.0]:

        error_gazeY, error_ballY, ballCoeff, gazeCoeff = myFunc(preBD, postBD)
        print('\npre: ', preBD, 'post: ', postBD)
        print('mean_gaze: ', np.round(np.mean(gazeCoeff, axis = 0),2))
        print('SD_gaze: ', np.round(np.std(gazeCoeff, axis = 0)/np.sqrt(10),2))
        print('mean_ball: ', np.round(np.mean(ballCoeff, axis = 0),2))
        print('SD_ball: ', np.round(np.std(ballCoeff, axis = 0)/np.sqrt(10),2))
        print('\n')
        
        
    