#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 02:43:53 2018

@author: kamranbinaee
This Python code is developed to find the similarity of gaze and ball trajectory
during the blank. Basically here I fit separate polynomials to the ball & gaze 
trajectory, find the the minimum degree required to model the gaze and ball. Finally
report their data using a confusion matrix.
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
 '2016-5-3-12-52', '2016-5-4-13-3', '2016-5-5-13-7', '2016-5-6-11-2', '2016-5-6-13-4']


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
        crIndex = slicedDF.ballCrossingIndex.values
        stIndex = slicedDF.trialStartIdx.values
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
            pr2g = 0
            pr2b = 0
            degG = 1
            degB = 1
            #fig = plt.figure(figsize=(5,5))
            #ax = fig.add_subplot(1,1,1)

            ''' Uncomment this part for plotting one trial data'''
            
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plt.gca().set_aspect('equal', 'box')#'datalim')            
            #plt.grid(True)
            # Major ticks every 20, minor ticks every 5
            major_ticks = np.arange(-60, 60, 2)
            minor_ticks = np.arange(-60, 60, 1 )
            
            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            ax.set_yticks(major_ticks)
            ax.set_yticks(minor_ticks, minor=True)
            
            # And a corresponding grid
            ax.grid(which='both')
            
            # Or if you want different settings for the grids:
            ax.grid(which='minor', alpha=0.1)
            ax.grid(which='major', alpha=0.5)

            #myMarker = ['-sb','-Dg','-vr', '-pm']
            myMarker = ['--b','--g','--r', '--m']            
            myLegend = ['$K_{gaze}=1$', '$K_{gaze}=2$', '$K_{gaze}=3$', '$K_{gaze}=4$']
            myLineWidth = 3
            #print('\n')
            # Pick the data during the blank to use for extrapolation
            idx = np.arange(slicedDF.ballOffIdx.values[trialID], slicedDF.ballOnIdx.values[trialID])    
            gDataX = gazeX[idx]
            gDataY = gazeY[idx]
            bDataX = ballX[idx]
            bDataY = ballY[idx]
            #for i in range(len(gDataX)):
            #    plt.plot([gDataX[i], bDataX[i]] , [gDataY[i], bDataY[i]], 'tab:gray', linewidth = 0.3)
            ''' Uncomment this part for plotting one trial data'''
            dataX = gazeX[idx]
            dataY = gazeY[idx]
            plt.plot(dataX, dataY, 'y', label = 'gaze', markersize = 3, linewidth = myLineWidth)

            dataX = ballX[idx]
            dataY = ballY[idx]
            plt.plot(dataX, dataY, 'c', alpha = 0.8, label = 'ball', markersize = 3, linewidth = myLineWidth)
            

            for degree in [2]:# [1,2,3,4]:

                dataX = gazeX[idx]
                dataY = gazeY[idx]
                
                coeffs = np.polyfit(dataX, dataY, degree)
                gazeCoeff = np.vstack((gazeCoeff,[coeffs]))

                # r-squared
                pX = np.poly1d(coeffs)
                # fit values, and mean
                yhat = pX(dataX)
                if (degree == 1):
                    diff = yhat[0] - dataY[0]
                    yhat = yhat - diff*0.8
                ''' Uncomment this part for plotting one trial data'''
                plt.plot(dataX, yhat, myMarker[degree-1], label = myLegend[degree-1], alpha = 0.7, markersize = 1, linewidth = myLineWidth-1)
                
                errorY = np.abs(yhat - dataY)
                ybar = np.sum(dataY)/len(dataY)          # or sum(y)/len(y)
                ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
                sstot = np.sum((dataY - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
                r2g = ssreg / sstot                

                if degree == 1:
                    scale = 1.0
                elif degree == 2:
                    scale = 1.05
                elif degree == 3:
                    scale = 1.8
                else:
                    scale = 3.5

                #adj_r2g = 1-((1-r2g)*(len(dataX)-1))/(len(dataX) - degree - 32)
                adj_r2g = 1-(1-r2g)*scale
                r2g = r2g * 100
                adj_r2g = adj_r2g * 100

                #print('%d) Gaze R^2 = %2.3f  Adj_R^2 = %2.3f' %(degree, r2g, adj_r2g))
                #r2g = adj_r2g                
                #print(pX)
                if (r2g - pr2g) > 0:
                    degG = degree
                    pr2g = r2g
#                if (np.max(errorY)<25):
#                    error_gazeY = np.vstack((error_gazeY, errorY))
#                else:
#                    print('Exception Gaze', np.max(errorY))
                
                # Pick the data during the blank to use for extrapolation
                dataX = ballX[idx]
                dataY = ballY[idx]
                coeffs = np.polyfit(dataX, dataY, degree)
                ballCoeff = np.vstack((ballCoeff,[coeffs]))
                        
                # r-squared
                pX = np.poly1d(coeffs)
                # fit values, and mean
                yhat = pX(dataX)
                errorY = np.abs(yhat - dataY)
                ybar = np.sum(dataY)/len(dataY)          # or sum(y)/len(y)
                ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
                sstot = np.sum((dataY - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
                r2b = ssreg/ sstot                
                if degree == 1:
                    scale = 1.0
                elif degree == 2:
                    scale = 1.1
                elif degree == 3:
                    scale = 100.0
                else:
                    scale = 1000
                    #scale = 0.001

                #adj_r2b = 1-((1-r2b)*(len(dataX)))/(len(dataX) - 5*degree - 30)
                #adj_r2b = 1-((1-r2b)*(38))/scale
                adj_r2b = 1-(1-r2b)*scale
                r2b = r2b * 100
                adj_r2b = adj_r2b * 100
                #if abs(adj_r2b) > 100:
                #    adj_r2b = 99.99
                #print('%d %d) Ball R^2 = %2.3f  Adj_R^2 = %2.3f' %(scale,degree, r2b, adj_r2b))
                #r2b = adj_r2b
                #print(pX, '\n')
                if (r2b - pr2b) > 0:
                    degB = degree
                    pr2b = r2b
#                if (np.max(errorY)<25):
#                    error_ballY = np.vstack((error_ballY, errorY))
#                else:
#                    print('Exception Ball', np.max(errorY))
                
                #print('degB : ', degB)
                gCol = 'gaze_'+str(degree)
                bCol = 'ball_'+str(degree)
                #myDataFrame[bCol].loc[mainCounter] = abs(adj_r2b)
                #myDataFrame[gCol].loc[mainCounter] = abs(adj_r2g)
                myDataFrame[bCol].loc[mainCounter] = abs(r2b)
                myDataFrame[gCol].loc[mainCounter] = abs(r2g)
            
            mainCounter = mainCounter + 1
            #print('Gaze:', degG, ' Ball:', degB)
            img[degB-1,degG-1] = img[degB-1,degG-1] + 1
            
            ''' New Plots
            m1 = min(dataX)
            m2 = min(dataY)
            myMin = int(min(m1,m2)) - 4
            m1 = max(dataX)
            m2 = max(dataY)            
            myMax = int(max(m1,m2)) + 4
            print(myMin, myMax)
            major_ticks_x = np.arange(int(min(dataX)) - 4, int(max(dataX)) + 4, 2)
            major_ticks_y = np.arange(int(min(dataY)) - 4, int(max(dataY)) + 4, 2)
            
            ax.set_xticks(major_ticks_x)
            ax.set_yticks(major_ticks_y)
                        
            # Or if you want different settings for the grids:
            ax.grid(which='major', alpha=0.8)
            plt.axis('equal')
            #ax.grid(xdata=np.arange(int(min(dataX))-8, int(max(dataX))+8, 2.0), ydata=np.arange(int(min(dataY))-8, int(max(dataY))+8, 2.0))
            '''
            
            ''' Uncomment this part for plotting one trial data'''
            plt.xlim(-16,-9.5)
            plt.ylim(11,22)
            plt.xlabel('azimuth (degree)', fontsize = 13.5)
            plt.ylabel('elevation (degree)', fontsize = 13.5)
            plt.xticks( fontsize = 12)
            plt.yticks( fontsize = 12)
            #plt.grid(True)
            #leg = plt.legend(loc = [0.2, 0.1], fontsize = 12)
            #leg.get_frame().set_alpha(1)
            currentFile = os.getcwd()
            #plt.savefig(currentFile+'/Outputs/gazeBallMaching/sampleFigure'+str(trialID)+'.png',dpi=600)
            plt.close()
            
            #else:
                #print('Outlier Data Skipped for:\nTrial ID: ', trialID, '\n Subject ID: ', fileTime)
    #print('df:', myDataFrame)
    #print('TrialID', trialID)
    
    '''
    #print('IMG = \n', img)
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)
    #ax.set_xticks(np.arange(1,4))
    #ax.set_yticks(np.arange(1,4))
    #ax.set_xlabel([80,122])
    plt.imshow(img, interpolation='nearest', extent=[1,4,4,1])
    #plt.grid(xdata = [1,2,3], ydata = [1,2,3]) 
    plt.xticks([1,2,3,4], fontsize = 10)
    plt.yticks([1,2,3,4], fontsize = 10)
    plt.xlabel('Gaze Degree', fontsize = 16)
    plt.ylabel('Ball Degree', fontsize = 16)
    ax.set_xticks([1.75,2.5, 3.25], minor = True)
    ax.set_yticks([1.75,2.5, 3.25], minor = True)
    plt.colorbar()
    ax.grid(which='both', linewidth=4, color = 'k')
    ax.grid(which = 'major', alpha = 0.0)
    #ax.set_xticklabels('edges')
    #ax.set_yticklabels('edges')

    currentFile = os.getcwd()
    #plt.savefig(currentFile+'/Outputs/gazeBallMaching/PreBD_'+str(preBD)+'_PostBD_'+str(postBD)+'.png',dpi=600)
    plt.show()
    plt.close()
    '''

    #pickle.dump( img, open(currentFile + "/Outputs/gazeBallMaching/PreBD_"+str(preBD)+"_PostBD_"+str(postBD)+".p", "wb" ) )
    #results = pickle.load(( open(currentFile + "/Outputs/gazeBallMaching/PreBD_"+str(preBD)+"_PostBD_"+str(postBD)+".p", "rb" ) ))
    #print('Saved Pickle:\n',results)
    
    error_ballY = np.delete(error_ballY, 0,0)
    error_gazeY = np.delete(error_gazeY, 0,0)
    #myDataFrame.to_pickle(currentFile+'/Outputs/gazeBallMaching/results_forNow_'+str(preBD)+'_'+str(postBD)+'.pkl')
    #myDataFrame.to_csv(currentFile+'/Outputs/gazeBallMaching/results_forNow_'+str(preBD)+'_'+str(postBD)+'.csv')
    print('Final Size', error_ballY.shape)
    ballCoeff = np.delete(ballCoeff, 0, 0)
    gazeCoeff = np.delete(gazeCoeff, 0, 0)
    return error_gazeY, error_ballY, ballCoeff, gazeCoeff
#myDegree = 4
#postBD = 0.5
myResult = []
for postBD in [0.3, 0.4, 0.5]:
    for preBD in [0.6, 0.8, 1.0]:

        error_gazeY, error_ballY, ballCoeff, gazeCoeff = myFunc(preBD, postBD)
        print('pre: ', preBD, 'post: ', postBD)
        print('mean_gaze: ', np.mean(gazeCoeff, axis = 0))
        print('SD_gaze: ', np.std(gazeCoeff, axis = 0)/np.sqrt(10))
        print('mean_ball: ', np.mean(gazeCoeff, axis = 0))
        print('SD_ball: ', np.std(gazeCoeff, axis = 0)/np.sqrt(10))
        print('\n')
        
        
    