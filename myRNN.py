# ====================================================================================================================
# ====================================================================================================================

from __future__ import division

import PerformParser as pp
import pandas as pd
import numpy as np
from scipy import signal as sig
import performFun as pF

import bokeh.plotting as bkP
import bokeh.models as bkM
from bokeh.palettes import Spectral6
bkP.output_notebook() 

import cv2
import os
import scipy.io as sio
import matplotlib

#%matplotlib notebook
from ipywidgets import interact
import filterpy as fP
from bokeh.io import push_notebook

import Quaternion as qu

import plotly
from plotly.graph_objs import Scatter, Layout
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

print plotly.__version__

plotly.offline.init_notebook_mode()

import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
from scipy.fftpack import fft
from mpl_toolkits.mplot3d import Axes3D

bkP.output_notebook()

#bkP.output_file('timeSeries.html') 

#%pylab inline
#%matplotlib notebook

# ====================================================================================================================
# ====================================================================================================================



# List of subjects with good calibration quality
fileTimeList = ['2016-4-19-14-4', '2016-4-22-11-57', '2016-4-27-13-28', '2016-4-28-10-57', '2016-4-29-11-56',
                '2016-5-3-12-52', '2016-5-4-13-3', '2016-5-5-13-7', '2016-5-6-11-2', '2016-5-6-13-4']
fileTimeList = ['2016-4-19-14-4'] # 

rawDataFrame = pd.DataFrame()
processedDataFrame = pd.DataFrame()
calibDataFrame = pd.DataFrame()
trialInfoDataFrame =  pd.DataFrame()
#fileTime = '2016-4-22-11-57'
#fileTime = '2016-4-27-13-28'
#fileTime = '2016-4-28-10-57'
#fileTime = '2016-4-29-11-56'
#fileTime = '2016-5-3-12-52'
#fileTime = '2016-5-4-13-3'
#fileTime = '2016-5-5-13-7'
#fileTime = '2016-5-6-11-2'
#fileTime = '2016-5-6-13-4'
expCfgName = "gd_pilot.cfg"
sysCfgName = "PERFORMVR.cfg"

saveSubjectPickle = False
startFromScratch = False

if startFromScratch == True:
    for fileTime in fileTimeList:

        print 'extracting data for:', fileTime
        filePath = "../Data/exp/" + fileTime + "/"
        fileName = "exp_data-" + fileTime

        sessionDict = pF.loadSessionDict(filePath,fileName,expCfgName,sysCfgName,startFresh=False)
        sessionDict['processed'], sessionDict['trialInfo'] = pF.calculateCrossingFrame(sessionDict['raw'], sessionDict['processed'], sessionDict['trialInfo'])
        rawDataFrame = rawDataFrame.append(sessionDict['raw'], ignore_index=True)
        processedDataFrame = processedDataFrame.append(sessionDict['processed'], ignore_index=True)
        calibDataFrame = calibDataFrame.append(sessionDict['calibration'], ignore_index=True)
        trialInfoDataFrame = trialInfoDataFrame.append(sessionDict['trialInfo'], ignore_index=True)
else:
    print 'Reading the All Subject Pickle File'
    df = pd.read_pickle('AllSubjects_2.pickle')
    rawDataFrame = df['raw']
    processedDataFrame = df['processed']
    calibDataFrame = df['calibration']
    trialInfoDataFrame = df['trialInfo']
    

if saveSubjectPickle == True:
    sessionDict['raw'] = rawDataFrame
    sessionDict['processed'] = processedDataFrame
    sessionDict['calibration'] = calibDataFrame
    sessionDict['trialInfo'] = trialInfoDataFrame
    pd.to_pickle(sessionDict, 'AllSubjects_2.pickle')
    print 'All Subject Pickle Saved'

processedDataFrame.loc[:, ('headVelocity','')] = pF.calculateHeadVelocity(rawDataFrame, trialID = None, plottingFlag = False)
processedDataFrame.loc[:, ('ballVelocity','')] = pF.calculateBallVelocity(rawDataFrame, processedDataFrame, trialID = None, plottingFlag = False)

trialStartIdx = processedDataFrame[processedDataFrame['eventFlag'] == 'trialStart'].index.tolist()
ballOffIdx = processedDataFrame[processedDataFrame['eventFlag'] == 'ballRenderOff'].index.tolist()
ballOnIdx = processedDataFrame[processedDataFrame['eventFlag'] == 'ballRenderOn'].index.tolist()
ballOnPaddleIdx = processedDataFrame[processedDataFrame['eventFlag'] == 'ballOnPaddle'].index.tolist()

ballCrossingIdx = np.zeros(len(trialInfoDataFrame), dtype = int)
ballCrossingIdx[trialInfoDataFrame.ballCaughtQ.values == True] = processedDataFrame[processedDataFrame['eventFlag'] == 'ballOnPaddle'].index.tolist()
ballCrossingIdx[trialInfoDataFrame.ballCaughtQ.values == False] = processedDataFrame[processedDataFrame['eventFlag'] == 'ballCrossingPaddle'].index.tolist()

trialInfoDataFrame.loc[:, ('trialStartIdx','')] = trialStartIdx
trialInfoDataFrame.loc[:, ('ballOffIdx','')] = ballOffIdx
trialInfoDataFrame.loc[:, ('ballOnIdx','')] = ballOnIdx
trialInfoDataFrame.loc[:, ('ballCrossingIndex','')] = ballCrossingIdx
print 'Number of Successful Trials: ', len(ballOnPaddleIdx), 'out of', len(trialStartIdx)
print 'Done!'

# ====================================================================================================================
# ====================================================================================================================

eyeToScreenDistance = 0.0725
handPosition = np.zeros((1,3), dtype = float)
ballPosition = np.zeros((1,2), dtype = float)
gazePosition = np.zeros((1,2), dtype = float)
ballVelocity = np.array([], dtype = float)
gazeVelocity = np.array([], dtype = float)
renderingFlag = np.array([], dtype = float)
trialEndIndex = np.array([], dtype = float)

rawDataFrame.ix[rawDataFrame.isBallVisibleQ.values == False, ('ballPos','X')] = 0.0
rawDataFrame.ix[rawDataFrame.isBallVisibleQ.values == False, ('ballPos','Y')] = 0.0
rawDataFrame.ix[rawDataFrame.isBallVisibleQ.values == False, ('ballPos','Z')] = 0.0
processedDataFrame.ix[rawDataFrame.isBallVisibleQ.values == False, ('ballVelocity')] = 0.0
for i in range(len(trialInfoDataFrame)):
    if (trialInfoDataFrame.ballCaughtQ.values[i] == False):
        endFrame = trialInfoDataFrame.ballCrossingIndex.values[i]

        x = np.array(processedDataFrame.ballOnScreen.X.values[trialStartIdx[i]:endFrame])
        y = np.array(processedDataFrame.ballOnScreen.Y.values[trialStartIdx[i]:endFrame])
        processedDataFrame.loc[trialStartIdx[i]:endFrame-1, ('ballOnScreen','X')] = (180/np.pi)*np.arctan(x.astype(float)/eyeToScreenDistance)
        processedDataFrame.loc[trialStartIdx[i]:endFrame-1, ('ballOnScreen','Y')] = (180/np.pi)*np.arctan(y.astype(float)/eyeToScreenDistance)
        tempVar = processedDataFrame.ballOnScreen.values[trialStartIdx[i]:endFrame, 0:2]
        ballPosition = np.vstack((ballPosition, tempVar))

        tempVar = rawDataFrame.paddlePos.values[trialStartIdx[i]:endFrame]
        handPosition = np.vstack((handPosition, tempVar))

        x = processedDataFrame.gazePoint.X.values[trialStartIdx[i]:endFrame]
        y = processedDataFrame.gazePoint.Y.values[trialStartIdx[i]:endFrame]
        processedDataFrame.loc[trialStartIdx[i]:endFrame-1, ('gazePoint','X')] = (180/np.pi)*np.arctan(x/eyeToScreenDistance)
        processedDataFrame.loc[trialStartIdx[i]:endFrame-1, ('gazePoint','Y')] = (180/np.pi)*np.arctan(y/eyeToScreenDistance)
        tempVar = processedDataFrame.gazePoint.values[trialStartIdx[i]:endFrame, 0:2]
        gazePosition = np.vstack((gazePosition, tempVar))

        tempVar = processedDataFrame.ballVelocity.values[trialStartIdx[i]:endFrame]
        ballVelocity = np.hstack((ballVelocity, tempVar))

        tempVar = processedDataFrame.cycGazeVelocity.values[trialStartIdx[i]:endFrame]
        gazeVelocity = np.hstack((gazeVelocity, tempVar))

        tempVar = rawDataFrame.isBallVisibleQ.values[trialStartIdx[i]:endFrame]
        renderingFlag = np.hstack((renderingFlag, tempVar))
        tempVar = np.zeros(endFrame - trialStartIdx[i])
        tempVar[-1] = 1
        trialEndIndex = np.hstack((trialEndIndex, tempVar))
ballPosition = np.delete(ballPosition, 0,0)
handPosition = np.delete(handPosition, 0, 0)
gazePosition = np.delete(gazePosition, 0, 0)
print ballPosition.shape, handPosition.shape, gazePosition.shape, ballVelocity.shape, gazeVelocity.shape, renderingFlag.shape
dataBase = np.zeros((ballPosition.shape[0], 10))
dataBase[:,0:3] = handPosition
dataBase[:,3:5] = gazePosition[:,0:2]
dataBase[:,5] = gazeVelocity
dataBase[:,6:8] = ballPosition
dataBase[:,8] = ballVelocity
dataBase[:,9] = renderingFlag
print dataBase.shape
print dataBase.mean(axis = 0)
print dataBase.std(axis = 0)
print dataBase.max(axis = 0)
print dataBase.min(axis = 0)
print len(trialEndIndex)

# ====================================================================================================================
# ====================================================================================================================

print 'Data Size = ', dataBase.shape
print '\nMean Values = ', dataBase.mean(axis = 0)
print '\nSTD x = ', dataBase.std(axis = 0)
print '\nMax Values = ', dataBase.max(axis = 0)
print '\nMin Values = ', dataBase.min(axis = 0)
#pd.to_pickle(dataBase, 'DataSet_Unformatted_Fail.pickle')
np.save('DataSet_Unformatted_False', dataBase)

# ====================================================================================================================
# ====================================================================================================================

inputLength = range(5,21)
#inputLength = [20]

for count in inputLength:
    myCounter = 0
    formattedDataBase = np.zeros((10,count, dataBase.shape[0] - count))
    for i in range(dataBase.shape[0] - count ):
        #if (all(v == 0 for v in trialEndIndex[i:i+count])):
        formattedDataBase[:,:,i] = dataBase[i:i+count, :].T
        #elif trialEndIndex[i+count-1] == 1:
            #print trialEndIndex[i:i+count]
        #    formattedDataBase[:,:,i] = dataBase[i:i+count, :].T
        #    myCounter = myCounter + 1
    #print myCounter

    print 'Final DataBase Size = ', formattedDataBase.shape
    pd.to_pickle(formattedDataBase, 'DataSet_'+ str(count)+'_False.pickle')

sampleIndex = 0
print 'Just To make sure that the data is correctly formatted'
print 'First Sample =  \n',formattedDataBase[:,:,sampleIndex]
print '\n\nSecond Sample = \n', formattedDataBase[:,:,sampleIndex+1]
print '\n\nThird Sample = \n', formattedDataBase[:,:,sampleIndex+2]