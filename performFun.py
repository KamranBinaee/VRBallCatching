from __future__ import division
import PerformParser as pp
import pandas as pd
import numpy as np
import sys
#import bokeh.plotting as bkP
#import bokeh.models as bkM
from scipy import signal as sig
import catchE1Funs as expFun
import cv2
import Quaternion as qu
import matplotlib.pyplot as plt

from configobj import ConfigObj
from configobj import flatten_errors
from validate import Validator
def plotMainEffect(dF,xVar,yVar, plotTitle = False):
    groupedBy =  dF.groupby(xVar)
    pre_mean = []
    pre_std = []
    for gNum, gr in groupedBy:
        pre_mean.append(np.mean(gr[yVar].values))
        pre_std.append(np.std(gr[yVar].values))
    xs = np.unique(dF[xVar].round(decimals =2))
    yerr = pre_std
    ys = pre_mean
    p = bkP.figure(width=800, height=400)
    p.line(xs, ys, color='orange',line_width=2)
    errorbar(p,xs,ys, yerr=yerr,point_kwargs={'size': 10}, error_kwargs={'line_width': 3})
    p.xaxis.axis_label = xVar
    p.yaxis.axis_label = ''.join(yVar)
    if( plotTitle ):
        p.title = plotTitle
    return p
def errorbar(fig, x, y, xerr=None, yerr=None, color='red', point_kwargs={}, error_kwargs={}):
    fig.circle(x, y, color=color, **point_kwargs)
    if xerr is not None:
      x_err_x = []
      x_err_y = []
      for px, py, err in zip(x, y, xerr):
          x_err_x.append((px - err, px + err))
          x_err_y.append((py, py))
      fig.multi_line(x_err_x, x_err_y, color=color, **error_kwargs)
    if yerr is not None:
      y_err_x = []
      y_err_y = []
      for px, py, err in zip(x, y, yerr):
          y_err_x.append((px, px))
          y_err_y.append((py - err, py + err))
      fig.multi_line(y_err_x, y_err_y, color=color, **error_kwargs)
def plotInteraction(trialInfoDF,xVar,yVar, lineVar):
    lineNames = np.array(np.unique(trialInfoDF[lineVar].round(decimals =2)),dtype=np.str)
    groupedByPre =  trialInfoDF.groupby([lineVar,xVar])
    mean_cond = []
    std_cond  = []
    for gNum, gr in groupedByPre:
        mean_cond.append(np.mean(gr[yVar].values))
        std_cond.append(np.std(gr[yVar].values))
    mean_pre_post = np.reshape(mean_cond,[3,3])
    std_pre_post = np.reshape(std_cond,[3,3])
    xs = [np.unique(trialInfoDF[xVar].round(decimals =2))]*3
    ys = [np.array(xyz,dtype=np.float) for xyz in mean_pre_post]
    yerr = [np.array(xyz,dtype=np.float) for xyz in std_pre_post]
    p = bkP.figure(width=800, height=400)
    off = [-.01, 0, .01]
    clist = ['blue','orange','green']
    for j in range(3):
        p.line( xs[j]+off[j], ys[j] ,line_width=3,color=clist[j],legend = lineVar + ' ' +  lineNames[j])
        errorbar(p,xs[j]+off[j],ys[j], yerr=yerr[j],color=clist[j], point_kwargs={'line_width':3,'size': 10}, 
                 error_kwargs={'line_width': 3})
    p.xaxis.axis_label = xVar
    p.yaxis.axis_label = ''.join(yVar)
    return p

def smi1x3_to_vizard1x4(dataIn_n_xyz, eyeOffsetX, numReps = 1):
    
    '''
    Converts dataIn_n_xyz into an Nx4 array, with eyeOffsetX added to the [0] column.  
    DataIn may be either a 3 element list (XYZ values) or an N x XYZ array, where N >1 (and equal to the number of rows of the original raw dataframe)

    Output is an nx4 array in which IOD has been added to the [0] column
    '''

    # If needed, tile dataIn_fr_xyz to match length of dataIn_fr_xyzw
    if(numReps == 0):
        
        raise NameError('numReps must be >0. To retain shape of current ')
        
    elif(numReps == 1):
        
        dataIn_fr_xyzw = np.tile([eyeOffsetX, 0, 0, 1],[len(dataIn_n_xyz),1])
        dataIn_fr_xyzw[:,:3] = dataIn_fr_xyzw[:,:3] + dataIn_n_xyz
        
    else:
        
        dataIn_fr_xyzw = np.tile([eyeOffsetX, 0, 0, 1],[numReps,1])
        dataIn_fr_xyzw[:,:3] = dataIn_fr_xyzw[:,:3] + np.tile(dataIn_n_xyz,[numReps,1])

    # Negate the SMI X axis for consistency with the Vizard FOR
    dataIn_fr_xyzw[:,0] = np.negative(dataIn_fr_xyzw[:,0])

    return dataIn_fr_xyzw

def EIH_to_GIW(rawDF,dataIn,eyeString,label):
    '''
    This function takes XYZ data in eye centered coordinates (XYZ) and transforms it into world centered coordinates.
    
    - rawDF must be the raw dataframe containing the transform matrix for the mainview
    
    - dataIn may be:
        - a dataframe of XYZ data
        - a 3 element list of XYZ data

    - eyeString is a string indicating which FOR contains dataIn, and may be of the values ['cyc','right','left'] 
    
    '''
    
    # Which eye?
    if( eyeString in ['cyc','right','left'] is False):
        raise NameError('Second argin (eyeType) must be cyc, right, or left')
    
    # Set IOD
    if(eyeString== 'cyc'):
        eyeOffsetX = 0.0
        
    elif(eyeString== 'right'):
        eyeOffsetX = -rawDF['IOD'].mean()/2/1000
        
    elif(eyeString== 'left'):
        eyeOffsetX = rawDF['IOD'].mean()/2/1000
        
    ######################################################################
    ######################################################################
    # Prepare the data.  After the next block of code, the data should be
    # an np.array of size [nFrames, XYZW]
    
    if( type(dataIn) == pd.DataFrame ):
    # dataIn is a dataframe
        
        vec_fr_XYZW = smi1x3_to_vizard1x4(dataIn.values,eyeOffsetX,1)
        
    elif( len(np.shape(dataIn)) == 1 and len(dataIn) == 3):
    # dataIn is a 3 element list
        vec_fr_XYZW = smi1x3_to_vizard1x4(dataIn,eyeOffsetX, len(rawDF.viewMat))

    else:
        raise NameError('Third argin (dataIn) must be either an Nx3 dataframe or np.array, where N = num rows in rawDF (the frist argument passed into this function)') 
    
    ######################################################################    
    ######################################################################
    
    # Convert viewmat data into 4x4 transformation matrix
    viewMat_fr_4x4 = [np.reshape(mat,[4,4]) for mat in rawDF.viewMat.values]
    
    # Take the dot product of vec_fr_XYZW and viewMat_fr_4x4
    dataOut_fr_XYZ = np.array([np.dot(vec_fr_XYZW[fr].T,viewMat_fr_4x4[fr])
                              for fr in range(len(vec_fr_XYZW))])
    
    # Discard the 4th column
    dataOut_fr_XYZ = dataOut_fr_XYZ[:,:3]
    
    # Turn it into a dataframe
    dataOutDf = pd.DataFrame(dataOut_fr_XYZ)

    # Rename the columns
    mIndex = pd.MultiIndex.from_tuples([(label,'X'),(label,'Y'),(label,'Z')])
    dataOutDf.columns = mIndex
    
    return dataOutDf    

def rawDataProcessing(sessionDict):
    rawDataFrame = sessionDict['raw']
    processedDataFrame = sessionDict['processed']
    calibDataFrame = sessionDict['calibration']
    trialInfoDataFrame = sessionDict['trialInfo']

    ################################################################################################################
    ################################## Copying Quaternion from Raw to Processed Data Frame #########################
    processedDataFrame.loc[:,('viewQuat', 'X')] = rawDataFrame['viewQuat']['X']
    processedDataFrame.loc[:,('viewQuat', 'Y')] = rawDataFrame['viewQuat']['Y']
    processedDataFrame.loc[:,('viewQuat', 'Z')] = rawDataFrame['viewQuat']['Z']
    processedDataFrame.loc[:,('viewQuat', 'W')] = rawDataFrame['viewQuat']['W']

    ################################################################################################################
    ####################################### Calculate True POR on the Screen #######################################
    # Creating an Empty Data Frame to store the filtered Data and attach it to the processed Data Frame
    print ('\nLow Pass/Median Filtering Gaze Data...')
    cyc_X = rawDataFrame['cycEyeOnScreen']['X'].values
    cyc_Y = rawDataFrame['cycEyeOnScreen']['Y'].values
    #print('===> Left Eye Hacked Due to poor calibration (KAMRAN)')
    #cyc_X = rawDataFrame['rightEyeOnScreen']['X'].values
    #cyc_Y = rawDataFrame['rightEyeOnScreen']['Y'].values
    #rightEyeOnScreen

    medianCyc_X = pd.rolling_median(cyc_X, 3, min_periods = 0)
    medianCyc_Y = pd.rolling_median(cyc_Y, 3, min_periods = 0)

    processedDataFrame.loc[:,('medFilt3_cycEyeOnScreen', 'X')] = pd.rolling_median(cyc_X, 3, min_periods = 0)
    processedDataFrame.loc[:,('medFilt3_cycEyeOnScreen', 'Y')] = pd.rolling_median(cyc_Y, 3, min_periods = 0)
    processedDataFrame.loc[:,('medFilt5_cycEyeOnScreen', 'X')] = pd.rolling_median(cyc_X, 5, min_periods = 0)
    processedDataFrame.loc[:,('medFilt5_cycEyeOnScreen', 'Y')] = pd.rolling_median(cyc_Y, 5, min_periods = 0)
    processedDataFrame.loc[:,('medFilt7_cycEyeOnScreen', 'X')] = pd.rolling_median(cyc_X, 7, min_periods = 0)
    processedDataFrame.loc[:,('medFilt7_cycEyeOnScreen', 'Y')] = pd.rolling_median(cyc_Y, 7, min_periods = 0)

    processedDataFrame.loc[:,('avgFilt3_cycEyeOnScreen', 'X')] = pd.rolling_mean(medianCyc_X, 3, min_periods = 0)
    processedDataFrame.loc[:,('avgFilt3_cycEyeOnScreen', 'Y')] = pd.rolling_mean(medianCyc_Y, 3, min_periods = 0)
    processedDataFrame.loc[:,('avgFilt5_cycEyeOnScreen', 'X')] = pd.rolling_mean(medianCyc_X, 5, min_periods = 0)
    processedDataFrame.loc[:,('avgFilt5_cycEyeOnScreen', 'Y')] = pd.rolling_mean(medianCyc_Y, 5, min_periods = 0)
    print ('... Done!')

    ################################################################################################################
    ####################################### Calculate Cyclopean Gaze Velocity ######################################
    print ('\nCalculating Cyc Gaze Velocity...')
    processedDataFrame.loc[:, ('cycGazeVelocity', '')] = calculateCycGazeVelocity(processedDataFrame,
                                                                                    trialID = None, plottingFlag = False)
    print ('... Done!')

    print ('\nCalculating True POR on Screen...')
    truePOR = find2DTruePOR(calibDataFrame)
    calibDataFrame.loc[:, ('cycTruePOR', 'X')] = truePOR[:,0]
    calibDataFrame.loc[:, ('cycTruePOR', 'Y')] = truePOR[:,1]
    calibDataFrame.loc[:, ('cycTruePOR', 'Z')] = truePOR[:,2]
    print ('\nCalculating True POR on Screen...')
    #print 'Size of Calibration Data Frame: ', len(calibDataFrame)

    ################################################################################################################
    ####################################### Calculate Linear Calibration Homography ################################
    print ('\nCalculating Calibration Homography...')
    H = calculateLinearHomography(calibDataFrame, plottingFlag = False)
    print ('H = ', H)
    tempMatrix = np.zeros((len(processedDataFrame),3,3))
    tempMatrix[:,0:3,0:3] = H
    tempMatrix = tempMatrix.reshape((len(processedDataFrame),9))
    #print 'H =',H
    #print 'M =', tempMatrix

    processedDataFrame.loc[:, ('linearHomography','0')] = tempMatrix[:,0]
    processedDataFrame.loc[:, ('linearHomography','1')] = tempMatrix[:,1]
    processedDataFrame.loc[:, ('linearHomography','2')] = tempMatrix[:,2]
    processedDataFrame.loc[:, ('linearHomography','3')] = tempMatrix[:,3]
    processedDataFrame.loc[:, ('linearHomography','4')] = tempMatrix[:,4]
    processedDataFrame.loc[:, ('linearHomography','5')] = tempMatrix[:,5]
    processedDataFrame.loc[:, ('linearHomography','6')] = tempMatrix[:,6]
    processedDataFrame.loc[:, ('linearHomography','7')] = tempMatrix[:,7]
    processedDataFrame.loc[:, ('linearHomography','8')] = tempMatrix[:,8]
    print ('... Done!')

    ################################################################################################################
    ####################################### Calculate Gaze & Ball Position on the Screen ###########################

    trialID = None

    print ('\nCalculating Gaze & Ball On Screen...')
    [gazePoint_fr_XYZ, rotatedGazePoint_fr_XYZ,
     ballOnScreen_fr_XYZ, rotatedBallOnScreen_fr_XYZ] = calculateGazeBallOnScreen(rawDataFrame, processedDataFrame, trialID)
    processedDataFrame.loc[:, ('gazePoint', 'X')] = gazePoint_fr_XYZ[:,0]
    processedDataFrame.loc[:, ('gazePoint', 'Y')] = gazePoint_fr_XYZ[:,1]
    processedDataFrame.loc[:, ('gazePoint', 'Z')] = gazePoint_fr_XYZ[:,2]

    processedDataFrame.loc[:, ('rotatedGazePoint', 'X')] = rotatedGazePoint_fr_XYZ[:,0]
    processedDataFrame.loc[:, ('rotatedGazePoint', 'Y')] = rotatedGazePoint_fr_XYZ[:,1]
    processedDataFrame.loc[:, ('rotatedGazePoint', 'Z')] = rotatedGazePoint_fr_XYZ[:,2]

    processedDataFrame.loc[:, ('ballOnScreen', 'X')] = ballOnScreen_fr_XYZ[:,0]
    processedDataFrame.loc[:, ('ballOnScreen', 'Y')] = ballOnScreen_fr_XYZ[:,1]
    processedDataFrame.loc[:, ('ballOnScreen', 'Z')] = ballOnScreen_fr_XYZ[:,2]

    processedDataFrame.loc[:, ('rotatedBallOnScreen', 'X')] = rotatedBallOnScreen_fr_XYZ[:,0]
    processedDataFrame.loc[:, ('rotatedBallOnScreen', 'Y')] = rotatedBallOnScreen_fr_XYZ[:,1]
    processedDataFrame.loc[:, ('rotatedBallOnScreen', 'Z')] = rotatedBallOnScreen_fr_XYZ[:,2]
    print ('... Done!')

    ################################################################################################################
    ############################################# Calculate Gaze Error for Eye In Head #############################

    print ('\nCalculating Gaze Error EIH ...')
    gazeX = processedDataFrame.gazePoint.X.values
    gazeY = processedDataFrame.gazePoint.Y.values
    gazeZ = processedDataFrame.gazePoint.Z.values

    ballX = processedDataFrame.ballOnScreen.X.values
    ballY = processedDataFrame.ballOnScreen.Y.values
    ballZ = processedDataFrame.ballOnScreen.Z.values

    gazeVector = np.array([gazeX, gazeY, gazeZ], dtype = float).T
    ballVector = np.array([ballX, ballY, ballZ], dtype = float).T
    #print gazeVector.shape
    angle = []
    for gV, bV in zip(gazeVector, ballVector):
        angle.append(vectorAngle(gV,bV))
    #len(angle)
    processedDataFrame.loc[:,('gazeError_HCS', '')] = angle
    print ('... Done!')

    ################################################################################################################
    ############################################# Calculate Gaze Error for Gaze in World ###########################
    print ('\nCalculating Gaze Error GIW ...')
    angleList = []
    for i in range(len(processedDataFrame)):
        temp = vectorAngle(processedDataFrame.rotatedGazePoint.values[i], processedDataFrame.rotatedBallOnScreen.values[i])
        angleList.append(temp)

    eyeToScreenDistance = 0.0725
    gazeError = np.array(processedDataFrame.rotatedGazePoint.values - processedDataFrame.rotatedBallOnScreen.values)
    x = gazeError[:,0]
    y = gazeError[:,1]
    z = gazeError[:,2]

    x = np.array(x,dtype = float)
    y = np.array(y,dtype = float)
    z = np.array(z,dtype = float)

    processedDataFrame.loc[:,('gazeError_WCS','X')] = (180/np.pi)*np.arctan(x/eyeToScreenDistance)
    processedDataFrame.loc[:,('gazeError_WCS','Y')] = (180/np.pi)*np.arctan(y/eyeToScreenDistance)
    processedDataFrame.loc[:,('gazeError_WCS','Z')] = (180/np.pi)*np.arctan(z/eyeToScreenDistance)
    processedDataFrame.loc[:,('gazeAngularError', '')] = np.array(angleList, dtype = float)
    print ('... Done!')

    ################################################################################################################
    ############################################# Calculate Head and Ball Angular Velocity in WCS ##################
    print ('\nCalculating Ball and Head Velocity ...')
    processedDataFrame.loc[:, ('headVelocity','')] = calculateHeadVelocity(rawDataFrame, trialID = None, plottingFlag = False)
    processedDataFrame.loc[:, ('ballVelocity','')] = calculateBallVelocity(rawDataFrame, processedDataFrame, trialID = None, plottingFlag = False)
    print ('.... Done!')

    ################################################################################################################
    ################################### Calculate Important Indexes during trial for further analysis ###############
    print ('\nCalculating Event Indexes ...')

    trialStartIdx = processedDataFrame[processedDataFrame['eventFlag'] == 'trialStart'].index.tolist()
    ballOffIdx = processedDataFrame[processedDataFrame['eventFlag'] == 'ballRenderOff'].index.tolist()
    ballOnIdx = processedDataFrame[processedDataFrame['eventFlag'] == 'ballRenderOn'].index.tolist()
    ballOnPaddleIdx = processedDataFrame[processedDataFrame['eventFlag'] == 'ballOnPaddle'].index.tolist()
    
    ballCrossingIdx = np.zeros(len(trialInfoDataFrame), dtype = int)
    ballCrossingIdx[trialInfoDataFrame.ballCaughtQ.values == True] = processedDataFrame[processedDataFrame['eventFlag'] == 'ballOnPaddle'].index.tolist()
    ballCrossingIdx[trialInfoDataFrame.ballCaughtQ.values == False] = processedDataFrame[processedDataFrame['eventFlag'] == 'ballCrossingPaddle'].index.tolist()
    
    trialInfoDataFrame.loc[:, 'trialStartIdx'] = trialStartIdx
    trialInfoDataFrame.loc[:, 'ballOffIdx'] = ballOffIdx
    trialInfoDataFrame.loc[:, 'ballOnIdx'] = ballOnIdx
    trialInfoDataFrame.loc[:, 'ballCrossingIndex'] = ballCrossingIdx
    print ('... Done!')
    ################################################################################################################

    sessionDict['raw'] = rawDataFrame
    sessionDict['processed'] = processedDataFrame
    sessionDict['calibration'] = calibDataFrame
    sessionDict['trialInfo'] = trialInfoDataFrame

    return sessionDict


def loadSessionDict(filePath,fileName,expCfgName,sysCfgName,startFresh = False):
    '''
    If startFresh is False, attempt to read in session dict from pickle.
    If pickle does not exist, or startFresh is True, 
    	- read session dict from raw data file
    	- create secondary dataframes
    '''
    if( startFresh == False):
        try:
            sessionDict = pd.read_pickle(filePath + fileName + '.pickle')
        except:
            print('SecDataFrame Called!')
            sessionDict = createSecondaryDataframes(filePath,fileName,expCfgName,sysCfgName)
            pd.to_pickle(sessionDict, filePath + fileName + '.pickle')
    else:
        sessionDict = createSecondaryDataframes(filePath,fileName,expCfgName,sysCfgName)
        pd.to_pickle(sessionDict, filePath + fileName + '.pickle')
    return sessionDict	
def createSecondaryDataframes(filePath,fileName,expCfgName,sysCfgName):
    '''
    Separates practice and calibration trials from main dataframe.
    Reads in exp and sys config.
    '''
    print('Creating Secondary Data Frame')
    sessionDf = pp.readPerformDict(filePath + fileName + ".dict")
    [sessionDf, calibDf] = seperateCalib(sessionDf)    

    expConfig =  createExpCfg(filePath + expCfgName)
    sysConfig =  createSysCfg(filePath + sysCfgName)
    practiceBlockIdx = [idx for idx, s in enumerate(expConfig['experiment']['blockList']) if s == 'practice']

    [sessionDf, practiceDf] =  seperatePractice(sessionDf,practiceBlockIdx)

    sessionDf = sessionDf.reset_index()
    sessionDf = sessionDf.rename(columns = {'index':'frameNumber'})

    trialInfoDf = expFun.initTrialInfo(sessionDf)

    procDataDf = expFun.initProcessedData(sessionDf)
    myHackVar = True
    if (myHackVar is False):
        paddleDF   = expFun.calcPaddleBasis(sessionDf)
        procDataDf = pd.concat([paddleDF,procDataDf],axis=1)
    else:
        print ('Hacked the Paddle Calculation (Kamran)')
    sessionDict = {'raw': sessionDf, 'processed': procDataDf, 'calibration': calibDf, 'practice': practiceDf, 
    'trialInfo': trialInfoDf,'expConfig': expConfig,'sysCfg': sysConfig}

    print('\nSecondary Data Frame Created!\n')
    return sessionDict

### Save calibration frames in a separate dataframe

def seperateCalib(sessionDf):
    calibFrames = sessionDf['trialNumber']>999
    calibDf = sessionDf[calibFrames]
    sessionDf = sessionDf.drop(sessionDf[calibFrames].index)
    return sessionDf, calibDf

def seperatePractice(sessionDf,practiceBlockIdx):
    
    practiceDf = pd.DataFrame()
    
    for bIdx in practiceBlockIdx:
    	#print 'Seperating practice block: ' + str(bIdx)    
        thisPracticeBlockDF = sessionDf[sessionDf['blockNumber']==bIdx]
        practiceDf = pd.concat([practiceDf,thisPracticeBlockDF],axis=0)
        sessionDf = sessionDf.drop(thisPracticeBlockDF.index)
        
    return sessionDf, practiceDf


def findFirstZeroCrossing(vecIn):
    '''
    This will return the index of the first zero crossing of the input vector
    '''
    return np.where(np.diff(np.sign(vecIn)))[0][0]

def findFirst(dataVec,targetVal):
    '''
    Reports the first occurance of targetVal in dataVec.
    If no occurances found, returns None
    '''
    return next((fr for fr, eF in enumerate(dataVec) if eF == targetVal),False)


def calcAngularVelocity(vector_fr,deltaTime_fr):
    '''
    Moving window takes cosine of the dot product for adjacent values.
    Appends a 0 onto the end of the vector.
    '''
        
    angularDistance_fr = np.array(  [ np.rad2deg(np.arccos( np.dot( vector_fr[fr,:],vector_fr[fr-1,:])))
         for fr in range(1,len(vector_fr))]) # if range starts at 0, fr-1 wil be -1.  Keep range from 1:len(vector)

    angularDistance_fr = np.append(0, angularDistance_fr)
    angularVelocity_fr = np.divide(angularDistance_fr,deltaTime_fr)

    return angularVelocity_fr


def timeSeries( frametime_fr=None, yDataList=None,yLabel=None,legendLabels=None,yLims = [0,300], events_fr=None,trialsStarts_tr=None,plotHeight=500,plotWidth = 1000):
    ''' 
    Creates a time-series plot of gaze data with Bokeh.
    dataFrame = a dataframe with field ['frameTime'], ['eventFlag'], and ['trialNumber'] 
    yLabel = A label for the Y axis. 
    yDataList = A list of vectors to be plotted on the Y axis as a line
    legendLabels = A list of names for data plotted on the Y axis
    yMax = Height of Y axidafdataFrames
    markEvents= Show vertical lines with labels at events in dataFrame['eventFlag']
    markTrials=Show vertical lines with labels at start of each trial
    '''
    from bokeh.palettes import Spectral6
    
    if( isinstance(yDataList, list) is False):
        raise TypeError('yDataList should be a list of lists.  Try [yData].')
    
    if( legendLabels and isinstance(legendLabels, list) is False):
        raise TypeError('legendLabels should be a list of lists.  Try [yLabelList].')
        
    #### Setup figure

    yRange = bkM.Range1d(yLims[0],yLims[1])

    p = bkP.figure(plot_width =plotWidth, plot_height=plotHeight,tools="xpan,reset,save,xwheel_zoom,resize,tap",
                   y_range=[0,500], 
                   x_range=[np.min(frametime_fr),np.max(frametime_fr)],
                   x_axis_label='time (s)', y_axis_label=yLabel)

    p.ygrid.grid_line_dash = [6, 4]

    #p.x_range = bkM.Range1d(dataFrame['frameTime'].values[0], dataFrame['frameTime'].values[0]+2)
    p.x_range = bkM.Range1d(np.min(frametime_fr), np.min(frametime_fr)+2)
    p.y_range = yRange

    ### Vertical lines at trial starts
    if( trialsStarts_tr ):
        
        X = [[startIdx]*2 for startIdx in trialsStarts_tr] #dataFrame.groupby('trialNumber')]
        Y = [[yLims[0],yLims[1]]] * len(trialsStarts_tr)
        p.multi_line(X,Y,color='red',alpha=0.6,line_width=2)

    ######################################################################################################
    ## Plot gaze velocity(s)
            
    for yIdx, yData in enumerate(yDataList):
        
        if(legendLabels and len(legendLabels) >= yIdx):
            p.line(frametime_fr,yData,line_width=3, alpha=.7,color=Spectral6[yIdx],legend=legendLabels[yIdx]) 
        else:
            p.line(frametime_fr,yData,line_width=3, alpha=.7,color=Spectral6[yIdx]) 

        #p.line(dataFrame['frameTime'].values,dataFrame['cycGIWVelocityRAW'].values,line_width=3,alpha=.7,color='green',legend="raw")
            

    ######################################################################################################
    ### Annotate events
    
    showHighBox = False
    
    #if( type(events_fr) is pd.Series ):
    if( events_fr.any() ):
        
        showHighBox = True
        X = frametime_fr[np.where(events_fr>2)]+.01
        Y = [yLims[1]*.9]*len(X)
        text = [str(event) for event in events_fr[np.where(events_fr>2)]]

        p.text(X,Y,text,text_font_size='8pt',text_font='futura')
        
        ### Vertical lines at events
        X = [ [X,X] for X in frametime_fr[np.where(events_fr>2)]]
        p.multi_line(X,[[yLims[0],yLims[1]*.9]] * len(X),color='red',alpha=0.6,line_width=2)

    if( trialsStarts_tr):
        
        showHighBox = True
        
        ### Annotate trial markers
        X = [trialStart+0.02 for trialStart in trialsStarts_tr]
        Y = [yLims[1]*.95] * len(trialsStarts_tr)
        text = [  'Tr ' + str(trIdx) for trIdx,trialStart in enumerate(trialsStarts_tr)]
        p.text(X,Y,text,text_font_size='10pt',text_font='futura',text_color='red')

        ### Vertical lines at trial starts
        X = [  [trialStart]*2 for trialStart in trialsStarts_tr]
        Y = [[yLims[0],yLims[1]*.9]] * len(trialsStarts_tr)
        p.multi_line(X,Y,color='red',alpha=0.6,line_width=4)

    if( showHighBox ):
        
        high_box = bkM.BoxAnnotation(plot=p, bottom = yLims[1]*.9, 
                                     top=yLims[1], fill_alpha=0.7, fill_color='green', level='underlay')
        p.renderers.extend([high_box])
        
    return p


def createExpCfg(expCfgPathAndName):
    """
    Parses and validates a config obj
    Variables read in are stored in configObj

    """

    print ("Loading experiment config file: " + expCfgPathAndName)
    
    from os import path
    filePath = path.dirname(path.abspath(expCfgPathAndName))

    # This is where the parser is called.
    expCfg = ConfigObj(expCfgPathAndName, configspec=filePath + '/expCfgSpec.ini', raise_errors = False, file_error = False)

    validator = Validator()
    expCfgOK = expCfg.validate(validator)
    if expCfgOK == True:
        print ("Experiment config file parsed correctly")
    else:
        print ('Experiment config file validation failed!')
        res = expCfg.validate(validator, preserve_errors=True)
        for entry in flatten_errors(expCfg, res):
        # 1each entry is a tuple
            section_list, key, error = entry
            if key is not None:
                section_list.append(key)
            else:
                section_list.append('[missing section]')
            section_string = ', '.join(section_list)
            if error == False:
                error = 'Missing value or section.'
            print (section_string, ' = ', error)
        sys.exit(1)
    if expCfg.has_key('_LOAD_'):
        for ld in expCfg['_LOAD_']['loadList']:
            print ('Loading: ' + ld + ' as ' + expCfg['_LOAD_'][ld]['cfgFile'])
            curCfg = ConfigObj(expCfg['_LOAD_'][ld]['cfgFile'], configspec = expCfg['_LOAD_'][ld]['cfgSpec'], raise_errors = True, file_error = True)
            validator = Validator()
            expCfgOK = curCfg.validate(validator)
            if expCfgOK == True:
                print ("Experiment config file parsed correctly")
            else:
                print ('Experiment config file validation failed!')
                res = curCfg.validate(validator, preserve_errors=True)
                for entry in flatten_errors(curCfg, res):
                # each entry is a tuple
                    section_list, key, error = entry
                    if key is not None:
                        section_list.append(key)
                    else:
                        section_list.append('[missing section]')
                    section_string = ', '.join(section_list)
                    if error == False:
                        error = 'Missing value or section.'
                    print (section_string, ' = ', error)
                sys.exit(1)
            expCfg.merge(curCfg)

    return expCfg


def createSysCfg(sysCfgPathAndName):
    """
    Set up the system config section (sysCfg)
    """

    # Get machine name
    #sysCfgName = platform.node()+".cfg"
    
    
    

    print ("Loading system config file: " + sysCfgPathAndName)

    # Parse system config file
    from os import path
    filePath = path.dirname(path.abspath(sysCfgPathAndName))
    
    sysCfg = ConfigObj(sysCfgPathAndName , configspec=filePath + '/sysCfgSpec.ini', raise_errors = False)

    validator = Validator()
    sysCfgOK = sysCfg.validate(validator)

    if sysCfgOK == True:
        print ("System config file parsed correctly")
    else:
        print ('System config file validation failed!')
        res = sysCfg.validate(validator, preserve_errors=True)
        for entry in flatten_errors(sysCfg, res):
        # each entry is a tuple
            section_list, key, error = entry
            if key is not None:
                section_list.append(key)
            else:
                section_list.append('[missing section]')
            section_string = ', '.join(section_list)
            if error == False:
                error = 'Missing value or section.'
            print (section_string, ' = ', error)
        #sys.exit(1)
    return sysCfg

def pixelsToMetric(x_pixel, y_pixel):
    '''
    0.126,0.071 = Screen size in meters according to SMI manual
    '''
    x = (0.126/1920)*np.subtract(x_pixel, 1920.0/2.0)
    y = (0.071/1080.0)*np.subtract(1080/2.0, y_pixel) # This line is diffetent than the one in Homography.py(KAMRAN)
    return x, y


def metricToPixels(x, y):
    x_pixel = (1920.0/0.126)*np.add(x, 0.126/2.0)
    y_pixel = (1080.0/0.071)*np.add(y, 0.071/2.0)
    return x_pixel, y_pixel

def plotMyData_Scatter3D(data, label, color, marker, axisLabels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,0], data[:,2], data[:,1], label = label, c = color, marker = marker)

    ax.set_xlabel(axisLabels[0])
    ax.set_ylabel(axisLabels[1])
    ax.set_zlabel(axisLabels[2])

    legend = plt.legend(loc=[1.,0.4], shadow=True, fontsize='small')# 'upper center'
    plt.show()

def plotMyData_2D(data, title, label, color, marker, axisLabels, dataRange = None):

        
    fig1 = plt.figure()
    plt.plot(data[:,0], data[:,1], label = label, c = color, marker = marker)

    #plt.xlim(xmin, xmax)
    #plt.ylim(ymin, ymax)
    plt.xlabel(axisLabels[0])
    plt.ylabel(axisLabels[1])
    plt.title(title)
    plt.grid(True)
    #plt.axis('equal')
    legend = plt.legend(loc=[1.,0.4], shadow=True, fontsize='small')# 'upper center'
    plt.show()

def dotproduct( v1, v2):
    r = sum((a*b) for a, b in zip(v1, v2))
    return r


def length(v):
    return np.sqrt(dotproduct(v, v))


def vectorAngle( v1, v2):
    r = (180.0/np.pi)*np.arccos((dotproduct(v1, v2)) / (length(v1) * length(v2)))#np.arccos((np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))))#
    return r

def plotFilteredGazeData(rawDataFrame, processedDataFrame):
    from bokeh.io import vplot
    from bokeh.plotting import figure, output_file, show

    frameRange = range(20000,21000)
    x0 = rawDataFrame['cycEyeOnScreen']['X'][frameRange].values
    x1 = processedDataFrame['medFilt3_cycEyeOnScreen']['X'][frameRange].values
    x2 = processedDataFrame['avgFilt3_cycEyeOnScreen']['X'][frameRange].values
    x3 = processedDataFrame['avgFilt5_cycEyeOnScreen']['X'][frameRange].values

    y0 = rawDataFrame['cycEyeOnScreen']['Y'][frameRange].values
    y1 = processedDataFrame['medFilt3_cycEyeOnScreen']['Y'][frameRange].values
    y2 = processedDataFrame['avgFilt3_cycEyeOnScreen']['Y'][frameRange].values
    y3 = processedDataFrame['avgFilt5_cycEyeOnScreen']['Y'][frameRange].values

    T = range(len(x1))

    dataColor = ['red','green', 'blue', 'yellow']
    dataLegend = ["Raw", "Med3", "Med5", "Med7"]
    p1 = figure(plot_width=500, plot_height=500)
    p1.multi_line(xs=[T,T,T,T], ys=[x0, x1, x2, x3],
                 color=dataColor)

    p2 = figure(plot_width=500, plot_height=500)
    p2.multi_line(xs=[T,T,T,T], ys=[y0, y1, y2, y3],
                 color=dataColor)

    p = vplot(p1,p2)
    show(p)


def calculateGazeBallOnScreen(rawDataFrame, processedDataFrame, trialID = None):

    minRange = 0
    if (trialID is None):
        rawTempDataFrame = rawDataFrame
        processedTempDataFrame = processedDataFrame
        maxRange = len(rawTempDataFrame)
    else:
        rawGBTrial = rawDataFrame.groupby(['trialNumber'])
        processedGBTrial = processedDataFrame.groupby(['trialNumber'])
        startFrame = 0
        maxRange = len(rawGBTrial.get_group(trialID))
        numberOfFrames = len(rawGBTrial.get_group(trialID))
        rawTempDataFrame = rawGBTrial.get_group(trialID)
        processedTempDataFrame = processedGBTrial.get_group(trialID)
    #print 'Trial ID = ', trialID, 'FrameCount = ', maxRange

    cycEyePosition_X = rawTempDataFrame['viewPos']['X'].values
    cycEyePosition_Y = rawTempDataFrame['viewPos']['Y'].values
    cycEyePosition_Z = rawTempDataFrame['viewPos']['Z'].values

    headQuat_X = rawTempDataFrame['viewQuat']['X'].values
    headQuat_Y = rawTempDataFrame['viewQuat']['Y'].values
    headQuat_Z = rawTempDataFrame['viewQuat']['Z'].values
    headQuat_W = rawTempDataFrame['viewQuat']['W'].values

    ballPosition_X = rawTempDataFrame['ballPos']['X'].values
    ballPosition_Y = rawTempDataFrame['ballPos']['Y'].values
    ballPosition_Z = rawTempDataFrame['ballPos']['Z'].values


    # =========================================================
    # === Instead of using raw POR we use the filtered one ====
    # =========================================================
    #cycPOR_X = tempDataFrame['cycEyeOnScreen']['X'].values
    #cycPOR_Y = tempDataFrame['cycEyeOnScreen']['Y'].values

    cycPOR_X = processedTempDataFrame['avgFilt3_cycEyeOnScreen']['X'].values
    cycPOR_Y = processedTempDataFrame['avgFilt3_cycEyeOnScreen']['Y'].values

    metricCycPOR_X = []
    metricCycPOR_Y = []
    metricCycPOR_Z = np.zeros(maxRange)
    constantValue = 1.0
    metricCycPOR_Z = metricCycPOR_Z + constantValue 

    cameraCenterPosition = np.array([0.0,0.0,0.0]) # in HCS
    planeNormal = np.array([0.0,0.0,1.0]) # in Both HCS and WCS
    eyetoScreenDistance = 0.0725 # This assumes that the Eye-Screen Distance is always constant
    screenCenterPosition = np.array([0.0,0.0,eyetoScreenDistance])

    #########################################################################
    #######  Pixel Values (SCS) ==> Metric Values (HCS)
    #########################################################################
    [metricCycPOR_X, metricCycPOR_Y] = pixelsToMetric(cycPOR_X, cycPOR_Y)

    #########################################################################
    ###  Extracting Rotation Matrix out of Quaternion for every Frame
    #########################################################################
    #viewRotMat_fr_mRow_mCol = [qu.Quat(np.array(q,dtype=np.float))._quat2transform() for q in tempDataFrame.viewQuat.values]    
    viewRotMat_fr_mRow_mCol = [quat2transform(q) for q in rawTempDataFrame.viewQuat.values]

    #########################################################################
    #######  Homogenous Coordinate Values of POR : [X,Y,1]
    #########################################################################
    metricCycPOR_fr_XYZ = np.array([metricCycPOR_X, metricCycPOR_Y, metricCycPOR_Z], dtype = float).T

    #########################################################################
    #######  Apply the Calibration MAtirx (Homography) on the POR Data (HCS)
    #########################################################################
    calibratedPOR_fr_XYZ = np.zeros((maxRange,3))
    #H = np.eye(3)######### Just to neglect the effect of Calibration ######
    H = processedDataFrame['linearHomography'].values[0].reshape((3,3))

    for i in range(maxRange):
        calibratedPOR_fr_XYZ[i,:] = np.dot(H, metricCycPOR_fr_XYZ[i,:])

    #########################################################################
    #######  Locating all the Calibrated POR points on the Screen (HCS)
    #######  From This point on they are all 3D Gaze Points
    #########################################################################
    calibratedPOR_fr_XYZ[:,2] = eyetoScreenDistance
    #print calibratedPOR_fr_XY[0:5, :]#metricCycPOR_fr_XYZ.shape

    #########################################################################
    #######  Apply the Rotation Matirx on the 3D Gaze Points (HCS)
    #######  The head rotation/orientation is now being taken into account
    #########################################################################
    gazePoint_fr_XYZ = np.array([ np.dot(viewRotMat_fr_mRow_mCol[fr], calibratedPOR_fr_XYZ[fr].T) 
         for fr in range(len(calibratedPOR_fr_XYZ))])

    #########################################################################
    #######  This Calculation is in "WCS"
    #######  Using Ray Tracing: BallOnCreen =  intersection of eye-ball line and the "Rotated" HMD plane (WCS)
    #########################################################################
    ballOnScreen_fr_XYZ = np.empty([1, 3], dtype = float)
    #truePOR.reshape((1,3))
    for i in range(maxRange):
        lineNormal = [np.subtract(rawTempDataFrame['ballPos'][i:i+1].values, rawTempDataFrame['viewPos'][i:i+1].values)]
        #print  'Line = ',lineNormal[0]
        rotatedNormalPlane = np.array( np.dot(viewRotMat_fr_mRow_mCol[i], planeNormal))
        rotatedScreenCenterPosition = np.dot(viewRotMat_fr_mRow_mCol[i], screenCenterPosition)
        tempPos = findLinePlaneIntersection( cameraCenterPosition, lineNormal[0], rotatedNormalPlane, rotatedScreenCenterPosition) 
        # TODO: I kinda cheated here by {lineNormal[0]}
        #print tempPos.shape
        #tempPos = np.subtract(tempPos, rotatedPoint1)
        ballOnScreen_fr_XYZ = np.vstack((ballOnScreen_fr_XYZ, tempPos[0].T))
    # TODO: I hate creating an empty variable and deleting it later on, there should be a better way
    ballOnScreen_fr_XYZ = np.delete(ballOnScreen_fr_XYZ, 0, 0)


    rotatedBallOnScreen_fr_XYZ = np.array([np.dot(np.linalg.inv(viewRotMat_fr_mRow_mCol[fr]), ballOnScreen_fr_XYZ[fr].T,) 
                                    for fr in range(len(ballOnScreen_fr_XYZ))])

    rotatedGazePoint_fr_XYZ = np.array([np.dot(np.linalg.inv(viewRotMat_fr_mRow_mCol[fr]), gazePoint_fr_XYZ[fr].T) 
                                    for fr in range(len(gazePoint_fr_XYZ))])
    #print ballOnScreen_fr_XYZ.shape

    return [gazePoint_fr_XYZ, rotatedGazePoint_fr_XYZ, ballOnScreen_fr_XYZ, rotatedBallOnScreen_fr_XYZ]

def find2DTruePOR(calibDataFrame):

    framesPerPoint = range(100)
    startFrame = 0
    endFrame = len(calibDataFrame)
    print ('Size of Calibration Data Frame is', endFrame)
    frameIndexRange = range(startFrame, endFrame)

    cameraCenterPosition = np.array([0.0,0.0,0.0])
    planeNormal = np.array([0.0,0.0,1.0])
    eyetoScreenDistance = 0.0725 # This assumes that the Eye-Screen Distance is always constant
    screenCenterPosition = np.array([0.0,0.0,eyetoScreenDistance])

    truePOR = np.empty([1, 3], dtype = float)
    for i in range(endFrame):
        lineNormal = calibDataFrame['calibrationPos'][['X','Y','Z']][i:i+1].values
        tempPos = findLinePlaneIntersection( cameraCenterPosition, lineNormal[0], planeNormal, screenCenterPosition ) # TODO: I kinda cheated here by {line[0]}
        truePOR = np.vstack((truePOR, tempPos))
    # TODO: I hate creating an empty variable and deleting it later on there should be a better way
    truePOR = np.delete(truePOR, 0, 0)
    print ('Size of TruePOR array:', truePOR.shape)

    return truePOR

def calculateLinearHomography (calibDataFrame, plottingFlag = False):
    gbCalibSession = calibDataFrame.groupby(['trialNumber'])

    numberOfCalibrationSession =  (max(calibDataFrame.trialNumber.values)%1000)//100

    trialOffset = 1000;
    frameOffset = 0
    numberOfCalibrationPoints = 27

    ## ===============================
    ## ========= HACK (KAMRAN) =======
    ## ===============================
    numberOfCalibrationSession = 0
    for i in range(numberOfCalibrationSession + 1):

        startTrialNumber = trialOffset + 100*i
        endTrialNumber = trialOffset + 100*i + numberOfCalibrationPoints - 1
        firstCalibrationSession = gbCalibSession.get_group(startTrialNumber)
        lastCalibrationSession = gbCalibSession.get_group(endTrialNumber)

        numberOfCalibrationFrames = max(lastCalibrationSession.index) - min(firstCalibrationSession.index) + 1

        dataRange = range(frameOffset, frameOffset + numberOfCalibrationFrames)

        #a = np.append(gbCalibSession.get_group(1103).index, gbCalibSession.get_group(1108).index)
        #b = np.append(gbCalibSession.get_group(1110).index, gbCalibSession.get_group(1116).index)
        #c = np.append(a,b)
        #d = np.append(gbCalibSession.get_group(1212).index, gbCalibSession.get_group(1217).index)
        #dataRange = np.append(c,d)

        print ('Frame Range =[', min(dataRange),' ', max(dataRange), ']'  )

        [metricCycPOR_X, metricCycPOR_Y] = pixelsToMetric(calibDataFrame['cycEyeOnScreen']['X'][dataRange].values, calibDataFrame['cycEyeOnScreen']['Y'][dataRange].values)
        cyclopeanPOR_XY = np.array([metricCycPOR_X, metricCycPOR_Y], dtype = float)
        cyclopeanPOR_XY = cyclopeanPOR_XY.T

        truePOR_XY = np.array([calibDataFrame['cycTruePOR']['X'][dataRange].values, calibDataFrame['cycTruePOR']['Y'][dataRange].values], dtype = float)
        truePOR_XY = truePOR_XY.T

        print (cyclopeanPOR_XY.shape)
        H = calibrateData(cyclopeanPOR_XY, truePOR_XY, cv2.RANSAC, 10, plottingFlag)
        frameOffset = numberOfCalibrationFrames

    print (startTrialNumber, endTrialNumber)
    len(calibDataFrame)

    return H

def calculateCycGazeVelocity(dataFrame, trialID, coordinateSystem = 'WorldCoordinateSystem', plottingFlag = False):

    if (trialID is None):
        tempDataFrame = dataFrame
        numberOfFrames = len(tempDataFrame)
    else:
        gbTrial = dataFrame.groupby(['trialNumber'])
        startFrame = 0
        endFrame = len(gbTrial.get_group(trialID))
        numberOfFrames = len(gbTrial.get_group(trialID))
        tempDataFrame = gbTrial.get_group(trialID)
    cycPOR_X = tempDataFrame['avgFilt3_cycEyeOnScreen','X']# [startFrame:endFrame]
    cycPOR_Y = tempDataFrame['avgFilt3_cycEyeOnScreen','Y']# [startFrame:endFrame]
    metricCycPOR_X = []
    metricCycPOR_Y = []
    metricCycPOR_Z = np.zeros(numberOfFrames)
    averageEyetoScreenDistance = 0.0725 # This assumes that the Eye-Screen Distance is always constant
    metricCycPOR_Z = metricCycPOR_Z + averageEyetoScreenDistance 
    # Calculating the Metric position of POR
    [metricCycPOR_X, metricCycPOR_Y] = pixelsToMetric(cycPOR_X.values, cycPOR_Y.values)

    viewRotMat_fr_mRow_mCol = [qu.Quat(np.array(q,dtype=np.float))._quat2transform() for q in tempDataFrame['viewQuat'].values]    
    #print np.shape(viewRotMat_fr_mRow_mCol)

    # Now these PORs are some 3D points on the Screen still in HCS
    metricCycPOR_fr_XYZ = np.array([metricCycPOR_X, metricCycPOR_Y, metricCycPOR_Z], dtype = float).T

    #print metricCycPOR_fr_XYZ.shape

    if ( coordinateSystem == 'WorldCoordinateSystem' ):
        # If we want to calculate the Gaze vector in WCS we need to take into account Head Orientation
        # So we neead to multiply all 3D POR points with the Rotation Matrix calculated by Quaternions
        global_metricCycPOR_fr_XYZ = np.array([ np.dot(metricCycPOR_fr_XYZ[fr].T,viewRotMat_fr_mRow_mCol[fr]) 
             for fr in range(len(metricCycPOR_fr_XYZ))])
    else:
        # If not then the Gaze vector would be calculated in HCS 
        global_metricCycPOR_fr_XYZ = metricCycPOR_fr_XYZ

    # Now we want to calculate the derivative of position so we need to shift the array
    metricCycPOR_fr_XYZ_shifted = np.roll(global_metricCycPOR_fr_XYZ, -1, axis=0)

    timeArray = tempDataFrame.frameTime.values
    timeArray_shifted = np.roll(timeArray, -1)
    timeDiff = timeArray_shifted -  timeArray
    #print timeArray_shifted[-1], '\n'
    #print timeArray[-1], '\n'
    #print timeDiff
    #print global_metricCycPOR_fr_XYZ[0:5,:], '\n'
    #print metricCycPOR_fr_XYZ_shifted[0:5,:]
    #plotMyData_Scatter3D(metricCycPOR_fr_XYZ_shifted, label = 'Cyc Gaze Points', color = 'r', marker = 'o', axisLabels=['X [m]', 'Y [m]', 'Z [m]'])
    gazeAngle_fr = []
    for v1,v2 in zip(global_metricCycPOR_fr_XYZ, metricCycPOR_fr_XYZ_shifted):
        gazeAngle_fr.append(vectorAngle(v1, v2))
    gazeVelocity_fr = np.divide(gazeAngle_fr, timeDiff)
    if (plottingFlag == True):
        plotMyData_2D(np.array([tempDataFrame.frameTime.values, gazeVelocity_fr]).T, 'Raw Gaze Velocity for Trial '+ str(trialID), label='Gaze Velocity', color = 'r', marker = None, axisLabels = ['T (s)','Angular Velocity degree/s'])
    return gazeVelocity_fr

def calculateHeadVelocity(dataFrame, trialID, plottingFlag = False):

    if (trialID is None):
        rawDataFrame = dataFrame
        numberOfFrames = len(rawDataFrame)
    else:
        gbTrial = dataFrame.groupby(['trialNumber'])
        startFrame = 0
        endFrame = len(gbTrial.get_group(trialID))
        numberOfFrames = len(gbTrial.get_group(trialID))
        rawDataFrame = gbTrial.get_group(trialID)

    maxRange = len(rawDataFrame)
    cycEyePosition_X = rawDataFrame['viewPos']['X'].values
    cycEyePosition_Y = rawDataFrame['viewPos']['Y'].values
    cycEyePosition_Z = rawDataFrame['viewPos']['Z'].values

    headQuat_X = rawDataFrame['viewQuat']['X'].values
    headQuat_Y = rawDataFrame['viewQuat']['Y'].values
    headQuat_Z = rawDataFrame['viewQuat']['Z'].values
    headQuat_W = rawDataFrame['viewQuat']['W'].values

    ballPosition_X = rawDataFrame['ballPos']['X'].values
    ballPosition_Y = rawDataFrame['ballPos']['Y'].values
    ballPosition_Z = rawDataFrame['ballPos']['Z'].values

    metricHeadDir_X = np.zeros(maxRange)
    metricHeadDir_Y = np.zeros(maxRange)
    metricHeadDir_Z = np.zeros(maxRange)
    constantValue = 1.0
    metricHeadDir_Z = metricHeadDir_Z + constantValue

    cameraCenterPosition = np.array([0.0,0.0,0.0]) # in HCS
    planeNormal = np.array([0.0,0.0,1.0]) # in Both HCS and WCS
    eyetoScreenDistance = 0.0725 # This assumes that the Eye-Screen Distance is always constant
    screenCenterPosition = np.array([0.0,0.0,eyetoScreenDistance])

    #########################################################################
    ###  Extracting Rotation Matrix out of Quaternion for every Frame
    #########################################################################
    #viewRotMat_fr_mRow_mCol = [qu.Quat(np.array(q,dtype=np.float))._quat2transform() for q in tempDataFrame.viewQuat.values]    
    viewRotMat_fr_mRow_mCol = [qu.Quat(np.array(q,dtype=np.float))._quat2transform() for q in rawDataFrame.viewQuat.values]

    #########################################################################
    #######  Homogenous Coordinate Values of Head Dir Vector : [X,Y,1]
    #########################################################################
    metricHeadDir_fr_XYZ = np.array([metricHeadDir_X, metricHeadDir_Y, metricHeadDir_Z], dtype = float).T
    #########################################################################
    #######  Apply the Rotation Matirx on the 3D Head Dir Points (HCS)
    #######  The head rotation/orientation is now being taken into account
    #########################################################################

    headDirection_fr_XYZ = np.array([ np.dot(viewRotMat_fr_mRow_mCol[fr], metricHeadDir_fr_XYZ[fr].T) 
         for fr in range(len(metricHeadDir_fr_XYZ))])
    #myEulers_fr_XYZ = np.array([transformations.euler_from_quaternion(rawDataFrame.viewQuat.values[fr])
    #     for fr in range(len(metricHeadDir_fr_XYZ))])

    # Now we want to calculate the derivative of position so we need to shift the array
    headDirection_fr_XYZ_shifted = np.roll(headDirection_fr_XYZ, -1, axis=0)
    timeArray = rawDataFrame.frameTime.values[0:maxRange]
    timeArray_shifted = np.roll(timeArray, -1)
    timeDiff = timeArray_shifted -  timeArray
    headAngle_fr = []
    for v1,v2 in zip(headDirection_fr_XYZ, headDirection_fr_XYZ_shifted):
        headAngle_fr.append(vectorAngle(v1, v2))
    headVelocity_fr = np.divide(headAngle_fr, timeDiff)

    return headVelocity_fr

def calculateBallVelocity(rawDataFrame, processedDataFrame, trialID, plottingFlag = False):

    if (trialID is None):
        #processedDataFrame = sessionDict['processed']
        #rawDataFrame = sessionDict['raw']
        numberOfFrames = len(rawDataFrame)
    else:
        gbTrial = dataFrame.groupby(['trialNumber'])
        startFrame = 0
        endFrame = len(gbTrial.get_group(trialID))
        numberOfFrames = len(gbTrial.get_group(trialID))
        processedDataFrame = gbTrial.get_group(trialID)

    ballPosition_fr_XYZ = np.array([processedDataFrame['ballOnScreen']['X'].values,
                                    processedDataFrame['ballOnScreen']['Y'].values,
                                    processedDataFrame['ballOnScreen']['Z'].values], dtype = float).T
    ballPosition_fr_XYZ_shifted = np.roll(ballPosition_fr_XYZ, -1, axis=0)

    timeArray = rawDataFrame.frameTime.values
    timeArray_shifted = np.roll(timeArray, -1)
    timeDiff = timeArray_shifted -  timeArray
    #print timeArray_shifted[-1], '\n'
    #print timeArray[-1], '\n'
    #print timeDiff
    #print global_metricCycPOR_fr_XYZ[0:5,:], '\n'
    #print metricCycPOR_fr_XYZ_shifted[0:5,:]
    #plotMyData_Scatter3D(metricCycPOR_fr_XYZ_shifted, label = 'Cyc Gaze Points', color = 'r', marker = 'o', axisLabels=['X [m]', 'Y [m]', 'Z [m]'])
    ballAngle_fr = []
    for v1,v2 in zip(ballPosition_fr_XYZ, ballPosition_fr_XYZ_shifted):
        ballAngle_fr.append(vectorAngle(v1, v2))
    ballVelocity_fr = np.divide(ballAngle_fr, timeDiff)

    return ballVelocity_fr

def calculateCrossingFrame(rawDataFrame, processedDataFrame, trialInfoDataFrame):
    for tr in range(len(trialInfoDataFrame)):
        if (trialInfoDataFrame.ballCaughtQ.values[tr] == False):
            distance = rawDataFrame.ballPos.Z.values[rawDataFrame.trialNumber.values == tr] - rawDataFrame.paddlePos.Z.values[rawDataFrame.trialNumber.values == tr]
            for i in range(len(distance)):
                if (distance[i] < 0):
                    #print 'Passing Frame', i - 1
                    break
            passingFrameNumber = i - 1
            trialInfoDataFrame.loc[tr,('ballCaughtFr')] = trialInfoDataFrame.firstFrame.values[tr] + i - 1
            processedDataFrame.loc[trialInfoDataFrame.firstFrame.values[tr] + i - 1,('eventFlag')] = 'ballCrossingPaddle'
        #else:
            #print 'This was a Success Trial : ', tr
    ballOnPaddleIdx = processedDataFrame[processedDataFrame['eventFlag'] == 'ballOnPaddle'].index.tolist()

    ballCrossingIdx = np.zeros(len(trialInfoDataFrame), dtype = int)
    ballCrossingIdx[trialInfoDataFrame.ballCaughtQ.values == True] = processedDataFrame[processedDataFrame['eventFlag'] == 'ballOnPaddle'].index.tolist()
    ballCrossingIdx[trialInfoDataFrame.ballCaughtQ.values == False] = processedDataFrame[processedDataFrame['eventFlag'] == 'ballCrossingPaddle'].index.tolist()
    trialInfoDataFrame.loc[:, 'ballCrossingIndex'] = ballCrossingIdx

    return (processedDataFrame, trialInfoDataFrame)
#trialInfoDataFrame = calculateCrossingFrame()
#result = rawDataFrame.ballPos.values[j] -  rawDataFrame.paddlePos.values[j]
#print 'Result = ', result
                                                                                                                  

def createLine( point0, point1 ):
        
    unitVector = np.subtract(point0, point1)/length(np.subtract(point0, point1))
    #print 'unitVector', unitVector
    return unitVector

def findLinePlaneIntersection(point_0, line, planeNormal, point_1):
    
    s = point_1 - point_0
    numerator = dotproduct(s, planeNormal)
    denumerator = np.inner(line, planeNormal)
    if (denumerator == 0):
        print ('No Intersection')
        return None
    #print numerator, denumerator
    d = np.divide(numerator, denumerator)
    intersectionPoint = np.multiply(d, line) + point_0
    #print 'result', d, intersectionPoint
    return intersectionPoint

def findResidualError(projectedPoints, referrencePoints):
    e2 = np.zeros((projectedPoints.shape[0],2))
    for i in range(projectedPoints.shape[0]):
        temp = np.subtract(projectedPoints[i], referrencePoints[i])
        #print 'temp', temp
        e2[i,:] = np.power(temp[0:2], 2)
    return [np.sqrt(sum(sum(e2[:])))]

def calibrateData(cyclopeanPOR_XY, truePOR_XY, method = cv2.LMEDS, threshold = 10, plottingFlag = False):

    result = cv2.findHomography(cyclopeanPOR_XY, truePOR_XY)#, method , ransacReprojThreshold = threshold)
    #print result[0]
    #print 'size', len(result[1]),'H=', result[1]
    totalFrameNumber = truePOR_XY.shape[0]
    arrayOfOnes = np.ones((totalFrameNumber,1), dtype = float)

    homogrophy = result[0]
    #print 'H=', homogrophy, '\n'
    #print 'Res', result[1]
    cyclopeanPOR_XY = np.hstack((cyclopeanPOR_XY, arrayOfOnes))
    truePOR_XY = np.hstack((truePOR_XY, arrayOfOnes))
    projectedPOR_XY = np.zeros((totalFrameNumber,3))
    
    for i in range(totalFrameNumber):
        projectedPOR_XY[i,:] = np.dot(homogrophy, cyclopeanPOR_XY[i,:])
        #print cyclopeanPOR_XY[i,:]
    
    projectedPOR_XY[:, 0], projectedPOR_XY[:, 1] = metricToPixels(projectedPOR_XY[:, 0], projectedPOR_XY[:, 1])
    cyclopeanPOR_XY[:, 0], cyclopeanPOR_XY[:, 1] = metricToPixels(cyclopeanPOR_XY[:, 0], cyclopeanPOR_XY[:, 1])
    truePOR_XY[:, 0], truePOR_XY[:, 1] = metricToPixels(truePOR_XY[:, 0], truePOR_XY[:, 1])
    data = projectedPOR_XY
    frameCount = range(len(cyclopeanPOR_XY))

    if( plottingFlag == True ):
        xmin = 550#min(cyclopeanPOR_XY[frameCount,0])
        xmax = 1350#max(cyclopeanPOR_XY[frameCount,0])
        ymin = 250#min(cyclopeanPOR_XY[frameCount,1])
        ymax = 800#max(cyclopeanPOR_XY[frameCount,1])
        #print xmin, xmax, ymin, ymax
        fig1 = plt.figure()
        plt.plot(data[frameCount,0], data[frameCount,1], 'bx', label='Calibrated POR')
        plt.plot(cyclopeanPOR_XY[frameCount,0], cyclopeanPOR_XY[frameCount,1], 'g2', label='Uncalibrated POR')
        plt.plot(truePOR_XY[frameCount,0], truePOR_XY[frameCount,1], 'r8', label='Ground Truth POR')
        #l1, = plt.plot([],[])
        
        #plt.xlim(xmin, xmax)
        #plt.ylim(ymin, ymax)
        plt.xlabel('X')
        plt.ylabel('Y')
        if ( method == cv2.RANSAC):
            methodTitle = ' RANSAC '
        elif( method == cv2.LMEDS ):
            methodTitle = ' Least Median '
        elif( method == 0 ):
            methodTitle = ' Homography '
        plt.title('Calibration Result using'+ methodTitle+'\nWith System Calibration ')
        plt.grid(True)
        #plt.axis('equal')
        #line_ani = animation.FuncAnimation(fig1, update_line1, frames = 11448, fargs=(sessionData, l1), interval=14, blit=True)
        legend = plt.legend(loc=[0.8,0.92], shadow=True, fontsize='small')# 'upper center'
        plt.show()

    print ('MSE_after = ', findResidualError(projectedPOR_XY, truePOR_XY))
    print ('MSE_before = ', findResidualError(cyclopeanPOR_XY, truePOR_XY))
    return homogrophy

def quat2transform(q):
    """
    Transform a unit quaternion into its corresponding rotation matrix (to
    be applied on the right side).

    :returns: transform matrix
    :rtype: numpy array

    """
    x, y, z, w = q
    xx2 = 2 * x * x
    yy2 = 2 * y * y
    zz2 = 2 * z * z
    xy2 = 2 * x * y
    wz2 = 2 * w * z
    zx2 = 2 * z * x
    wy2 = 2 * w * y
    yz2 = 2 * y * z
    wx2 = 2 * w * x

    rmat = np.empty((3, 3), float)
    rmat[0,0] = 1. - yy2 - zz2
    rmat[0,1] = xy2 - wz2
    rmat[0,2] = zx2 + wy2
    rmat[1,0] = xy2 + wz2
    rmat[1,1] = 1. - xx2 - zz2
    rmat[1,2] = yz2 - wx2
    rmat[2,0] = zx2 - wy2
    rmat[2,1] = yz2 + wx2
    rmat[2,2] = 1. - xx2 - yy2

    return rmat