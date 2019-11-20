#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 14:40:49 2018

The goal of this python code is to read the interpolation results and plot the mean/std
@author: kamranbinaee
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
#pickle.dump( myResult, open( "interPolationResult.p", "wb" ) )
results = pickle.load(( open( "interPolationResult.p", "rb" ) ))

#[myWindow, myDegree, preBD, postBD, meanBallX[-1], meanBallY[-1] ,stdBallX[-1], stdBallY[-1]]
print(len(results))
print(len(results[0]))
#print(results)
plt.figure(figsize = (9,6))

#f, axarr = plt.subplots(1,4, sharey=True, figsize = (9,6))

t1 = []
t2 = []
t3 = []
x1 = []
x2 = []
x3 = []
xs1 = []
xs2 = []
xs3 = []
y1 = []
y2 = []
y3 = []
ys1 = []
ys2 = []
ys3 = []

myWindow = 15
myDegree = 4
for items in results:
    
    if not (items[0] == myWindow):
        continue
    if not (items[1] == myDegree):
        continue
    idx = items[1]
    print(items[0],items[1])
    if (items[2] ==  0.6):
        marker = 'o'
        t1.append(items[3]-0.005)
        x1.append(items[4])
        xs1.append(items[6])
        y1.append(items[5])
        ys1.append(items[7])        
    elif(items[2] ==  0.8):
        marker = 'v'
        t2.append(items[3])
        x2.append(items[4])
        xs2.append(items[6])
        y2.append(items[5])
        ys2.append(items[7])        
    elif(items[2] ==  1.0):
        marker = '^'
        t3.append(items[3]+0.005)
        x3.append(items[4])
        xs3.append(items[6])
        y3.append(items[5])
        ys3.append(items[7])
    else:
        print('Not Valid PreBD!!!!')
        
    #plt.errorbar(items[3], items[4], yerr=items[6], marker = marker, mfc = 'red')
    #plt.plot(items[3], items[4], 'r')    
#plt.errorbar(t1, x1, yerr=xs1, fmt = '-o', capthick=6, markersize = 10, lw = 3, mfc = 'blue', mec = 'blue', label = 'PreBD = 0.6')
#plt.errorbar(t2, x2, yerr=xs2, fmt = '-v', capthick=6, markersize = 10, lw = 3, mfc = 'red', mec = 'red', label = 'PreBD = 0.8')
#plt.errorbar(t3, x3, yerr=xs3, fmt = '-^', capthick=6, markersize = 10, lw = 3, mfc = 'green', mec = 'green', label = 'PreBD = 1.0')

plt.errorbar(t1, y1, yerr=ys1, fmt = '-o', capthick=6, markersize = 10, lw = 3, mfc = 'blue', mec = 'blue', label = 'PreBD = 0.6')
plt.errorbar(t2, y2, yerr=ys2, fmt = '-v', capthick=6, markersize = 10, lw = 3, mfc = 'red', mec = 'red', label = 'PreBD = 0.8')
plt.errorbar(t3, y3, yerr=ys3, fmt = '-^', capthick=6, markersize = 10, lw = 3, mfc = 'green', mec = 'green', label = 'PreBD = 1.0')

#plt.plot(items[3], items[5], 'b')
plt.ylim(-1,10)
plt.grid(True)
plt.legend(fontsize = 12)# loc=[0.05,0.86],
#axarr[idx].set_xlabel('TOR [s]', fontsize = 20)
plt.title("Degree "+str(idx), fontsize = 20)
plt.ylabel('Error [degree]', fontsize = 20)
plt.xlabel('TOR [s]', fontsize = 20)
plt.xticks([0.3, 0.4, 0.5])
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
currentFile = os.getcwd()
#print(currentFile)
#plt.savefig(currentFile+'/Outputs/GazeBallInterpolation/myWindow_'+str(myWindow)+'_PreBD_'+str(preBD)+'_PostBD_'+str(postBD)+'_degree_'+str(myDegree)+'.png',dpi=600)
plt.savefig(currentFile+'/Outputs/GazeBallInterpolation/allSubjectResult_'+str(myWindow)+'_'+str(myDegree)+'_Y.png', dpi=600)
plt.show()
#plt.close()