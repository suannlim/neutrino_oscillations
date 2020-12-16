"""
This file contains all functions required to produce the graphs required to
visualise the data
"""

import matplotlib.pyplot as plt
import numpy as np


def readData(data):
    """
    function to read in data from data files

    """
    #opening data files 
    fileObject=open(data,"r")
    fileObject1=open(data,"r")

    #read in data
    experimentData=fileObject.readlines()[2:202]
    simulatedData=fileObject1.readlines()[205:405]


    #appending data into arrays
    expData=[]
    for i in experimentData:
        expData.append(float(i))
    simData=[]
    for i in simulatedData:
        simData.append(float(i))
    
    return(expData,simData)



def histogramPlot(title,xValues,barHeight,xlabel,ylabel):
    plt.figure(title)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.bar(xValues,barHeight,0.05,align='edge')
    

def linePlot(title,xValues,yValues,xlabel,ylabel):
    plt.figure(title)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(xValues,yValues)

def singlePoint(xVal,yVal):
    plt.plot(xVal,yVal,"x")
    

def likelihoodContourPlot(title,varyingVarVals,func,xlabel,ylabel):
    plt.figure(title)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax=plt.axes()
    X,Y=np.meshgrid(varyingVarVals[0],varyingVarVals[1])
    likelihoodVals=func(X,Y,1,'2dContour')
    ax.contour(X,Y,likelihoodVals,50,cmap='binary')
    

#also include one to show the contour plot + heat map to include the alpha axis
