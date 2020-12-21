"""
This file contains all functions required to produce the graphs required to
visualise the data
"""
#Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np


def readData(data):
    """
    readData reads in the data from the text file in the folder

    INPUTS:
    data - Textfile

    RETURNS:
    expData - Experimental data as an array
    simData - Simulated data as an array

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
    """
    Function to automatically plot data as a histogram

    INPUTS:
    title - String, title of graph
    xValues - Array, bins on the xaxis
    barHeight - Array, y values of bins
    xlabel - String, x axis label
    ylabel - String, y axis label

    """

    plt.figure(title)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.bar(xValues,barHeight,0.05,align='edge')

    return
    

def linePlot(title,xValues,yValues,xlabel,ylabel):
    """
    Function to automatically plot data as a linegraph

    INPUTS:
    title - String, title of the graph
    xValues - Array, x axis values
    yValues - Array, y axis values
    xlabel - String, label on x axis
    ylabel - String, label on y axis

    """

    plt.figure(title)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(xValues,yValues)
    plt.grid(True)

    return

def singlePoint(xVal,yVal):
    """
    Plot a single point to show position of the minimum

    INPUTS:
    xVal - Float, x value
    yVal - Float, y value


    """
    plt.plot(xVal,yVal,"x")

    return
    

def likelihoodContourPlot(title,varyingVarVals,func,xlabel,ylabel):
    """
    Function to automatically produce a contour plot of varying mixing angle and square mass difference

    INPUTS:
    title - String, title of graph
    varyingVarVals - 2d array of theta and mass values
    func - NLL function
    xlabel - label on x axis
    ylabel - label on y axis

    """
    plt.figure(title)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    X,Y=np.meshgrid(varyingVarVals[0],varyingVarVals[1])
    likelihoodVals=func(X,Y,1,'2dContour')
    plt.contour(X,Y,likelihoodVals,50,cmap='hot')
    plt.colorbar()

    return

def ContourPath(title,varyingVarVals,func,xlabel,ylabel,points,points1,points2):
    """
    Function to automatically produce a contour plot of varying mixing angle and square mass difference

    INPUTS:
    title - String, title of graph
    varyingVarVals - 2d array of theta and mass values
    func - NLL function
    xlabel - label on x axis
    ylabel - label on y axis

    """
    thetaPoints=[]
    massPoints=[]
    for i in range(len(points)):
        thetaPoints.append(points[i][0])
        massPoints.append(points[i][1])

    thetaPoints1=[]
    massPoints1=[]
    for i in range(len(points1)):
        thetaPoints1.append(points1[i][0])
        massPoints1.append(points1[i][1])

    thetaPoints2=[]
    massPoints2=[]
    for i in range(len(points2)):
        thetaPoints2.append(points2[i][0])
        massPoints2.append(points2[i][1])

    plt.figure(title)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    X,Y=np.meshgrid(varyingVarVals[0],varyingVarVals[1])
    likelihoodVals=func(X,Y,1,'2dContour')
    plt.contour(X,Y,likelihoodVals,50,cmap='hot')
    plt.plot(thetaPoints,massPoints,label="Univariate")
    plt.plot(thetaPoints1,massPoints1,label="Gradient")
    plt.plot(thetaPoints2,massPoints2,label="Quasi Newton")
    plt.legend()
    plt.colorbar()

    return
    

