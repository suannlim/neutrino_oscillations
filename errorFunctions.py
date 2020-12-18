"""
This file contains all the functions required to find the error in our values of minimum
"""

#Import necessary libraries
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



def NLLgaussianError(param,varyingParamVals,minimum):
    """
    By approximating the curve at the minimum point as a Gaussian, the standard deviation can be found

    INPUTS:
    param - Name of parameter of which uncertainty is being found
    varyingParamVals - Array of parameter values 
    minimum - Float, estimated minimum of the parameter

    RETURNS:
    sigma - Uncertainty in result of minimum

    """
    #Finding the length of the array
    N=len(varyingParamVals)

    #Calculating standard deviation
    sigma=minimum/np.sqrt(N)

    print("The error of ", param, "is", minimum, "+/-", sigma, " using the gaussian error")

    return(sigma)

def NLLshiftError(func,minPoint,minNLL,form):
    """
    The values of the parameter when the NLL value is shifted up by 0.5, each correspond to a shift of one standard deviation

    INPUTS:
    func - NLL function
    minPoint - array of minimum point (must either be len 1,2,3 depending on what dimension the NLL has been minimised in). Must be put in order theta,mass,alpha
    minNLL - Value of NLL at the minimum point
    form - String, can take 'theta1d','theta2d','delM2d','theta3d','delM3d' or 'alpha3d' to specify what error is being found

    RETURNS:
    sigma - Error of parameter


    """


    shift=minNLL+0.5

    if form=='theta1d':
        fixedMass=2.4e-3
        x0plus=minPoint[0]+0.01
        x1plus=minPoint[0]+0.02
        x0minus=minPoint[0]-0.01
        x1minus=minPoint[0]-0.02

    if form=='theta2d':
        fixedMass=minPoint[1]
        x0plus=minPoint[0]+0.01
        x1plus=minPoint[0]+0.02
        x0minus=minPoint[0]-0.01
        x1minus=minPoint[0]-0.02
    if form=='delM2d':
        fixedTheta=minPoint[0]
        x0plus=minPoint[1]+1e-4
        x1plus=minPoint[1]+2e-4
        x0minus=minPoint[1]-1e-4
        x1minus=minPoint[1]-2e-4
    if form=='theta3d':
        fixedMass=minPoint[1]
        fixedAlpha=minPoint[2]
        x0plus=minPoint[0]+0.01
        x1plus=minPoint[0]+0.02
        x0minus=minPoint[0]-0.01
        x1minus=minPoint[0]-0.02
    if form=='delM3d':
        fixedTheta=minPoint[0]
        fixedAlpha=minPoint[2]
        x0plus=minPoint[1]+0.0001
        x1plus=minPoint[1]+0.0002
        x0minus=minPoint[1]-0.0001
        x1minus=minPoint[1]-0.0002
    if form=='alpha3d':
        fixedTheta=minPoint[0]
        fixedMass=minPoint[1]
        x0plus=minPoint[2]+0.1
        x1plus=minPoint[2]+0.2
        x0minus=minPoint[2]-0.1
        x1minus=minPoint[2]-0.2
    #shift the NLL down by (min-0.5) then find the roots of that equation
    diffplus=100
    diffminus=100
    while diffplus>0.001 and diffminus>0.001:
        if form=='theta1d':
            x2plus=x1plus-(func(x1plus,fixedMass,1,'singular2d')-shift)*((x1plus-x0plus)/(func(x1plus,fixedMass,1,'singular2d')-shift -func(x0plus,fixedMass,1,'singular2d')-shift))
            x0plus=x1plus
            x1plus=x2plus
            diffplus=x1plus-x0plus
            x2minus=x1minus-(func(x1minus,fixedMass,1,'singular2d')-shift)*((x1minus-x0minus)/(func(x1minus,fixedMass,1,'singular2d')-shift -func(x0minus,fixedMass,1,'singular2d')-shift))
            x0minus=x1minus
            x1minus=x2minus
            diffminus=x1minus-x0minus
        if form=='theta2d':
            x2plus=x1plus-(func(x1plus,fixedMass,1,'singular2d')-shift)*((x1plus-x0plus)/(func(x1plus,fixedMass,1,'singular2d')-shift -func(x0plus,fixedMass,1,'singular2d')-shift))
            x0plus=x1plus
            x1plus=x2plus
            diffplus=x1plus-x0plus
            x2minus=x1minus-(func(x1minus,fixedMass,1,'singular2d')-shift)*((x1minus-x0minus)/(func(x1minus,fixedMass,1,'singular2d')-shift -func(x0minus,fixedMass,1,'singular2d')-shift))
            x0minus=x1minus
            x1minus=x2minus
            diffminus=x1minus-x0minus
            
        if form=='delM2d':
            x2plus=x1plus-(func(fixedTheta,x1plus,1,'singular2d')-shift)*((x1plus-x0plus)/(func(fixedTheta,x1plus,1,'singular2d')-shift -func(fixedTheta,x0plus,1,'singular2d')-shift))
            x0plus=x1plus
            x1plus=x2plus
            diffplus=x1plus-x0plus
            x2minus=x1minus-(func(fixedTheta,x1minus,1,'singular2d')-shift)*((x1minus-x0minus)/(func(fixedTheta,x1minus,1,'singular2d')-shift -func(fixedTheta,x0minus,1,'singular2d')-shift))
            x0minus=x1minus
            x1minus=x2minus
            diffminus=x1minus-x0minus
            

        if form=='theta3d':
            x2plus=x1plus-(func(x1plus,fixedMass,fixedAlpha,'singular3d')-shift)*((x1plus-x0plus)/(func(x1plus,fixedMass,fixedAlpha,'singular3d')-shift -func(x0plus,fixedMass,fixedAlpha,'singular3d')-shift))
            x0plus=x1plus
            x1plus=x2plus
            diffplus=x1plus-x0plus
            x2minus=x1minus-(func(x1minus,fixedMass,fixedAlpha,'singular3d')-shift)*((x1minus-x0minus)/(func(x1minus,fixedMass,fixedAlpha,'singular3d')-shift -func(x0minus,fixedMass,fixedAlpha,'singular3d')-shift))
            x0minus=x1minus
            x1minus=x2minus
            diffminus=x1minus-x0minus
        if form=='delM3d':
            x2plus=x1plus-(func(fixedTheta,x1plus,fixedAlpha,'singular3d')-shift)*((x1plus-x0plus)/(func(fixedTheta,x1plus,fixedAlpha,'singular3d')-shift -func(fixedTheta,x0plus,fixedAlpha,'singular3d')-shift))
            x0plus=x1plus
            x1plus=x2plus
            diffplus=x1plus-x0plus
            x2minus=x1minus-(func(fixedTheta,x1minus,fixedAlpha,'singular3d')-shift)*((x1minus-x0minus)/(func(fixedTheta,x1minus,fixedAlpha,'singular3d')-shift -func(fixedTheta,x0minus,fixedAlpha,'singular3d')-shift))
            x0minus=x1minus
            x1minus=x2minus
            diffminus=x1minus-x0minus

        if form=='alpha3d':
            x2plus=x1plus-(func(fixedTheta,fixedMass,x1plus,'singular3d')-shift)*((x1plus-x0plus)/(func(fixedTheta,fixedMass,x1plus,'singular3d')-shift -func(fixedTheta,fixedMass,x0plus,'singular3d')-shift))
            x0plus=x1plus
            x1plus=x2plus
            diffplus=x1plus-x0plus
            x2minus=x1minus-(func(fixedTheta,fixedMass,x1minus,'singular3d')-shift)*((x1minus-x0minus)/(func(fixedTheta,fixedMass,x1minus,'singular3d')-shift -func(fixedTheta,fixedMass,x0minus,'singular3d')-shift))
            x0minus=x1minus
            x1minus=x2minus
            diffminus=x1minus-x0minus

    print(x1plus,x1minus)
    sigma=(x1plus-x1minus)/2
    if form=='theta2d' or form=='theta3d' or form=='theta1d':
        minVal=minPoint[0]
    if form=='delM2d' or form=='delM3d':
        minVal=minPoint[1]
    if form=='alpha3d':
        minVal=minPoint[2]


    print("The error of", form, "is", minVal, "+/-", sigma, "using the shift error")
    return sigma



    


