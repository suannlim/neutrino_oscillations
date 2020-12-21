"""
This file contains all the functions required to find the error in our values of minimum
"""

#Import necessary libraries
import numpy as np



def NLLgaussianError(func,minimum):
    """
    By approximating the curve at the minimum point as a Gaussian, the standard deviation can be found

    INPUTS:
    param - Name of parameter of which uncertainty is being found
    varyingParamVals - Array of parameter values 
    minimum - Float, estimated minimum of the parameter

    RETURNS:
    sigma - Uncertainty in result of minimum

    """
    
    #defining step size
    h=1e-7

    #calculating the second derivative using a finite difference scheme
    secDeriv=(func(minimum+h,2.4e-3,1,'singular2d')-(func(minimum+h,2.4e-3,1,'singular2d')*2) \
        + func(minimum-h,2.4e-3,1,'singular2d'))/(h**2)

    #Calculating standard deviation
    sigma=1/secDeriv

    print("The error of theta is", minimum, "+/-", sigma, " using the gaussian error")

    return(sigma)

def NLLshiftError(func,minPoint,minNLL,form):
    """
    The values of the parameter when the NLL value is shifted up by 0.5, each correspond to a shift of one standard deviation

    INPUTS:
    func - NLL function
    minPoint - array of minimum point (must either be len 1,2,3 depending on what dimension the NLL has been minimised in).
                 Must be put in order theta,mass,alpha
    minNLL - Value of NLL at the minimum point
    form - String, can take 'theta1d','theta2d','delM2d','theta3d','delM3d' or 'alpha3d' to specify what error is being found

    RETURNS:
    sigma - Error of parameter


    """

    #calculating the new value of the func to find the corresponding param values
    shift=minNLL+0.5

    #determining the next point for iteration based on the initial parameter
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
    
    diffplus=100
    diffminus=100

    #using the secant method to find the new roots of the equation
    while diffplus>0.001 and diffminus>0.001:
        if form=='theta1d':
            x2plus=x1plus-(func(x1plus,fixedMass,1,'singular2d')-shift)*((x1plus-x0plus)/\
                (func(x1plus,fixedMass,1,'singular2d')-shift -func(x0plus,fixedMass,1,'singular2d')-shift))
            x0plus=x1plus
            x1plus=x2plus
            diffplus=x1plus-x0plus
            x2minus=x1minus-(func(x1minus,fixedMass,1,'singular2d')-shift)*((x1minus-x0minus)/\
                (func(x1minus,fixedMass,1,'singular2d')-shift -func(x0minus,fixedMass,1,'singular2d')-shift))
            x0minus=x1minus
            x1minus=x2minus
            diffminus=x1minus-x0minus
        if form=='theta2d':
            x2plus=x1plus-(func(x1plus,fixedMass,1,'singular2d')-shift)*((x1plus-x0plus)/\
                (func(x1plus,fixedMass,1,'singular2d')-shift -func(x0plus,fixedMass,1,'singular2d')-shift))
            x0plus=x1plus
            x1plus=x2plus
            diffplus=x1plus-x0plus
            x2minus=x1minus-(func(x1minus,fixedMass,1,'singular2d')-shift)*((x1minus-x0minus)/\
                (func(x1minus,fixedMass,1,'singular2d')-shift -func(x0minus,fixedMass,1,'singular2d')-shift))
            x0minus=x1minus
            x1minus=x2minus
            diffminus=x1minus-x0minus
            
        if form=='delM2d':
            x2plus=x1plus-(func(fixedTheta,x1plus,1,'singular2d')-shift)*((x1plus-x0plus)/\
                (func(fixedTheta,x1plus,1,'singular2d')-shift -func(fixedTheta,x0plus,1,'singular2d')-shift))
            x0plus=x1plus
            x1plus=x2plus
            diffplus=x1plus-x0plus
            x2minus=x1minus-(func(fixedTheta,x1minus,1,'singular2d')-shift)*((x1minus-x0minus)/\
                (func(fixedTheta,x1minus,1,'singular2d')-shift -func(fixedTheta,x0minus,1,'singular2d')-shift))
            x0minus=x1minus
            x1minus=x2minus
            diffminus=x1minus-x0minus
            

        if form=='theta3d':
            x2plus=x1plus-(func(x1plus,fixedMass,fixedAlpha,'singular3d')-shift)*((x1plus-x0plus)/\
                (func(x1plus,fixedMass,fixedAlpha,'singular3d')-shift -func(x0plus,fixedMass,fixedAlpha,'singular3d')-shift))
            x0plus=x1plus
            x1plus=x2plus
            diffplus=x1plus-x0plus
            x2minus=x1minus-(func(x1minus,fixedMass,fixedAlpha,'singular3d')-shift)*((x1minus-x0minus)/\
                (func(x1minus,fixedMass,fixedAlpha,'singular3d')-shift -func(x0minus,fixedMass,fixedAlpha,'singular3d')-shift))
            x0minus=x1minus
            x1minus=x2minus
            diffminus=x1minus-x0minus
        if form=='delM3d':
            x2plus=x1plus-(func(fixedTheta,x1plus,fixedAlpha,'singular3d')-shift)*((x1plus-x0plus)/\
                (func(fixedTheta,x1plus,fixedAlpha,'singular3d')-shift -func(fixedTheta,x0plus,fixedAlpha,'singular3d')-shift))
            x0plus=x1plus
            x1plus=x2plus
            diffplus=x1plus-x0plus
            x2minus=x1minus-(func(fixedTheta,x1minus,fixedAlpha,'singular3d')-shift)*((x1minus-x0minus)/\
                (func(fixedTheta,x1minus,fixedAlpha,'singular3d')-shift -func(fixedTheta,x0minus,fixedAlpha,'singular3d')-shift))
            x0minus=x1minus
            x1minus=x2minus
            diffminus=x1minus-x0minus

        if form=='alpha3d':
            x2plus=x1plus-(func(fixedTheta,fixedMass,x1plus,'singular3d')-shift)*((x1plus-x0plus)/\
                (func(fixedTheta,fixedMass,x1plus,'singular3d')-shift -func(fixedTheta,fixedMass,x0plus,'singular3d')-shift))
            x0plus=x1plus
            x1plus=x2plus
            diffplus=x1plus-x0plus
            x2minus=x1minus-(func(fixedTheta,fixedMass,x1minus,'singular3d')-shift)*((x1minus-x0minus)/\
                (func(fixedTheta,fixedMass,x1minus,'singular3d')-shift -func(fixedTheta,fixedMass,x0minus,'singular3d')-shift))
            x0minus=x1minus
            x1minus=x2minus
            diffminus=x1minus-x0minus

    #finding the average value of the two shifts as the error may be assymmetric
    sigma=(x1plus-x1minus)/2
    if form=='theta2d' or form=='theta3d' or form=='theta1d':
        minVal=minPoint[0]
    if form=='delM2d' or form=='delM3d':
        minVal=minPoint[1]
    if form=='alpha3d':
        minVal=minPoint[2]


    print("The error of", form, "is", minVal, "+/-", sigma, "using the shift error")
    return sigma



    


