"""
This file contains the functions to produce the NLL as well as perform
the minimisation
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


def noOscProb(E,mixAng,diffSqrMass,L):
    """
    Function to find the probability that the tau neutrino will remain as a tau neutrino after travelling a distance L and not
    oscillate into a muon neutrino

    INPUTS:
    E - Array of neutrino energy values
    mixAng - Float, value of mixing angle
    diffSqrMass - Float, value of neutrino square mass difference
    L - Distance the neutrino travels

    RETURNS:
    probVals - Array of probabilities corresponding to energy bins

    """

    #Create an array to store probability values
    probVals=[]


    #Loop through energy array, finding probability corresponding to each energy
    for i in E:
        sinVal=(1.267*diffSqrMass*L)/i
        sin1=np.sin(2*mixAng)
        sin2=np.sin(sinVal)
        prob=1-((sin1**2)*(sin2**2))
        probVals.append(prob)

    return probVals

def oscEventRate(noDecayProb,simData):
    """
    Function to multiply probability and simulated Data to find the Oscillation event rate

    INPUTS:
    noDecayProb - Array of probability values
    simData - Array of simulated flux values

    RETURNS:
    EventRate - Array of event rate values
    """
    #Create array to store values
    EventRate=[]

    #Loop through decay probability values and multiply by simulated flux values
    for i in range(len(noDecayProb)):
        EventRate.append((noDecayProb[i])*simData[i])
    return EventRate

def newEventRate(EventRate,alpha,E):
    """
    Finding the event rate when neutrino oscillation cross section is taken into account

    INPUTS:
    EventRate -  Array of event rate values calculated previously
    Alpha - Float, scaling the neutrino cross section linearly with energy
    E - Array of energy values

    RETURNS:
    EventRateNew - Array of event rate with neutrino oscillation cross section taken into account
    """
    #Create an array to store event rate values
    EventRateNew=[]
    
    #Iterate through event rate values and multiply by alpha and E
    for i in range(len(EventRate)):
            lambdaNew=EventRate[i]*alpha*E[i]
            EventRateNew.append(lambdaNew)
    return EventRateNew

def NLL(mixAng,diffSqrMass,alpha,form='general'):
    
    """
    This function will either produce an array of negative log likelihood values with varying parameters or the negative log likelihood
    at a specific point

    INPUTS:
    mixAng - Array/float of neutrino mixing angle values
    diffSqrMass - Array/float of difference between neutrino square mass values
    alpha - Array/float of alpha values
    form - String, can take 'theta','delM','alpha','2dContour','singular2d','singular3d' depending on situation
    
    RETURNS:
    likelihood - Array of likelihood values
    """

    #Create array to store likelihood values
    likelihood=[]

    #read in the data
    expData=readData("dataFile.txt")[0]
    simData=readData("dataFile.txt")[1]

    #set the constants
    L=295
    E=np.arange(0.025,10,0.05)

    if form=='theta':
        #if form=='theta, the NLL function will produce likelihood values of varying theta and fixed mass
        for i in range(len(mixAng)):
            #Finding probabilities corresponding to specific paramaters
            noDecayProb=noOscProb(E,mixAng[i],diffSqrMass,L)

            #Setting a sum to find NLL
            NLLsum=0

            #Finding event rate values
            OscillationEventRate=oscEventRate(noDecayProb,simData)

            #Iterate through Event Rate values and sum to find NLL value
            for j in range(len(noDecayProb)):
                m=expData[j]
                Lamda=OscillationEventRate[j]
                if m==0:
                    NLLsum+=Lamda
                else:
                    NLLsum+=(Lamda-m+(m*np.log(m/Lamda)))
            likelihood.append(NLLsum)

    if form=='delM':
        #if form=='delm' , the NLL function will produce likelihood values of varying mass and fixed theta
        for i in range(len(diffSqrMass)):
            #Finding probabilities corresponding to specific paramaters
            noDecayProb=noOscProb(E,mixAng,diffSqrMass[i],L)

            #Setting a sum to track find NLL
            NLLsum=0

            #Finding event rate values
            OscillationEventRate=oscEventRate(noDecayProb,simData)

            #Iterate through Event Rate values and sum to find NLL value
            for j in range(len(noDecayProb)):
                m=expData[j]
                Lamda=OscillationEventRate[j]
                if m==0:
                    NLLsum+=Lamda
                else:
                    NLLsum+=(Lamda-m+(m*np.log(m/Lamda)))
            likelihood.append(NLLsum)

    if form == 'alpha':
        #if form=='alpha', the NLL function will produce likelihood values of varying alpha and fixed theta and mass
        for i in range(len(alpha)):
            #Finding probabilities corresponding to specific paramaters
            noDecayProb=noOscProb(E,mixAng,diffSqrMass,L)

            #Setting a sum to track find NLL
            NLLsum=0

            #Finding event rate values - taking into account cross section
            OscillationEventRate=oscEventRate(noDecayProb,simData)
            OscillationEventRate=newEventRate(OscillationEventRate,alpha[i],E)
            for j in range(len(noDecayProb)):
                m=expData[j]
                Lamda=OscillationEventRate[j]
                if m==0:
                    NLLsum+=Lamda
                else:
                    NLLsum+=(Lamda-m+(m*np.log(m/Lamda)))
            likelihood.append(NLLsum) 
    if form=='2dContour':
        #if form=='2dContour' , the NLL function will produce likelihood values of varying mass and theta for a contour plot
        for i in range(len(mixAng)):
            #Finding probabilities corresponding to specific paramaters
            noDecayProb=noOscProb(E,mixAng[i],diffSqrMass[i],L)

            #Setting a sum to track find NLL
            NLLsum=0

            #Finding event rate values - taking into account cross section
            OscillationEventRate=oscEventRate(noDecayProb,simData)
            for j in range(len(noDecayProb)):
                m=expData[j]
                Lamda=OscillationEventRate[j]
                if m==0:
                    NLLsum+=Lamda
                else:
                    NLLsum+=(Lamda-m+(m*np.log(m/Lamda)))
            likelihood.append(NLLsum)
   
    if form=='singular2d':
        #if form=='singular2d', the NLL function will produce one NLL value at the point of fixed theta and mass

        #Finding probabilities corresponding to specific paramaters
        noDecayProb=noOscProb(E,mixAng,diffSqrMass,L)

        #Setting a sum to track find NLL
        NLLsum=0

        #Finding event rate values
        OscillationEventRate=oscEventRate(noDecayProb,simData)
        for j in range(len(noDecayProb)):
            m=expData[j]
            Lamda=OscillationEventRate[j]
            if m==0:
                NLLsum+=Lamda
                
            else:
                NLLsum+=(Lamda-m+(m*np.log(m/Lamda)))
        likelihood = NLLsum
     

    if form=='singular3d':
        #if form=='singular3d', the NLL function will produce one NLL value at the point of fixed theta,mass and alpha

        #Finding probabilities corresponding to specific paramaters
        noDecayProb=noOscProb(E,mixAng,diffSqrMass,L)

        #Setting a sum to track find NLL
        NLLsum=0

        #Finding event rate values
        OscillationEventRate=oscEventRate(noDecayProb,simData)
        OscillationEventRate=newEventRate(OscillationEventRate,alpha,E)
        for j in range(len(noDecayProb)):
            m=expData[j]
            Lamda=OscillationEventRate[j]
            if m==0:
                NLLsum+=Lamda
            else:
                NLLsum+=(Lamda-m+(m*np.log(m/Lamda)))
        likelihood = NLLsum

    return likelihood

def minimiser_parabolic(func,param,initPoint,dim):
    """
    The parabolic minimiser function will take 3 initial guesses of the minimum and iterate until the local minimum is found 
    using lagrange second order polynomials

    INPUTS:
    func - Function to be minimised
    param - 2d/3d array of parameter values, only one array of len>1 which will be the direction the function is minimised in
    initPoint - Initial guess of minimum point
    dim - String, either '2d' or '3d', depending on how many dimensions the function corresponds to

    RETURNS:
    x3 - minimum point of minimised parameter
    y3 - minimum value of function

    
    """
    #Minimising in 2 varying parametes
    if dim=='2d':
        form='singular2d'
        #Extracting the values from inputs
        mixAng=param[0]
        diffSqrMass=param[1]
        xGuess=[]
        massLength=len(diffSqrMass)
        thetaLength=len(mixAng)
        alphaLength=1
    
    #Minimising in 3 varying parameters
    if dim=='3d':
        #Extracting the values from inputs
        form='singular3d'
        mixAng=param[0]
        diffSqrMass=param[1]
        alpha=param[2]
        #print(len(diffSqrMass))
        xGuess=[]
        massLength=len(diffSqrMass)
        thetaLength=len(mixAng)
        alphaLength=len(alpha)
    
    #minimising in theta direction
    if massLength==1 and alphaLength==1: 
        diffSqrMass=diffSqrMass[0]
        if dim=='2d':
            alpha=1
        if dim=='3d':
            alpha=alpha[0]
        #append initial points
        xGuess.append(initPoint[0])
        firstGuess=initPoint[0]+0.001
        secGuess=initPoint[0]+0.002

        #Condition to find suitable initial points
        while func(firstGuess,diffSqrMass,alpha,form)>= func(secGuess,diffSqrMass,alpha,form):
            firstGuess+=0.001
            secGuess+=0.001
    
        xGuess.append(firstGuess)
        xGuess.append(secGuess)

    #minimising in delm direction
    if thetaLength==1 and alphaLength==1: 
        mixAng=mixAng[0]
        if dim=='2d':
            alpha=1
        if dim=='3d':
            alpha=alpha[0]

        #Appending the initial points
        xGuess.append(initPoint[1])
        firstGuess=initPoint[1] + 0.0005
        secGuess=initPoint[1] + 0.001

        #Condition to find suitable initial points
        while func(mixAng,firstGuess,alpha,form)>= func(mixAng,secGuess,alpha,form):
            firstGuess+=0.0001
            secGuess+=0.0001
        xGuess.append(firstGuess)
        xGuess.append(secGuess)

    #Minimising in the alpha direction
    if thetaLength==1 and massLength==1:
        mixAng=mixAng[0]
        diffSqrMass=diffSqrMass[0]

        #Appending the initial points
        xGuess.append(initPoint[2])
        firstGuess=initPoint[2] + 0.01
        secGuess=initPoint[2] + 0.02

        #Condition to find suitable initial points
        while func(mixAng,diffSqrMass,firstGuess,form)>=func(mixAng,diffSqrMass,secGuess,form):
            firstGuess+=0.01
            secGuess+=0.01
        xGuess.append(firstGuess)
        xGuess.append(secGuess)


    #continue iterating until x3 changes by less than 0.001
    diffx3=100
    prevX = 0

    while diffx3>0.000001:

        xFinal=xGuess

        #find all relevant y values
        if massLength==1 and alphaLength==1:
            y0=func(xGuess[0],diffSqrMass,alpha,form)
            y1=func(xGuess[1],diffSqrMass,alpha,form)
            y2=func(xGuess[2],diffSqrMass,alpha,form)
        if thetaLength==1 and alphaLength==1:
            y0=func(mixAng,xGuess[0],alpha,form)
            y1=func(mixAng,xGuess[1],alpha,form)
            y2=func(mixAng,xGuess[2],alpha,form)
        if thetaLength==1 and massLength==1:
            y0=func(mixAng,diffSqrMass,xGuess[0],form)
            y1=func(mixAng,diffSqrMass,xGuess[1],form)
            y2=func(mixAng,diffSqrMass,xGuess[2],form)


        #break up calculation
        numFirst=(xGuess[2]**2 - xGuess[1]**2)*y0
        numSec=(xGuess[0]**2 - xGuess[2]**2)*y1
        numThird=(xGuess[1]**2 - xGuess[0]**2)*y2
        denomFirst=(xGuess[2] - xGuess[1])*y0
        denomSec=(xGuess[0] - xGuess[2])*y1
        denomThird=(xGuess[1] - xGuess[0])*y2
        
        totalNum=numFirst+numSec+numThird
        totalDenom=denomFirst+denomSec+denomThird

        #finding minimum of lagrange polynomial and appending to xGuess
        x3=0.5*(totalNum/totalDenom)

        #Finding the difference in current and old x point
        diffx3=abs(x3 - prevX)

        #Finding the corresponding y value of the minimum
        if massLength==1 and alphaLength==1:
            y3=func(x3,diffSqrMass,alpha,form)
        if thetaLength==1 and alphaLength==1:
            y3=func(mixAng,x3,alpha,form)
        if thetaLength==1 and massLength==1:
            y3=func(mixAng,diffSqrMass,x3,form)
        

        xGuess.append(x3)
        prevX = x3
        
        #loop through xGuess to remove the x value that corresponds to the largest y
        maxY = 0
        maxX = 0
        for i in xGuess:
            if massLength==1 and alphaLength==1:
                if func(i,diffSqrMass,alpha,form)>maxY:
                    maxY=func(i,diffSqrMass,alpha,form)
                    maxX=i
            if thetaLength==1 and alphaLength==1:
                if func(mixAng,i,alpha,form)>maxY:
                    maxY=func(mixAng,i,alpha,form)
                    maxX=i
            if thetaLength==1 and massLength==1:
                if func(mixAng,diffSqrMass,i,form)>maxY:
                    maxY=func(mixAng,diffSqrMass,i,form)
                    maxX=i
            
        xGuess.remove(maxX)

    return(x3,y3)

def univariate(func,param,initPoint,dim):
    """
    The univariate minimises the function in one paramater, then the other and oscillates between the two (or three depending on dimension) until
    a local minimum is found

    INPUTS:
    func - Function to be minimised 
    param - 2D/3D array of parameter values
    initPoint - Array containing initial point of minimisation
    dim - String, either '2d' or '3d' minimisation
    
    RETURNS:
    xtheta - Minimum theta value
    xdelM - Minimum mass value
    xalpha - Minimum alpha value
    NLLmin - Minimum value of NLL
    """

    #Create array to store points
    points=[]

    #Extracing data from inputs
    mixAng=param[0]
    delm=param[1]
    alpha=param[2]
    param=[mixAng,np.array([initPoint[1]]),np.array([initPoint[2]])]
    
    #Setting initial variables to find differences in points
    if dim=='2d':
        xtheta=0
        xdelm=0
        points.append([initPoint[0],initPoint[1]])
    if dim=='3d':           
        xtheta=0
        xdelm=0
        xalpha=0
    
    #Loop through until function changes by small amount
    diff=100
    i=0
    while diff>0.0000001:
        if dim=='2d':
            
            old=np.array([[xtheta],[xdelm]])
            #Minimising in theta direction
            xtheta=minimiser_parabolic(func,param,initPoint,'2d')[0]
            param=[np.array([xtheta]),delm]

            #Storing the points to plot the path of convergence
            if i==0:
                points.append([xtheta,initPoint[1]])

            if i>0:
                points.append([xtheta,xdelm])

            #Minimising in mass direction
            xdelm=minimiser_parabolic(func,param,initPoint,'2d')[0]

            #Storing the points to plot the path of convergence
            points.append([xtheta,xdelm])
            initPoint=[xtheta,xdelm]

            param=[mixAng,np.array([xdelm])]
            new=np.array([[xtheta],[xdelm]])

            #Finding function difference in old and new point
            diff=abs(func(new[0][0],new[1][0],1,'singular2d') - func(old[0][0],old[1][0],1,'singular2d'))

            i=i+1


        if dim=='3d':
            old=np.array([[xtheta],[xdelm],[xalpha]])

            #Minimising in theta direction
            xtheta=minimiser_parabolic(func,param,initPoint,'3d')[0]
            param=[np.array([xtheta]),delm,np.array([initPoint[2]])]

            #Minimising in mass direction
            xdelm=minimiser_parabolic(func,param,initPoint,'3d')[0]
            param=[np.array([xtheta]),np.array([xdelm]),alpha]

            #Minimising in alpha direction
            xalpha=minimiser_parabolic(func,param,initPoint,'3d')[0]
            initPoint=[xtheta,xdelm,xalpha]
            param=[mixAng,np.array([xdelm]),np.array([xalpha])]
            new=np.array([[xtheta],[xdelm],[xalpha]])

            #Finding difference in function value for old and new point
            diff=abs(func(new[0][0],new[1][0],new[2][0],'singular3d') - func(old[0][0],old[1][0],old[2][0],'singular3d'))
            
            i+=1
            points.append([xtheta,xdelm])


    print("The univariate method took ", i ,"iterations")
    if dim=='2d':
        print(func(xtheta,xdelm,1,'singular2d'))
        print("The minimum occurs at",[xtheta,xdelm])
        return(xtheta,xdelm,func(xtheta,xdelm,1,'singular2d'),points)
    if dim=='3d':
        print(func(xtheta,xdelm,xalpha,'singular3d'))
        print("The minimum occurs at",[xtheta,xdelm,xalpha])
        return(xtheta,xdelm,xalpha,func(xtheta,xdelm,xalpha,'singular3d'),points)

def gradMin(func,initPoint,alpha,dim):
    """
    The gradient descent method folllows the negative gradient of the function until a minimum is found

    INPUTS:
    func - Function to be minimised
    initPoint - Initial point of minimisation
    alpha - Float, scaling factor of gradient
    dim - String, either '2d' or '3d' depending on the function

    RETURNS:
    theta - Minimum theta value
    delM - Minimum mass value
    alpha - Minimum alpha value
    NLLmin - Minimum value of function
    """
    #Initialise array to store points
    points=[]

    #Setting the step size of the central difference method
    delta=1e-5

    #Extracing data from inputs
    if dim=='2d':
        theta=initPoint[0]
        delM=initPoint[1]

    if dim=='3d':
        theta=initPoint[0]
        delM=initPoint[1]
        varAlpha=initPoint[2]
    diff=100
    i=0

    #Find the scaling ratio such that the initial points always match the magnitude of the first dimension
    scale1=theta/delM
    delM=delM*scale1
    if dim=='3d':
        scale2=theta/varAlpha
        varAlpha=varAlpha*scale2

    #Iterate until NLL changes by a small value
    while diff>0.00000001:
        #creating an array to append the gradient vector
        grad=[]
        if dim=='2d':
            #Storing the points to plot the path of convergence
            points.append([theta,delM/scale1])

            #Finding the gradients using the central difference scheme
            gradTheta=(func((theta+delta),delM/scale1,1,'singular2d')-func((theta-delta),delM/scale1,1,'singular2d'))/(2*delta)
            grad.append([gradTheta])
            graddelM=(func(theta,(delM+delta)/scale1,1,'singular2d')-func(theta,(delM-delta)/scale1,1,'singular2d'))/(2*delta)
            grad.append([graddelM])
            grad=np.array(grad)

            xNew=np.array([[theta],[delM]])- (alpha*grad)
            
            j=1
            #this while loop checks that the new point is lower than the old point, if not reduce alpha
            while func(xNew[0][0],xNew[1][0]/scale1,1,'singular2d')>func(theta,delM/scale1,1,'singular2d'):
                newAlpha=alpha-(j*alpha*1e-1)
                xNew=np.array([[theta],[delM]])-(newAlpha*grad)
                j+=1

            #finding the difference between last and new point
            old=[[theta],[delM]]
            diff=abs(func(xNew[0][0],xNew[1][0]/scale1,1,'singular2d') - func(theta,delM/scale1,1,'singular2d'))

            #replacing the vector with the new points
            theta=xNew[0][0]    
            delM=xNew[1][0]
            i+=1
            

        if dim=='3d':

            #find the gradients using the central difference scheme
            gradTheta=(func((theta+delta),delM/scale1,varAlpha/scale2,'singular3d')-func((theta-delta),delM/scale1,varAlpha/scale2,'singular3d'))/(2*delta)
            grad.append([gradTheta])
            graddelM=(func(theta,(delM+delta)/scale1,varAlpha/scale2,'singular3d')-func(theta,(delM-delta)/scale1,varAlpha/scale2,'singular3d'))/(2*delta)
            grad.append([graddelM])
            gradAlpha=(func(theta,delM/scale1,(varAlpha+delta)/scale2,'singular3d')-func(theta,delM/scale1,(varAlpha-delta)/scale2,'singular3d'))/(2*delta)
            grad.append([gradAlpha])
            grad=np.array(grad)

            #calculating the new point using the grad vector
            xNew=np.array([[theta],[delM],[varAlpha]])- (alpha*grad)
            
            j=1
            #this while loop checks that the new point is lower than the old point, if not reduce alpha
            
            while func(xNew[0][0],xNew[1][0]/scale1,xNew[2][0]/scale2,'singular3d')>func(theta,delM/scale1,varAlpha/scale2,'singular3d'):
                newAlpha=alpha-(j*alpha*1e-1)
                xNew=np.array([[theta],[delM],[varAlpha]])-(newAlpha*grad)
                j+=1
            
            #finding the difference between last and new point
            old=[[theta],[delM],[varAlpha]]
            diff=abs(func(xNew[0][0],xNew[1][0]/scale1,xNew[2][0]/scale2,'singular3d') -func(theta,delM/scale1,varAlpha/scale2,'singular3d'))

            #replacing the vector with the new points
            theta=xNew[0][0]    
            delM=xNew[1][0]
            varAlpha=xNew[2][0]
            i+=1
    
    #Scale back the variables
    delM=delM/scale1
    if dim=='3d':
        varAlpha=varAlpha/scale2


    print("The gradient method took ", i , "iterations")
    if dim=='2d':
        print(func(theta,delM,1,'singular2d'))
        print("The minimum occurs at", [theta,delM])
        return(theta,delM,func(theta,delM,1,'singular2d'),points)
    if dim=='3d':
        print(func(theta,delM,varAlpha,'singular3d'))
        print("The minimum occurs at", [theta,delM,varAlpha])
        return(theta,delM,varAlpha,func(theta,delM,varAlpha,'singular3d'))

def quasiNewtonMin(func,initPoint,alpha,dim):
    """
    The Quasi Newton minimiser is more efficient than the gradient method as it scales the gradient by the approximation
    of the inverse hessian

    INPUTS:
    func - Function to be minimised
    initPoint - Initial point of minimisation
    alpha - Float, scaling of the gradient and approximation of inverse hessian
    dim - String, either '2d' or '3d' depending on the function

    RETURNS:
    theta - Minimum theta value
    delM - Minimum mass value
    alpha - Minimum alpha value
    NLLmin - Minimum value of function
    """
    #Initialise array to store plotting points
    points=[]

    #Setting the step size of the central difference approximation
    delta=1e-5

    #Extracting data from inputs
    if dim=='2d':
        theta=initPoint[0]
        delM=initPoint[1]
    if dim=='3d':
        theta=initPoint[0]
        delM=initPoint[1]
        varAlpha=initPoint[2]
   
    diff=100
    i=0


    #Find the scaling ratio such that the initial points always match the magnitude of the first dimension
    scale1=theta/delM
    delM=delM*scale1
    if dim=='3d':
        scale2=theta/varAlpha
        varAlpha=varAlpha*scale2
    
    #Iterate until function value changes by a small value
    while diff>0.00000001:

        if dim=='2d':
            #Store points to plot the path of convergence
            points.append([theta,delM/scale1])

            #find gradient using central difference approximation
            gradTheta=(func((theta+delta),delM/scale1,1,'singular2d')-func((theta-delta),delM/scale1,1,'singular2d'))/(2*delta)
            graddelM=(func(theta,(delM+delta)/scale1,1,'singular2d')-func(theta,(delM-delta)/scale1,1,'singular2d'))/(2*delta)
            grad=np.array([[gradTheta],[graddelM]])

            #If condition to find approximation of inverse hessian on first and successive loops
            if i==0:
                G=np.identity(2)
            else:
                outerProd=np.outer(xdelta,xdelta)
                G=G + (outerProd)*(1/np.dot(gamma.transpose(),xdelta)) - (np.matmul(G,np.matmul(outerProd,G)))*(1/np.dot(gamma.transpose(),np.matmul(G,gamma)))
            
            #Finding xdelta (diff between points) to calculate inverse hessian approx. on next loop
            old=np.array([[theta],[delM]])
            newPoint=old - (alpha * np.matmul(G,grad))
            xdelta=newPoint - old

            #Calculating difference between old and new grad to find gamma to calculate inverse hessian approx on next loop
            gradThetaNew=(func((newPoint[0][0]+delta),newPoint[1][0]/scale1,1,'singular2d')-func((newPoint[0][0]-delta),newPoint[1][0]/scale1,1,'singular2d'))/(2*delta)
            graddelMNew=(func(newPoint[0][0],(newPoint[1][0]+delta)/scale1,1,'singular2d')-func(newPoint[0][0],(newPoint[1][0]-delta)/scale1,1,'singular2d'))/(2*delta)
            newGrad=np.array([[gradThetaNew],[graddelMNew]])
            gamma=newGrad-grad

            #Finding the difference between the old and new function
            diff =abs(func(newPoint[0][0],newPoint[1][0]/scale1,1,'singular2d') - func(theta,delM/scale1,1,'singular2d'))

            #Setting the new points
            theta=newPoint[0][0]
            delM=newPoint[1][0]

            i+=1
           
     

        if dim=='3d':
            #find grad using central difference approximation
            gradTheta=(func((theta+delta),delM/scale1,varAlpha/scale2,'singular3d')-func((theta-delta),delM/scale1,varAlpha/scale2,'singular3d'))/(2*delta)
            graddelM=(func(theta,(delM+delta)/scale1,varAlpha/scale2,'singular3d')-func(theta,(delM-delta)/scale1,varAlpha/scale2,'singular3d'))/(2*delta)
            gradAlpha=(func(theta,delM/scale1,(varAlpha+delta)/scale2,'singular3d')-func(theta,delM/scale1,(varAlpha-delta)/scale2,'singular3d'))/(2*delta)
            grad=np.array([[gradTheta],[graddelM],[gradAlpha]])

            #If condition to find approximation of inverse hessian on first and successive loops
            if i==0:
                G=np.identity(3)
            else:
                outerProd=np.outer(xdelta,xdelta)
                G=G + (outerProd)*(1/np.dot(gamma.transpose(),xdelta)) - (np.matmul(G,np.matmul(outerProd,G)))*(1/np.dot(gamma.transpose(),np.matmul(G,gamma)))

            #Finding xdelta (diff between points) to calculate inverse hessian approx. on next loop
            old=np.array([[theta],[delM],[varAlpha]])
            newPoint=old - (alpha * np.matmul(G,grad))
            xdelta=newPoint - old

            #Calculating difference between old and new grad to find gamma to calculate inverse hessian approx on next loop
            gradThetaNew=(func((newPoint[0][0]+delta),newPoint[1][0]/scale1,newPoint[2][0]/scale2,'singular3d')-func((newPoint[0][0]-delta),newPoint[1][0]/scale1,newPoint[2][0]/scale2,'singular3d'))/(2*delta)
            graddelMNew=(func(newPoint[0][0],(newPoint[1][0]+delta)/scale1,newPoint[2][0]/scale2,'singular3d')-func(newPoint[0][0],(newPoint[1][0]-delta)/scale1,newPoint[2][0]/scale2,'singular3d'))/(2*delta)
            gradAlphaNew=(func(newPoint[0][0],newPoint[1][0]/scale1,(newPoint[2][0]+delta)/scale2,'singular3d')-func(newPoint[0][0],newPoint[1][0]/scale1,(newPoint[2][0]-delta)/scale2,'singular3d'))/(2*delta)
            newGrad=np.array([[gradThetaNew],[graddelMNew],[gradAlphaNew]])
            gamma=newGrad-grad

            #Finding the difference between the old and new function
            diff =abs(func(newPoint[0][0],newPoint[1][0]/scale1,newPoint[2][0]/scale2,'singular3d') - func(theta,delM/scale1,varAlpha/scale2,'singular3d'))

            #Setting the new points
            theta=newPoint[0][0]
            delM=newPoint[1][0]
            varAlpha=newPoint[2][0]

            i+=1


    #Scale back the variables
    delM=delM/scale1
    if dim=='3d':
        varAlpha=varAlpha/scale2


    print("The QuasiNewton method took ", i, "iterations")
    if dim=='2d':
        print(func(theta,delM,1,'singular2d'))
        print("The minimum occurs at",[theta,delM])
        return(theta,delM,func(theta,delM,1,'singular2d'),points)
    if dim=='3d':
        print(func(theta,delM,varAlpha,'singular3d'))
        print("The minimum occurs at", [theta,delM,varAlpha])
        return(theta,delM,varAlpha,func(theta,delM,varAlpha,'singular3d'))



