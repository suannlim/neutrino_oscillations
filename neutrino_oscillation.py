#%%
#3.1 The Data
import numpy as np
import matplotlib.pyplot as plt

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

expData=readData("dataFile.txt")[0]
simData=readData("dataFile.txt")[1]

"""
Plotting the actual and predicted event rates
"""
#creating X array to represent energy bins
energyBins=np.arange(0.05,10.05,0.05,)

plt.figure(1)
plt.title("Simulated Data")
plt.ylabel("Event Rate")
plt.xlabel("Energy/GeV")
plt.bar(energyBins,simData,0.05,align='edge')


plt.figure(2)
plt.title("Experimental Data")
plt.ylabel("Event Rate")
plt.xlabel("Energy/GeV")
plt.bar(energyBins,expData,0.05,align='edge')


plt.show
#%%
#3.2 Fit Function
import numpy as np

def noOscProb(E,mixAng,diffSqrMass,L):
    """
    This is the probability that the muon neutrino will be observed as a muon neutrino
    and will not have oscillated into a tau neutrino
    """
    probVals=[]
    for i in E:
        sinVal=(1.267*diffSqrMass*L)/i
        sin1=np.sin(2*mixAng)
        sin2=np.sin(sinVal)
        prob=1-((sin1**2)*(sin2**2))
        probVals.append(prob)

    return probVals

#creating an array of neutrino energy values, i take the energy as the midpoint of the bin
E=np.arange(0.025,10,0.05)

#trying tonys value of E
newE=np.linspace(0,10-0.05,200)


#finding the probability of the neutrino not decaying
noDecayProb=noOscProb(E,np.pi/4,2.4e-3,295)

#plotting probability values on a graph
plt.figure(1)
plt.title("Probability values wrt neutrino energy")
plt.plot(E,noDecayProb)

#now we find the oscillated event rate probability by multiplying the simulated data by the
#probability that the neutrino will not oscilalte from a muon neutrino to tau neutrino


def oscEventRate(noDecayProb,simData):
    EventRate=[]
    for i in range(len(noDecayProb)):
        EventRate.append((noDecayProb[i])*simData[i])
    return EventRate

eventRate=oscEventRate(noDecayProb,simData)

plt.figure(2)
plt.title("Oscillated event rate prediction vs energy")
plt.xlabel("Energy/GeV")
plt.ylabel("Oscillated event rate prediction")
plt.plot(E,eventRate)


#%%
#3.3 Likelihood function

def negLogLike(expdata,simdata,mixAng):
    #create array to store negative Log Likelihood values for varying mix angles
    likelihood=[]

    #set the constants
    diffSqrMass=2.4e-3
    L=295
    E=np.arange(0.025,10,0.05)
    for i in mixAng:
        #finding the array of probabilities for a specific mix ang (prob vals depend on E)
        noDecayProb=noOscProb(E,i,diffSqrMass,L)
        NLLsum=0
        OscillationEventRate=oscEventRate(noDecayProb,simdata)
        for j in range(len(noDecayProb)):
            m=expdata[j]
            Lamda=OscillationEventRate[j]
            if m==0:
                NLLsum+=Lamda
            else:
                NLLsum+=(Lamda-m+(m*np.log(m/Lamda)))
        likelihood.append(NLLsum)

    return likelihood

#create an array of mixing angle values

mixAng=np.linspace(np.pi/32,np.pi/2,200)

likelihoodVals=negLogLike(expData,simData,mixAng)

#plotting the NLL values against the mixing angle to find the approx minimum
plt.title("Negative log likelihood with varying mixing angle")
plt.xlabel("Mixing angle")
plt.ylabel("Negative Log Likelihood")
plt.plot(mixAng,likelihoodVals)

print(mixAng,likelihoodVals)





#%%
#3.4 Minimisation

#this minimiser is only effective when the approximate positionof the minimum is known
def parabolicMinimiser(xVals,yVals,initPoint,initPointX):
    """
    Parabolic minimisation takes 3 points(x0,x1,x2) and fits a lagrange polynomial across
    those points. The polynomial minimum can be found to give x3.Keep the
    3 lowest points then repeat, eventually x3 will converge to the true
    minimum
    """
    #create an array to store the points we are considering
    xGuess=[]

    #creating a dictionary between the x and y array
    dictionary=dict(zip(xVals,yVals))

    #need to find 3 initial points f(x1),f(x2) and f(x3) such that f(x2) is the lowest point

    if initPoint==True:
        xGuess.append(initPointX)
    else:
        xGuess.append(xVals[10])
    firstGuess=15
    secGuess=20
    while dictionary[xVals[firstGuess]]>= dictionary[xVals[secGuess]]:
        firstGuess+=1
        secGuess+=1
    
    xGuess.append(xVals[firstGuess])
    xGuess.append(xVals[secGuess])
    
    #continue iterating until x3 changes by less than 0.001
    diffx3=100
    prevX = 0

    #store the xGuess values used to make the last polynomial
    xFinal=[]

    while diffx3>0.000001:

        xFinal=xGuess

        #find all relevant y values
        y0=dictionary[xGuess[0]]
        y1=dictionary[xGuess[1]]
        y2=dictionary[xGuess[2]]

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
        diffx3=abs(x3 - prevX)

        #finding the y value corresponding to this x value
        LagrangeFirst=((x3-xGuess[1])*(x3-xGuess[2])*y0)/((xGuess[0]-xGuess[1])*(xGuess[0]-xGuess[2]))
        LagrangeSec=((x3-xGuess[0])*(x3-xGuess[2])*y1)/((xGuess[1]-xGuess[0])*(xGuess[1]-xGuess[2]))
        LagrangeThird=((x3-xGuess[0])*(x3-xGuess[1])*y2)/((xGuess[2]-xGuess[0])*(xGuess[2]-xGuess[1]))

        y3=LagrangeFirst + LagrangeSec + LagrangeThird

        #append the new found y3 value to the dictionary
        dictionary[x3]=y3

        xGuess.append(x3)
        prevX = x3
        
        #loop through xGuess to remove the x value that corresponds to the largest y
        maxY = 0
        maxX = 0
        for i in xGuess:
            if dictionary[i]>maxY:
                maxY=dictionary[i]
                maxX=i
        xGuess.remove(maxX)

        print(x3,y3)
    np.append(xVals,x3)
    np.append(yVals,y3)

    return(x3,y3,dictionary,xVals,yVals)


plt.figure("Minimising Neg log like")
xVals=parabolicMinimiser(mixAng,likelihoodVals,False,0)[3]
yVals=parabolicMinimiser(mixAng,likelihoodVals,False,0)[4]
x=parabolicMinimiser(mixAng,likelihoodVals,False,0)[0]
y=parabolicMinimiser(mixAng,likelihoodVals,False,0)[1]
#plotting the NLL values against the mixing angle to find the approx minimum
plt.title("Negative log likelihood with varying mixing angle")
plt.xlabel("Mixing angle")
plt.ylabel("Negative Log Likelihood")
plt.plot(xVals,yVals)
plt.plot(x,y,"x")




#%%
#3.5 Find accuracy of fit result

#By shifting the value of the NLL at the minimum by +0.5, the two corresponding
#theta values's range give one standard deviation

minX=parabolicMinimiser(mixAng,likelihoodVals)[0]
minY=parabolicMinimiser(mixAng,likelihoodVals)[1]
dictionary=parabolicMinimiser(mixAng,likelihoodVals)[2]
xFinal=parabolicMinimiser(mixAng,likelihoodVals)[3]

def findNearest(array,value):
    difference=value-array
    idx=np.argmin(np.abs(difference))
    if difference[idx]>0:
        return(idx,idx+1)
    if difference[idx]<0:
        return(idx,idx-1)



def NLLshifterror(minY,xVals,yVals):

    #in order to find the values of x at y+-0.5, use the last lagrange polynomial used
    #to find the minimum in the minimiser function.
    #Interpolate between the two nearest Y values
    newY=minY+0.5
    y1_idx,y2_idx=findNearest(yVals,newY)
    y1=yVals[y1_idx]
    y2=yVals[y2_idx]
    print(y1,y2)
    x1=xVals[y1_idx]
    x2=xVals[y2_idx]
    print(x1,x2)
    numerator=newY*(x2-x1) - x2*y1 + x1*y2
    denominator=y2-y1
    thetaPlus=numerator/denominator

    return(thetaPlus)



def NLLgaussianError(xVals,yVals,minX):
    #by approximating the pdf as a gaussian, we can find the uncertainty of the measurement
    #the best estimate of the min is found using our minimising function
    N=len(xVals)
    sigma=minX/np.sqrt(N)
    print("The error of ", minX, "is +/- ", sigma)
    return(sigma)

NLLgaussianError(mixAng,likelihoodVals,minX)
NLLshifterror(minY,mixAng,likelihoodVals)


#%%
#4.1 The Univariate Method

from mpl_toolkits import mplot3d

#first we need to code an oscillation probability function that takes into account
#mix angle and square mass difference

def NLL_varying(expdata,simdata,mixAng,diffSqrMass):
    #create array to store negative Log Likelihood values for varying mix angles
    #this NLL varying works for 2d and 3d as 
    likelihood=[]

    #set the constants
    L=295
    E=np.arange(0.025,10,0.05)
    for i in range(len(mixAng)):
        #finding the array of probabilities for a specific mix ang (prob vals depend on E)
        noDecayProb=noOscProb(E,mixAng[i],diffSqrMass[i],L)
        NLLsum=0
        OscillationEventRate=oscEventRate(noDecayProb,simdata)
        for j in range(len(noDecayProb)):
            m=expdata[j]
            Lamda=OscillationEventRate[j]
            if m==0:
                NLLsum+=Lamda
            else:
                NLLsum+=(Lamda-m+(m*np.log(m/Lamda)))
        likelihood.append(NLLsum)

    return likelihood

diffSqrMass=np.linspace(1e-3,4.8e-3,200)
Likelihood_2d=NLL_varying(expData,simData,mixAng,diffSqrMass)

#finding the minimum using the univariate method means finding the min in one direction and then the other 
#direction then iterating

def univariateMinimisation(angVals,delmVals,likelihood):
    #the univariate method finds the minimum in one dimension (given an initial point), uses that minimum
    #and finds the minimum in the other direction. repeat this iteration and it will eventually converge

    #find first iteration in min theta direction
    first_x_theta= parabolicMinimiser(angVals,likelihood,False,0)[0]

    #now we use this point as the initial point for the minimisation in delta m direction
    first_x_delm=parabolicMinimiser(delmVals,likelihood,True,first_x_theta)[0]

    return(first_x_delm)


#univariateMinimisation(mixAng,diffSqrMass,Likelihood_2d)
xdelm=parabolicMinimiser(diffSqrMass,Likelihood_2d,False,0)[0]
ydelm=parabolicMinimiser(diffSqrMass,Likelihood_2d,False,0)[1]


plt.figure("NLL vs mix ang")
plt.plot(diffSqrMass,Likelihood_2d)
plt.plot(xdelm,ydelm,"x")
plt.xlabel("Diff sqr mass")
plt.ylabel("NLL")

fig = plt.figure("2d")
ax = plt.axes()
X, Y = np.meshgrid(mixAng, diffSqrMass)
plt.title("Univariate method for 2d")
likelihood_countour=NLL_varying(expData,simData,X,Y)
ax.contour(X, Y, likelihood_countour, 50, cmap='binary')

plt.figure("3d")
plt.title("Univariate method for 3d")
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, likelihood_countour, 50, cmap='binary')



#%%
diffSqrMass=np.linspace(1e-3,4.8e-3,200)
mixAng=np.linspace(np.pi/32,np.pi/2,200)
#iffSqrMass=np.array([2.4e-3])



#try to rewrite minimiser function so it works for 2d and 3d
#first have to rewrite NLL function so it works for 2d and 3d
def NLL(mixAng,diffSqrMass,form='general'):
    """
    create array to store negative Log Likelihood values for varying mix angles
    this NLL varying works for 2d and 3d as 
    form can take 4 parameters : twodimension,threedimension,singular_theta,singular_delm
    """
    likelihood=[]

    #read in the data
    expData=readData("dataFile.txt")[0]
    simData=readData("dataFile.txt")[1]

    #set the constants
    L=295
    E=np.arange(0.025,10,0.05)

    #if function for singular values of likelihood
    if form=='twodimension':
        for i in range(len(mixAng)):
            #finding the array of probabilities for a specific mix ang (prob vals depend on E)
            noDecayProb=noOscProb(E,mixAng[i],diffSqrMass[0],L)
            NLLsum=0
            OscillationEventRate=oscEventRate(noDecayProb,simData)
            for j in range(len(noDecayProb)):
                m=expData[j]
                Lamda=OscillationEventRate[j]
                if m==0:
                    NLLsum+=Lamda
                else:
                    NLLsum+=(Lamda-m+(m*np.log(m/Lamda)))
            likelihood.append(NLLsum)
    elif form=='threedimension':
        for i in range(len(mixAng)):
            #finding the array of probabilities for a specific mix ang (prob vals depend on E)
            noDecayProb=noOscProb(E,mixAng[i],diffSqrMass[i],L)
            NLLsum=0
            OscillationEventRate=oscEventRate(noDecayProb,simData)
            for j in range(len(noDecayProb)):
                m=expData[j]
                Lamda=OscillationEventRate[j]
                if m==0:
                    NLLsum+=Lamda
                else:
                    NLLsum+=(Lamda-m+(m*np.log(m/Lamda)))
            likelihood.append(NLLsum)
    elif form=='singular':
        noDecayProb=noOscProb(E,mixAng,diffSqrMass,L)
        NLLsum=0
        OscillationEventRate=oscEventRate(noDecayProb,simData)
        for j in range(len(noDecayProb)):
            m=expData[j]
            Lamda=OscillationEventRate[j]
            if m==0:
                NLLsum+=Lamda
            else:
                NLLsum+=(Lamda-m+(m*np.log(m/Lamda)))
        likelihood = NLLsum

    return likelihood


def minimiser_parabolic(func,param):
    """
    This function will take the function that wants to be minimised along with the 
    parameters, should work for 2d and 3d functions

    PARAM MUST BE A 2D ARRAY

    
    """
    form='singular'
    mixAng=param[0]
    diffSqrMass=param[1]
    #print(len(diffSqrMass))
    xGuess=[]
    zen=len(diffSqrMass)
    xen=len(mixAng)
    
    if zen==1: #minimising in theta direction
        diffSqrMass=diffSqrMass[0]
        #append initial points
        xGuess.append(mixAng[10])
        firstGuess=15
        secGuess=20
        while func(mixAng[firstGuess],diffSqrMass,form)>= func(mixAng[secGuess],diffSqrMass,form):
            firstGuess+=1
            secGuess+=1
    
        xGuess.append(mixAng[firstGuess])
        xGuess.append(mixAng[secGuess])
    if xen==1: #minimising in delm direction
        mixAng=mixAng[0]
        xGuess.append(diffSqrMass[10])
        firstGuess=15
        secGuess=20
        while func(mixAng,diffSqrMass[firstGuess],form)>= func(mixAng,diffSqrMass[secGuess],form):
            firstGuess+=1
            secGuess+=1
    
        xGuess.append(diffSqrMass[firstGuess])
        xGuess.append(diffSqrMass[secGuess])
        

    #continue iterating until x3 changes by less than 0.001
    diffx3=100
    prevX = 0

    while diffx3>0.000001:

        xFinal=xGuess

        #find all relevant y values
        if zen==1:
            y0=func(xGuess[0],diffSqrMass,form)
            y1=func(xGuess[1],diffSqrMass,form)
            y2=func(xGuess[2],diffSqrMass,form)
        if xen==1:
            y0=func(mixAng,xGuess[0],form)
            y1=func(mixAng,xGuess[1],form)
            y2=func(mixAng,xGuess[2],form)


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
        diffx3=abs(x3 - prevX)

        if zen==1:
            y3=func(x3,diffSqrMass,form)
        if xen==1:
            y3=func(mixAng,x3,form)

        xGuess.append(x3)
        prevX = x3
        
        #loop through xGuess to remove the x value that corresponds to the largest y
        maxY = 0
        maxX = 0
        for i in xGuess:
            if zen==1:
                if func(i,diffSqrMass,form)>maxY:
                    maxY=func(i,diffSqrMass,form)
                    maxX=i
            if xen==1:
                if func(mixAng,i,form)>maxY:
                    maxY=func(mixAng,i,form)
                    maxX=i
        xGuess.remove(maxX)



    return(x3,y3)

def univariate(func,param):
    #first minimise in theta direction, take that point then minimise in mix ang direction
    #repeat this process 
    mixAng=param[0]
    delm=param[1]
    param=[mixAng,np.array([delm[0]])]
    
    for i in range(10):
        xtheta=minimiser_parabolic(func,param)[0]
         #putting it into array
        param=[np.array([xtheta]),delm]
        xdelm=minimiser_parabolic(func,param)[0]
        param=[mixAng,np.array([xdelm])]
    print(NLL(xtheta,xdelm,'singular'))
    return(xtheta,xdelm)

diffSqrMass=np.linspace(1e-3,4.8e-3,200)
mixAng=np.linspace(np.pi/32,np.pi/2,200)
#diffSqrMass=np.array([2.4e-3])



#x=minimiser_parabolic(NLL,[mixAng,diffSqrMass])[0]
#y=minimiser_parabolic(NLL,[mixAng,diffSqrMass])[1]
print(univariate(NLL,[mixAng,diffSqrMass]))

fig = plt.figure("2d")
ax = plt.axes()
X, Y = np.meshgrid(mixAng, diffSqrMass)
NLLvals=NLL(X,Y,'threedimension')
plt.title("Contour plot of likelihood")
ax.contour(X, Y, NLLvals, 50, cmap='binary')




#%%


#%%
