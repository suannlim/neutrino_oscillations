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
def parabolicMinimiser(xVals,yVals):
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
        
    return(x3,y3,dictionary,xFinal)


plt.figure("Minimising Neg log like")
parabolicMinimiser(mixAng,likelihoodVals)
x=parabolicMinimiser(mixAng,likelihoodVals)[0]
y=parabolicMinimiser(mixAng,likelihoodVals)[1]
#plotting the NLL values against the mixing angle to find the approx minimum
plt.title("Negative log likelihood with varying mixing angle")
plt.xlabel("Mixing angle")
plt.ylabel("Negative Log Likelihood")
plt.plot(mixAng,likelihoodVals)
plt.plot(x,y,"x")

plt.figure("test")
parabolicMinimiser(E,noDecayProb)
x1=parabolicMinimiser(E,noDecayProb)[0]
y1=parabolicMinimiser(E,noDecayProb)[1]
plt.plot(E,noDecayProb)
plt.plot(x1,y1,"x")



#%%
#3.5 Find accuracy of fit result

#By shifting the value of the NLL at the minimum by +0.5, the two corresponding
#theta values's range give one standard deviation

minX=parabolicMinimiser(mixAng,likelihoodVals)[0]
minY=parabolicMinimiser(mixAng,likelihoodVals)[1]
dictionary=parabolicMinimiser(mixAng,likelihoodVals)[2]
xFinal=parabolicMinimiser(mixAng,likelihoodVals)[3]

#def NLLshifterror(minY,xVals,yVals,dictionary):

    #in order to find the values of x at y+-0.5, use the last lagrange polynomial used
    #to find the minimum in the minimiser function
    #newY=minY+0.5



def NLLgaussianError(xVals,yVals,minX):
    #by approximating the pdf as a gaussian, we can find the uncertainty of the measurement
    #the best estimate of the min is found using our minimising function
    N=len(xVals)
    sigma=minX/np.sqrt(N)
    print("The error of ", minX, "is +/- ", sigma)
    return(sigma)

NLLgaussianError(mixAng,likelihoodVals,minX)


#%%
#4.1 The Univariate Method

from mpl_toolkits import mplot3d

#first we need to code an oscillation probability function that takes into account
#mix angle and square mass difference

def NLL_varying(expdata,simdata,mixAng,diffSqrMass):
        #create array to store negative Log Likelihood values for varying mix angles
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

diffSqrMass=np.linspace(1e-3,2.4e-3,200)
Likelihood_2d=NLL_varying(expData,simData,mixAng,diffSqrMass)

plt.figure("NLL vs mix ang")
plt.plot(diffSqrMass,Likelihood_2d)
plt.xlabel("Diff sqr mass")
plt.ylabel("NLL")

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(mixAng, diffSqrMass, Likelihood_2d)



    

#%%

