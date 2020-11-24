#%%
#3.1 The Data
import numpy as np
import matplotlib.pyplot as plt

def readData(data,eventRate):
    """
    function to read in data from data files

    """
    #opening data files 
    fileObject=open(data,"r")
    fileObject1=open(eventRate,"r")

    #read in data
    experimentData=fileObject.read().split("\n")
    eventRateData=fileObject1.read().split("\n")

    #appending data into arrays
    expData=[]
    for i in experimentData:
        expData.append(float(i))
    rateData=[]
    for i in eventRateData:
        rateData.append(float(i))


    return(expData,rateData)

expData=readData("data.txt","eventRate.txt")[0]
simData=readData("data.txt","eventRate.txt")[1]

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
        prob=1-(sin1**2)*(sin2**2)
        probVals.append(prob)

    return probVals

#creating an array of neutrino energy values, i take the energy as the midpoint of the bin
E=np.arange(0.025,10,0.05)

#finding the probability of the neutrino not decaying
noDecayProb=noOscProb(E,np.pi/4,2.4e-3,295)

#plotting probability values on a graph
plt.figure(1)
plt.title("Probability values wrt neutrino energy")
plt.plot(E,noDecayProb)

#now we find the oscillated event rate probability by multiplying the simulated data by the
#probability that the neutrino will oscilalte from a muon neutrino to tau neutrino


def oscEventRate(noDecayProb,simData):
    EventRate=[]
    for i in range(len(noDecayProb)):
        EventRate.append((1-noDecayProb[i])*simData[i])
    return EventRate

eventRate=oscEventRate(noDecayProb,simData)

plt.figure(2)
plt.title("Oscillated event rate prediction vs energy")
plt.xlabel("Energy/GeV")
plt.ylabel("Oscillated event rate prediction")
plt.plot(E,eventRate)


#%%
#3.3 Likelihood function

def negLogLike(data,mixAng):
    #create array to store negative Log Likelihood values for varying mix angles
    likelihood=[]

    #set the constants
    diffSqrMass=2.4e-3
    L=295
    E=np.arange(0.025,10,0.05)

    for i in mixAng:
        print(i)
        noDecayProb=noOscProb(E,i,diffSqrMass,L)
        print(noDecayProb)
        sum=0
        for j in range(len(noDecayProb)):
            OscillationEventRate=oscEventRate(noDecayProb,data)
            m=data[j]
            Lamda=OscillationEventRate[j]
            sum+=Lamda-m+(m*np.log(m/Lamda))
        likelihood.append(sum)

    return likelihood

#create an array of mixing angle values

mixAng=np.linspace(np.pi/32,np.pi/2,100)

likelihoodVals=negLogLike(simData,mixAng)

#plotting the NLL values against the mixing angle to find the approx minimum
plt.title("Negative log likelihood with varying mixing angle")
plt.xlabel("Mixing angle")
plt.ylabel("Negative Log Likelihood")
plt.plot(mixAng,likelihoodVals)


#%%
#3.4 Minimise

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

    #append the first 3 points 
    for i in range(3):
        xGuess.append(xVals[i])
    
    #continue iterating until x3 changes by less than 0.001
    diffx3=100

    for x in range(100):
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
        xGuess.append(x3)
        print(xGuess)
        #search through xGuess and remove the x value corresponding to the largest y
        yNew=[]
        for i in xGuess:
            yNew.append(dictionary[i])
        positionMax=yNew.index(max(yNew))
        xGuess.remove(xGuess[positionMax])
        print(xGuess)
     
        
    print(x3)
    return(x3)

parabolicMinimiser(E,noDecayProb)

#the issue is that the new value of x3 found does not correspond to a 
#specific value of y as our x values are not continous and infinite








#%%
