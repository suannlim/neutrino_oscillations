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
        prob=1-((sin1**2)*(sin2**2))
        probVals.append(prob)

    return probVals

#creating an array of neutrino energy values, i take the energy as the midpoint of the bin
E=np.arange(0.025,10,0.05)

#trying tonys value of E
newE=np.linspace(0,10-0.05,200)


#finding the probability of the neutrino not decaying
noDecayProb=noOscProb(newE,np.pi/4,2.4e-3,295)

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
        noDecayProb=noOscProb(E,i,diffSqrMass,L)
        NLLsum=0
        for j in range(len(noDecayProb)):
            OscillationEventRate=oscEventRate(noDecayProb,simdata)
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





#%%
#3.4 Minimisation


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

    #append the first, middle and last value
    xGuess.append(xVals[70])
    xGuess.append(xVals[80])
    xGuess.append(xVals[90])
    
    #continue iterating until x3 changes by less than 0.001
    diffx3=100
    prevX = 0

    while diffx3>0.00000001:

        print(xGuess)
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
        print("found x3: ", x3)
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
        print("Setting prev3 to: ", prevX)
        print(xGuess)
        
        #loop through xGuess to remove the x value that corresponds to the largest y
        maxY = 0
        maxX = 0
        for i in xGuess:
            if dictionary[i]>maxY:
                maxY=dictionary[i]
                maxX=i
        xGuess.remove(maxX)
        print(xGuess)
        
    print(x3)
    return(x3,y3)



parabolicMinimiser(mixAng,likelihoodVals)
x=parabolicMinimiser(mixAng,likelihoodVals)[0]
y=parabolicMinimiser(mixAng,likelihoodVals)[1]
#plotting the NLL values against the mixing angle to find the approx minimum
plt.title("Negative log likelihood with varying mixing angle")
plt.xlabel("Mixing angle")
plt.ylabel("Negative Log Likelihood")
plt.plot(mixAng,likelihoodVals)
plt.plot(x,y,"x")


#%%
