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

noOscillationProbValues=noOscProb(E,np.pi/4,2.4**-3,295)
print(noOscillationProbValues)

#plotting probability values on a graph
plt.figure(1)
plt.title("Probability values wrt neutrino energy")
plt.plot(E,noOscillationProbValues)

#now we find the oscillated event rate probability by multiplying the simulated data by the
#probability that the neutrino will oscilalte from a muon neutrino to tau neutrino


def oscEventRate(noOscProbVals,simData):
    EventRate=[]
    for i in range(len(noOscProbVals)):
        EventRate.append((1-noOscProbVals[i])*simData[i])
    return EventRate

oscEventRatePredic=oscEventRate(noOscillationProbValues,simData)

plt.figure(2)
plt.title("Oscillated event rate prediction vs energy")
plt.xlabel("Energy/GeV")
plt.ylabel("Oscillated event rate prediction")
plt.plot(E,oscEventRatePredic)


#%%
#3.3 Likelihood function

def negLogLike(data,mixAng):
    #create array to store negative Log Likelihood values for varying mix angles
    likelihood=[]

    #set the constants
    diffSqrMass=2.4**-3
    L=295
    E=np.arange(0.025,10,0.05)

    for i in mixAng:
        probNoOsc=noOscProb(E,i,diffSqrMass,L)
        print(probNoOsc)
        print(len(probNoOsc))
        sum=0
        for j in range(len(probNoOsc)):
            OscillationEventRate=oscEventRate(probNoOsc,data)
            m=data[j]
            Lamda=OscillationEventRate[j]
            sum+=Lamda-m+(m*np.log(m/Lamda))
        likelihood.append(sum)

    print(likelihood)
    return likelihood

#create an array of mixing angle values
mixAng=np.arange(np.pi/4,100,np.pi/4)


likelihoodVals=negLogLike(simData,mixAng)





#%%
