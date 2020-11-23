#%%
#3.1 The Data
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
rateData=readData("data.txt","eventRate.txt")[1]

import matplotlib.pyplot as plt
import numpy as np

"""
Plotting the actual and predicted event rates
"""

#creating X array for energy bin values
#the energy bins are caused by the masses of the neutrinos
energyBins=np.linspace(0,10,200)

plt.figure(1)
plt.title("Experimental Data")
plt.ylabel("Event Rate")
plt.xlabel("Energy/GeV")
plt.plot(energyBins,expData)


plt.figure(2)
plt.title("Predicted Values")
plt.xlabel("Energy/GeV")
plt.plot(energyBins,rateData)

plt.show
#%%
#3.2 Fit Function
import numpy as np

def osProb(E,mixAng,diffSqrMass,L):
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

    print(probVals)
    return probVals

#creating an array of neutrino energy values
E=np.linspace(0,10,200)
oscillationValues=osProb(E,np.pi/4,2.4**-3,295)

#plotting probability values on a graph
plt.title("Probability values wrt neutrino energy")
plt.plot(E,oscillationValues)

#%%
