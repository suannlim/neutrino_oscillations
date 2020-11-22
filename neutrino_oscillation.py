#%%
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


print(rateData)


#%%
import matplotlib.pyplot as plt
import numpy as np

#creating X array for energy bin values
energyBins=np.linspace(0,10,200)

plt.figure(1)
plt.title("Experimental Data")
plt.xlabel("Energy/GeV")
plt.plot(energyBins,expData)


plt.figure(2)
plt.title("Predicted Values")
plt.xlabel("Energy/GeV")
plt.plot(energyBins,rateData)

plt.show
#%%
