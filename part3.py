#%%
import matplotlib.pyplot as plt
import neutrino_functions as f
import numpy as np

#3.1 The Data

expData=f.readData("dataFile.txt")[0]
simData=f.readData("dataFile.txt")[1]

"""
Plotting the actual and predicted event rates
"""
#creating X array to represent energy bins
energyBins=np.arange(0.05,10.05,0.05,)

plt.figure('Simulated Data')
plt.title("Simulated Data")
plt.ylabel("Event Rate")
plt.xlabel("Energy/GeV")
plt.bar(energyBins,simData,0.05,align='edge')


plt.figure('Experimental Data')
plt.title("Experimental Data")
plt.ylabel("Event Rate")
plt.xlabel("Energy/GeV")
plt.bar(energyBins,expData,0.05,align='edge')

plt.show()

#3.2 Fit Function

#creating an array of neutrino energy values, i take the energy as the midpoint of the bin
E=np.arange(0.025,10,0.05)

#finding the probability of the neutrino not decaying
noDecayProb=f.noOscProb(E,np.pi/4,2.4e-3,295)

#plotting probability values on a graph
plt.figure('Probability')
plt.title("Probability values wrt neutrino energy")
plt.plot(E,noDecayProb)

eventRate=f.oscEventRate(noDecayProb,simData)

plt.figure('Graph of Lamda')
plt.title("Oscillated event rate prediction vs energy")
plt.xlabel("Energy/GeV")
plt.ylabel("Oscillated event rate prediction")
plt.plot(E,eventRate)

plt.show()

#3.3 Likelihood Function
print(i)
mixAng=np.linspace(np.pi/32,np.pi/2,200)
diffSqrMass=np.array([2.4e-3])
NLLthetaVals=f.NLL(mixAng,diffSqrMass,'twodimension')
plt.figure('NLL Function (theta)')
plt.ylabel('Likelihood Values')
plt.xlabel('Mixing Angle Values')
plt.plot(mixAng,NLLthetaVals)
plt.show()


#3.4 Minimisation
thetaMin=f.minimiser_parabolic(NLL,[mixAng,diffSqrMass])[0]
NLLMin=f.minimiser_parabolic(NLL,[mixAng,diffSqrMass])[1]
plt.figure('NLL Function Minimised')
plt.ylabel('Likelihood Values')
plt.xlabel('Mixing Angle Values')
plt.plot(mixAng,NLLthetaVals)
plt.plot(thetaMin,NLLMin,"x")
plt.show()

#3.5 Accuracy of Fit






#%%
