"""
This is the only required file to be run which produces the answers
The file will contain functions that call other functions
This file will only run functions defined within the file
"""
#import all the relevant modules
import minimisationFunctions as minimiser
import errorFunctions as error
import visualisationFunctions as vis 
import numpy as np 
import matplotlib.pyplot as plt 


def Part3():
    #3.1 Import the Data and present it as a histogram
    expData=vis.readData("dataFile.txt")[0]
    simData=vis.readData("dataFile.txt")[1]

    #creating X array to represent energy bins
    energyBins=np.arange(0.05,10.05,0.05)

    #plot the histograms
    vis.histogramPlot('Simulated Data',energyBins,simData,"Energy/GeV","Neutrino Flux")
    vis.histogramPlot('Experimental Data',energyBins,expData,"Energy/GeV","Neutrino Flux")

    #3.2 Fit Function
    #creating a list of energy values that take the midpoints of the energy bins
    E=np.arange(0.025,10,0.05)
    #finding the probability values
    noDecayProb=minimiser.noOscProb(E,np.pi/4,2.4e-3,295)
    vis.linePlot("Probability Values",E,noDecayProb,"Energy/GeV","Probability of Oscillation")

    #finding the Event Rate values
    eventRate=minimiser.oscEventRate(noDecayProb,simData)
    vis.linePlot("Oscillated Event Rate",E,eventRate,"Energy/GeV","Event Rate")


    #3.3 Likelihood function
    #defining a range of our mixAngle
    mixAng=np.linspace(np.pi/32,np.pi/2,200)
    #finding likelihood values
    likelihoodVals=minimiser.NLL(mixAng,2.4e-3,1,'theta')
    vis.linePlot("Likelihood with varying Theta",mixAng,likelihoodVals,"Mixing Angle","Negative Log Like")

    
    #3.4 Minimise
    thetaMin,likelihoodMin=minimiser.minimiser_parabolic(minimiser.NLL,[mixAng,[2.4e-3],[1]],[0.4,2.4e-3,1],'2d')
    vis.linePlot("Position of minimum",mixAng,likelihoodVals,"Mixing Angle","Negative Log Like")
    vis.singlePoint(thetaMin,likelihoodMin)

    #3.5 Accuracy of Fit Result

    #Uncertainty found by approximating our last parabolic estimate as a Gaussian
    error.NLLgaussianError('Theta',mixAng,thetaMin)
    #Uncertainty found by shifting the NLL values by 0.5
    error.ShiftError(minimiser.NLL,'theta2d',mixAng,thetaMin)

    #plt.show()

    return

def Part4():
    #4.1 The Univariate Method

    #defining the array of values of theta and mass
    mixAng=np.linspace(np.pi/32,np.pi/2,200)
    diffSqrMass=np.linspace(1e-3,4.8e-3,200)

    #plotting the NLL with varying theta and mass on a contour plot
    vis.likelihoodContourPlot("Contour plot of theta and mass",[mixAng,diffSqrMass],minimiser.NLL,"Theta","Mass")

    #Plot mass vs likelihood
    likelihoodVals=minimiser.NLL(np.pi/4,diffSqrMass,1,'delM')
    vis.linePlot("Likelihood with varying mass",diffSqrMass,likelihoodVals,"Neutrino mass square difference","Likelihood Values")
    

    #minimise using the univariate method
    thetaMin,delMMin=minimiser.univariate(minimiser.NLL,[mixAng,diffSqrMass,[1]],[0.4,0.001,1],'2d')


    #4.2 Simultaneous Minimisation

    #Minimising using the Grad Method
    minimiser.gradMin(minimiser.NLL,[0.4,0.001],1e-9,'2d')

    #Minimising using the Quasi Newton Method
    minimiser.quasiNewtonMin(minimiser.NLL,[0.4,0.001],1e-9,'2d')

    #find the error of our estimates
    error.ShiftError(minimiser.NLL,'theta2d',mixAng,thetaMin)
    error.NLLgaussianError('theta',mixAng,thetaMin)
    error.ShiftError(minimiser.NLL,'delM2d',diffSqrMass,delMMin)
    error.NLLgaussianError('delM',diffSqrMass,delMMin)


    plt.show()

    return


def Part5():
    #5 Neutrino Interaction Cross Section

    #Defining our varying variables
    alpha=np.linspace(0,5,200)
    diffSqrMass=np.linspace(1e-3,4.8e-3,200)
    mixAng=np.linspace(np.pi/32,np.pi/2,200)

    #Minimising using the univariate method with varying alpha
    minimiser.univariate(minimiser.NLL,[mixAng,diffSqrMass,alpha],[0.4,0.001,1.1],'3d')
    
    #Minimising using the grad method with varying alpha
    minimiser.gradMin(minimiser.NLL,[0.4,0.001,1.1],1e-9,'3d')

    #Minimising using the quasi newton method with varying alpha
    thetaMin,delMMin,alphaMin=minimiser.quasiNewtonMin(minimiser.NLL,[0.4,0.001,1.1],1e-9,'3d')

    #finding the error in our estimates
    error.ShiftError(minimiser.NLL,'theta3d',mixAng,thetaMin)
    error.NLLgaussianError('theta',mixAng,thetaMin)
    error.ShiftError(minimiser.NLL,'delM3d',diffSqrMass,delMMin)
    error.NLLgaussianError('delM',diffSqrMass,delMMin)
    error.ShiftError(minimiser.NLL,'alpha3d',alpha,alphaMin)
    error.NLLgaussianError('alpha',alpha,alphaMin)

    return


Part4()
