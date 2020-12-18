"""
This is the only required file to be run which produces the answers
The file will contain functions that call other functions
This file will only run functions defined within the file
"""
#import all the relevant modules
import minimisationFunctions as minimiser
import errorFunctions as error
import visualisationFunctions as vis 
import validationFunctions as validate
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
    error.NLLshiftError(minimiser.NLL,[thetaMin],likelihoodMin,'theta1d')


    return

def Part4():
    #4.1 The Univariate Method

    #defining the array of values of theta and mass
    mixAng=np.linspace(np.pi/32,np.pi/2,200)
    diffSqrMass=np.linspace(1e-3,4.8e-3,200)


    #Plot mass vs likelihood
    likelihoodVals=minimiser.NLL(np.pi/4,diffSqrMass,1,'delM')
    vis.linePlot("Likelihood with varying mass",diffSqrMass,likelihoodVals,"Neutrino mass square difference","Likelihood Values")
    
    #minimise using the univariate method
    thetaMin,delMMin,likelihoodMin,path=minimiser.univariate(minimiser.NLL,[mixAng,diffSqrMass,[1]],[0.4,0.001,1],'2d')

    #plotting the NLL with varying theta and mass on a contour plot
    vis.likelihoodContourPlot("Contour plot of theta and mass",[mixAng,diffSqrMass],minimiser.NLL,"Theta","Mass")

   
    #4.2 Simultaneous Minimisation

    #Minimising using the Grad Method
    thetaMin,delMMin,likelihoodMin,points1=minimiser.gradMin(minimiser.NLL,[0.4,0.001],1e-4,'2d')

    #Minimising using the Quasi Newton Method
    thetaMin,delMMin,likelihoodMin,path2=minimiser.quasiNewtonMin(minimiser.NLL,[0.4,0.001],1e-4,'2d')

    vis.ContourPath("Contour Plot with Path",[mixAng,diffSqrMass],minimiser.NLL,'Theta','Mass',path,points1,path2)

    #find the error of our estimates
    error.NLLgaussianError('theta',mixAng,thetaMin)
    error.NLLshiftError(minimiser.NLL,[thetaMin,delMMin],likelihoodMin,'theta2d')
    error.NLLgaussianError('delM',diffSqrMass,delMMin)
    error.NLLshiftError(minimiser.NLL,[thetaMin,delMMin],likelihoodMin,'delM2d')

    
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
    minimiser.gradMin(minimiser.NLL,[0.4,0.001,1.1],1e-4,'3d')

    #Minimising using the quasi newton method with varying alpha
    thetaMin,delMMin,alphaMin,likelihoodMin=minimiser.quasiNewtonMin(minimiser.NLL,[0.4,0.001,1.1],1e-4,'3d')

    #finding the error in our estimates
    error.NLLgaussianError('theta',mixAng,thetaMin)
    error.NLLshiftError(minimiser.NLL,[thetaMin,delMMin,alphaMin],likelihoodMin,'theta3d')
    error.NLLgaussianError('delM',diffSqrMass,delMMin)
    error.NLLshiftError(minimiser.NLL,[thetaMin,delMMin,alphaMin],likelihoodMin,'delM3d')
    error.NLLgaussianError('alpha',alpha,alphaMin)
    error.NLLshiftError(minimiser.NLL,[thetaMin,delMMin,alphaMin],likelihoodMin,'alpha3d')

    return


def validations():

    #Validate the parabolic minimiser with test function y=x^2
    x=np.linspace(-10,10,100)
    minimiser.minimiser_parabolic(validate.xSqr,[x,[1],[1]],[0,1,1],'2d')

    #Validate the Univariate minimiser with test function x^2+y^2
    y=np.linspace(-10,10,100)
    minimiser.univariate(validate.contourFunc,[x,y,1],[1,1,1],'2d')

    #Validate the Gradient minimiser with test function x^2 +y^2
    minimiser.gradMin(validate.contourFunc,[10,10,1],1e-1,'2d')

    #Validate the Quasi Newton minimiser with test function x^2 +y^2
    minimiser.quasiNewtonMin(validate.contourFunc,[10,10,1],1e-1,'2d')

    z=np.linspace(-10,10,100)

    #Validate the Univaraite minimiser with test function x^2+y^2+z^2
    minimiser.univariate(validate.threeDim,[x,y,z],[1,1,1],'3d')

    #Validate the Gradient minimiser with test function x^2+y^2+z^2
    minimiser.gradMin(validate.threeDim,[20,30,40],1e-2,'3d')

    #Validate the Quasi Newton minimiser with test function x^2+y^2+z^2
    minimiser.quasiNewtonMin(validate.threeDim,[10,20,10],1e-1,'3d')

    return

Part4()



