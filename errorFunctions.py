"""
This file contains all the functions required to find the error in our values of minimum
"""
import numpy as np

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



def NLLgaussianError(param,varyingParamVals,minimum):
    #by approximating the pdf as a gaussian, we can find the uncertainty of the measurement
    #the best estimate of the min is found using our minimising function
    N=len(varyingParamVals)
    sigma=minimum/np.sqrt(N)
    print("The error of ", param, "is", minimum, "+/-", sigma, " using the gaussian error")
    return(sigma)

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

def oscEventRate(noDecayProb,simData):
    EventRate=[]
    for i in range(len(noDecayProb)):
        EventRate.append((noDecayProb[i])*simData[i])
    return EventRate

def newEventRate(EventRate,alpha,E):
    #this function finds the new event rate by scaling the neutrino cross section linearly with energy
    EventRateNew=[]
    
    for i in range(len(EventRate)):
            lambdaNew=EventRate[i]*alpha*E[i]
            EventRateNew.append(lambdaNew)
    return EventRateNew



def ShiftError(func,paramName,param,minimum):
    #read in the data
    expData=readData("dataFile.txt")[0]
    simData=readData("dataFile.txt")[1]

    #set the constants
    L=295
    E=np.arange(0.025,10,0.05)
    if paramName=='theta2d':

        #find the new NLL value
        newNLL=func(minimum,2.4e-3,1,'singular2d')
        shiftedX=[]

        #this new NLL will correspond to two theta values
        for i in range(len(param)):
            #finding the array of probabilities for a specific mix ang (prob vals depend on E)
            noDecayProb=noOscProb(E,param[i],2.4e-3,L)
            NLLsum=0
            OscillationEventRate=oscEventRate(noDecayProb,simData)
            for j in range(len(noDecayProb)):
                m=expData[j]
                Lamda=OscillationEventRate[j]
                if m==0:
                    NLLsum+=Lamda
                else:
                    NLLsum+=(Lamda-m+(m*np.log(m/Lamda)))
                
            if (NLLsum-newNLL)<=0.5:
                shiftedX.append(param[i])
            else:
                continue
        #Want the first and last value of the shifted X array as that is the point closes to NLL shifted by 0.5
        End=len(shiftedX)
        sigmaPoint=[shiftedX[0],shiftedX[End-1]]

        sigma=abs(sigmaPoint[0]-sigmaPoint[1])

    if paramName=='delM2d':

        #find the new NLL value
        newNLL=func(np.pi/4,minimum,1,'singular2d')
        shiftedX=[]

        #this new NLL will correspond to two theta values
        for i in range(len(param)):
            #finding the array of probabilities for a specific mix ang (prob vals depend on E)
            noDecayProb=noOscProb(E,np.pi/4,param[i],L)
            NLLsum=0
            OscillationEventRate=oscEventRate(noDecayProb,simData)
            for j in range(len(noDecayProb)):
                m=expData[j]
                Lamda=OscillationEventRate[j]
                if m==0:
                    NLLsum+=Lamda
                else:
                    NLLsum+=(Lamda-m+(m*np.log(m/Lamda)))
                
            if (NLLsum-newNLL)<=0.5:
                shiftedX.append(param[i])
            else:
                continue
        #Want the first and last value of the shifted X array as that is the point closes to NLL shifted by 0.5
        End=len(shiftedX)
        sigmaPoint=[shiftedX[0],shiftedX[End-1]]

        sigma=abs(sigmaPoint[0]-sigmaPoint[1])

    if paramName=='theta3d':

        #find the new NLL value
        newNLL=func(minimum,2.4e-3,1,'singular3d')
        shiftedX=[]

        #this new NLL will correspond to two theta values
        for i in range(len(param)):
            #finding the array of probabilities for a specific mix ang (prob vals depend on E)
            noDecayProb=noOscProb(E,param[i],2.4e-3,L)
            NLLsum=0
            OscillationEventRate=oscEventRate(noDecayProb,simData)
            OscillationEventRate=newEventRate(OscillationEventRate,1,E)
            for j in range(len(noDecayProb)):
                m=expData[j]
                Lamda=OscillationEventRate[j]
                if m==0:
                    NLLsum+=Lamda
                else:
                    NLLsum+=(Lamda-m+(m*np.log(m/Lamda)))
                
            if (NLLsum-newNLL)<=0.5:
                shiftedX.append(param[i])
            else:
                continue
        #Want the first and last value of the shifted X array as that is the point closes to NLL shifted by 0.5
        End=len(shiftedX)
        sigmaPoint=[shiftedX[0],shiftedX[End-1]]

        sigma=abs(sigmaPoint[0]-sigmaPoint[1])
    if paramName=='delM3d':

        #find the new NLL value
        newNLL=func(np.pi/4,minimum,1,'singular3d')
        shiftedX=[]

        #this new NLL will correspond to two theta values
        for i in range(len(param)):
            #finding the array of probabilities for a specific mix ang (prob vals depend on E)
            noDecayProb=noOscProb(E,np.pi/4,param[i],L)
            NLLsum=0
            OscillationEventRate=oscEventRate(noDecayProb,simData)
            OscillationEventRate=newEventRate(OscillationEventRate,1,E)
            for j in range(len(noDecayProb)):
                m=expData[j]
                Lamda=OscillationEventRate[j]
                if m==0:
                    NLLsum+=Lamda
                else:
                    NLLsum+=(Lamda-m+(m*np.log(m/Lamda)))
                
            if (NLLsum-newNLL)<=0.5:
                shiftedX.append(param[i])
            else:
                continue
        #Want the first and last value of the shifted X array as that is the point closes to NLL shifted by 0.5
        End=len(shiftedX)
        sigmaPoint=[shiftedX[0],shiftedX[End-1]]

        sigma=abs(sigmaPoint[0]-sigmaPoint[1])
    if paramName=='alpha3d':

        #find the new NLL value
        newNLL=func(np.pi/4,2.4e-3,minimum,'singular3d')
        shiftedX=[]

        #this new NLL will correspond to two theta values
        for i in range(len(param)):
            #finding the array of probabilities for a specific mix ang (prob vals depend on E)
            noDecayProb=noOscProb(E,np.pi/4,2.4e-3,L)
            NLLsum=0
            OscillationEventRate=oscEventRate(noDecayProb,simData)
            OscillationEventRate=newEventRate(OscillationEventRate,param[i],E)
            for j in range(len(noDecayProb)):
                m=expData[j]
                Lamda=OscillationEventRate[j]
                if m==0:
                    NLLsum+=Lamda
                else:
                    NLLsum+=(Lamda-m+(m*np.log(m/Lamda)))
                
            if (NLLsum-newNLL)<=0.5:
                shiftedX.append(param[i])
            else:
                continue
        #Want the first and last value of the shifted X array as that is the point closes to NLL shifted by 0.5
        End=len(shiftedX)
        sigmaPoint=[shiftedX[0],shiftedX[End-1]]

        sigma=abs(sigmaPoint[0]-sigmaPoint[1])


    print("The error of ", paramName, "is ",minimum,"+/-", sigma, " using the shift error")
    return sigma