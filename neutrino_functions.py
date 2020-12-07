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

def NLL(mixAng,diffSqrMass,form='general'):
    """
    create array to store negative Log Likelihood values for varying mix angles
    this NLL varying works for 2d and 3d as 
    form can take 4 parameters : twodimension,threedimension,singular_theta,singular_delm
    """
    likelihood=[]

    #read in the data
    expData=readData("dataFile.txt")[0]
    simData=readData("dataFile.txt")[1]

    #set the constants
    L=295
    E=np.arange(0.025,10,0.05)

    #if function for singular values of likelihood
    if form=='twodimension':
        for i in range(len(mixAng)):
            #finding the array of probabilities for a specific mix ang (prob vals depend on E)
            noDecayProb=noOscProb(E,mixAng[i],diffSqrMass[0],L)
            NLLsum=0
            OscillationEventRate=oscEventRate(noDecayProb,simData)
            for j in range(len(noDecayProb)):
                m=expData[j]
                Lamda=OscillationEventRate[j]
                if m==0:
                    NLLsum+=Lamda
                else:
                    NLLsum+=(Lamda-m+(m*np.log(m/Lamda)))
            likelihood.append(NLLsum)
    elif form=='threedimension':
        for i in range(len(mixAng)):
            #finding the array of probabilities for a specific mix ang (prob vals depend on E)
            noDecayProb=noOscProb(E,mixAng[i],diffSqrMass[i],L)
            NLLsum=0
            OscillationEventRate=oscEventRate(noDecayProb,simData)
            for j in range(len(noDecayProb)):
                m=expData[j]
                Lamda=OscillationEventRate[j]
                if m==0:
                    NLLsum+=Lamda
                else:
                    NLLsum+=(Lamda-m+(m*np.log(m/Lamda)))
            likelihood.append(NLLsum)
    elif form=='singular':
        noDecayProb=noOscProb(E,mixAng,diffSqrMass,L)
        NLLsum=0
        OscillationEventRate=oscEventRate(noDecayProb,simData)
        for j in range(len(noDecayProb)):
            m=expData[j]
            Lamda=OscillationEventRate[j]
            if m==0:
                NLLsum+=Lamda
            else:
                NLLsum+=(Lamda-m+(m*np.log(m/Lamda)))
        likelihood = NLLsum

    return likelihood


def minimiser_parabolic(func,param):
    """
    This function will take the function that wants to be minimised along with the 
    parameters, should work for 2d and 3d functions

    PARAM MUST BE A 2D ARRAY

    
    """
    form='singular'
    mixAng=param[0]
    diffSqrMass=param[1]
    #print(len(diffSqrMass))
    xGuess=[]
    zen=len(diffSqrMass)
    xen=len(mixAng)
    
    if zen==1: #minimising in theta direction
        diffSqrMass=diffSqrMass[0]
        #append initial points
        xGuess.append(mixAng[10])
        firstGuess=15
        secGuess=20
        while func(mixAng[firstGuess],diffSqrMass,form)>= func(mixAng[secGuess],diffSqrMass,form):
            firstGuess+=1
            secGuess+=1
    
        xGuess.append(mixAng[firstGuess])
        xGuess.append(mixAng[secGuess])
    if xen==1: #minimising in delm direction
        mixAng=mixAng[0]
        xGuess.append(diffSqrMass[10])
        firstGuess=15
        secGuess=20
        while func(mixAng,diffSqrMass[firstGuess],form)>= func(mixAng,diffSqrMass[secGuess],form):
            firstGuess+=1
            secGuess+=1
    
        xGuess.append(diffSqrMass[firstGuess])
        xGuess.append(diffSqrMass[secGuess])
        

    #continue iterating until x3 changes by less than 0.001
    diffx3=100
    prevX = 0

    while diffx3>0.000001:

        xFinal=xGuess

        #find all relevant y values
        if zen==1:
            y0=func(xGuess[0],diffSqrMass,form)
            y1=func(xGuess[1],diffSqrMass,form)
            y2=func(xGuess[2],diffSqrMass,form)
        if xen==1:
            y0=func(mixAng,xGuess[0],form)
            y1=func(mixAng,xGuess[1],form)
            y2=func(mixAng,xGuess[2],form)


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

        if zen==1:
            y3=func(x3,diffSqrMass,form)
        if xen==1:
            y3=func(mixAng,x3,form)

        xGuess.append(x3)
        prevX = x3
        
        #loop through xGuess to remove the x value that corresponds to the largest y
        maxY = 0
        maxX = 0
        for i in xGuess:
            if zen==1:
                if func(i,diffSqrMass,form)>maxY:
                    maxY=func(i,diffSqrMass,form)
                    maxX=i
            if xen==1:
                if func(mixAng,i,form)>maxY:
                    maxY=func(mixAng,i,form)
                    maxX=i
        xGuess.remove(maxX)



    return(x3,y3)

def univariate(func,param):
    #first minimise in theta direction, take that point then minimise in mix ang direction
    #repeat this process 
    mixAng=param[0]
    delm=param[1]
    param=[mixAng,np.array([delm[0]])]
    
    for i in range(10):
        xtheta=minimiser_parabolic(func,param)[0]
         #putting it into array
        param=[np.array([xtheta]),delm]
        xdelm=minimiser_parabolic(func,param)[0]
        param=[mixAng,np.array([xdelm])]
    print(NLL(xtheta,xdelm,'singular'))
    return(xtheta,xdelm)
