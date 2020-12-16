#%%
#3.1 The Data
import numpy as np
import matplotlib.pyplot as plt

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

expData=readData("dataFile.txt")[0]
simData=readData("dataFile.txt")[1]

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


#finding the probability of the neutrino not decaying
noDecayProb=noOscProb(E,np.pi/4,2.4e-3,295)

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
        #finding the array of probabilities for a specific mix ang (prob vals depend on E)
        noDecayProb=noOscProb(E,i,diffSqrMass,L)
        NLLsum=0
        OscillationEventRate=oscEventRate(noDecayProb,simdata)
        for j in range(len(noDecayProb)):
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

print(mixAng,likelihoodVals)





#%%
#3.4 Minimisation

#this minimiser is only effective when the approximate positionof the minimum is known
def parabolicMinimiser(xVals,yVals,initPoint,initPointX):
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

    #need to find 3 initial points f(x1),f(x2) and f(x3) such that f(x2) is the lowest point

    if initPoint==True:
        xGuess.append(initPointX)
    else:
        xGuess.append(xVals[10])
    firstGuess=15
    secGuess=20
    while dictionary[xVals[firstGuess]]>= dictionary[xVals[secGuess]]:
        firstGuess+=1
        secGuess+=1
    
    xGuess.append(xVals[firstGuess])
    xGuess.append(xVals[secGuess])
    
    #continue iterating until x3 changes by less than 0.001
    diffx3=100
    prevX = 0

    #store the xGuess values used to make the last polynomial
    xFinal=[]

    while diffx3>0.000001:

        xFinal=xGuess

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
        
        #loop through xGuess to remove the x value that corresponds to the largest y
        maxY = 0
        maxX = 0
        for i in xGuess:
            if dictionary[i]>maxY:
                maxY=dictionary[i]
                maxX=i
        xGuess.remove(maxX)

        print(x3,y3)
    np.append(xVals,x3)
    np.append(yVals,y3)

    return(x3,y3,dictionary,xVals,yVals)


plt.figure("Minimising Neg log like")
xVals=parabolicMinimiser(mixAng,likelihoodVals,False,0)[3]
yVals=parabolicMinimiser(mixAng,likelihoodVals,False,0)[4]
x=parabolicMinimiser(mixAng,likelihoodVals,False,0)[0]
y=parabolicMinimiser(mixAng,likelihoodVals,False,0)[1]
#plotting the NLL values against the mixing angle to find the approx minimum
plt.title("Negative log likelihood with varying mixing angle")
plt.xlabel("Mixing angle")
plt.ylabel("Negative Log Likelihood")
plt.plot(xVals,yVals)
plt.plot(x,y,"x")




#%%
#4.1 The Univariate Method

from mpl_toolkits import mplot3d

#first we need to code an oscillation probability function that takes into account
#mix angle and square mass difference

def NLL_varying(expdata,simdata,mixAng,diffSqrMass):
    #create array to store negative Log Likelihood values for varying mix angles
    #this NLL varying works for 2d and 3d as 
    likelihood=[]

    #set the constants
    L=295
    E=np.arange(0.025,10,0.05)
    for i in range(len(mixAng)):
        #finding the array of probabilities for a specific mix ang (prob vals depend on E)
        noDecayProb=noOscProb(E,mixAng[i],diffSqrMass[i],L)
        NLLsum=0
        OscillationEventRate=oscEventRate(noDecayProb,simdata)
        for j in range(len(noDecayProb)):
            m=expdata[j]
            Lamda=OscillationEventRate[j]
            if m==0:
                NLLsum+=Lamda
            else:
                NLLsum+=(Lamda-m+(m*np.log(m/Lamda)))
        likelihood.append(NLLsum)

    return likelihood

diffSqrMass=np.linspace(1e-3,4.8e-3,200)
Likelihood_2d=NLL_varying(expData,simData,mixAng,diffSqrMass)



#univariateMinimisation(mixAng,diffSqrMass,Likelihood_2d)
xdelm=parabolicMinimiser(diffSqrMass,Likelihood_2d,False,0)[0]
ydelm=parabolicMinimiser(diffSqrMass,Likelihood_2d,False,0)[1]


plt.figure("NLL vs mix ang")
plt.plot(diffSqrMass,Likelihood_2d)
plt.plot(xdelm,ydelm,"x")
plt.xlabel("Diff sqr mass")
plt.ylabel("NLL")

fig = plt.figure("2d")
ax = plt.axes()
X, Y = np.meshgrid(mixAng, diffSqrMass)
plt.title("Univariate method for 2d")
likelihood_countour=NLL_varying(expData,simData,X,Y)
ax.contour(X, Y, likelihood_countour, 50, cmap='binary')

plt.figure("3d")
plt.title("Univariate method for 3d")
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, likelihood_countour, 50, cmap='binary')



#%%
diffSqrMass=np.linspace(1e-3,4.8e-3,200)
mixAng=np.linspace(np.pi/32,np.pi/2,200)
#iffSqrMass=np.array([2.4e-3])



#try to rewrite minimiser function so it works for 2d and 3d
#first have to rewrite NLL function so it works for 2d and 3d
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

    alpha=np.linspace(0,4,200)

    #if function for singular values of likelihood
    if form=='twodimension':
        for i in range(len(mixAng)):
            #finding the array of probabilities for a specific mix ang (prob vals depend on E)
            noDecayProb=noOscProb(E,mixAng[i],diffSqrMass,L)
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
    if form=='threedimension':
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
    if form=='singular':
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


def minimiser_parabolic(func,param,initPoint):
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
    massLength=len(diffSqrMass)
    thetaLength=len(mixAng)
    
    if massLength==1: #minimising in theta direction
        diffSqrMass=diffSqrMass[0]
        #append initial points
        xGuess.append(initPoint[0])
        firstGuess=initPoint[0]+0.001
        secGuess=initPoint[0]+0.002
        while func(firstGuess,diffSqrMass,form)>= func(secGuess,diffSqrMass,form):
            firstGuess+=0.001
            secGuess+=0.001
    
        xGuess.append(firstGuess)
        xGuess.append(secGuess)
    if thetaLength==1: #minimising in delm direction
        mixAng=mixAng[0]
        xGuess.append(initPoint[1])
        firstGuess=initPoint[1] + 0.0005
        secGuess=initPoint[1] + 0.001
        while func(mixAng,firstGuess,form)>= func(mixAng,secGuess,form):
            firstGuess+=0.0001
            secGuess+=0.0001
    
        xGuess.append(firstGuess)
        xGuess.append(secGuess)
        

    #continue iterating until x3 changes by less than 0.001
    diffx3=100
    prevX = 0

    while diffx3>0.000001:

        xFinal=xGuess

        #find all relevant y values
        if massLength==1:
            y0=func(xGuess[0],diffSqrMass,form)
            y1=func(xGuess[1],diffSqrMass,form)
            y2=func(xGuess[2],diffSqrMass,form)
        if thetaLength==1:
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

        if massLength==1:
            y3=func(x3,diffSqrMass,form)
        if thetaLength==1:
            y3=func(mixAng,x3,form)

        xGuess.append(x3)
        prevX = x3
        
        #loop through xGuess to remove the x value that corresponds to the largest y
        maxY = 0
        maxX = 0
        for i in xGuess:
            if massLength==1:
                if func(i,diffSqrMass,form)>maxY:
                    maxY=func(i,diffSqrMass,form)
                    maxX=i
            if thetaLength==1:
                if func(mixAng,i,form)>maxY:
                    maxY=func(mixAng,i,form)
                    maxX=i
        xGuess.remove(maxX)



    return(x3,y3)

def univariate(func,param,initPoint):
    #first minimise in theta direction, take that point then minimise in mix ang direction
    #repeat this process 
    mixAng=param[0]
    delm=param[1]
    param=[mixAng,np.array([initPoint[1]])]
    
    
    xtheta=0
    xdelm=0
    new=np.array([[100],[100]])
    old=np.array([[xtheta],[xdelm]])
    diff=abs(np.linalg.norm(new-old))
    i=0
    while diff>0.000001:
        old=np.array([[xtheta],[xdelm]])
        xtheta=minimiser_parabolic(func,param,initPoint)[0]
         #putting it into array
        param=[np.array([xtheta]),delm]
        xdelm=minimiser_parabolic(func,param,initPoint)[0]
        initPoint=[xtheta,xdelm]
        param=[mixAng,np.array([xdelm])]
        new=np.array([[xtheta],[xdelm]])
        diff=abs(np.linalg.norm(new-old))
        i+=1

    print("The univariate method took ", i ,"iterations")
    print(NLL(xtheta,xdelm,'singular'))
    return(xtheta,xdelm)

diffSqrMass=np.linspace(1e-3,4.8e-3,200)
mixAng=np.linspace(np.pi/32,np.pi/2,200)



univariate(NLL,[mixAng,diffSqrMass],[0.4,0.001])

#x=minimiser_parabolic(NLL,[mixAng,diffSqrMass])[0]
#y=minimiser_parabolic(NLL,[mixAng,diffSqrMass])[1]

"""

fig = plt.figure("2d")
ax = plt.axes()
X, Y = np.meshgrid(mixAng, diffSqrMass)
NLLvals=NLL(X,Y,'threedimension')


plt.title("Contour plot of likelihood")
ax.contour(X, Y, NLLvals, 50, cmap='binary')


plt.figure("3d")
plt.title("Univariate method for 3d")
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, NLLvals, 50, cmap='binary')
plt.show()

"""



"""
plt.figure(1)
plt.plot(mixAng,NLL(mixAng,diffSqrMass,'threedimension'))
plt.figure(2)
plt.plot(diffSqrMass,NLL(mixAng,diffSqrMass,'threedimension'))
plt.show()
"""   


#%%
#3.5 Accuracy of fit result

def NLLgaussianError(mixAng,minX):
    #by approximating the pdf as a gaussian, we can find the uncertainty of the measurement
    #the best estimate of the min is found using our minimising function
    N=len(mixAng)
    sigma=minX/np.sqrt(N)
    print("The error of ", minX, "is +/- ", sigma)
    return(sigma)

def ShiftError(func,mintheta,mixAng,delM):
    #find the new NLL value
    newNLL=func(mintheta,delM,'singular')

    likelihood=[]

    #read in the data
    expData=readData("dataFile.txt")[0]
    simData=readData("dataFile.txt")[1]

    #set the constants
    L=295
    E=np.arange(0.025,10,0.05)

    shiftedX=[]

    #this new NLL will correspond to two theta values
    for i in range(len(mixAng)):
        #finding the array of probabilities for a specific mix ang (prob vals depend on E)
        noDecayProb=noOscProb(E,mixAng[i],delM,L)
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
            shiftedX.append(mixAng[i])
        else:
            continue
        likelihood.append(NLLsum)    
    
    shiftedX[0:2]
    sigma=abs(shiftedX[0]-shiftedX[1])

    print("The error of ", mintheta, "is +/- ", sigma)
    return sigma


mixAng=np.linspace(np.pi/32,np.pi/2,200)
mintheta=minimiser_parabolic(NLL,[mixAng,[2.4e-3]])[0]
minNLL=minimiser_parabolic(NLL,[mixAng,[2.4e-3]])[1]
NLLgaussianError(mixAng,mintheta)
shiftedVals=ShiftError(NLL,mintheta,mixAng,2.4e-3)
print(shiftedVals)






#%%

#4.2 Newton Method

#essentially the same as the grad method but instead of mulltiplying by alpha, multiply by the
#inverse hessian instead

def newtonMin(func,initPoint):
    #the hessian is the NxN matrix of the second order derivatives, use eq 5.13 from the notes
    theta=initPoint[0][0]
    delM=initPoint[1][0]
    diff=100
    delta=1e-5

    theta=theta*1e-2

    while diff>0.0001:
        #calculate first derivatives
        gradTheta=(NLL((theta+delta)*1e2,delM,'singular')-NLL((theta-delta)*1e2,delM,'singular'))/(2*delta)
        graddelM=(NLL(theta*1e2,delM+delta,'singular')-NLL(theta*1e2,delM-delta,'singular'))/(2*delta)

        #appending grad to 2d matrix
        grad=np.array([[gradTheta],[graddelM]])
        print("this is the gradient")
        print(grad)

        #calculate second derivatives
        secDerivTheta=(NLL((theta+delta)*1e2,delM,'singular')-(2*NLL(theta*1e2,delM,'singular'))+NLL((theta-delta)*1e2,delM,'singular'))/(delta**2)
        secDerivMass=(NLL(theta*1e2,delM+delta,'singular')-(2*NLL(theta*1e2,delM,'singular'))+NLL(theta*1e2,delM-delta,'singular'))/(delta**2)
        secDerivBoth=(NLL((theta+delta)*1e2,delM+delta,'singular')-NLL((theta+delta)*1e2,delM-delta,'singular')-NLL((theta-delta)*1e2,delM+delta,'singular')+NLL((theta-delta)*1e2,delM-delta,'singular'))/(4*(delta**2))

        #appending the hessian to a 2d matrix
        hessian=np.array([[secDerivTheta,secDerivBoth],[secDerivBoth,secDerivMass]])
        print("this is the hessian")
        print(hessian)
        

        #invert the hessian
        invhessian=np.linalg.inv(hessian)
        print("this is the inv hessian")
        print(invhessian)
        
        

        old=np.array([[theta],[delM]])
        

        newPoint=old - np.matmul(invhessian,grad)
        

        diff=abs(np.linalg.norm(newPoint-old))

        theta=newPoint[0][0]
        delM=newPoint[1][0]

    theta=theta*1e2
    print(newPoint)

    return(theta,delM)

newtonMin(NLL,[[0.9],[0.003]])





#%%

#%%
#5 Neutrino interaction cross section
#creating an array of neutrino energy values, i take the energy as the midpoint of the bin
E=np.arange(0.025,10,0.05)

def newEventRate(EventRate,alpha,E):
    #this function finds the new event rate by scaling the neutrino cross section linearly with energy
    EventRateNew=[]
    
    for i in range(len(EventRate)):
            lambdaNew=EventRate[i]*alpha*E[i]
            EventRateNew.append(lambdaNew)
    return EventRateNew

alpha=np.linspace(0,5,200)
diffSqrMass=np.linspace(1e-3,4.8e-3,200)
mixAng=np.linspace(np.pi/32,np.pi/2,200)


#%%



#try to rewrite minimiser function so it works for 2d and 3d
#first have to rewrite NLL function so it works for 2d and 3d
def NLL(mixAng,diffSqrMass,alpha,form='general'):
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
            noDecayProb=noOscProb(E,mixAng[i],diffSqrMass,L)
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
    if form=='threedimension':
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
    if form == 'fourdimension':
        for i in range(len(mixAng)):
            #finding the array of probabilities for a specific mix ang (prob vals depend on E)
            noDecayProb=noOscProb(E,mixAng[i],diffSqrMass[i],L)
            
            NLLsum=0
            OscillationEventRate=oscEventRate(noDecayProb,simData)
            OscillationEventRate=newEventRate(OscillationEventRate,alpha[i],E)
            for j in range(len(noDecayProb)):
                m=expData[j]
                Lamda=OscillationEventRate[j]
                if m==0:
                    NLLsum+=Lamda
                else:
                    NLLsum+=(Lamda-m+(m*np.log(m/Lamda)))
            likelihood.append(NLLsum)    
    if form=='singular':
        noDecayProb=noOscProb(E,mixAng,diffSqrMass,L)
        NLLsum=0
        OscillationEventRate=oscEventRate(noDecayProb,simData)
        OscillationEventRate=newEventRate(OscillationEventRate,alpha,E)
        for j in range(len(noDecayProb)):
            m=expData[j]
            Lamda=OscillationEventRate[j]
            if m==0:
                NLLsum+=Lamda
            else:
                NLLsum+=(Lamda-m+(m*np.log(m/Lamda)))
        likelihood = NLLsum

    return likelihood


def minimiser_parabolic(func,param,initPoint):
    """
    This function will take the function that wants to be minimised along with the 
    parameters, should work for 2d and 3d functions

    
    """
    form='singular'
    mixAng=param[0]
    diffSqrMass=param[1]
    alpha=param[2]
    #print(len(diffSqrMass))
    xGuess=[]
    massLength=len(diffSqrMass)
    thetaLength=len(mixAng)
    alphaLength=len(alpha)
    
    if massLength==1 and alphaLength==1: #minimising in theta direction
        diffSqrMass=diffSqrMass[0]
        alpha=alpha[0]
        #append initial points
        xGuess.append(initPoint[0])
        firstGuess=initPoint[0]+0.001
        secGuess=initPoint[0]+0.002
        while func(firstGuess,diffSqrMass,alpha,form)>= func(secGuess,diffSqrMass,alpha,form):
            firstGuess+=0.001
            secGuess+=0.001
    
        xGuess.append(firstGuess)
        xGuess.append(secGuess)
    if thetaLength==1 and alphaLength==1: #minimising in delm direction
        mixAng=mixAng[0]
        alpha=alpha[0]
        xGuess.append(initPoint[1])
        firstGuess=initPoint[1] + 0.0005
        secGuess=initPoint[1] + 0.001
        while func(mixAng,firstGuess,alpha,form)>= func(mixAng,secGuess,alpha,form):
            firstGuess+=0.0001
            secGuess+=0.0001
        xGuess.append(firstGuess)
        xGuess.append(secGuess)

    if thetaLength==1 and massLength==1:
        mixAng=mixAng[0]
        diffSqrMass=diffSqrMass[0]

        xGuess.append(initPoint[2])
        firstGuess=initPoint[2] + 0.01
        secGuess=initPoint[2] + 0.02

        while func(mixAng,diffSqrMass,firstGuess,form)>=func(mixAng,diffSqrMass,secGuess,form):
            firstGuess+=0.01
            secGuess+=0.01
        xGuess.append(firstGuess)
        xGuess.append(secGuess)


    #continue iterating until x3 changes by less than 0.001
    diffx3=100
    prevX = 0

    while diffx3>0.000001:

        xFinal=xGuess

        #find all relevant y values
        if massLength==1 and alphaLength==1:
            y0=func(xGuess[0],diffSqrMass,alpha,form)
            y1=func(xGuess[1],diffSqrMass,alpha,form)
            y2=func(xGuess[2],diffSqrMass,alpha,form)
        if thetaLength==1 and alphaLength==1:
            y0=func(mixAng,xGuess[0],alpha,form)
            y1=func(mixAng,xGuess[1],alpha,form)
            y2=func(mixAng,xGuess[2],alpha,form)
        if thetaLength==1 and massLength==1:
            y0=func(mixAng,diffSqrMass,xGuess[0],form)
            y1=func(mixAng,diffSqrMass,xGuess[1],form)
            y2=func(mixAng,diffSqrMass,xGuess[2],form)


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

        if massLength==1 and alphaLength==1:
            y3=func(x3,diffSqrMass,alpha,form)
        if thetaLength==1 and alphaLength==1:
            y3=func(mixAng,x3,alpha,form)
        if thetaLength==1 and massLength==1:
            y3=func(mixAng,diffSqrMass,x3,form)
        

        xGuess.append(x3)
        prevX = x3
        
        #loop through xGuess to remove the x value that corresponds to the largest y
        maxY = 0
        maxX = 0
        for i in xGuess:
            if massLength==1 and alphaLength==1:
                if func(i,diffSqrMass,alpha,form)>maxY:
                    maxY=func(i,diffSqrMass,alpha,form)
                    maxX=i
            if thetaLength==1 and alphaLength==1:
                if func(mixAng,i,alpha,form)>maxY:
                    maxY=func(mixAng,i,alpha,form)
                    maxX=i
            if thetaLength==1 and massLength==1:
                if func(mixAng,diffSqrMass,i,form)>maxY:
                    maxY=func(mixAng,diffSqrMass,i,form)
                    maxX=i
            
        xGuess.remove(maxX)

    return(x3,y3)

#%%

def univariate(func,param,initPoint):
    #first minimise in theta direction, take that point then minimise in mix ang direction
    #repeat this process 
    mixAng=param[0]
    delm=param[1]
    alpha=param[2]
    param=[mixAng,np.array([initPoint[1]]),np.array([initPoint[2]])]
    
    
    xtheta=0
    xdelm=0
    xalpha=0
    
    diff=100
    i=0
    while diff>0.0001:
        old=np.array([[xtheta],[xdelm],[xtheta]])
        xtheta=minimiser_parabolic(func,param,initPoint)[0]
         #putting it into array
        param=[np.array([xtheta]),delm,np.array([initPoint[2]])]

        xdelm=minimiser_parabolic(func,param,initPoint)[0]

        param=[np.array([xtheta]),np.array([xdelm]),alpha]

        xalpha=minimiser_parabolic(func,param,initPoint)[0]

        initPoint=[xtheta,xdelm,xalpha]
        param=[mixAng,np.array([xdelm]),np.array([xalpha])]
        new=np.array([[xtheta],[xdelm],[xalpha]])
        diff=abs(func(new[0][0],new[1][0],new[2][0],'singular') - func(old[0][0],old[1][0],old[2][0],'singular'))
        i+=1
        print(NLL(xtheta,xdelm,xalpha,'singular'))


    print("The univariate method took ", i ,"iterations")
    print(NLL(xtheta,xdelm,xalpha,'singular'))
    return(xtheta,xdelm)

diffSqrMass=np.linspace(1e-3,4.8e-3,200)
mixAng=np.linspace(np.pi/32,np.pi/2,200)
alpha=np.linspace(1,3,200)



univariate(NLL,[mixAng,diffSqrMass,alpha],[0.4,0.001,1.1])



#%%
def gradMin(func,initPoint,alpha):
    delta=1e-5
    theta=initPoint[0]
    delM=initPoint[1]
    varAlpha=initPoint[2]
    diff=100
    i=0

    #scale variables to speed up convergence
    theta=theta*1e-2
    varAlpha=varAlpha*1e-3

    while diff>0.000001:


        #creating an array to append the grad vector
        grad=[]
        
        #find grad using central difference approximation
        gradTheta=(func((theta+delta)*1e2,delM,varAlpha*1e3,'singular')-func((theta-delta)*1e2,delM,varAlpha*1e3,'singular'))/(2*delta)
        grad.append([gradTheta])
        graddelM=(func(theta*1e2,delM+delta,varAlpha*1e3,'singular')-func(theta*1e2,delM-delta,varAlpha*1e3,'singular'))/(2*delta)
        grad.append([graddelM])
        gradAlpha=(func(theta*1e2,delM,(varAlpha+delta)*1e3,'singular')-func(theta*1e2,delM,(varAlpha-delta)*1e3,'singular'))/(2*delta)
        grad.append([gradAlpha])
        grad=np.array(grad)

        #calculating the new point using the grad vector
        #scale the theta value so it matches in magnitude to delm
        xNew=np.array([[theta],[delM],[varAlpha]])- (alpha*grad)
        
        j=1
        #this while loop checks that the new point is lower than the old point, if not reduce alpha
        
        while NLL(xNew[0][0]*1e2,xNew[1][0],xNew[2][0]*1e3,'singular')>NLL(theta*1e2,delM,varAlpha*1e3,'singular'):
            newAlpha=alpha-(j*1e-10)
            xNew=np.array([[theta],[delM],[varAlpha]])-(newAlpha*grad)
            j+=1
        
        #finding the difference between last and new point
        old=[[theta],[delM],[varAlpha]]
        diff=abs(NLL(xNew[0][0]*1e2,xNew[1][0],xNew[2][0]*1e3,'singular') - NLL(theta*1e2,delM,varAlpha*1e3,'singular'))

        #replacing the vector with the new points
        theta=xNew[0][0]    
        delM=xNew[1][0]
        varAlpha=xNew[2][0]
        i+=1

        print(i)
        print(NLL(theta*1e2,delM,varAlpha*1e3,'singular'))
    print("The gradient method took ", i , "iterations")
    print(NLL(theta*1e2,delM,varAlpha*1e3,'singular'))
    return(theta*1e2,delM,varAlpha*1e3)

gradMin(NLL,[0.4,0.001,1.3],1e-9)


#%%

def quasiNewtonMin(func,initPoint,alpha):
    #the quasi Newton minimum search is like a combination of the gradient method and newton method
    delta=1e-5
    theta=initPoint[0]
    delM=initPoint[1]
    varAlpha=initPoint[2]
    diff=100
    i=0

    #scaling the variables to make convergence faster
    theta=theta*1e-2
    varAlpha=varAlpha*1e-3

    while diff>0.000001:


        #find grad using central difference approximation
        gradTheta=(NLL((theta+delta)*1e2,delM,varAlpha*1e3,'singular')-NLL((theta-delta)*1e2,delM,varAlpha*1e3,'singular'))/(2*delta)
        graddelM=(NLL(theta*1e2,delM+delta,varAlpha*1e3,'singular')-NLL(theta*1e2,delM-delta,varAlpha*1e3,'singular'))/(2*delta)
        gradAlpha=(NLL(theta*1e2,delM,(varAlpha+delta)*1e3,'singular')-NLL(theta*1e2,delM,(varAlpha-delta)*1e3,'singular'))/(2*delta)
        grad=np.array([[gradTheta],[graddelM],[gradAlpha]])

        if i==0:
            G=np.identity(3)
        else:
            outerProd=np.outer(xdelta,xdelta)
            G=G + (outerProd)*(1/np.dot(gamma.transpose(),xdelta)) - (np.matmul(G,np.matmul(outerProd,G)))*(1/np.dot(gamma.transpose(),np.matmul(G,gamma)))

        old=np.array([[theta],[delM],[varAlpha]])
        newPoint=old - (alpha * np.matmul(G,grad))

        xdelta=newPoint - old
        gradThetaNew=(NLL((newPoint[0][0]+delta)*1e2,newPoint[1][0],newPoint[2][0]*1e3,'singular')-NLL((newPoint[0][0]-delta)*1e2,newPoint[1][0],newPoint[2][0]*1e3,'singular'))/(2*delta)
        graddelMNew=(NLL(newPoint[0][0]*1e2,newPoint[1][0]+delta,newPoint[2][0]*1e3,'singular')-NLL(newPoint[0][0]*1e2,newPoint[1][0]-delta,newPoint[2][0]*1e3,'singular'))/(2*delta)
        gradAlphaNew=(NLL(newPoint[0][0]*1e2,newPoint[1][0],(newPoint[2][0]+delta)*1e3,'singular')-NLL(newPoint[0][0]*1e2,newPoint[1][0],(newPoint[2][0]-delta)*1e3,'singular'))/(2*delta)
        newGrad=np.array([[gradThetaNew],[graddelMNew],[gradAlphaNew]])
        gamma=newGrad-grad
        diff =abs(NLL(newPoint[0][0]*1e2,newPoint[1][0],newPoint[2][0]*1e3,'singular') - NLL(theta*1e2,delM,varAlpha*1e3,'singular'))
        theta=newPoint[0][0]
        delM=newPoint[1][0]
        varAlpha=newPoint[2][0]

        i+=1
        print(i)
        print(NLL(theta*1e2,delM,varAlpha*1e3,'singular'))

    print("The QuasiNewton method took ", i, "iterations")
    print(NLL(theta*1e2,delM,varAlpha*1e3,'singular'))
    return(theta*1e2,delM,varAlpha*1e3)

quasiNewtonMin(NLL,[0.4,0.001,1.3],1e-9)


#%%
