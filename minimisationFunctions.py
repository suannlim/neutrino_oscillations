"""
This file contains the functions to produce the NLL as well as perform
the minimisation

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

def NLL(mixAng,diffSqrMass,alpha,form='general'):
    
    #create array to store negative Log Likelihood values 
    likelihood=[]

    #read in the data
    expData=readData("dataFile.txt")[0]
    simData=readData("dataFile.txt")[1]

    #set the constants
    L=295
    E=np.arange(0.025,10,0.05)

    if form=='theta':
        #if form=='theta, the NLL function will produce likelihood values of varying theta and fixed mass
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
    if form=='delM':
        #if form=='delm' , the NLL function will produce likelihood values of varying mass and fixed theta
        for i in range(len(diffSqrMass)):
            noDecayProb=noOscProb(E,mixAng,diffSqrMass[i],L)
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
    if form == 'alpha':
        #if form=='alpha', the NLL function will produce likelihood values of varying alpha and fixed theta and mass
        for i in range(len(alpha)):
            noDecayProb=noOscProb(E,mixAng,diffSqrMass,L)
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
    if form=='2dContour':
        #if form=='delm' , the NLL function will produce likelihood values of varying mass and fixed theta
        for i in range(len(mixAng)):
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
   
    if form=='singular2d':
        #if form=='singular, the NLL function will produce one NLL value at the point of fixed theta,mass and alpha
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

    if form=='singular3d':
        #if form=='singular, the NLL function will produce one NLL value at the point of fixed theta,mass and alpha
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

def minimiser_parabolic(func,param,initPoint,dim):
    """
    The parabolic minimiser will minimise along the direction that is input as an array
    The input must consist of a 3d array with array length 200 and the other two with length 1

    
    """
    if dim=='2d':
        form='singular2d'
        mixAng=param[0]
        diffSqrMass=param[1]
        #print(len(diffSqrMass))
        xGuess=[]
        massLength=len(diffSqrMass)
        thetaLength=len(mixAng)
        alphaLength=1
    if dim=='3d':
        form='singular3d'
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
        if dim=='2d':
            alpha=1
        if dim=='3d':
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
        print(xGuess)
    if thetaLength==1 and alphaLength==1: #minimising in delm direction
        mixAng=mixAng[0]
        if dim=='2d':
            alpha=1
        if dim=='3d':
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

def univariate(func,param,initPoint,dim):
    """
    The univariate minimises in the theta direction, then the mass direction, then the alpha direction
    The iterations are then repeated to find the final minimum point
    """

    mixAng=param[0]
    delm=param[1]
    alpha=param[2]
    param=[mixAng,np.array([initPoint[1]]),np.array([initPoint[2]])]
    
    if dim=='2d':
        xtheta=0
        xdelm=0
    if dim=='3d':
        xtheta=0
        xdelm=0
        xalpha=0
    
    diff=100
    i=0
    while diff>0.0000001:
        if dim=='2d':
            old=np.array([[xtheta],[xdelm]])
            xtheta=minimiser_parabolic(func,param,initPoint,'2d')[0]
             #putting it into array
            param=[np.array([xtheta]),delm]

            xdelm=minimiser_parabolic(func,param,initPoint,'2d')[0]
            
            initPoint=[xtheta,xdelm]
            param=[mixAng,np.array([xdelm])]

            new=np.array([[xtheta],[xdelm]])

            diff=abs(func(new[0][0],new[1][0],1,'singular2d') - func(old[0][0],old[1][0],1,'singular2d'))

            i+=1
            #print(NLL(xtheta,xdelm,1,'singular2d'))

        if dim=='3d':
            old=np.array([[xtheta],[xdelm],[xalpha]])
            xtheta=minimiser_parabolic(func,param,initPoint,'3d')[0]
            #putting it into array
            param=[np.array([xtheta]),delm,np.array([initPoint[2]])]

            xdelm=minimiser_parabolic(func,param,initPoint,'3d')[0]

            param=[np.array([xtheta]),np.array([xdelm]),alpha]

            xalpha=minimiser_parabolic(func,param,initPoint,'3d')[0]

            initPoint=[xtheta,xdelm,xalpha]
            param=[mixAng,np.array([xdelm]),np.array([xalpha])]
            new=np.array([[xtheta],[xdelm],[xalpha]])
            diff=abs(func(new[0][0],new[1][0],new[2][0],'singular3d') - func(old[0][0],old[1][0],old[2][0],'singular3d'))
            
            i+=1
            #print(NLL(xtheta,xdelm,xalpha,'singular3d'))


    print("The univariate method took ", i ,"iterations")
    if dim=='2d':
        print(NLL(xtheta,xdelm,1,'singular2d'))
        print("The minimum occurs at",[xtheta,xdelm])
        return(xtheta,xdelm)
    if dim=='3d':
        print(NLL(xtheta,xdelm,xalpha,'singular3d'))
        print("The minimum occurs at",[xtheta,xdelm,xalpha])
        return(xtheta,xdelm,xalpha)

def gradMin(func,initPoint,alpha,dim):
    delta=1e-5

    if dim=='2d':
        theta=initPoint[0]
        delM=initPoint[1]

    if dim=='3d':
        theta=initPoint[0]
        delM=initPoint[1]
        varAlpha=initPoint[2]
    diff=100
    i=0

    #scale variables to speed up convergence
    theta=theta*1e-2
    if dim=='3d':
        varAlpha=varAlpha*1e-3

    while diff>0.000001:
        #creating an array to append the grad vector
        grad=[]
        if dim=='2d':
            gradTheta=(func((theta+delta)*1e2,delM,1,'singular2d')-func((theta-delta)*1e2,delM,1,'singular2d'))/(2*delta)
            grad.append([gradTheta])
            graddelM=(func(theta*1e2,delM+delta,1,'singular2d')-func(theta*1e2,delM-delta,1,'singular2d'))/(2*delta)
            grad.append([graddelM])
            grad=np.array(grad)
            #calculating the new point using the grad vector
            #scale the theta value so it matches in magnitude to delm

            xNew=np.array([[theta],[delM]])- (alpha*grad)
            
            j=1
            #this while loop checks that the new point is lower than the old point, if not reduce alpha
            
            while func(xNew[0][0]*1e2,xNew[1][0],1,'singular2d')>func(theta*1e2,delM,1,'singular2d'):
                newAlpha=alpha-(j*1e-10)
                xNew=np.array([[theta],[delM]])-(newAlpha*grad)
                j+=1

            #finding the difference between last and new point
            old=[[theta],[delM]]
            diff=abs(func(xNew[0][0]*1e2,xNew[1][0],1,'singular2d') - func(theta*1e2,delM,1,'singular2d'))

            #replacing the vector with the new points
            theta=xNew[0][0]    
            delM=xNew[1][0]
            i+=1

            #print(i)
            #print(func(theta*1e2,delM,1,'singular2d'))



        if dim=='3d':
            #find grad using central difference approximation
            gradTheta=(func((theta+delta)*1e2,delM,varAlpha*1e3,'singular3d')-func((theta-delta)*1e2,delM,varAlpha*1e3,'singular3d'))/(2*delta)
            grad.append([gradTheta])
            graddelM=(func(theta*1e2,delM+delta,varAlpha*1e3,'singular3d')-func(theta*1e2,delM-delta,varAlpha*1e3,'singular3d'))/(2*delta)
            grad.append([graddelM])
            gradAlpha=(func(theta*1e2,delM,(varAlpha+delta)*1e3,'singular3d')-func(theta*1e2,delM,(varAlpha-delta)*1e3,'singular3d'))/(2*delta)
            grad.append([gradAlpha])
            grad=np.array(grad)

            #calculating the new point using the grad vector
            #scale the theta value so it matches in magnitude to delm

            xNew=np.array([[theta],[delM],[varAlpha]])- (alpha*grad)
            
            j=1
            #this while loop checks that the new point is lower than the old point, if not reduce alpha
            
            while func(xNew[0][0]*1e2,xNew[1][0],xNew[2][0]*1e3,'singular3d')>func(theta*1e2,delM,varAlpha*1e3,'singular3d'):
                newAlpha=alpha-(j*1e-10)
                xNew=np.array([[theta],[delM],[varAlpha]])-(newAlpha*grad)
                j+=1
            
            #finding the difference between last and new point
            old=[[theta],[delM],[varAlpha]]
            diff=abs(func(xNew[0][0]*1e2,xNew[1][0],xNew[2][0]*1e3,'singular3d') -func(theta*1e2,delM,varAlpha*1e3,'singular3d'))

            #replacing the vector with the new points
            theta=xNew[0][0]    
            delM=xNew[1][0]
            varAlpha=xNew[2][0]
            i+=1

            #print(i)
            #print(func(theta*1e2,delM,varAlpha*1e3,'singular3d'))
    
    print("The gradient method took ", i , "iterations")
    if dim=='2d':
        print(NLL(theta*1e2,delM,1,'singular2d'))
        print("The minimum occurs at", [theta*1e2,delM])
        return(theta*1e2,delM)
    if dim=='3d':
        print(NLL(theta*1e2,delM,varAlpha*1e3,'singular3d'))
        print("The minimum occurs at", [theta*1e2,delM,varAlpha*1e3])
        return(theta*1e2,delM,varAlpha*1e3)

def quasiNewtonMin(func,initPoint,alpha,dim):
    #the quasi Newton minimum search is like a combination of the gradient method and newton method
    delta=1e-5
    if dim=='2d':
        theta=initPoint[0]
        delM=initPoint[1]
    if dim=='3d':
        theta=initPoint[0]
        delM=initPoint[1]
        varAlpha=initPoint[2]
   
    diff=100
    i=0

    #scaling the variables to make convergence faster
    theta=theta*1e-2
    if dim=='3d':
        varAlpha=varAlpha*1e-3

    while diff>0.000001:

        if dim=='2d':
            #find grad using central difference approximation
            gradTheta=(func((theta+delta)*1e2,delM,1,'singular2d')-func((theta-delta)*1e2,delM,1,'singular2d'))/(2*delta)
            graddelM=(func(theta*1e2,delM+delta,1,'singular2d')-func(theta*1e2,delM-delta,1,'singular2d'))/(2*delta)
            grad=np.array([[gradTheta],[graddelM]])

            if i==0:
                G=np.identity(2)
            else:
                outerProd=np.outer(xdelta,xdelta)
                G=G + (outerProd)*(1/np.dot(gamma.transpose(),xdelta)) - (np.matmul(G,np.matmul(outerProd,G)))*(1/np.dot(gamma.transpose(),np.matmul(G,gamma)))

            old=np.array([[theta],[delM]])
            newPoint=old - (alpha * np.matmul(G,grad))

            xdelta=newPoint - old
            gradThetaNew=(func((newPoint[0][0]+delta)*1e2,newPoint[1][0],1,'singular2d')-func((newPoint[0][0]-delta)*1e2,newPoint[1][0],1,'singular2d'))/(2*delta)
            graddelMNew=(func(newPoint[0][0]*1e2,newPoint[1][0]+delta,1,'singular2d')-NLL(newPoint[0][0]*1e2,newPoint[1][0]-delta,1,'singular2d'))/(2*delta)
        
            newGrad=np.array([[gradThetaNew],[graddelMNew]])
            gamma=newGrad-grad
            diff =abs(func(newPoint[0][0]*1e2,newPoint[1][0],1,'singular2d') - NLL(theta*1e2,delM,1,'singular2d'))
            theta=newPoint[0][0]
            delM=newPoint[1][0]

            i+=1
            #print(i)
            #print(func(theta*1e2,delM,1,'singular2d'))

        if dim=='3d':
            #find grad using central difference approximation
            gradTheta=(func((theta+delta)*1e2,delM,varAlpha*1e3,'singular3d')-func((theta-delta)*1e2,delM,varAlpha*1e3,'singular3d'))/(2*delta)
            graddelM=(func(theta*1e2,delM+delta,varAlpha*1e3,'singular3d')-func(theta*1e2,delM-delta,varAlpha*1e3,'singular3d'))/(2*delta)
            gradAlpha=(func(theta*1e2,delM,(varAlpha+delta)*1e3,'singular3d')-func(theta*1e2,delM,(varAlpha-delta)*1e3,'singular3d'))/(2*delta)
            grad=np.array([[gradTheta],[graddelM],[gradAlpha]])

            if i==0:
                G=np.identity(3)
            else:
                outerProd=np.outer(xdelta,xdelta)
                G=G + (outerProd)*(1/np.dot(gamma.transpose(),xdelta)) - (np.matmul(G,np.matmul(outerProd,G)))*(1/np.dot(gamma.transpose(),np.matmul(G,gamma)))

            old=np.array([[theta],[delM],[varAlpha]])
            newPoint=old - (alpha * np.matmul(G,grad))

            xdelta=newPoint - old
            gradThetaNew=(func((newPoint[0][0]+delta)*1e2,newPoint[1][0],newPoint[2][0]*1e3,'singular3d')-func((newPoint[0][0]-delta)*1e2,newPoint[1][0],newPoint[2][0]*1e3,'singular3d'))/(2*delta)
            graddelMNew=(func(newPoint[0][0]*1e2,newPoint[1][0]+delta,newPoint[2][0]*1e3,'singular3d')-func(newPoint[0][0]*1e2,newPoint[1][0]-delta,newPoint[2][0]*1e3,'singular3d'))/(2*delta)
            gradAlphaNew=(func(newPoint[0][0]*1e2,newPoint[1][0],(newPoint[2][0]+delta)*1e3,'singular3d')-func(newPoint[0][0]*1e2,newPoint[1][0],(newPoint[2][0]-delta)*1e3,'singular3d'))/(2*delta)
            newGrad=np.array([[gradThetaNew],[graddelMNew],[gradAlphaNew]])
            gamma=newGrad-grad
            diff =abs(func(newPoint[0][0]*1e2,newPoint[1][0],newPoint[2][0]*1e3,'singular3d') - func(theta*1e2,delM,varAlpha*1e3,'singular3d'))
            theta=newPoint[0][0]
            delM=newPoint[1][0]
            varAlpha=newPoint[2][0]

            i+=1
            #print(i)
            #print(func(theta*1e2,delM,varAlpha*1e3,'singular3d'))

    print("The QuasiNewton method took ", i, "iterations")
    if dim=='2d':
        print(func(theta*1e2,delM,1,'singular2d'))
        print("The minimum occurs at",[theta*1e2,delM])
        return(theta*1e2,delM)
    if dim=='3d':
        print(func(theta*1e2,delM,varAlpha*1e3,'singular3d'))
        print("The minimum occurs at", [theta*1e2,delM,varAlpha*1e3])
        return(theta*1e2,delM,varAlpha*1e3)



