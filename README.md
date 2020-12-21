# Computational Physics - Neutrino Oscillations Project

This project uses simulated and experimental data of neutrino oscillations to extract the relevant neutrino oscillation parameters of mixing angle and square mass difference. It also investigates the linear rate of increase of the neutrino event rate with energy. The extractions of these parameters are done by finding the minimum point of a negative log-likelihood function to find a set of parameters that fit the data given.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing 
purposes.

### Prerequisites

Modules needed in order to run the program: 

```
NumPy
Matplotlib
```
Numpy is necessary to load data and manipulate matrices. Matplotlib is used to visualise the contour graphs and finish points of the minimisation function.

## Usage guide

First, make sure you have the above prerequisites installed. Next, make sure you have a way to run python code. Open up this project in your IDE.

For each of the parts, I have implemented easy to use functions. The place where you will run the functions is in main.py. 
Everything else is contained within the other files.

The data is read in from the data.txt file and plots of the data are produced. The event rate and neutrino oscillation probability is plotted as a function of energy. Minimisation in the 1D mixing angle parameter is carried out and the relevant error is found.
```
Part3()
```

Minimisation in 2D is carried out with respect to mixing angle and difference square mass via all 3 minimisers. A contour plot showing the path of minimisation is also plotted. Errors of all minimum points are found.
```
Part4()
```

Minimisation in 3D is carried out with respect to mixing angle, difference square mass and alpha via all 3 minimisers. The error values corresponding to each minimum is found.
```
Part5()
```

## Subfunctions (If you want to know how it works)

These are the functions which go into making the whole thing work. The return values and inputs are specified here, with very brief descriptions of their purpose as well.

#### minimisationFunctions.py:

##### noOscProb

This function loops through an array of E values to find the survival probability with given mixing angle and difference square mass values.
```
noOscProb(E,mixAng,diffSqrMass,L)
```

##### oscEventRate
This function multiplies the survival probability by the simulated neutrino flux to calculate the expected event rate.

```
oscEventRate(noDecayProb,simData)
```

##### newEventRate
This function takes into account that event rate linearly scales with energy, multiplying the old event rate array by a factor alpha and energy.

```
newEventRate(EventRate,alpha,E)
```

##### NLL
This function calculates the Negative Log Likelihood based on the set of input parameters. The function can take different 'form' strings to return arrays varying in one paramater, varying two parameters or singular points in 2D or 3D minimisation.

```
NLL(mixAng,diffSqrMass,alpha,form='general')
```

##### minimiser_parabolic
This function takes an initial point and iterates through until two other points are found that satifies the function condition. A second order lagrange polynomial is found and minimised, the lowest 3 points are keep and the iteration continues until the function value changes by a small amount - finding the minimum of the input function.

```
minimiser_parabolic(func,param,initPoint,dim)
```

##### univariate
This function extends parabolic minimisation into multi dimensional cases. The function checks the input values and depending on the length of the array, the function decides which parameter to minimise in.

```
univariate(func,param,initPoint,dim):
```

##### gradMin
This function finds the minimum of the input function by following the descent of the gradient.

```
gradMin(func,initPoint,alpha,dim):
```

##### quasiNewtonMin
This function finds the minimum of the input function by following the descent of the gradient and modifying it by the approximation of the inverse Hessian.

```
quasiNewtonMin(func,initPoint,alpha,dim):
```

#### errorFunctions.py:

##### NLLgaussianError
This function calculates the standard deviation by approximating the curvature at the minimum as a Gaussian. 

```
NLLgaussianError(param,varyingParamVals,minimum)
```

##### NLLshiftError
This function finds the corresponding parameter values when the function value is shifted up by 0.5. The average difference between the parameter values give one standard deviation.

```
NLLshiftError(func,minPoint,minNLL,form)
```

#### visualisationFunctions.py:

##### histogramPlot
Returns a histogram of input values.

```
histogramPlot(title,xValues,barHeight,xlabel,ylabel)
```

##### linePlot
Returns a line graph of input values.

```
linePlot(title,xValues,yValues,xlabel,ylabel)
```

##### singlePoint
Returns a singular point.

```
singlePoint(xVal,yVal)
```


##### likelihoodContourPlot
Returns a contour graph of 3 arrays.

```
likelihoodContourPlot(title,varyingVarVals,func,xlabel,ylabel)
```

##### ContourPath
Returns a contour graph of 3 arrays as well as the path of minimisation.

```
ContourPath(title,varyingVarVals,func,xlabel,ylabel,points,points1,points2)
```

## Built With

* [NumPy](https://numpy.org/) - Used for retrieving data from .txt file and matrix manipulation.
* [Matplotlib](https://matplotlib.org/) - Used to visualise the graphs.

## Author

* Su Ann Lim

## Acknowledgments

* Thank you to my lecturers for providing us this coursework and my demonstrators who were very patient with answering all my queries. 
