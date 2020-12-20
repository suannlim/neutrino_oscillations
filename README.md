# Computational Physics - Neutrino Oscillations Project

This project is essentially split into two parts. The first is about implementing a decision tree algorithm and using 
it to determine an indoor location based on what WIFI strengths have been obtained. The second is to create and use 
evaluation functions to test/modify the accuracy of our algorithm. We are using k-fold cross validation in order to 
test our algorithm and pruning to modify our generated decision trees. 

Replace above with your own description of project!!!

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

Replace below with description of main functions!!!

Set the dataset variables, which contains the path to the clean and noisy dataset.
```
clean_dataset = "venv/wifi_db/clean_dataset.txt"
noisy_dataset = "venv/wifi_db/noisy_dataset.txt"
```

Create the trees using the `step_two_create_trees` function.
```
step_two_create_trees(clean_dataset)
step_two_create_trees(noisy_dataset)
```

Evaluate the model using the `step_three_eval` function.
```
step_three_eval(clean_dataset)
step_three_eval(noisy_dataset)
```

Add in the pruning as well as additional validation fold evaluation using the `step_four_prune_and_eval` function.
```
step_four_prune_and_eval('venv/wifi_db/clean_dataset.txt')
step_four_prune_and_eval('venv/wifi_db/noisy_dataset.txt')
```

## Subfunctions (If you want to know how it works)

These are the functions which go into making the whole thing work. The return values and inputs are specified here, with very brief descriptions of their purpose as well.

#### minimisationFunctions.py:

##### noOscProb
One/two sentences on what this does.

```
noOscProb(E,mixAng,diffSqrMass,L)
```

##### oscEventRate
One/two sentences on what this does.

```
oscEventRate(noDecayProb,simData)
```

##### newEventRate
One/two sentences on what this does.

```
newEventRate(EventRate,alpha,E)
```

##### NLL
One/two sentences on what this does.

```
NLL(mixAng,diffSqrMass,alpha,form='general')
```

##### minimiser_parabolic
One/two sentences on what this does.

```
minimiser_parabolic(func,param,initPoint,dim)
```

##### univariate
One/two sentences on what this does.

```
univariate(func,param,initPoint,dim):
```

##### gradMin
One/two sentences on what this does.

```
gradMin(func,initPoint,alpha,dim):
```

##### quasiNewtonMin
One/two sentences on what this does.

```
quasiNewtonMin(func,initPoint,alpha,dim):
```

#### errorFunctions.py:

##### NLLgaussianError
One/two sentences on what this does.

```
NLLgaussianError(param,varyingParamVals,minimum)
```

##### NLLshiftError
One/two sentences on what this does.

```
NLLshiftError(func,minPoint,minNLL,form)
```

#### visualisationFunctions.py:

##### histogramPlot
One/two sentences on what this does.

```
histogramPlot(title,xValues,barHeight,xlabel,ylabel)
```

##### linePlot
One/two sentences on what this does.

```
linePlot(title,xValues,yValues,xlabel,ylabel)
```

##### singlePoint
One/two sentences on what this does.

```
singlePoint(xVal,yVal)
```


##### likelihoodContourPlot
One/two sentences on what this does.

```
likelihoodContourPlot(title,varyingVarVals,func,xlabel,ylabel)
```

##### ContourPath
One/two sentences on what this does.

```
ContourPath(title,varyingVarVals,func,xlabel,ylabel,points,points1,points2)
```

## Built With

* [NumPy](https://numpy.org/) - Used for retrieving data from .txt file and matrix manipulation.
* [Matplotlib](https://matplotlib.org/) - Used to visualise the graphs.

## Author

* Su Ann Lim

## Acknowledgments
[//]: <> (Add in acknowledgments for initial code given by lecturers)
* Thank you to our lecturers for providing us this coursework.
