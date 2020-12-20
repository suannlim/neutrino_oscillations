"""
This file will contain the functions that provide validations for my function outputs
"""

#import relevant libraries
import numpy as np


def oneDvalidation(x,y,z,form):
    """
    This function will output values of y =x^2 to show that the parabolic minimiser will correctly converge at zero

    """
    return(x**3 +x**2)

def contourFunc(x,y,z,form):
    """
    This function will output values of x^2+y^2 to show that the univariate, gradient and quasi newton minimisers work
    """

    return(x**2 + y**2)

def threeDim(x,y,z,form):

    return(x**2 + y**2 + z**2)

 



