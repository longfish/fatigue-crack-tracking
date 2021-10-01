import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

N = [80000, 90000, 100000, 105000, 115000, 125000,
     130000, 135000, 140000, 150000, 160000]
a = [22, 32, 37, 40, 42, 62, 66, 80, 98, 120, 166]

N5 = np.array(N[0:5])
a5 = np.array(a[0:5])


def func(params, x):
    '''Standard Form of Quadratic Function'''
    a, b, c = params
    return a * x * x + b * x + c


def error(params, x, y):
    '''Error function, that is, the difference between the value obtained by fitting curve and the actual value'''
    return func(params, x) - y


def solvePara():
    '''Solving parameters'''
    p0 = [1, 1, 1]
    para = leastsq(error, p0, args=(N5, a5))
    return para
