import numpy as np


def calc_b(a, h, D):
    denominator_sq = -2*D*D+2*D*np.sqrt(D*D-4*h*h)+4*a*a+4*h*h
    return 2*a*h/np.sqrt(denominator_sq)


print(calc_b(1, 0.9, 2))
