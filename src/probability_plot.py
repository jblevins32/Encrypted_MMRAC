import numpy as np
import matplotlib.pyplot as plt
from dyers import *
from tuning import *

'''Probability of attack detection stuff
Args:
    X = largest plaintext value with encoding factors accounted for
    d = degree of polynomial
    N = number of variables
'''
class prob_plot:
    def __init__(self, X, d, N):
        self.prob_vec = np.array([])
        self.bit_length, self.rho, self.rho_, self.p_min, _ = tuning(X, d, N)

    # Get probability vectors for plotting
    def calc_prob(self):
        bit_vec = np.arange(self.bit_length, self.bit_length + 15)
        for i in bit_vec:
            _, p = keygen(i, self.rho, self.rho_)
            self.prob_vec = np.append(self.prob_vec, (p - self.p_min) / p)
        return bit_vec, self.prob_vec