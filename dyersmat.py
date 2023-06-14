import numpy as np
from dyers import *

def mat_enc(matrix,kappa, p, modulus, delta):
    rows = len(matrix)
    cols = len(matrix[0])
    enc_matrix = np.zeros((int(rows), int(cols)), dtype = object)
    for i in range(rows):
        for ii in range(cols):
            enc_matrix[i][ii] = enc(matrix[i][ii], kappa, p, modulus, delta)
    return enc_matrix

def mat_dec(matrix,kappa, p, delta):
    rows = len(matrix)
    cols = len(matrix[0])
    dec_matrix = np.zeros((int(rows), int(cols)), dtype=object)
    for i in range(rows):
        for ii in range(cols):
            dec_matrix[i][ii] = dec(matrix[i][ii], kappa, p, delta)
    return dec_matrix