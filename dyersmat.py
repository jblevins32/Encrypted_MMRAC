import numpy as np
from dyers import *

def mat_enc(matrix, kappa, p, modulus, delta):
    if isinstance(matrix, list): # turns list types into ndarray types
        matrix = np.array(matrix)
    if len(matrix.shape) == 1: # Condition where only one column
        rows = matrix.shape[0]
        cols = 1
        enc_matrix = np.zeros((int(rows), int(cols)), dtype=object)
        for i in range(rows):
            enc_matrix[i] = enc(matrix[i], kappa, p, modulus, delta)
        return enc_matrix
    else: # Condition with multiple columns
        rows, cols = np.shape(matrix)
        enc_matrix = np.zeros((int(rows), int(cols)), dtype=object)
        for i in range(rows):
            for ii in range(cols):
                enc_matrix[i][ii] = enc(matrix[i][ii], kappa, p, modulus, delta)
        return enc_matrix

def mat_dec(matrix, kappa, p, delta):
    if isinstance(matrix, list):
        matrix = np.array(matrix)
    if len(matrix.shape) == 1:
        rows = matrix.shape[0]
        cols = 1
        dec_matrix = np.zeros((int(rows), int(cols)), dtype=object)
        for i in range(rows):
            dec_matrix[i] = dec(matrix[i], kappa, p, delta)
        return dec_matrix
    else:
        rows, cols = np.shape(matrix)
        dec_matrix = np.zeros((int(rows), int(cols)), dtype=object)
        for i in range(rows):
            for ii in range(cols):
                dec_matrix[i][ii] = dec(matrix[i][ii], kappa, p, delta)
        return dec_matrix
