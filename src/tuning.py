from dyersmat import *
import math

def tuning(X, d, n):

    b = math.ceil(math.log2(X+1))
    M = (2**b) - 1
    kappa_min = ((n+1)**d)*(M**d)
    nu = math.log2(kappa_min+1)

    bit_length = 900
    rho = 32
    rho_ = math.ceil(nu)+rho
    kappa, _ = keygen(bit_length, rho, rho_)
    p_min = ((n)**d)*(M+(kappa**2))**d
    lam = int(math.ceil(math.log2(p_min+1)))

    return lam, rho, rho_, p_min, kappa_min