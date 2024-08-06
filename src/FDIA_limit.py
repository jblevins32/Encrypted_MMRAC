import math
import numpy as np
import matplotlib.pyplot as plt
from dyersmat import *
from tuning import *
# matplotlib.use('TkAgg')  # Use TkAgg backend for better interactivity

bit_length = 888
rho = 32
rho_ = 143
delta = 1
kappa, p = keygen(bit_length, rho, rho_)
mod, q = pgen(bit_length, rho_, p)
d = 4
n = 1
max_num = 100000000
lam, rho, rho_ = tuning(max_num, d, n)
p_log = math.log(p)

m = 100
C = enc(m, kappa, p, mod, 1) # depth 1
O2 = mult(C, C, mod) # depth 2
O2_log = math.log(O2)

O3 = mult(mult(C, C, mod), C, mod) # depth 3
D3 = dec(O3, kappa, p, 1)
O3_log = math.log(O3)
msk = math.log(O3 % (p))
# p1_check = math.log(((n+1)**d)*(D1+kappa**2)**d)

O4 = mult(mult(mult(C, C, mod), C, mod), C, mod) # depth 4
D4 = dec(O4, kappa, p, 1)
D4_manual = (O4 % p) % kappa
O4_log = math.log(O4)
msk = math.log(O4 % (p))

O5 = mult(mult(mult(mult(C, C, mod), C, mod), C, mod), C, mod)
D5 = dec(O5, kappa, p, 1)
test = (O5 % p) % kappa
msk = math.log(O5 % (p))
N = math.log(p*q)


O4 = mult(mult(mult(mult(mult(C, C, mod), C, mod), C, mod), C, mod), C, mod)
D4 = dec(O4, kappa, p, 1)

O5 = mult(mult(mult(mult(mult(mult(C, C, mod), C, mod), C, mod), C, mod), C, mod), C, mod)
D5 = dec(O5, kappa, p, 1)
real = (117**6)