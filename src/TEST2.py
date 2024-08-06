import math
import numpy as np
import matplotlib.pyplot as plt
from dyersmat import *
from tuning import *

d = 2
n = 2
m = 4294967295
lam, rho, rho_, M = tuning(m, d, n)

bit_length = 270
rho = 32
rho_ = 100
delta = 1
kappa, p = keygen(bit_length, rho, rho_)
mod, q = pgen(bit_length, rho_, p)

test = (kappa-(((n+1)**d) * (M**d)))/kappa

x = 4294967295
y = 4294967295
x_enc = enc(x, kappa, p, mod, 1)
y_enc = enc(y, kappa, p, mod, 1)
c = mult(x_enc,y_enc,mod)
dec_m = dec(c, kappa, p, delta)
val = (x*y) + (x*kappa + y*kappa + kappa**3)*kappa
check = abs(dec_m) > M**d

p_log = math.log(p)
val_log = math.log(val)

# move lambda from 272 to 271 to see p change enough to cause overflow
if val_log > p_log:
    print('Overflow has occurred')

test = 1

