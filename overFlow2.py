from dyersmat import *
import math

max_num = 134217727
d = 4  # Degree of polynomial
n = 1  # Number of variables in polynomial

kappa_ineq = ((n+1)**d)*(max_num**d)
nu = math.log2(kappa_ineq+1)

bit_length = 900
rho = 32
rho_ = math.ceil(nu)+rho
delta = 1
kappa, p = keygen(bit_length, rho, rho_)
mod = pgen(bit_length, rho_, p)

p_ineq = ((n+1)**d)*(max_num+(kappa**2))**d
lam = math.log2(p_ineq+1)

# Equations
x = 100
u = x**d
enc_x = enc(x, kappa, p, mod, delta)
enc_u = mult(mult(mult(enc_x, enc_x, mod), enc_x, mod), enc_x, mod)
dec_u = dec(enc_u, kappa, p, delta**d)

print(u)
print(dec_u)

stop = 1