from dyers import *

bit_length = 400
rho = 16
rho_ = 64
delta = .1
kappa, p = keygen(bit_length, rho, rho_)
mod = pgen(bit_length, rho_, p)

# Equations
x = 10000
y = 20000
z = 300
A = 10
B = 20
enc_x = enc(x, kappa, p, mod, delta)
enc_y = enc(y, kappa, p, mod, delta)
enc_z = enc(z, kappa, p, mod, delta**2)
enc_A = enc(A, kappa, p, mod, delta**2)
enc_B = enc(B, kappa, p, mod, delta**2)

u = x**2 + y*x + z + 15
x_dot = A*x + B*u

enc_u = longadd([longmult([enc_x, enc_x], mod), mult(enc_x, enc_y, mod), enc_z, enc(15, kappa, p, mod, delta**2)], mod)
enc_x_dot = add(mult(enc_A, enc_x, mod), mult(enc_B, enc_u, mod), mod)

d = 3  # Degree polynomial
n = 3  # Number of variables in polynomial

v1 = (kappa ** (1/d))/(n + 1)
v2 = ((p ** (1/d))/(n + 1)) - (kappa ** 2)

max_num = abs(floor(min(v1, v2)))

dec_x_dot = dec(enc_x_dot, kappa, p, delta**4)

stop = 1