from dyersmat import *

bit_length = 700
rho = 16
rho_ = 64
delta = .1
kappa, p = keygen(bit_length, rho, rho_)
mod = pgen(bit_length, rho_, p)

# Equations
x = 100
y = 20000
z = 300
A = 10
B = 20
enc_x = enc(x, kappa, p, mod, delta)
enc_y = enc(y, kappa, p, mod, delta)
enc_z = enc(z, kappa, p, mod, delta**2)
enc_A = enc(A, kappa, p, mod, delta**3)
enc_B = enc(B, kappa, p, mod, delta)

# u = x**4 + y*x + z + 15
u = x**3
x_dot = A*x + B*u # encoding depth 3

enc_u = mult(mult(enc_x, enc_x, mod), enc_x, mod)
# enc_u = longadd([longmult([enc_x, enc_x], mod), mult(enc_x, enc_y, mod), enc_z, enc(15, kappa, p, mod, delta**2)], mod)
enc_x_dot = add(mult(enc_A, enc_x, mod), mult(enc_B, enc_u, mod), mod)

d = 4  # Degree of polynomial
n = 4  # Number of variables in polynomial

test1 = ((n+1)**d)*(x_dot**d)
test2 = ((n+1)**d)*((x_dot+(kappa**2))**d)

if (kappa > test1):
    print("Condition 1 satisfied")
else:
    print("Condition 1 NOT satisfied")

if (p > test2):
    print("Condition 2 satisfied")
else:
    print("Condition 2 NOT satisfied")

dec_x_dot = dec(enc_x_dot, kappa, p, delta**4)

print(x_dot)
print(dec_x_dot)

stop = 1