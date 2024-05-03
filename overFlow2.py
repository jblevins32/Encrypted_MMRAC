from dyersmat import *

bit_length = 700
rho = 16
rho_ = 100
delta = .1
kappa, p = keygen(bit_length, rho, rho_)
mod = pgen(bit_length, rho_, p)

d = 4  # Degree of polynomial
n = 1  # Number of variables in polynomial
# 8e32

# Equations
x = 100
enc_x = enc(x, kappa, p, mod, delta)

u = x**d
enc_u = mult(mult(mult(enc_x, enc_x, mod), enc_x, mod), enc_x, mod)


test1 = ((n+1)**d)*(u**d)
test2 = ((n+1)**d)*((u+(kappa**2))**d)

if (kappa > test1):
    print("Condition 1 satisfied")
else:
    print("Condition 1 NOT satisfied")

if (p > test2):
    print("Condition 2 satisfied")
else:
    print("Condition 2 NOT satisfied")

dec_u = dec(enc_u, kappa, p, delta**d)

print(u)
print(dec_u)

stop = 1