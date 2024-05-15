from dyersmat import *
import math

M = 100000000 #134217727 # STEP 1 & 2: max number in the algorithm
d = 4  # Degree of polynomial: x^d
n = 1  # Number of variables in polynomial

kappa_min = ((n+1)**d)*(M**d) # STEP 3
nu = math.log2(kappa_min+1) # STEP 4

bit_length = 890
rho = 32 # STEP 4 setting rho
rho_ = math.ceil(nu)+rho # STEP 4 determining rho'
delta = 1 # using integers so no encoding factor right now
kappa, p = keygen(bit_length, rho, rho_)

p_ineq = ((n+1)**d)*(M+(kappa**2))**d # STEP 5
lam = math.log2(p_ineq+1) # STEP 6
print(lam) # this is telling me what about bit_length variable needs to be now to cause overflow. Chane the bit_length variable above according to this result.

mod = pgen(bit_length, rho_, p)

# Equations
x = 100
u = x**d # actual u
enc_x = enc(x, kappa, p, mod, delta)
enc_u = mult(mult(mult(enc_x, enc_x, mod), enc_x, mod), enc_x, mod) # Cyphertext operations

#FDIA: uncomment below to see overflow happen
# attack = 2
# enc_u = mult(enc_u, enc(attack, kappa, p, mod, delta), mod)

# decrypting final u
dec_u = dec(enc_u, kappa, p, delta**d) # decrypted u

# IF THESE ARE THE SAME, OVERFLOW HAS NOT HAPPENED
print(u)
print(dec_u)

if u != dec_u:
    print('FDIA Detected')
else:
    print('No FDIA Detected')