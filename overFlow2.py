from dyersmat import *
import math
import matplotlib.pyplot as plt
from mpmath import mp

M = 100000000 #134217727 # STEP 1 & 2: max number in the algorithm
d = 4  # Degree of polynomial: x^d
n = 1  # Number of variables in polynomial

kappa_min = ((n+1)**d)*(M**d) # STEP 3
nu = math.log2(kappa_min+1) # STEP 4

bit_length = 888
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
original_u = enc_u

#FDIA: uncomment below to see overflow happen
attack = 10000
enc_u = mult(enc_u, attack, mod)

# decrypting final u
dec_u = dec(enc_u, kappa, p, delta**d) # decrypted u

# IF THESE ARE THE SAME, OVERFLOW HAS NOT HAPPENED
print(u)
print(dec_u)

if u != dec_u:
    print('FDIA Detected')
else:
    print('No FDIA Detected')

enc_u = mp.mpf(int(enc_u))
enc_u = mp.log10(enc_u)
original_u = mp.mpf(int(original_u))
original_u = mp.log10(original_u)
print(enc_u)
print(original_u)

# Plotting u and M over time
plt.figure(1, figsize=(8, 4.5))
plt.plot([0, 1], [original_u, original_u], label='Allowed')  # Plot u
# plt.plot([0, 1], [enc_u, enc_u], color='r', linestyle='--', label='Attacked')  # Plot M
# plt.ylim(0, max(u, M)*1.1)  # Set y-axis limit to include 0 and extend slightly beyond max(u, M)
plt.xlim(0, 1)
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel('Plaintext Value', fontsize=16)
plt.legend()
plt.show()