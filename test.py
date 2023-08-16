from dyers import *

bit_length = 128
rho = 1
rho_ = 64
delta = .1
kappa, p = keygen(bit_length, rho, rho_)
mod = pgen(bit_length, rho_, p)

m = 5

c = enc(m, kappa, p, mod, delta)
result = dec(c, kappa, p, delta)

print(result)