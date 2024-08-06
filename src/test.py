from dyers import *

theta0 = 0  # Initial roll angle (rad)
p0 = 0  # Initial roll rate (rad/s)
x = np.array([theta0, p0])
theta = np.array([1, 0.2314, 0.6918, 0.6245, 0.1, 0.214])
phi = np.array([1, x[0], x[1], np.abs(x[0]) * x[0], np.abs(x[1]) * x[1], x[0] ** 3])


bit_length = 400
rho = 16
rho_ = 64
delta = .1
kappa, p = keygen(bit_length, rho, rho_)
mod = pgen(bit_length, rho_, p)

# This test is for comparing multiplication noise error with
m1 = 5
m2 = 2
c1 = enc(m1, kappa, p, mod, delta)
c2 = enc(m2, kappa, p, mod, delta)
k = 10
test = 2

if test == 1:
    for i in range(k):
        c1 = mult(c1, c2, mod)
    result = dec(c1, kappa, p, delta ** (k + 1))

else:
    for i in range(k):
        for ii in range(m2-1):
            c1 = add(c1, c1, mod)  # add c1 to itself c2 times so c2 is matrix A
    result = dec(c1, kappa, p, delta)

print(result)
