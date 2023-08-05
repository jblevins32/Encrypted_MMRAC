import numpy as np
import matplotlib.pyplot as plt
from dyersmat import *
import math
from compare import *
import pdb

class Encrypter():
    def __init__(s, enc_method):
        # Encryption
        s.bit_length = 256
        s.rho = 1
        s.rho_ = 32
        s.delta = 0.00000000001
        s.kappa, s.p = keygen(s.bit_length, s.rho, s.rho_)
        s.mod = pgen(s.bit_length, s.rho_, s.p)
        s.reset_xr = 1  # Reset Encryption of xr
        s.reset_reg_eps = 1  # Reset Encryption of epsilon and regressor generator
        s.reset_par = 1  # Reset Encryption of par

        # system parameters
        s.m = 1
        s.b = 2
        s.k = 0

        s.zeta = .8  # Damping Ratio
        s.wn = .5  # Natural Frequency
        s.beta_1 = 1  # 2 * zeta * wn
        s.beta_0 = 1  # wn * wn

        # Creating plant state space
        s.A = np.array([[0, 1], [-s.k / s.m, -s.b / s.m]])
        s.B = np.array([[0], [1 / s.m]])

        # Creating reference state space
        s.Ar = np.array([[0, 1], [-s.beta_0, -s.beta_1]])
        s.Br = np.array([[0], [s.beta_0]])

        # Initial Conditions
        s.x = np.array([[0], [0]])
        s.xr = np.array([[0], [0]])

        # Other variables
        # s.c = np.array([[2, 3.125]])
        s.c = np.array([[.5, 1]])
        s.gam1 = 15
        s.gam2 = 1
        s.gains = np.array([[-s.gam1, 0], [0, -s.gam2]])
        s.u = 0
        s.r = 0
        s.Ts = .1
        s.Ts_time = s.Ts
        s.enc_par_dot_vec = []
        s.enc_x_vec = []
        s.enc_xr_vec = []
        s.enc_e_vec = []
        s.enc_reg_vec = []
        s.enc_u_vec = []
        s.enc_par_vec = []
        s.reg_vec = []
        s.par_dot_vec = []
        s.par_dot_vec_test = []
        s.x_vec = []
        s.xr_vec = []
        s.e_vec = []
        s.par_vec = []
        s.u_vec = []
        s.r_vec = []
        s.enc_par = np.array([[0], [0]])
        s.par = np.array([[0], [0]])

        s.t = 0  # time
        s.Encrypt = enc_method  # Encrypt? 0 = none, 1 = encode, 2 = encrypt

    def encrypt(s):
        for k in range(1, 28):
            if s.Encrypt == 2:
                s.enc_ada(k)
            elif s.Encrypt == 1:
                s.encode_ada(k)
            elif s.Encrypt == 0:
                s.ada(k)
        s.t = np.arange(s.Ts, (k+1) * s.Ts, s.Ts)

    def enc_ada(s, k):
        # encryption of matrices and variables
        if k == 1:
            enc_Ts = enc(s.Ts, kappa, p, mod, delta)  # d1
            enc_d = enc(1 / delta, kappa, p, mod, delta)  # for balancing encoding depths
            enc_Ar = mat_enc(s.Ar, kappa, p, mod, delta)  # d1
            enc_Br = mat_enc(s.Br, kappa, p, mod, delta)  # d1
            enc_xr = mat_enc(s.xr, kappa, p, mod, delta)  # d1
            enc_r = enc(s.r, kappa, p, mod, delta)  # d1
            enc_c = mat_enc(s.c, kappa, p, mod, delta)  # d1
            enc_beta_0 = enc(s.beta_0, kappa, p, mod, delta)  # d1
            enc_beta_1 = enc(s.beta_1, kappa, p, mod, delta)  # d1
            enc_gains = mat_enc(s.gains, kappa, p, mod, delta)  # d1

        # Calculating next encrypted reference state
        enc_xr = add(mat_mult(enc_Ar, enc_xr, mod), mat_mult(enc_Br, enc_r, mod), mod)  # d2
        r_vec.append(dec(enc_r, kappa, p, delta))  # d0

        # For resetting xr
        if reset_xr == 1:
            xr = mat_dec(enc_xr, kappa, p, delta)
            enc_xr = mat_enc(xr, kappa, p, mod, delta ** 2)  # d2
            neg_enc_xr = mat_enc(-xr, kappa, p, mod, delta ** 2)  # d2 for subtracting values later

        # PLANT: Calculating next output based on the input
        x = np.dot(A, x) + np.dot(B, u)  # Only the output of this needs to be encrypted
        enc_x = mat_enc(x, kappa, p, mod, delta ** 2)  # d2

        # Error and filtered error calculation
        enc_e = add(enc_x, neg_enc_xr, mod)  # d2
        enc_eps = mat_mult(enc_c, enc_e, mod)  # d3

        # Need these vectors later for plotting
        enc_x_vec.append(enc_x.flatten())  # d2
        enc_xr_vec.append(enc_xr.flatten())  # d2
        enc_e_vec.append(enc_e.flatten())  # d2

        # Regressor Generator split into parts for debugging
        r = -math.sin(t + math.pi / 2) + 1
        enc_r = enc(r, kappa, p, mod, delta)  # d1
        enc_reg = np.array([mult([enc_x[1][0]], enc_d, mod), [
            add(mult(add(mult(enc_r, enc_d, mod), -enc_x[0][0], mod), enc_beta_0, mod),
                mult(-enc_x[1][0], enc_beta_1, mod), mod)]])  # d3
        enc_reg_vec.append(enc_reg.flatten())

        # Resetting reg and eps because of overflow
        if reset_reg_eps == 1:
            reg = mat_dec(enc_reg, kappa, p, delta ** 3)
            # reg = np.array([reg[0][0] / delta, reg[1][0]]) * delta * delta  # Correcting encoding depth
            reg_vec.append(reg.flatten())
            enc_reg = mat_enc(reg, kappa, p, mod, delta)  # d1
            eps = mat_dec(enc_eps, kappa, p, delta ** 3)
            enc_eps = enc(eps, kappa, p, mod, delta)  # d1

        # Parameter adaptation
        enc_par_mult = mat_mult(enc_eps, enc_reg, mod)  # d2
        enc_par_dot = mat_mult(enc_gains, enc_par_mult, mod)  # d3
        enc_par_dot_vec.append(enc_par_dot.flatten())  # d3

        # Integrator
        if k != 1:
            enc_par = add(mat_mult(enc_par, enc_d, mod),
                          mat_mult(enc_par_dot_vec[k - 2].reshape((2, 1)), enc_Ts, mod), mod)  # d3
        enc_par_vec.append(enc_par.flatten())  # d3

        # Resetting par because of overflow
        if reset_par == 1:
            par = mat_dec(enc_par, kappa, p, delta ** 4)
            enc_par = mat_enc(par, kappa, p, mod, delta)

        # Calculating next input if par and reg were not exposed
        if reset_par == 0 & reset_reg_eps == 0:
            enc_u = mat_mult(enc_reg.transpose(), enc_par, mod, kappa, p, delta)
            enc_u_vec.append(enc_u.flatten())
            u = dec(enc_u, kappa, p, delta)
            u_vec.append(u)
        else:
            # Calculating next input if par and reg are exposed/reset
            s.u = [np.dot(reg.transpose(), par)]  # dec(enc_u, kappa, p, delta) * delta
            s.u_vec.append([float(x) for x in u])

        s.t = s.t + s.Ts

    def encode_ada(s, k):
        if k == 1:
            s.Ts = encode(s.Ts, s.delta)  # d1
            s.c = mat_encode(s.c, s.delta)  # d1
            s.beta_0 = encode(s.beta_0, s.delta)  # d1
            s.beta_1 = encode(s.beta_1, s.delta)  # d1
            s.gains = mat_encode(s.gains, s.delta)  # d1
            s.Ar = mat_encode(s.Ar, s.delta)  # d1
            s.xr = mat_encode(s.xr, s.delta)  # d1
            s.Br = mat_encode(s.Br, s.delta)  # d1
            s.r = encode(s.r, s.delta)  # d1

        s.xr = np.dot(s.Ar, s.xr) + np.dot(s.Br, s.r)  # d2
        s.x = np.dot(s.A, s.x) + np.dot(s.B, s.u)
        e = s.x - s.xr
        eps = np.dot(s.c.flatten(), e)  # d3

        # Need these vectors later for plotting
        s.x_vec.append(s.x.flatten()*(s.delta**2))  # d0
        s.xr_vec.append(s.xr.flatten()*(s.delta**2))  # d0
        s.e_vec.append(e.flatten()*(s.delta**2))  # d2

        # Regressor Generator split into parts for debugging
        s.r = (-math.sin(s.t + math.pi / 2) + 1)
        s.r = encode(s.r, s.delta)  # d1
        s.r_vec.append(decode(s.r,s.delta))  # d1
        p1 = s.x[1][0] / (s.delta**2)
        p2 = ((s.r / s.delta) - s.x[0][0]) * s.beta_0 - s.x[1][0] * s.beta_1
        reg = np.array([[p1], [p2]])  # d3
        s.reg_vec.append(mat_decode(reg.flatten(), s.delta**3))  # d3

        par_mult = eps * reg  # d6
        par_mult = par_mult.reshape((2, 1))  # Reshape to a 2x1 vector

        # Parameter adaptation
        par_dot = np.dot(s.gains, par_mult)  # d7
        s.par_dot_vec.append(par_dot.flatten())  # d7
        s.par_dot_vec_test.append(mat_decode(par_dot.flatten(), s.delta**7))  # d7

        # Integrator
        if k != 1:
            s.par = mat_encode(s.par, s.delta**7) + np.dot(s.par_dot_vec[k - 1].reshape((2, 1)), s.Ts_time) # d7 # Ts might have to be different here than rest of code. IDK why.
        s.par_vec.append(s.par.flatten()*(s.delta**7))  # d7

        s.u = float(np.dot(reg.transpose(), s.par))  # d10

        # decoding
        s.u = decode(s.u, s.delta**10)
        s.u_vec.append(s.u)

        s.x = decode(s.x, s.delta**2)
        s.xr = decode(s.xr, s.delta**2)
        s.par = decode(s.par, s.delta**7)
        # recoding for next iteration
        s.u = encode(s.u, s.delta)
        s.x = mat_encode(s.x, s.delta)
        s.xr = mat_encode(s.xr, s.delta)

        s.t = s.t + s.Ts_time

    def ada(s, k):
        s.xr = np.dot(s.Ar, s.xr) + np.dot(s.Br, s.r)
        s.x = np.dot(s.A, s.x) + np.dot(s.B, s.u)  # d2 - Only the output of this needs to be encrypted
        e = s.x - s.xr
        eps = np.dot(s.c.flatten(), e)

        # Need these vectors later for plotting
        s.x_vec.append(s.x.flatten())
        s.xr_vec.append(s.xr.flatten())
        s.e_vec.append(e.flatten())

        # Regressor Generator split into parts for debugging
        s.r = (-math.sin(s.t + math.pi / 2) + 1)
        s.r_vec.append(s.r)
        reg = np.array([[s.x[1][0]], [(s.r - s.x[0][0]) * s.beta_0 - s.x[1][0] * s.beta_1]])
        s.reg_vec.append(reg.flatten())

        par_mult = eps * reg
        par_mult = par_mult.reshape((2, 1))  # Reshape to a 2x1 vector

        # Parameter adaptation
        par_dot = np.dot(s.gains, par_mult)
        s.par_dot_vec.append(par_dot.flatten())

        # Integrator
        if k != 1:
            s.par = s.par + np.dot(s.par_dot_vec[k - 1].reshape((2, 1)),
                               s.Ts)  # Ts might have to be different here than rest of code. IDK why.
        s.par_vec.append(s.par.flatten())

        s.u = float(np.dot(reg.transpose(), s.par))
        s.u_vec.append(s.u)

        s.t = s.t + s.Ts
