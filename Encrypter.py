import numpy as np
import matplotlib.pyplot as plt
from dyersmat import *
from integrator import *
import math
from compare import *
import pdb

class Encrypter():
    def __init__(s, enc_method):
        # Encryption
        s.bit_length = 256
        s.rho = 1
        s.rho_ = 32
        s.delta = 0.0001
        s.kappa, s.p = keygen(s.bit_length, s.rho, s.rho_)
        s.mod = pgen(s.bit_length, s.rho_, s.p)
        s.reset_xr = 1  # Reset Encryption of xr
        s.reset_reg_eps = 1  # Reset Encryption of epsilon and regressor generator
        s.reset_par = 1  # Reset Encryption of par
        s.encode_reset_xr = 1  # Reset encoding of xr
        s.encode_reset_reg_eps = 0  # Reset encoding of epsilon and regressor generator
        s.encode_reset_par = 0  # Reset encoding of par

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
        s.par_dot_depth = 0
        s.x_vec = []
        s.xr_vec = []
        s.e_vec = []
        s.par_vec = []
        s.u_vec = []
        s.r_vec = []
        s.par = np.array([[0], [0]])
        s.enc_par = mat_enc(s.par, s.kappa, s.p, s.mod, s.delta)

        s.t = 0  # time
        s.Encrypt = enc_method  # Encrypt? 0 = none, 1 = encode, 2 = encrypt

    def encrypt(s):
        for k in range(1, 200):
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
            s.enc_d = enc(1 / s.delta, s.kappa, s.p, s.mod, s.delta)  # for balancing encoding depths
            s.enc_Ar = mat_enc(s.Ar, s.kappa, s.p, s.mod, s.delta)  # d1
            s.enc_Br = mat_enc(s.Br, s.kappa, s.p, s.mod, s.delta)  # d1
            s.enc_xr = mat_enc(s.xr, s.kappa, s.p, s.mod, s.delta)  # d1
            s.enc_r = enc(s.r, s.kappa, s.p, s.mod, s.delta)  # d1
            s.enc_c = mat_enc(s.c, s.kappa, s.p, s.mod, s.delta)  # d1
            s.enc_beta_0 = enc(s.beta_0, s.kappa, s.p, s.mod, s.delta)  # d1
            s.enc_beta_1 = enc(s.beta_1, s.kappa, s.p, s.mod, s.delta)  # d1
            s.enc_gains = mat_enc(s.gains, s.kappa, s.p, s.mod, s.delta)  # d1

        # Calculating next encrypted reference state
        s.enc_xr = add(mat_mult(s.enc_Ar, s.enc_xr, s.mod), mat_mult(s.enc_Br, s.enc_r, s.mod), s.mod)  # d2
        s.r_vec.append(dec(s.enc_r, s.kappa, s.p, s.delta))  # d0

        # For resetting xr
        if s.reset_xr == 1: # for the purpose of removing encryption depth, not encoding depth
            s.xr = mat_dec(s.enc_xr, s.kappa, s.p, s.delta ** 2)  # d0
            s.enc_xr = mat_enc(s.xr, s.kappa, s.p, s.mod, s.delta ** 2)  # d2
            neg_enc_xr = mat_enc(-s.xr, s.kappa, s.p, s.mod, s.delta ** 2)  # d2 for subtracting values later

        # PLANT: Calculating next output based on the input
        s.x = np.dot(s.A, s.x) + np.dot(s.B, s.u)  # Only the output of this needs to be encrypted
        enc_x = mat_enc(s.x, s.kappa, s.p, s.mod, s.delta ** 2)  # d2
        neg_enc_x = mat_enc(-s.x, s.kappa, s.p, s.mod, s.delta ** 2)  # d2

        # Error and filtered error calculation
        enc_e = add(enc_x, neg_enc_xr, s.mod)  # d2
        enc_eps = mat_mult(s.enc_c, enc_e, s.mod)  # d3

        # Need these vectors later for plotting
        s.enc_x_vec.append(enc_x.flatten())  # d2
        s.enc_xr_vec.append(s.enc_xr.flatten())  # d2
        s.enc_e_vec.append(enc_e.flatten())  # d2

        # Regressor Generator split into parts for debugging
        s.r = -math.sin(s.t + math.pi / 2) + 1
        enc_r = enc(s.r, s.kappa, s.p, s.mod, s.delta)  # d1
        p1 = mult(enc_x[1][0], s.enc_d, s.mod)  # d3
        p2 = add(mult(add(enc_r, neg_enc_x[0][0], s.mod), s.enc_beta_0, s.mod), mult(neg_enc_x[1][0], s.enc_beta_1, s.mod), s.mod)
        enc_reg = np.array([[p1], [p2]])  # d3
        s.enc_reg_vec.append(enc_reg.flatten())

        # Resetting reg and eps because of overflow
        if s.reset_reg_eps == 1:
            reg = mat_dec(enc_reg, s.kappa, s.p, s.delta ** 3)
            s.reg_vec.append(reg.flatten())
            enc_reg = mat_enc(reg, s.kappa, s.p, s.mod, s.delta)  # d1
            eps = mat_dec(enc_eps, s.kappa, s.p, s.delta ** 3)
            enc_eps = enc(eps, s.kappa, s.p, s.mod, s.delta)  # d1

        # Parameter adaptation
        enc_par_mult = mat_mult(enc_eps, enc_reg, s.mod)  # d2
        enc_par_dot = mat_mult(s.enc_gains, enc_par_mult, s.mod)  # d3
        s.enc_par_dot_vec.append(enc_par_dot.flatten())  # d3

        s.enc_par = integrator(k, s.enc_par, 2, s.enc_d, s.enc_par_dot_vec, s.par_dot_depth, s.Ts_time, s.mod)
        s.enc_par_vec.append(s.enc_par.flatten())  # d3

        # Resetting par because of overflow
        if s.reset_par == 1:
            s.par = mat_dec(s.enc_par, s.kappa, s.p, s.delta ** 4)
            s.enc_par = mat_enc(s.par, s.kappa, s.p, s.mod, s.delta)

        # Calculating next input if par and reg were not exposed
        if s.reset_par == 0 & s.reset_reg_eps == 0:
            enc_u = mat_mult(enc_reg.transpose(), s.enc_par, s.mod, s.kappa, s.p, s.delta)
            s.enc_u_vec.append(enc_u.flatten())
            u = dec(enc_u, kappa, p, delta)
            s.u_vec.append(u)
        else:
            # Calculating next input if par and reg are exposed/reset
            s.u = [np.dot(reg.transpose(), s.par)]  # dec(enc_u, kappa, p, delta) * delta
            s.u_vec.append([float(x) for x in s.u])

        s.t = s.t + s.Ts

    def encode_ada(s, k):
        if k == 1:
            s.c = mat_encode(s.c, s.delta)  # d1
            s.beta_0 = encode(s.beta_0, s.delta)  # d1
            s.beta_1 = encode(s.beta_1, s.delta)  # d1
            s.gains = mat_encode(s.gains, s.delta)  # d1
            s.Ar = mat_encode(s.Ar, s.delta)  # d1
            s.xr = mat_encode(s.xr, s.delta)  # d1
            s.Br = mat_encode(s.Br, s.delta)  # d1
            s.r = encode(s.r, s.delta)  # d1

        s.xr = np.dot(s.Ar, s.xr) + np.dot(s.Br, s.r)  # d2
        # For resetting xr
        if s.encode_reset_xr == 1:
            s.xr = mat_decode(s.xr, s.delta**2)  # d0
            s.xr = mat_encode(s.xr, s.delta**2)  # d2

        s.x = np.dot(s.A, s.x) + np.dot(s.B, s.u)  # d0
        s.x = mat_encode(s.x, s.delta**2)  # d2
        e = s.x - s.xr  # d2

        # Testing for if tolerated error is reached
        e_tol = .001  # tolerable error
        if abs(decode(e[0][0],s.delta**2)) <= e_tol:
            print(f"State 1 error becomes < {e_tol} at iteration {k}, time {round(s.t, 2)}")

        eps = np.dot(s.c.flatten(), e)  # d3

        # Need these vectors later for plotting
        s.x_vec.append(s.x.flatten()*(s.delta**2))  # d0
        s.xr_vec.append(s.xr.flatten()*(s.delta**2))  # d0
        s.e_vec.append(e.flatten()*(s.delta**2))  # d0

        # Regressor Generator split into parts for debugging
        s.r = (-math.sin(s.t + math.pi / 2) + 1)  # d0
        s.r = encode(s.r, s.delta**2)  # d2
        s.r_vec.append(decode(s.r, s.delta))  # d1
        p1 = encode(s.x[1][0], s.delta)  # d3
        p2 = ((s.r - s.x[0][0]) * s.beta_0) - (s.x[1][0] * s.beta_1)  # d3
        reg = np.array([[p1], [p2]])  # d3
        s.reg_vec.append(mat_decode(reg.flatten(), s.delta**3))  # d0

        if s.encode_reset_reg_eps == 1:
            reg = mat_decode(reg, s.delta ** 3)  # d0
            reg = mat_encode(reg, s.delta)  # d1
            eps = mat_decode(eps, s.delta ** 3)  # d0
            eps = encode(eps, s.delta)  # d1
            s.par_dot_depth = 3
        else:
            s.par_dot_depth = 7

        par_mult = eps * reg  # d6 or d2
        par_mult = par_mult.reshape((2, 1))  # Reshape to a 2x1 vector

        # Parameter adaptation
        par_dot = np.dot(s.gains, par_mult)  # d7 or d3
        s.par_dot_vec.append(par_dot.flatten())  # d7 or d3 keeping this at d7 or d3 for par calculation
        s.par_dot_vec_test.append(mat_decode(par_dot.flatten(), s.delta**s.par_dot_depth))  # d0 this one is used for test data

        s.par = integrator(k, s.par, 1, s.delta, s.par_dot_vec, s.par_dot_depth, s.Ts_time, s.mod)
        s.par_vec.append(s.par.flatten()*(s.delta**s.par_dot_depth))  # d0

        if s.encode_reset_par == 1:
            s.par = mat_decode(s.par, s.delta ** s.par_dot_depth)  # d0
            s.par = mat_encode(s.par, s.delta)  # d1

        s.u = float(np.dot(reg.transpose(), s.par))  # d10 or d2

        if (s.encode_reset_par == 1) & (s.encode_reset_reg_eps == 1):
            u_depth = 2
        elif (s.encode_reset_par == 0) & (s.encode_reset_reg_eps == 0):
            u_depth = 10
        elif (s.encode_reset_par == 0) & (s.encode_reset_reg_eps == 1):
            u_depth = 8
        elif (s.encode_reset_par == 1) & (s.encode_reset_reg_eps == 0):
            u_depth = 4

        # decoding
        s.u = decode(s.u, s.delta ** u_depth)  # d0
        s.u_vec.append(s.u)

        if s.encode_reset_par == 1:
            s.par = decode(s.par, s.delta)  # d0
        elif s.encode_reset_par == 0:
            s.par = decode(s.par, s.delta**7)  # d0


        s.x = mat_decode(s.x, s.delta**2)  # d0
        s.xr = mat_decode(s.xr, s.delta**2)  # d0 Have to decode all the way before encoding again or the numbers will be too big
        s.xr = mat_encode(s.xr, s.delta)  # d1
        s.r = decode(s.r, s.delta)  # d1

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

        s.par = integrator(k, s.par, 0, s.delta, s.par_dot_vec, s.par_dot_depth, s.Ts_time, s.mod)
        s.par_vec.append(s.par.flatten())

        s.u = float(np.dot(reg.transpose(), s.par))
        s.u_vec.append(s.u)

        s.t = s.t + s.Ts_time
