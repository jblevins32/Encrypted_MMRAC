import math

from integrator import *

class Encrypter():
    def __init__(s, enc_method):
        # Encryption
        s.bit_length = 500
        s.rho = 1
        s.rho_ = 64
        s.delta = 0.00001
        s.kappa, s.p = keygen(s.bit_length, s.rho, s.rho_)
        s.mod = pgen(s.bit_length, s.rho_, s.p)
        s.reset_xr = 1  # Reset Encryption of xr
        s.reset_reg_eps = 1  # Reset Encryption of epsilon and regressor generator
        s.reset_par = 1  # Reset Encryption of par
        s.reset_par_mult = 1
        s.reset_par_dot = 1

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
        s.e_tol = .001  # tolerable error
        s.c = np.array([[.5, 1]])
        s.gam1 = 15
        s.gam2 = 1
        s.gains = np.array([[-s.gam1, 0], [0, -s.gam2]])
        s.u = 0
        s.r = 0
        s.Ts = .1
        s.enc_Ts = enc(s.Ts, s.kappa, s.p, s.mod, s.delta)
        s.encode_Ts = encode(s.Ts, s.delta)
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
        s.enc_par = mat_enc(s.par, s.kappa, s.p, s.mod, s.delta ** 4)

        s.t = 0  # time
        s.Encrypt = enc_method  # Encrypt? 0 = none, 1 = encode, 2 = encrypt

    def encrypt(s):
        for k in range(1, 100):
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
            s.enc_c = mat_enc(s.c, s.kappa, s.p, s.mod, s.delta)  # d1
            s.enc_beta_0 = enc(s.beta_0, s.kappa, s.p, s.mod, s.delta)  # d1
            s.enc_beta_1 = enc(s.beta_1, s.kappa, s.p, s.mod, s.delta)  # d1
            s.enc_gains = mat_enc(s.gains, s.kappa, s.p, s.mod, s.delta)  # d1
            s.enc_Ar = mat_enc(s.Ar, s.kappa, s.p, s.mod, s.delta)  # d1
            s.enc_xr = mat_enc(s.xr, s.kappa, s.p, s.mod, s.delta)  # d1
            s.enc_Br = mat_enc(s.Br, s.kappa, s.p, s.mod, s.delta)  # d1
            s.enc_r = enc(s.r, s.kappa, s.p, s.mod, s.delta)  # d1
            s.enc_d = enc(1, s.kappa, s.p, s.mod, s.delta)  # d1 for balancing encoding depths

        # Calculating next encrypted reference state
        s.enc_xr = add(mat_mult(s.enc_Ar, s.enc_xr, s.mod), mat_mult(s.enc_Br, s.enc_r, s.mod), s.mod)  # d2

        # For resetting xr
        if s.reset_xr == 1:  # for the purpose of removing encryption depth, not encoding depth
            s.xr = mat_dec(s.enc_xr, s.kappa, s.p, s.delta ** 2)  # d0
            neg_enc_xr = mat_enc(-s.xr, s.kappa, s.p, s.mod, s.delta ** 2)  # d2 for subtracting values later
            s.enc_xr = mat_enc(s.xr, s.kappa, s.p, s.mod, s.delta ** 2)  # d2

        # PLANT: Calculating next output based on the input
        s.x = np.dot(s.A, s.x) + np.dot(s.B, s.u)  # d0 Only the output of this needs to be encrypted
        enc_x = mat_enc(s.x, s.kappa, s.p, s.mod, s.delta ** 2)  # d2
        neg_enc_x = mat_enc(-s.x, s.kappa, s.p, s.mod, s.delta)  # d1 for subtracting values later in reg

        # Error and filtered error calculation
        enc_e = add(enc_x, neg_enc_xr, s.mod)  # d2
        enc_eps = mat_mult(s.enc_c, enc_e, s.mod)  # d3

        # Testing for if tolerated error is reached
        if abs(dec(enc_e[0][0], s.kappa, s.p, s.delta ** 2)) <= s.e_tol:
            print(f"State 1 error becomes < {s.e_tol} at iteration {k}, time {round(s.t, 2)}")

        # Need these vectors later for plotting
        s.x_vec.append(mat_dec(enc_x.flatten(), s.kappa, s.p, s.delta ** 2))  # d0
        s.xr_vec.append(mat_dec(s.enc_xr.flatten(), s.kappa, s.p, s.delta ** 2))  # d0
        s.e_vec.append(mat_dec(enc_e.flatten(), s.kappa, s.p, s.delta ** 2))  # d0

        # Regressor Generator split into parts for debugging
        s.r = -math.sin(s.t + math.pi / 2) + 1  # d0
        s.enc_r = enc(s.r, s.kappa, s.p, s.mod, s.delta)  # d1
        s.r_vec.append(dec(s.enc_r, s.kappa, s.p, s.delta))  # d0
        p1 = enc_x[1][0]  # d2
        p2 = add(mult(add(s.enc_r, neg_enc_x[0][0], s.mod), s.enc_beta_0, s.mod), mult(neg_enc_x[1][0], s.enc_beta_1, s.mod), s.mod)  # d2
        enc_reg = np.array([[p1], [p2]])  # d2
        s.reg_vec.append(mat_dec(enc_reg.flatten(), s.kappa, s.p, s.delta ** 2))  # d0

        # Resetting reg and eps because of overflow
        if s.reset_reg_eps == 1:
            reg = mat_dec(enc_reg, s.kappa, s.p, s.delta ** 2)  # d0
            enc_reg = mat_enc(reg, s.kappa, s.p, s.mod, s.delta)  # d1
            eps = dec(enc_eps, s.kappa, s.p, s.delta ** 3)  # d0
            enc_eps = enc(eps, s.kappa, s.p, s.mod, s.delta)  # d1
            s.par_dot_depth = 3
        else:
            s.par_dot_depth = 6

        # Parameter adaptation
        enc_par_mult = mult(enc_eps, enc_reg, s.mod)  # d5 or d2

        if s.reset_par_mult == 1:
            par_mult = mat_dec(enc_par_mult, s.kappa, s.p, s.delta**(s.par_dot_depth - 1))  # d0
            enc_par_mult = mat_enc(par_mult, s.kappa, s.p, s.mod, s.delta)  # d1
            s.par_dot_depth = 2

        enc_par_dot = mat_mult(s.enc_gains, enc_par_mult, s.mod)  # d6 or d3

        if s.reset_par_dot == 1:
            par_dot = mat_dec(enc_par_dot, s.kappa, s.p, s.delta ** s.par_dot_depth)  # d0
            enc_par_dot = mat_enc(par_dot, s.kappa, s.p, s.mod, s.delta)  # d1
            s.par_dot_depth = 1

        s.enc_par_dot_vec.append(enc_par_dot.flatten())  # d6 or d3
        s.par_dot_vec_test.append(mat_dec(enc_par_dot.flatten(), s.kappa, s.p, s.delta ** s.par_dot_depth))  # d0

        s.enc_par = integrator(k, s.enc_par, 2, s.enc_par_dot_vec, s.enc_Ts, s.mod) # increases encode depth by 1
        s.par_vec.append(mat_dec(s.enc_par.flatten(), s.kappa, s.p, s.delta ** (s.par_dot_depth + 1)))  # d0

        # Resetting par because of overflow
        if s.reset_par == 1:
            s.par = mat_dec(s.enc_par, s.kappa, s.p, s.delta ** (s.par_dot_depth + 1))  # d0
            s.enc_par = mat_enc(s.par, s.kappa, s.p, s.mod, s.delta)  # d1

        enc_u = mat_mult(enc_reg.transpose(), s.enc_par, s.mod)

        if (s.reset_par == 1) & (s.reset_reg_eps == 1):
            u_depth = 2
        elif (s.reset_par == 0) & (s.reset_reg_eps == 0):
            u_depth = 9
        elif (s.reset_par == 0) & (s.reset_reg_eps == 1):
            u_depth = 8 # not correct right now
        elif (s.reset_par == 1) & (s.reset_reg_eps == 0):
            u_depth = 4 # not correct right now

        # Decrypting
        s.u = dec(enc_u, s.kappa, s.p, s.delta ** u_depth)
        s.u_vec.append(s.u)

        # setting up par for next iteration
        if s.reset_par == 1:
            s.par = mat_dec(s.enc_par, s.kappa, s.p, s.delta)  # d0
        elif s.reset_par == 0:
            s.par = mat_dec(s.enc_par, s.kappa, s.p, s.delta ** (s.par_dot_depth + 1))  # d0
        s.enc_par = mat_enc(s.par, s.kappa, s.p, s.mod, s.delta ** (s.par_dot_depth + 1))

        s.x = mat_dec(enc_x, s.kappa, s.p, s.delta**2)  # d0
        s.xr = mat_dec(s.enc_xr, s.kappa, s.p, s.delta**2)  # d0
        s.enc_xr = mat_enc(s.xr, s.kappa, s.p, s.mod, s.delta)  # d1

        s.t = s.t + s.Ts_time

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
        if s.reset_xr == 1:
            s.xr = mat_decode(s.xr, s.delta**2)  # d0
            s.xr = mat_encode(s.xr, s.delta**2)  # d2

        s.x = np.dot(s.A, s.x) + np.dot(s.B, s.u)  # d0
        x_d1 = mat_encode(s.x, s.delta)  # d2
        s.x = mat_encode(s.x, s.delta**2)  # d2
        e = s.x - s.xr  # d2
        eps = np.dot(s.c.flatten(), e)  # d3

        # Testing for if tolerated error is reached
        if abs(decode(e[0][0],s.delta**2)) <= s.e_tol:
            print(f"State 1 error becomes < {s.e_tol} at iteration {k}, time {round(s.t, 2)}")

        # Need these vectors later for plotting
        s.x_vec.append(s.x.flatten()*(s.delta**2))  # d0
        s.xr_vec.append(s.xr.flatten()*(s.delta**2))  # d0
        s.e_vec.append(e.flatten()*(s.delta**2))  # d0

        # Regressor Generator split into parts for debugging
        s.r = (-math.sin(s.t + math.pi / 2) + 1)  # d0
        s.r = encode(s.r, s.delta)  # d1
        s.r_vec.append(decode(s.r, s.delta))  # d1
        p1 = s.x[1][0]  # d2
        p2 = ((s.r - x_d1[0][0]) * s.beta_0) - (x_d1[1][0] * s.beta_1)  # d2
        reg = np.array([[p1], [p2]])  # d2
        s.reg_vec.append(mat_decode(reg.flatten(), s.delta**2))  # d0

        # Resetting reg and eps because of overflow
        if s.reset_reg_eps == 1:
            reg = mat_decode(reg, s.delta ** 2)  # d0
            reg = mat_encode(reg, s.delta)  # d1
            eps = decode(eps, s.delta ** 3)  # d0
            eps = encode(eps, s.delta)  # d1
            s.par_dot_depth = 3
        else:
            s.par_dot_depth = 6

        # Parameter adaptation
        par_mult = eps * reg  # d5 or d2

        if s.reset_par_mult == 1:
            par_mult = mat_decode(par_mult, s.delta**(s.par_dot_depth - 1))  # d0
            par_mult = mat_encode(par_mult, s.delta)  # d1
            s.par_dot_depth = 2

        par_dot = np.dot(s.gains, par_mult)  # d6 or d3 or d2
        s.par_dot_vec.append(par_dot.flatten())  # d6 or d3 or d2 keeping this at d6 or d3 or d2 for par calculation
        s.par_dot_vec_test.append(mat_decode(par_dot.flatten(), s.delta**s.par_dot_depth))  # d0 this one is used for test data

        s.par = integrator(k, s.par, 1, s.par_dot_vec, s.encode_Ts, s.mod)
        s.par_vec.append(s.par.flatten()*(s.delta**(s.par_dot_depth + 1)))  # d0

        if s.reset_par == 1:
            s.par = mat_decode(s.par, s.delta ** (s.par_dot_depth + 1))  # d0
            s.par = mat_encode(s.par, s.delta)  # d1

        s.u = float(np.dot(reg.transpose(), s.par))  # d9 or d2

        if (s.reset_par == 1) & (s.reset_reg_eps == 1):
            u_depth = 2
        elif (s.reset_par == 0) & (s.reset_reg_eps == 0):
            u_depth = 9
        elif (s.reset_par == 0) & (s.reset_reg_eps == 1):
            u_depth = 8
        elif (s.reset_par == 1) & (s.reset_reg_eps == 0):
            u_depth = 4

        # decoding
        s.u = decode(s.u, s.delta ** u_depth)  # d0
        s.u_vec.append(s.u)

        # setting up par for next iteration
        if s.reset_par == 1:
            s.par = mat_decode(s.par, s.delta)  # d0
        elif s.reset_par == 0:
            s.par = mat_decode(s.par, s.delta**(s.par_dot_depth+1))  # d0
        s.par = mat_encode(s.par, s.delta**(s.par_dot_depth+1))

        s.x = mat_decode(s.x, s.delta**2)  # d0
        s.xr = mat_decode(s.xr, s.delta**2)  # d0 Have to decode all the way before encoding again or the numbers will be too big
        s.xr = mat_encode(s.xr, s.delta)  # d1

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

        s.par = integrator(k, s.par, 0, s.par_dot_vec, s.Ts_time, s.mod)
        s.par_vec.append(s.par.flatten())

        s.u = float(np.dot(reg.transpose(), s.par))
        s.u_vec.append(s.u)

        s.t = s.t + s.Ts_time
