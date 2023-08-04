import numpy as np
import matplotlib.pyplot as plt
from dyersmat import *
import math
from compare import *
import pdb

class Encrypter():
    def __init__(self, enc_method):
        # Encryption
        self.bit_length = 256
        self.rho = 1
        self.rho_ = 32
        self.delta = 0.01
        self.kappa, self.p = keygen(self.bit_length, self.rho, self.rho_)
        self.mod = pgen(self.bit_length, self.rho_, self.p)
        self.reset_xr = 1  # Reset Encryption of xr
        self.reset_reg_eps = 1  # Reset Encryption of epsilon and regressor generator
        self.reset_par = 1  # Reset Encryption of par

        # system parameters
        self.m = 1
        self.b = 2
        self.k = 0

        self.zeta = .8  # Damping Ratio
        self.wn = .5  # Natural Frequency
        self.beta_1 = 1  # 2 * zeta * wn
        self.beta_0 = 1  # wn * wn

        # Creating plant state space
        self.A = np.array([[0, 1], [-self.k / self.m, -self.b / self.m]])
        self.B = np.array([[0], [1 / self.m]])

        # Creating reference state space
        self.Ar = np.array([[0, 1], [-self.beta_0, -self.beta_1]])
        self.Br = np.array([[0], [self.beta_0]])

        # Initial Conditions
        self.x = np.array([[0], [0]])
        self.xr = np.array([[0], [0]])

        # Other variables
        # self.c = np.array([[2, 3.125]])
        self.c = np.array([[.5, 1]])
        self.gam1 = 15
        self.gam2 = 1
        self.gains = np.array([[-self.gam1, 0], [0, -self.gam2]])
        self.u = 0
        self.r = 0
        self.Ts = .1
        self.enc_par_dot_vec = []
        self.enc_x_vec = []
        self.enc_xr_vec = []
        self.enc_e_vec = []
        self.enc_reg_vec = []
        self.enc_u_vec = []
        self.enc_par_vec = []
        self.reg_vec = []
        self.par_dot_vec = []
        self.x_vec = []
        self.xr_vec = []
        self.e_vec = []
        self.par_vec = []
        self.u_vec = []
        self.r_vec = []
        self.enc_par = np.array([[0], [0]])
        self.par = np.array([[0], [0]])

        self.t = 0  # time
        self.Encrypt = enc_method  # Encrypt? 1 = yes, 0 = no, 2 = encode only

    def encrypt(self):
        for k in range(1, 200):
            if self.Encrypt == 1:
                self.enc_ada(k)
            elif self.Encrypt == 2:
                self.encode_ada(k)
            elif self.Encrypt == 0:
                self.ada(k)
        self.t = np.arange(self.Ts, (k+1) * self.Ts, self.Ts)

    def enc_ada(self, k):
        # encryption of matrices and variables
        if k == 1:
            enc_Ts = enc(self.Ts, kappa, p, mod, delta)  # d1
            enc_d = enc(1 / delta, kappa, p, mod, delta)  # for balancing encoding depths
            enc_Ar = mat_enc(self.Ar, kappa, p, mod, delta)  # d1
            enc_Br = mat_enc(self.Br, kappa, p, mod, delta)  # d1
            enc_xr = mat_enc(self.xr, kappa, p, mod, delta)  # d1
            enc_r = enc(self.r, kappa, p, mod, delta)  # d1
            enc_c = mat_enc(self.c, kappa, p, mod, delta)  # d1
            enc_beta_0 = enc(self.beta_0, kappa, p, mod, delta)  # d1
            enc_beta_1 = enc(self.beta_1, kappa, p, mod, delta)  # d1
            enc_gains = mat_enc(self.gains, kappa, p, mod, delta)  # d1

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
            self.u = [np.dot(reg.transpose(), par)]  # dec(enc_u, kappa, p, delta) * delta
            self.u_vec.append([float(x) for x in u])

        self.t = self.t + self.Ts

    def encode_ada(self, k):
        if k == 1:
            self.Ts = encode(self.Ts, self.delta)  # d1
            self.c = mat_encode(self.c, self.delta)  # d1
            self.beta_0 = encode(self.beta_0, self.delta)  # d1
            self.beta_1 = encode(self.beta_1, self.delta)  # d1
            self.gains = mat_encode(self.gains, self.delta)  # d1
            self.Ar = mat_encode(self.Ar, self.delta)  # d1
            self.xr = mat_encode(self.xr, self.delta)  # d1
            self.Br = mat_encode(self.Br, self.delta)  # d1
            self.r = encode(self.r, self.delta)  # d1

        self.xr = np.dot(self.Ar, self.xr) + np.dot(self.Br, self.r)  # d2
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        self.x = mat_encode(self.x, self.delta**2)  # d2
        e = self.x - self.xr
        eps = np.dot(self.c.flatten(), e)  # d3

        # Need these vectors later for plotting
        self.x_vec.append(self.x.flatten())  # d2
        self.xr_vec.append(self.xr.flatten())  # d2
        self.e_vec.append(e.flatten())  # d2

        # Regressor Generator split into parts for debugging
        self.r = (-math.sin(self.t + math.pi / 2) + 1)
        self.r = encode(self.r, self.delta)  # d1
        self.r_vec.append(self.r)  # d1
        p1 = self.x[1][0] / self.delta
        p2 = ((self.r / self.delta) - self.x[0][0]) * self.beta_0 - self.x[1][0] * self.beta_1
        reg = np.array([[p1], [p2]])  # d3
        self.reg_vec.append(reg.flatten())  # d3

        par_mult = eps * reg  # d6
        par_mult = par_mult.reshape((2, 1))  # Reshape to a 2x1 vector

        # Parameter adaptation
        par_dot = np.dot(self.gains, par_mult)  # d7
        self.par_dot_vec.append(par_dot.flatten())  # d7

        # Integrator
        if k != 1:
            self.par = mat_encode(self.par, self.delta**7) + np.dot(self.par_dot_vec[k - 1].reshape((2, 1)),
                                                   self.Ts) # d7 # Ts might have to be different here than rest of code. IDK why.
        self.par_vec.append(self.par.flatten())  # d7

        self.u = float(np.dot(reg.transpose(), self.par))  # d10
        self.u = decode(self.u, self.delta**10)  # d0 bringing back encoding depth to 0
        self.u_vec.append(self.u)

        self.t = self.t + self.Ts

    def ada(self, k):
        self.xr = np.dot(self.Ar, self.xr) + np.dot(self.Br, self.r)
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)  # d2 - Only the output of this needs to be encrypted
        e = self.x - self.xr
        eps = np.dot(self.c.flatten(), e)

        # Need these vectors later for plotting
        self.x_vec.append(self.x.flatten())
        self.xr_vec.append(self.xr.flatten())
        self.e_vec.append(e.flatten())

        # Regressor Generator split into parts for debugging
        self.r = (-math.sin(self.t + math.pi / 2) + 1)
        self.r_vec.append(self.r)
        reg = np.array([[self.x[1][0]], [(self.r - self.x[0][0]) * self.beta_0 - self.x[1][0] * self.beta_1]])
        self.reg_vec.append(reg.flatten())

        par_mult = eps * reg
        par_mult = par_mult.reshape((2, 1))  # Reshape to a 2x1 vector

        # Parameter adaptation
        par_dot = np.dot(self.gains, par_mult)
        self.par_dot_vec.append(par_dot.flatten())

        # Integrator
        if k != 1:
            self.par = self.par + np.dot(self.par_dot_vec[k - 1].reshape((2, 1)),
                               self.Ts)  # Ts might have to be different here than rest of code. IDK why.
        self.par_vec.append(self.par.flatten())

        self.u = float(np.dot(reg.transpose(), self.par))
        self.u_vec.append(self.u)

        self.t = self.t + self.Ts
