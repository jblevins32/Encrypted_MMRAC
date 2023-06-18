import numpy as np
import matplotlib.pyplot as plt
from dyers import *
from dyersmat import *
import math

# reg_vec isn't giving correct output because beta 0 is losing data to encoding. Fixed with delta = .01
# enc_c is erroring

# Encryption
bit_length = 256
rho = 1
rho_ = 32
delta = 0.01
kappa, p = keygen(bit_length, rho, rho_)
modulus = pgen(bit_length, rho_, p)

def main():
    # Reference inputs
    r = 1
    r_dot = 0
    r_ddot = 0

    m = 1
    b = 0.2
    k = 0

    zeta = 0.8
    wn = 0.5
    beta_1 = 2 * zeta * wn
    beta_0 = wn * wn
    gamma = 1

    # Creating plant state space
    A = np.array([[0, 1], [-k / m, -b / m]])
    B = np.array([[0], [1 / m]])
    C = np.eye(2)
    D = np.zeros((2, 1))

    # Creating reference state space
    Ar = np.array([[0, 1], [-beta_0, -beta_1]])
    Br = np.array([[0], [beta_0]])
    Cr = np.eye(2)
    Dr = np.zeros((2, 1))

    # Initial Conditions
    x = np.array([[0], [1]])
    xr = np.array([[0], [0]])

    # Other variables
    c = np.array([2, 3.125])
    gam1 = 100
    gam2 = 10
    gains = np.array([[-gam1, 0], [0, -gam2]])
    u = 0
    r = 0
    Ts = 0.2
    par_dot_vec = []
    enc_x_vec = []
    enc_xr_vec = []
    enc_e_vec = []
    e_vec = []
    enc_reg_vec = []
    u_vec = []
    par_vec = []
    par = np.array([[0], [0]])

    # encryption of matricies and variables
    enc_d = enc(1/delta, kappa, p, modulus, delta) # for balancing encoding depths
    enc_Ar = mat_enc(Ar, kappa, p, modulus, delta)
    enc_Br = mat_enc(Br, kappa, p, modulus, delta)
    enc_A = mat_enc(A, kappa, p, modulus, delta)
    enc_B = mat_enc(B, kappa, p, modulus, delta)
    enc_xr = mat_enc(xr, kappa, p, modulus, delta)
    enc_x = mat_enc(x, kappa, p, modulus, delta)
    enc_r = enc(r, kappa, p, modulus, delta)
    enc_u = enc(u, kappa, p, modulus, delta)
    #enc_c = mat_enc(c, kappa, p, modulus, delta)
    enc_beta_0 = enc(beta_0, kappa, p, modulus, delta)
    enc_beta_1 = enc(beta_1, kappa, p, modulus, delta)
    enc_r_dot = enc(r_dot, kappa, p, modulus, delta)
    enc_r_ddot = enc(r_ddot, kappa, p, modulus, delta)

    k = 0  # step
    t = 0  # time
    while k <= 50:
        enc_xr = np.dot(enc_Ar, enc_xr) + np.dot(enc_Br, enc_r) # d2 - encrypted plant calculation
        enc_x = np.dot(enc_A, enc_x) + np.dot(enc_B, enc_u) # d2 - Only the output of this needs to be encrypted
        enc_x_vec.append(enc_x.flatten())
        enc_xr_vec.append(enc_xr.flatten())
        enc_e = enc_x - enc_xr # d2
        enc_e_vec.append(enc_e.flatten())
        #eps = np.dot(enc_c, enc_e)

        x = mat_dec(enc_x, kappa, p, delta)/delta
        e_vec = mat_dec(enc_e_vec, kappa, p, delta)

        # Regressor Generator
        r = -math.sin(t + math.pi / 2) + 1
        enc_r = enc(r, kappa, p, modulus, delta)
        enc_p1 = enc_r * enc_beta_0 * enc_d
        enc_p2 = enc_r_dot * enc_beta_1 * enc_d
        enc_p3 = enc_r_ddot * enc_d * enc_d
        enc_p4 = mult(enc_beta_0, enc_x[0][0], modulus)
        enc_p5 = mult(enc_beta_1, enc_x[1][0], modulus)
        enc_reg = np.array([[enc_x[1][0]], [enc_p1 + enc_p2 + enc_p3 - enc_p4 - enc_p5]]) #[enc_r * enc_beta_0 * enc_d + enc_r_dot * enc_beta_1 * enc_d + enc_r_ddot * enc_d * enc_d - enc_beta_0 * enc_x[0][0] - enc_beta_1 * enc_x[1][0]]])
        enc_reg_vec.append(enc_reg.flatten())

        # Debug stuff
        dec_beta_0 = dec(enc_beta_0, kappa, p, delta)
        dec_beta_1 = dec(enc_beta_1, kappa, p, delta)
        dec_xx = dec(enc_x[0][0], kappa, p, delta)
        p4 = beta_0 * x[0][0]
        p5 = beta_1 * x[1][0]
        dec_p4 = dec(enc_p4, kappa, p, delta)
        dec_p5 = dec(enc_p5, kappa, p, delta)

        reg_vec = mat_dec(enc_reg_vec, kappa, p, delta)

        # Parameter adaptation
        par_dot = np.dot(gains, eps * reg)
        par_dot_vec.append(par_dot.flatten())
        if k != 0:
            par = par + np.dot(par_dot_vec[k].reshape((2, 1)), Ts)
        par_vec.append(par.flatten())

        u = np.dot(reg.transpose(), par)
        u_vec.append(u.flatten())
        k = k + 1
        t = t + Ts

    t = np.arange(Ts, (k + 1) * Ts, Ts)

    # Plot the Results
    plt.figure(1)
    plt.subplot(311)
    plt.plot(t, np.array(e_vec)[:, 0])
    plt.xlabel('Time (sec)')
    plt.ylabel('Tracking Error')
    plt.title('Gains: gam1=100, gam2=10')
    plt.subplot(312)
    plt.plot(t, np.array(par_vec))
    plt.xlabel('Time (sec)')
    plt.ylabel('Parameters')
    plt.subplot(313)
    plt.plot(t, np.array(x_vec)[:, 0])
    plt.xlabel('Time (sec)')
    plt.ylabel('Output')

    plt.figure(2)
    plt.plot(t, np.array(x_vec)[:, 0], t, np.array(xr_vec)[:, 0])
    plt.title('Actual vs Reference Model')
    plt.xlabel('Time')
    plt.show()

if __name__ == '__main__':
    main()