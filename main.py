import numpy as np
import matplotlib.pyplot as plt
from dyersmat import *
import math
from compare import *
import pdb

#pdb.set_trace()

# Possible solutions:
# Use variable delta: If I see overflow coming, change delta to accommodate
# Somehow apply quantizer stuff

# Need to turn to discrete space
# Need to understand stability...stability of gradient descent or of MRAC?
# Do the problem with larger numbers to see if I get the same instability... I assume more overflow
# Need to watch adaptive lectures to understand it more

# The issue with decrypting and doing math outside of cyphertext is that I loose data. enc(.0111)*enc(.05) = .000555 but enc(.0111*.05) = 0
# The other issue is that I am trying to make error go to 0; therefore, as error gets smaller, my delta value cannot track those small numbers. Why not use huge delta? Apparently it's hard to encrypt large numbers
# par mult of iteration 3 is where small numbers become an issue

# Encryption
bit_length = 256
rho = 1
rho_ = 32
delta = 0.01
kappa, p = keygen(bit_length, rho, rho_)
modulus = pgen(bit_length, rho_, p)
reset_xr = 1  # Reset Encryption of xr
reset_reg_eps = 1  # Reset Encryption of epsilon and regressor generator
reset_par = 1  # Reset Encryption of par
# cannot get correct results if both reg_eps and par are not reset. Need to make u calculation conditions for these cases

def main():
    # Reference inputs
    r = 0
    r_dot = 0
    r_ddot = 0

    # system parameters
    m = .2
    b = .2
    k = 0

    zeta = 2  # Damping Ratio
    wn = .5  # Natural Frequency
    beta_1 = 2 * zeta * wn
    beta_0 = wn * wn

    # Creating plant state space
    A = np.array([[0, 1], [-k / m, -b / m]])
    B = np.array([[0], [1 / m]])

    # Creating reference state space
    Ar = np.array([[0, 1], [-beta_0, -beta_1]])
    Br = np.array([[0], [beta_0]])

    # Initial Conditions
    x = np.array([[0], [1]])
    xr = np.array([[0], [1]])

    # Other variables
    c = np.array([[2, 3.125]])
    gam1 = 100
    gam2 = 10
    gains = np.array([[-gam1, 0], [0, -gam2]])
    u = 0
    r = 0
    Ts = 0.1
    enc_par_dot_vec = []
    enc_x_vec = []
    enc_xr_vec = []
    enc_e_vec = []
    enc_reg_vec = []
    enc_u_vec = []
    enc_par_vec = []
    reg_vec = []
    par_dot_vec = []
    x_vec = []
    xr_vec = []
    e_vec = []
    par_vec = []
    u_vec = []
    r_vec = []
    enc_par = np.array([[0], [0]])
    par = np.array([[0], [0]])

    # encryption of matricies and variables
    enc_Ts = enc(.01, kappa, p, modulus, delta)
    enc_d = enc(1/delta, kappa, p, modulus, delta) # for balancing encoding depths
    enc_Ar = mat_enc(Ar, kappa, p, modulus, delta)
    enc_Br = mat_enc(Br, kappa, p, modulus, delta)
    enc_xr = mat_enc(xr, kappa, p, modulus, delta)
    enc_r = enc(r, kappa, p, modulus, delta)
    enc_c = mat_enc(c, kappa, p, modulus, delta)
    enc_beta_0 = enc(beta_0, kappa, p, modulus, delta)
    enc_beta_1 = enc(beta_1, kappa, p, modulus, delta)
    enc_r_dot = enc(r_dot, kappa, p, modulus, delta)
    enc_r_ddot = enc(r_ddot, kappa, p, modulus, delta)
    enc_gains = mat_enc(gains, kappa, p, modulus, delta)

    k = 1  # step
    t = 0  # time
    Encrypt = 0  # Encrypt? 1 = yes, 0 = no

    while k <= 999:

        if Encrypt == 1:
            # Calculating next encrypted reference state
            enc_xr = add(mat_mult(enc_Ar, enc_xr, modulus, kappa, p, delta), mat_mult(enc_Br, enc_r, modulus, kappa, p, delta), modulus)  # ed2
            r_vec.append(dec(enc_r, kappa, p, delta))

            # For resetting xr
            if reset_xr == 1:
                xr = mat_dec(enc_xr, kappa, p, delta)
                enc_xr = mat_enc(xr, kappa, p, modulus, delta)
                neg_enc_xr = mat_enc(-xr, kappa, p, modulus, delta) # for subtracting values later

            # PLANT: Calculating next output based on the input
            x = np.dot(A, x) + np.dot(B, u)  # d2 - Only the output of this needs to be encrypted
            enc_x = mat_enc(x, kappa, p, modulus, delta * delta)

            # Error and filtered error calculation
            enc_e = add(enc_x, neg_enc_xr, modulus)  # d2
            enc_eps = mat_mult(enc_c, enc_e, modulus, kappa, p, delta)

            # Need these vectors later for plotting
            enc_x_vec.append(enc_x.flatten())
            enc_xr_vec.append(enc_xr.flatten())
            enc_e_vec.append(enc_e.flatten())

            # Regressor Generator split into parts for debugging
            r = (-math.sin(t + math.pi / 2) + 1)
            enc_r = enc(r, kappa, p, modulus, delta)
            enc_p1 = mult(enc_r, mult(enc_beta_0, enc_d, modulus), modulus)
            enc_p2 = mult(enc_r_dot, mult(enc_beta_1, enc_d, modulus), modulus)
            enc_p3 = mult(enc_r_ddot, mult(enc_d, enc_d, modulus), modulus)
            # enc_p1 = mat_mult(enc_r, mat_mult(enc_beta_0, enc_d, modulus, kappa, p, delta), modulus, kappa, p, delta)
            # enc_p2 = mat_mult(enc_r_dot, mat_mult(enc_beta_1, enc_d, modulus, kappa, p, delta), modulus, kappa, p, delta)
            # enc_p3 = mat_mult(enc_r_ddot, mult(enc_d, enc_d, modulus, kappa, p, delta), modulus, kappa, p, delta)
            enc_p4 = mat_mult(enc_beta_0, enc_x[0][0], modulus, kappa, p, delta)
            enc_p5 = mat_mult(enc_beta_1, enc_x[1][0], modulus, kappa, p, delta)
            enc_reg = np.array([[enc_x[1][0]], [enc_p1 + enc_p2 + enc_p3 - enc_p4 - enc_p5]]) #[enc_r * enc_beta_0 * enc_d + enc_r_dot * enc_beta_1 * enc_d + enc_r_ddot * enc_d * enc_d - enc_beta_0 * enc_x[0][0] - enc_beta_1 * enc_x[1][0]]])
            enc_reg_vec.append(enc_reg.flatten())

            # Resetting reg and eps because of overflow
            if reset_reg_eps == 1:
                reg = mat_dec(enc_reg, kappa, p, delta) # encoding depth of 2
                reg = np.array([reg[0][0] / delta, reg[1][0]]) * delta * delta  # Correcting encoding depth
                reg_vec.append(reg.flatten())
                enc_reg = mat_enc(reg, kappa, p, modulus, delta)
                eps = mat_dec(enc_eps, kappa, p, delta) * delta * delta # encoding depth of 2
                enc_eps = enc(eps, kappa, p, modulus, delta)

            # Parameter adaptation
            enc_par_mult = mat_mult(enc_eps, enc_reg, modulus, kappa, p, delta) #mat_enc(par_mult, kappa, p, modulus, delta)
            enc_par_dot = mat_mult(enc_gains, enc_par_mult, modulus, kappa, p, delta)
            enc_par_dot_vec.append(enc_par_dot.flatten())

            # Integrator
            if k != 1:
                enc_par = add(mat_mult(enc_par, enc_d, modulus, kappa, p, delta), mat_mult(enc_par_dot_vec[k-2].reshape((2, 1)), enc_Ts, modulus, kappa, p, delta), modulus)
            enc_par_vec.append(enc_par.flatten())

            # Resetting par because of overflow
            if reset_par == 1:
                par = mat_dec(enc_par, kappa, p, delta) * delta * delta * delta
                enc_par = mat_enc(par, kappa, p, modulus, delta)

            # Calculating next input if par and reg were not exposed
            if reset_par == 0 & reset_reg_eps == 0:
                enc_u = mat_mult(enc_reg.transpose(), enc_par, modulus, kappa, p, delta)
                enc_u_vec.append(enc_u.flatten())
                u = dec(enc_u, kappa, p, delta)
                u_vec.append(u)
            else:
                # Calculating next input if par and reg are exposed/reset
                u = [np.dot(reg.transpose(), par)]    # dec(enc_u, kappa, p, delta) * delta
                u_vec.append([float(x) for x in u])

            k = k + 1
            t = t + Ts

        if Encrypt == 0:
            xr = np.dot(Ar, xr) + np.dot(Br, r)
            x = np.dot(A, x) + np.dot(B, u)  # d2 - Only the output of this needs to be encrypted
            e = x - xr
            eps = np.dot(c.flatten(), e)

            # Need these vectors later for plotting
            x_vec.append(x.flatten())
            xr_vec.append(xr.flatten())
            e_vec.append(e.flatten())

            # Regressor Generator split into parts for debugging
            r = (-math.sin(t + math.pi / 2) + 1)
            r_vec.append(r)
            p1 = r * beta_0
            p2 = r_dot * beta_1
            p3 = r_ddot
            p4 = beta_0 * x[0][0]
            p5 = beta_1 * x[1][0]
            reg = np.array([[x[1][0]], [p1 + p2 + p3 - p4 - p5]])
            reg_vec.append(reg.flatten())

            par_mult = eps * reg
            par_mult = par_mult.reshape((2, 1))  # Reshape to a 2x1 vector

            # Parameter adaptation
            par_dot = np.dot(gains, par_mult)
            par_dot_vec.append(par_dot.flatten())

            # Integrator
            if k != 0:
                par = par + np.dot(par_dot_vec[k - 1].reshape((2, 1)), .01)
            par_vec.append(par.flatten())

            u = float(np.dot(reg.transpose(), par))
            u_vec.append(u)

            k = k + 1
            t = t + Ts

    t = np.arange(Ts, k * Ts, Ts)

    # Analyze the data
    if Encrypt == 1:
        x_vec = mat_dec(enc_x_vec, kappa, p, delta * delta)
        xr_vec = mat_dec(enc_xr_vec, kappa, p, delta * delta)
        e_vec = mat_dec(enc_e_vec, kappa, p, delta * delta)
        par_vec = mat_dec(enc_par_vec, kappa, p, delta * delta * delta * delta)
        par_dot_vec = mat_dec(enc_par_dot_vec, kappa, p, delta * delta * delta)
        write_matrices_to_csv([r_vec, x_vec, xr_vec, e_vec, par_vec, reg_vec, par_dot_vec, u_vec],
                              ['r_vec', 'x_vec', 'xr_vec', 'e_vec', 'par_vec', 'reg_vec', 'par_dot_vec', 'u_vec'], 'enc_data.csv')
    else:
        write_matrices_to_csv([r_vec, x_vec, xr_vec, e_vec, par_vec, reg_vec, par_dot_vec, u_vec], ['r_vec', 'x_vec', 'xr_vec', 'e_vec', 'par_vec', 'reg_vec', 'par_dot_vec', 'u_vec'], 'un_enc_data.csv')

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
    plt.plot(t, np.array(x_vec)[:, 0], label='Actual model')
    plt.plot(t, np.array(xr_vec)[:, 0], label='Reference model')
    plt.title('Actual vs Reference Model')
    plt.xlabel('Time')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == '__main__':
    main()