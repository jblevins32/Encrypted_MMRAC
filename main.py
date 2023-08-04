import numpy as np
import matplotlib.pyplot as plt
from dyersmat import *
import math
from compare import *
from Encrypter import Encrypter
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

"""# Encryption
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

"""

def main():
    e = Encrypter(enc_method=2)  # input enc_ada, encode_ada, or ada here. Instantiating the class
    e.encrypt()

    """    # system parameters
    m = 1
    b = 2
    k = 0

    zeta = .8  # Damping Ratio
    wn = .5  # Natural Frequency
    beta_1 = 1 #2 * zeta * wn
    beta_0 = 1 #wn * wn

    # Creating plant state space
    A = np.array([[0, 1], [-k / m, -b / m]])
    B = np.array([[0], [1 / m]])

    # Creating reference state space
    Ar = np.array([[0, 1], [-beta_0, -beta_1]])
    Br = np.array([[0], [beta_0]])

    # Initial Conditions
    x = np.array([[0], [0]])
    xr = np.array([[0], [0]])

    # Other variables
    #c = np.array([[2, 3.125]])
    c = np.array([[.5, 1]])
    gam1 = 15
    gam2 = 1
    gains = np.array([[-gam1, 0], [0, -gam2]])
    u = 0
    r = 0
    Ts = .1
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

    t = 0  # time
    Encrypt = 0  # Encrypt? 1 = yes, 0 = no, 2 = encode only

    for k in range(1, 200):
            
        # encryption of matrices and variables
        if k == 0:
            enc_Ts = enc(Ts, kappa, p, modulus, delta)  # d1
            enc_d = enc(1 / delta, kappa, p, modulus, delta)  # for balancing encoding depths
            enc_Ar = mat_enc(Ar, kappa, p, modulus, delta)  # d1
            enc_Br = mat_enc(Br, kappa, p, modulus, delta)  # d1
            enc_xr = mat_enc(xr, kappa, p, modulus, delta)  # d1
            enc_r = enc(r, kappa, p, modulus, delta)  # d1
            enc_c = mat_enc(c, kappa, p, modulus, delta)  # d1
            enc_beta_0 = enc(beta_0, kappa, p, modulus, delta)  # d1
            enc_beta_1 = enc(beta_1, kappa, p, modulus, delta)  # d1
            enc_gains = mat_enc(gains, kappa, p, modulus, delta)  # d1

        # Calculating next encrypted reference state
        enc_xr = add(mat_mult(enc_Ar, enc_xr, modulus), mat_mult(enc_Br, enc_r, modulus), modulus)  # d2
        r_vec.append(dec(enc_r, kappa, p, delta))  # d0

        # For resetting xr
        if reset_xr == 1:
            xr = mat_dec(enc_xr, kappa, p, delta)
            enc_xr = mat_enc(xr, kappa, p, modulus, delta**2)  # d2
            neg_enc_xr = mat_enc(-xr, kappa, p, modulus, delta**2)  # d2 for subtracting values later

        # PLANT: Calculating next output based on the input
        x = np.dot(A, x) + np.dot(B, u)  # Only the output of this needs to be encrypted
        enc_x = mat_enc(x, kappa, p, modulus, delta**2)  # d2

        # Error and filtered error calculation
        enc_e = add(enc_x, neg_enc_xr, modulus)  # d2
        enc_eps = mat_mult(enc_c, enc_e, modulus)  # d3

        # Need these vectors later for plotting
        enc_x_vec.append(enc_x.flatten())  # d2
        enc_xr_vec.append(enc_xr.flatten())  # d2
        enc_e_vec.append(enc_e.flatten())  # d2

        # Regressor Generator split into parts for debugging
        r = -math.sin(t + math.pi / 2) + 1
        enc_r = enc(r, kappa, p, modulus, delta)  # d1
        enc_reg = np.array([mult([enc_x[1][0]], enc_d, modulus), [add(mult(add(mult(enc_r, enc_d, modulus), -enc_x[0][0], modulus), enc_beta_0, modulus), mult(-enc_x[1][0], enc_beta_1, modulus), modulus)]])  # d3
        enc_reg_vec.append(enc_reg.flatten())

        # Resetting reg and eps because of overflow
        if reset_reg_eps == 1:
            reg = mat_dec(enc_reg, kappa, p, delta**3)
            #reg = np.array([reg[0][0] / delta, reg[1][0]]) * delta * delta  # Correcting encoding depth
            reg_vec.append(reg.flatten())
            enc_reg = mat_enc(reg, kappa, p, modulus, delta)  # d1
            eps = mat_dec(enc_eps, kappa, p, delta**3)
            enc_eps = enc(eps, kappa, p, modulus, delta)  # d1

        # Parameter adaptation
        enc_par_mult = mat_mult(enc_eps, enc_reg, modulus)  # d2
        enc_par_dot = mat_mult(enc_gains, enc_par_mult, modulus)  # d3
        enc_par_dot_vec.append(enc_par_dot.flatten())  # d3

        # Integrator
        if k != 1:
            enc_par = add(mat_mult(enc_par, enc_d, modulus), mat_mult(enc_par_dot_vec[k-2].reshape((2, 1)), enc_Ts, modulus), modulus) #d3
        enc_par_vec.append(enc_par.flatten())  # d3

        # Resetting par because of overflow
        if reset_par == 1:
            par = mat_dec(enc_par, kappa, p, delta**4)
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

        if Encrypt == 2:  # encode with no encryption case
            if k == 0:
                Ts = encode(Ts, delta) #d1
                c = mat_encode(c, delta) #d1
                beta_0 = encode(beta_0, delta) #d1
                beta_1 = encode(beta_1, delta) #d1
                gains = mat_encode(gains, delta) #d1
                Ar = mat_encode(Ar, delta) #d1
                xr = mat_encode(xr, delta) #d1
                Br = mat_encode(Br, delta) #d1
                r = encode(r, delta) #d1

            xr = np.dot(Ar, xr) + np.dot(Br, r) #d2
            x = np.dot(A, x) + np.dot(B, u)
            x = mat_encode(x, delta**3) #d3
            e = x - xr
            eps = np.dot(c.flatten(), e) #d3

            # Need these vectors later for plotting
            x_vec.append(x.flatten()) #d2
            xr_vec.append(xr.flatten()) #d2
            e_vec.append(e.flatten()) #d2

            # Regressor Generator split into parts for debugging
            r = (-math.sin(t + math.pi / 2) + 1)
            r = encode(r, delta) #d1
            r_vec.append(r) #d1
            reg = np.array([[x[1][0]], [(r*delta - x[0][0]) * beta_0 - x[1][0] * beta_1]]) #d3
            reg_vec.append(reg.flatten()) #d3

            par_mult = eps * reg #d6
            par_mult = par_mult.reshape((2, 1))  # Reshape to a 2x1 vector

            # Parameter adaptation
            par_dot = np.dot(gains, par_mult) #d7
            par_dot_vec.append(par_dot.flatten()) #d7

            # Integrator
            if k != 0:
                par = encode(par, delta**7) + np.dot(par_dot_vec[k - 1].reshape((2, 1)), Ts)  # Ts might have to be different here than rest of code. IDK why.
            par_vec.append(par.flatten()) #d7

            u = float(np.dot(reg.transpose(), par)) #d10
            u_vec.append(u)

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
            reg = np.array([[x[1][0]], [(r - x[0][0])*beta_0 - x[1][0]*beta_1]])
            reg_vec.append(reg.flatten())

            par_mult = eps * reg
            par_mult = par_mult.reshape((2, 1))  # Reshape to a 2x1 vector

            # Parameter adaptation
            par_dot = np.dot(gains, par_mult)
            par_dot_vec.append(par_dot.flatten())

            # Integrator
            if k != 0:
                par = par + np.dot(par_dot_vec[k - 1].reshape((2, 1)), Ts) # Ts might have to be different here than rest of code. IDK why.
            par_vec.append(par.flatten())

            u = float(np.dot(reg.transpose(), par))
            u_vec.append(u)

            k = k + 1
            t = t + Ts

    t = np.arange(Ts, k * Ts, Ts)"""

    # Analyze the data
    if e.Encrypt == 1:
        x_vec = mat_dec(e.enc_x_vec, e.kappa, e.p, e.delta * e.delta)
        xr_vec = mat_dec(e.enc_xr_vec, e.kappa, e.p, e.delta * e.delta)
        e_vec = mat_dec(e.enc_e_vec, e.kappa, e.p, e.delta * e.delta)
        par_vec = mat_dec(e.enc_par_vec, e.kappa, e.p, e.delta * e.delta * e.delta * e.delta)
        par_dot_vec = mat_dec(e.enc_par_dot_vec, e.kappa, e.p, e.delta * e.delta * e.delta)
        write_matrices_to_csv([e.r_vec, e.x_vec, e.xr_vec, e.e_vec, e.par_vec, e.reg_vec, e.par_dot_vec, e.u_vec],
                              ['r_vec', 'x_vec', 'xr_vec', 'e_vec', 'par_vec', 'reg_vec', 'par_dot_vec', 'u_vec'], 'enc_data.csv')
    elif e.Encrypt == 0:
        write_matrices_to_csv([e.r_vec, e.x_vec, e.xr_vec, e.e_vec, e.par_vec, e.reg_vec, e.par_dot_vec, e.u_vec], ['r_vec', 'x_vec', 'xr_vec', 'e_vec', 'par_vec', 'reg_vec', 'par_dot_vec', 'u_vec'], 'un_enc_data.csv')

    # Plot the Results
    plt.figure(1)
    plt.subplot(311)
    plt.plot(e.t, np.array(e.e_vec)[:, 0])
    plt.xlabel('Time (sec)')
    plt.ylabel('Tracking Error')
    plt.title('Gains: gam1=100, gam2=10')
    plt.subplot(312)
    plt.plot(e.t, np.array(e.par_vec))
    plt.xlabel('Time (sec)')
    plt.ylabel('Parameters')
    plt.subplot(313)
    plt.plot(e.t, np.array(e.x_vec)[:, 0])
    plt.xlabel('Time (sec)')
    plt.ylabel('Output')

    plt.figure(2)
    plt.plot(e.t, np.array(e.x_vec)[:, 0], label='Actual model')
    plt.plot(e.t, np.array(e.xr_vec)[:, 0], label='Reference model')
    plt.title('Actual vs Reference Model')
    plt.xlabel('Time')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == '__main__':
    main()