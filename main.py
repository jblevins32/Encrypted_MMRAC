import numpy as np
import matplotlib.pyplot as plt
from dyers import *

# Encryption
bit_length = 256
rho = 1
rho_ = 32
delta = 0.1
kappa, p = keygen(bit_length, rho, rho_)
modulus = pgen(bit_length, rho_, p)

def mat_encrypt(matrix, rows, cols):
    enc_matrix = np.zeros((rows, cols), dtype='object')

    for i in range(rows):
        for ii in range(cols):
            enc_matrix[i][ii] = int(encrypt(int(matrix[i][ii]), kappa, p, modulus))
            # print(int(encrypt(int(matrix[i][ii]), kappa, p, modulus)))
    return enc_matrix

def mat_decrypt(matrix,rows,cols):
    dec_matrix = np.zeros((int(rows), int(cols)))
    for i in range(rows):
        for ii in range(cols):
            dec_matrix[i][ii] = decrypt(int(matrix[i][ii]), kappa, p)
    return dec_matrix

def mat_encode(matrix,rows,cols):
    encode_matrix = np.zeros((int(rows), int(cols)))
    for i in range(rows):
        for ii in range(cols):
            encode_matrix[i][ii] = encode(matrix[i][ii], delta)
    return encode_matrix

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

    A = mat_encode(A, len(A), len(A[0]))
    print(A)
    enc_A = mat_encrypt(A, len(A), len(A[0]))
    print(enc_A)
    A = mat_decrypt(enc_A, len(A), len(A[0]))/10
    print(A)

    #
    # # Creating reference state space
    # Ar = np.array([[0, 1], [-beta_0, -beta_1]])
    # Br = np.array([[0], [beta_0]])
    # Cr = np.eye(2)
    # Dr = np.zeros((2, 1))
    #
    # # Initial Conditions
    # x = np.array([[0], [0]])
    # xr = np.array([[0], [0]])
    #
    # # Other variables
    # c = np.array([2, 3.125])
    # gam1 = 100
    # gam2 = 10
    # gains = np.array([[-gam1, 0], [0, -gam2]])
    # u = 0
    # r = 0
    # Ts = 0.2
    # par_dot_vec = []
    # x_vec = []
    # xr_vec = []
    # e_vec = []
    # reg_vec = []
    # u_vec = []
    # par_vec = []
    # par = np.array([[0], [0]])
    #
    # k = 0  # time step
    # while k <= 50:
    #     xr = np.dot(Ar, xr) + np.dot(Br, r)
    #     x = np.dot(A, x) + np.dot(B, u)
    #     x_vec.append(x.flatten())
    #     xr_vec.append(xr.flatten())
    #     e = x - xr
    #     e_vec.append(e.flatten())
    #     eps = np.dot(c, e)
    #
    #     # Regressor Generator
    #     r = 1
    #     reg = np.array([[x[1][0]], [r * beta_0 + r_dot * beta_1 + r_ddot - beta_0 * x[0][0] - beta_1 * x[1][0]]])
    #     reg_vec.append(reg.flatten())
    #
    #     # Parameter adaptation
    #     par_dot = np.dot(gains, eps * reg)
    #     par_dot_vec.append(par_dot.flatten())
    #     if k != 0:
    #         par = par + np.dot(par_dot_vec[k].reshape((2, 1)), Ts)
    #     par_vec.append(par.flatten())
    #
    #     u = np.dot(reg.transpose(), par)
    #     u_vec.append(u.flatten())
    #     k = k + 1
    #
    #     t = np.arange(Ts, (k + 1) * Ts, Ts)
    #
    # # Plot the Results
    # plt.figure(1)
    # plt.subplot(311)
    # plt.plot(t, np.array(e_vec)[:, 0])
    # plt.xlabel('Time (sec)')
    # plt.ylabel('Tracking Error')
    # plt.title('Gains: gam1=100, gam2=10')
    # plt.subplot(312)
    # plt.plot(t, np.array(par_vec))
    # plt.xlabel('Time (sec)')
    # plt.ylabel('Parameters')
    # plt.subplot(313)
    # plt.plot(t, np.array(x_vec)[:, 0])
    # plt.xlabel('Time (sec)')
    # plt.ylabel('Output')
    #
    # plt.figure(2)
    # plt.plot(t, np.array(x_vec)[:, 0], t, np.array(xr_vec)[:, 0])
    # plt.title('Actual vs Reference Model')
    # plt.xlabel('Time')
    # plt.show()


if __name__ == '__main__':
    main()