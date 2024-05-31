import math
import numpy as np
import matplotlib.pyplot as plt
from dyersmat import *
# matplotlib.use('TkAgg')  # Use TkAgg backend for better interactivity

def plot_limit(c, max_num, kappa, p, mod, q):
    limit_data = []
    s_tilde = math.log(kappa-1) # log values
    r_tilde = math.log(q-1)
    # test1 = math.log(kappa-1)
    # test2 = math.log(q-1)
    # for i in range(0, 3):
        # for ii in range(0, int(math.log(q-1))):
    C = (max_num+((kappa-1)*kappa)+((q-1)*p)) % mod
    # limit_data.append((i, 10**i, C))

    M = dec(C, kappa, p, 1)
            # print(M)
    # limit_data = np.array(limit_data)

    # Extracting i, ii, and C for plotting
    # i_values = limit_data[:, 0]
    # ii_values = limit_data[:, 1]
    # C_values = limit_data[:, 2]

    C = enc(100, kappa, p, mod, 1)
    o1 = mult(C, C, mod)
    dec_o1 = dec(o1, kappa, p, 1)
    Overflow1 = mult(mult(C, C, mod), C,mod)
    Decrypted1 = dec(Overflow1, kappa, p, 1)
    Overflow2 = mult(mult(mult(C, C, mod), C, mod), C, mod)
    Decrypted2 = dec(Overflow2, kappa, p, 1)
    test1 = math.log(mod)
    test2 = -math.log(Overflow2)
    lim = add(mod,Overflow2, mod)
    Overflow3 = mult(mult(mult(mult(C, C, mod), C, mod), C, mod), 5, mod)
    test1 = math.log(mod)
    test2 = math.log(Overflow3)
    lim = (math.log(mod)-math.log(Overflow3))
    Decrypted3 = dec(Overflow3, kappa, p, 1)
    Overflow4 = mult(mult(mult(mult(mult(C, C, mod), C, mod), C, mod), C, mod), C, mod)
    Decrypted4 = dec(Overflow4, kappa, p, 1)
    Overflow5 = mult(mult(mult(mult(mult(mult(C, C, mod), C, mod), C, mod), C, mod), C, mod), C, mod)
    Decrypted5 = dec(Overflow5, kappa, p, 1)
    real = (117**6)

    # Enabling interactive mode
    plt.ion()

    # Creating a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(i_values, ii_values, C_values, c=C_values, cmap='viridis', marker='o')

    ax.set_xlabel('i')
    ax.set_ylabel('ii')
    ax.set_zlabel('M')

    # Adding a color bar to show the value of M
    fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5)

    plt.show()

