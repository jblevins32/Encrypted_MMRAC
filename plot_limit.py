import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better interactivity

def plot_limit(c, max_num, kappa, p, mod, q):
    limit_data = []
    i = 0
    ii = 0
    test1 = int(math.log(kappa-1))
    test2 = int(math.log(q-1))
    for i in range(0, int(math.log(kappa-1))):
        for ii in range(0, int(math.log(q-1))):
            M = math.log((-(i*kappa) - (ii*p) + c) % mod)
            limit_data.append((i, ii, M))
    limit_data = np.array(limit_data)

    # c = (M + (i*kappa) + (ii*p)) % mod

    # Extracting i, ii, and M for plotting
    i_values = limit_data[:, 0]
    ii_values = limit_data[:, 1]
    M_values = limit_data[:, 2]

    # Enabling interactive mode
    plt.ion()

    # Creating a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(i_values, ii_values, M_values, c=M_values, cmap='viridis', marker='o')

    ax.set_xlabel('i')
    ax.set_ylabel('ii')
    ax.set_zlabel('M')

    # Adding a color bar to show the value of M
    fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5)

    plt.show()

