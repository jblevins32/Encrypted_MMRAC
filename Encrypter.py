import time
from mpmath import mp
from disc import *
from scipy.linalg import solve_discrete_lyapunov
from max_num import *
from tuning import *
from plot_limit import *


class MRAC_Encrypter():
    def __init__(s, enc_method):

        # Encryption
        s.Encrypt = enc_method  # Encrypt? 0 = none, 1 = encode, 2 = encrypt
        s.bit_length = 888
        rho_test = math.log2(s.bit_length) # see what rho should be atleast to prevent brute force attack
        s.rho = 32
        s.rho_ = 143
        s.delta = 0.01
        s.kappa, s.p = keygen(s.bit_length, s.rho, s.rho_)
        s.mod, q = pgen(s.bit_length, s.rho_, s.p)
        c = (100000000 + (s.kappa-1)*s.kappa + (q-1)*s.p) % s.mod #190787524
        kappa, lam = tuning(100000000, 4, 1)
        c2 = math.log(c)
        c3 = math.log(s.p*q)
        plot_limit(c, 100000000, s.kappa, s.p, s.mod, q)

        # Creating continuous plant state space
        s.A = np.array([[0, 1], [-4, -3]])
        s.B = np.array([[0], [3]])
        s.C = np.eye(2)
        s.D = np.array([[0], [0]])

        # Creating continuous reference state space
        s.Ar = np.array([[0, 1], [-5, -5]])
        s.Br = np.array([[0], [4]])
        s.Cr = np.eye(2)
        s.Dr = np.array([[0], [0]])

        # Converting to discrete
        s.Ts = .01
        s.A, s.B, s.C, s.D = disc(s.A, s.B, s.C, s.D, s.Ts)
        s.Ar, s.Br, s.Cr, s.Dr = disc(s.Ar, s.Br, s.Cr, s.Dr, s.Ts)
        # eigenvalues = np.linalg.eigvals(s.Ar)
        # abs_eigenvalues = np.abs(eigenvalues)

        # Initial Conditions
        theta0 = 0; #Initial roll angle(rad)
        p0 = 0; #Initial roll rate(rad / s)
        s.x = np.array([[theta0], [p0]])
        s.xr = np.array([[theta0], [p0]])

        # Disturbance
        # True but unknown coefficient of disturbance
        s.theta = np.array([1, 0.2314, 0.6918, 0.6245, 0.1, 0.214])

        # Lyapunov
        s.P = solve_discrete_lyapunov(np.transpose(s.Ar), np.eye(2))

        # Matrix sizes for generality
        s.n = s.Ar.shape[1]
        s.m = s.Br.shape[1]
        s.q = len(s.theta)

        # Unused? variables
        s.e_tol = .02  # tolerable error
        s.e_flag = 0
        s.ss_k = 0

        #Initial Command
        s.u = 0

        # Encryption specific variables
        s.enc_Ts = enc(s.Ts, s.kappa, s.p, s.mod, s.delta)
        s.encode_Ts = encode(s.Ts, s.delta)
        s.Ts_time = s.Ts

        # Initialized vectors and variables
        s.gains_vec = []
        s.x_vec = []
        s.xr_vec = []
        s.e_vec = []
        s.u_vec = []
        s.enc_u_vec = []
        s.r_vec = []
        s.max_vec = []
        s.enc_gains = 0 # for case 1

        # Define parameters
        s.gamma_x = np.diag([30, 10])
        s.gamma_r = 30
        s.gamma_theta = np.diag([20, 20, 20, 40, 40, 40])
        s.gains = 0

        # Precomputation
        v_1 = np.dot(s.B.T, s.P)
        v_2 = np.dot(np.dot(s.B.T, s.P), s.Ar)
        v_3 = np.dot(np.dot(s.B.T, s.P), s.Br)

        s.G_omega0_x = -s.Ts * v_1[0][0] * s.gamma_x
        s.G_omega1_x = -s.Ts * v_1[0][1] * s.gamma_x
        s.G_omega0_r = -s.Ts * v_1[0][0] * s.gamma_r
        s.G_omega1_r = -s.Ts * v_1[0][1] * s.gamma_r
        s.G_omega0_theta = -s.Ts * v_1[0][0] * s.gamma_theta
        s.G_omega1_theta = -s.Ts * v_1[0][1] * s.gamma_theta

        s.A_omega0_x = s.Ts * v_2[0][0] * s.gamma_x
        s.A_omega1_x = s.Ts * v_2[0][1] * s.gamma_x
        s.A_omega0_r = s.Ts * v_2[0][0] * s.gamma_r
        s.A_omega1_r = s.Ts * v_2[0][1] * s.gamma_r
        s.A_omega0_theta = s.Ts * v_2[0][0] * s.gamma_theta
        s.A_omega1_theta = s.Ts * v_2[0][1] * s.gamma_theta

        s.B_omega_x = s.Ts * v_3 * s.gamma_x
        s.B_omega_r = s.Ts * v_3 * s.gamma_r
        s.B_omega_theta = s.Ts * v_3 * s.gamma_theta

        s.t = 0  # time

    # Run timers and chosen algorithm
    def encrypt(s):
        start_time = time.time()
        iterations = 500
        for k in range(1, iterations):
            # print(k)
            if s.Encrypt == 2:
                s.enc_ada(k)
            elif s.Encrypt == 1:
                s.overflow_ada(k)
            elif s.Encrypt == 0:
                s.ada(k)
        s.t = np.arange(s.Ts, (k+1) * s.Ts, s.Ts)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time/iterations} seconds")
        print(f"State 1 error becomes < {s.e_tol} permanently at iteration {s.ss_k} and time {(s.ss_k/iterations) * s.t[-1]}")
        avg = abs(np.array(s.e_vec)[:, 0])
        print(f"State 1 average error is {np.mean(avg)}")
        max_num = int(np.max(s.max_vec))
        print(f"Max value is {max_num}")
        lam, rho, rho_ = tuning(max_num, 4, 6)        # lam, rho, rho_ = tuning(14748022240000000000, 9, 13)
        print(f"Tuning parameters are {lam, rho, rho_}")

    def enc_ada(s, k):
        # Plant
        phi = np.array([[1, s.x[0][0], s.x[1][0], np.abs(s.x[0][0]) * s.x[0][0], np.abs(s.x[1][0]) * s.x[1][0], s.x[0][0] ** 3]]).T
        phi = phi.astype(int)
        r = math.sin(s.t)

        # Changing plant at 20 seconds
        # if (k == 2000):
        #     s.A = np.array([[0, 1], [-8, -8]])
        #     s.B = np.array([[0], [3]])
        #     s.A, s.B, s.C, s.D = disc(s.A, s.B, s.C, s.D, s.Ts)

        s.xr = np.dot(s.Ar, s.xr) + s.Br * r
        s.x = np.dot(s.A, s.x) + s.B * (s.u + np.dot(s.theta, phi))
        e = s.x - s.xr # error monitor

        # Testing for if tolerated error is reached
        if abs(e[0][0]) <= s.e_tol:
            if s.e_flag == 0:
                s.ss_k = k  # iteration at which e_tol remains true
            s.e_flag = 1
            print(f"State 1 error becomes < {s.e_tol} at iteration {k}, time {round(s.t, 2)}")
        else:
            s.e_flag = 0

        # encryption of matrices and variables
        if k == 1: # all d1
            s.enc_Ar = mat_enc(s.Ar, s.kappa, s.p, s.mod, s.delta)
            s.enc_Br = mat_enc(s.Br, s.kappa, s.p, s.mod, s.delta)
            s.enc_G_omega0_x = mat_enc(s.G_omega0_x, s.kappa, s.p, s.mod, s.delta)
            s.enc_G_omega1_x = mat_enc(s.G_omega1_x, s.kappa, s.p, s.mod, s.delta)
            s.enc_G_omega0_r = enc(s.G_omega0_r, s.kappa, s.p, s.mod, s.delta)
            s.enc_G_omega1_r = enc(s.G_omega1_r, s.kappa, s.p, s.mod, s.delta)
            s.enc_G_omega0_theta = mat_enc(s.G_omega0_theta, s.kappa, s.p, s.mod, s.delta)
            s.enc_G_omega1_theta = mat_enc(s.G_omega1_theta, s.kappa, s.p, s.mod, s.delta)
            s.enc_A_omega0_x = mat_enc(s.A_omega0_x, s.kappa, s.p, s.mod, s.delta)
            s.enc_A_omega1_x = mat_enc(s.A_omega1_x, s.kappa, s.p, s.mod, s.delta)
            s.enc_A_omega0_r = enc(s.A_omega0_r, s.kappa, s.p, s.mod, s.delta)
            s.enc_A_omega1_r = enc(s.A_omega1_r, s.kappa, s.p, s.mod, s.delta)
            s.enc_A_omega0_theta = mat_enc(s.A_omega0_theta, s.kappa, s.p, s.mod, s.delta)
            s.enc_A_omega1_theta = mat_enc(s.A_omega1_theta, s.kappa, s.p, s.mod, s.delta)
            s.enc_B_omega_x = mat_enc(s.B_omega_x, s.kappa, s.p, s.mod, s.delta)
            s.enc_B_omega_r = mat_enc(s.B_omega_r, s.kappa, s.p, s.mod, s.delta)
            s.enc_B_omega_theta = mat_enc(s.B_omega_theta, s.kappa, s.p, s.mod, s.delta)
            s.enc_Ts = enc(s.Ts, s.kappa, s.p, s.mod, s.delta)  # d1 for balancing encoding depths
            s.enc_d2 = enc(1, s.kappa, s.p, s.mod, s.delta ** 2)

        enc_r = enc(r, s.kappa, s.p, s.mod, s.delta)  # d1
        enc_xr = mat_enc(s.xr, s.kappa, s.p, s.mod, s.delta)  # d1
        enc_x = mat_enc(s.x, s.kappa, s.p, s.mod, s.delta)  # d1
        enc_phi = mat_enc(phi, s.kappa, s.p, s.mod, s.delta)

        # Cloud
        enc_y_omega_x = add(mat_mult(s.enc_G_omega0_x, enc_x[0][0], s.mod), mat_mult(s.enc_G_omega1_x, enc_x[1][0], s.mod), s.mod)
        enc_y_omega_r = add(mat_mult(s.enc_G_omega0_r, enc_x[0][0], s.mod), mat_mult(s.enc_G_omega1_r, enc_x[1][0], s.mod), s.mod)
        enc_y_omega_theta = add(mat_mult(s.enc_G_omega0_theta, enc_x[0][0], s.mod), mat_mult(s.enc_G_omega1_theta, enc_x[1][0], s.mod), s.mod)

        enc_yr_omega_x = add(add(mat_mult(s.enc_A_omega0_x, enc_xr[0][0], s.mod), mat_mult(s.enc_A_omega1_x, enc_xr[1][0], s.mod), s.mod), mat_mult(s.enc_B_omega_x, enc_r, s.mod), s.mod)
        enc_yr_omega_r = add(add(mat_mult(s.enc_A_omega0_r, enc_xr[0][0], s.mod), mat_mult(s.enc_A_omega1_r, enc_xr[1][0], s.mod), s.mod), mat_mult(s.enc_B_omega_r, enc_r, s.mod), s.mod)
        enc_yr_omega_theta = add(add(mat_mult(s.enc_A_omega0_theta, enc_xr[0][0], s.mod), mat_mult(s.enc_A_omega1_theta, enc_xr[1][0], s.mod), s.mod), mat_mult(s.enc_B_omega_theta, enc_r, s.mod), s.mod)

        enc_eps_x = add(enc_y_omega_x, enc_yr_omega_x, s.mod)
        enc_eps_r = add(enc_y_omega_r, enc_yr_omega_r, s.mod)
        enc_eps_theta = add(enc_y_omega_theta, enc_yr_omega_theta, s.mod)

        enc_z_x = mat_mult(enc_x.T,enc_eps_x, s.mod)
        enc_z_r = mat_mult(enc_r,enc_eps_r, s.mod)
        enc_z_theta = mat_mult(enc_phi.T,enc_eps_theta, s.mod)

        enc_z_vec = np.concatenate((enc_z_x.flatten(), enc_z_r.flatten(), enc_z_theta.flatten())).flatten()

        s.enc_gains = add(s.enc_gains, enc_z_vec.reshape(-1, 1), s.mod)
        # if k % 500 == 0:
        #     s.gains = mat_dec(s.enc_gains, s.kappa, s.p, s.delta**3)
        #     s.gains = mat_enc(s.gains, s.kappa, s.p, s.mod, s.delta ** 3)

        s.gains_plot = mat_dec(s.enc_gains,  s.kappa, s.p, s.delta ** 3) # decrypt to update plaintext gains vector and reset integration increasing depth

        enc_u = add(add(mat_mult(s.enc_gains[:s.n].reshape(1, -1), enc_x, s.mod), mat_mult(s.enc_gains[s.n:s.n + s.m].reshape(1, -1), enc_r, s.mod), s.mod), mat_mult(s.enc_gains[s.n + s.m:s.n + s.m + s.q].reshape(1, -1), enc_phi, s.mod), s.mod)

        # Beta attack
        enc_u = s.attack(enc_u, k)

        s.u = dec(enc_u,  s.kappa, s.p, s.delta ** 4)

        # Need these vectors later for plotting
        s.x_vec.append(s.x.flatten())
        s.xr_vec.append(s.xr.flatten())
        s.e_vec.append(e.flatten())
        s.r_vec.append(r)
        s.gains_vec.append(s.gains_plot.flatten())

        # Getting max number in each iteration
        max_num(s, s.max_vec, enc_y_omega_x)
        max_num(s, s.max_vec, enc_y_omega_r)
        max_num(s, s.max_vec, enc_y_omega_theta)
        max_num(s, s.max_vec, enc_yr_omega_x)
        max_num(s, s.max_vec, enc_yr_omega_r)
        max_num(s, s.max_vec, enc_yr_omega_theta)
        max_num(s, s.max_vec, enc_eps_x)
        max_num(s, s.max_vec, enc_eps_r)
        max_num(s, s.max_vec, enc_eps_theta)
        max_num(s, s.max_vec, enc_z_x)
        max_num(s, s.max_vec, enc_z_r)
        max_num(s, s.max_vec, enc_z_theta)
        max_num(s, s.max_vec, s.enc_gains)
        max_num(s, s.max_vec, enc_u)
        s.max_vec.append(np.max(phi))
        s.max_vec.append(np.max(s.x))
        s.max_vec.append(r)
        s.max_vec.append(np.max(s.xr))

        # Time step
        s.t = s.t + s.Ts_time

        # Decrypting for encryption plotting
        s.enc_u = mp.mpf(int(enc_u))
        s.enc_u = mp.log10(s.enc_u)
        s.enc_u_vec.append(s.enc_u)
        s.u_vec.append(s.u)

    def overflow_ada(s, k):
        # Plant
        phi = np.array(
            [[1, s.x[0][0], s.x[1][0], np.abs(s.x[0][0]) * s.x[0][0], np.abs(s.x[1][0]) * s.x[1][0], s.x[0][0] ** 3]]).T
        r = math.sin(s.t)
        s.x = np.dot(s.A, s.x) + s.B * (
                    s.u + np.dot(s.theta, phi))  # d2 - Only the output of this needs to be encrypted

        # Encrypting
        enc_r = enc(r, s.kappa, s.p, s.mod, s.delta)
        enc_phi = mat_enc(phi, s.kappa, s.p, s.mod, s.delta)
        enc_xr = mat_enc(s.xr, s.kappa, s.p, s.mod, s.delta)
        enc_x = mat_enc(s.x, s.kappa, s.p, s.mod, s.delta)
        enc_Ar = mat_enc(s.Ar, s.kappa, s.p, s.mod, s.delta)
        enc_Br = mat_enc(s.Br, s.kappa, s.p, s.mod, s.delta)
        enc_gamma_x = mat_enc(s.gamma_x, s.kappa, s.p, s.mod, s.delta)
        enc_gamma_r = enc(s.gamma_r, s.kappa, s.p, s.mod, s.delta)
        enc_gamma_theta = mat_enc(s.gamma_theta, s.kappa, s.p, s.mod, s.delta)
        enc_P = mat_enc(s.P, s.kappa, s.p, s.mod, s.delta)
        enc_B = mat_enc(s.B, s.kappa, s.p, s.mod, s.delta)
        enc_Ts = enc(s.Ts, s.kappa, s.p, s.mod, s.delta)
        enc_del = enc(s.delta, s.kappa, s.p, s.mod, s.delta)
        enc_del2 = enc(s.delta, s.kappa, s.p, s.mod, s.delta ** 2)
        enc_n1 = enc(-1, s.kappa, s.p, s.mod, s.delta)

        s.xr = np.dot(s.Ar, s.xr) + s.Br * r
        e = s.x - s.xr  # error monitor

        # Testing for if tolerated error is reached
        if abs(e[0][0]) <= s.e_tol:
            if s.e_flag == 0:
                s.ss_k = k  # iteration at which e_tol remains true
            s.e_flag = 1
            print(f"State 1 error becomes < {s.e_tol} at iteration {k}, time {round(s.t, 2)}")
        else:
            s.e_flag = 0


        # Cloud
        enc_xr = add(mat_mult(enc_Ar, enc_xr, s.mod),mat_mult(enc_Br,enc_r,s.mod),s.mod)
        enc_e = add(mat_mult(enc_del2, enc_x, s.mod), mat_mult(enc_n1, enc_xr, s.mod), s.mod)
        enc_x_Kplus = mat_mult(mat_mult(mat_mult(mat_mult(mat_mult(enc_n1,enc_gamma_x,s.mod), enc_x, s.mod), enc_e.T, s.mod), enc_P, s.mod), enc_B, s.mod)
        enc_r_Kplus = mat_mult(mat_mult(mat_mult(mat_mult(mat_mult(enc_n1,enc_gamma_r,s.mod), enc_r, s.mod), enc_e.T, s.mod), enc_P, s.mod), enc_B, s.mod)
        enc_theta_Kplus = mat_mult(mat_mult(mat_mult(mat_mult(mat_mult(enc_n1,enc_gamma_theta,s.mod), enc_phi, s.mod), enc_e.T, s.mod), enc_P, s.mod), enc_B, s.mod)
        enc_Kplus_vec = np.concatenate((enc_x_Kplus.flatten(), enc_r_Kplus.flatten(), enc_theta_Kplus.flatten())).flatten()
        s.enc_gains = add(s.enc_gains, mat_mult(enc_Kplus_vec, enc_Ts, s.mod), s.mod)

        enc_u = add(add(mat_mult(s.enc_gains[:s.n].reshape(1, -1), enc_x, s.mod),mat_mult(s.enc_gains[s.n:s.n + s.m].reshape(1, -1), enc_r, s.mod), s.mod), mat_mult(
        s.enc_gains[s.n + s.m:s.n + s.m + s.q].reshape(1, -1), enc_phi, s.mod), s.mod)

        s.gains = mat_dec(s.enc_gains, s.kappa, s.p, s.delta ** 9)
        s.u = dec(enc_u, s.kappa, s.p, s.delta ** 10)

        # Need these vectors later for plotting
        s.x_vec.append(s.x.flatten())
        s.xr_vec.append(s.xr.flatten())
        s.e_vec.append(e.flatten())
        s.r_vec.append(r)
        s.gains_vec.append(s.gains.flatten())

        s.t = s.t + s.Ts_time

        # Decrypting
        s.enc_u = mp.mpf(int(enc_u))
        s.enc_u = mp.log10(s.enc_u)
        s.enc_u_vec.append(s.enc_u)
        s.u_vec.append(s.u)

    def ada(s, k):
        # Plant
        phi = np.array([[1, s.x[0][0], s.x[1][0], np.abs(s.x[0][0]) * s.x[0][0], np.abs(s.x[1][0]) * s.x[1][0], s.x[0][0] ** 3]]).T
        r = math.sin(s.t)

        s.xr = np.dot(s.Ar, s.xr) + s.Br * r
        s.x = np.dot(s.A, s.x) + s.B * (s.u + np.dot(s.theta,phi))  # d2 - Only the output of this needs to be encrypted
        e = s.x - s.xr # error monitor

        # Testing for if tolerated error is reached
        if abs(e[0][0]) <= s.e_tol:
            if s.e_flag == 0:
                s.ss_k = k  # iteration at which e_tol remains true
            s.e_flag = 1
            print(f"State 1 error becomes < {s.e_tol} at iteration {k}, time {round(s.t, 2)}")
        else:
            s.e_flag = 0

        int_var = np.dot(np.dot(s.B.T, s.P), s.x)
        y_omega_x = -s.gamma_x * int_var
        y_omega_r = -s.gamma_r * int_var
        y_omega_theta = -s.gamma_theta * int_var

        # Cloud
        yr_omega_x = s.A_omega0_x*s.xr[0][0] + s.A_omega1_x*s.xr[1][0] + s.B_omega_x*r
        yr_omega_r = s.A_omega0_r*s.xr[0][0] + s.A_omega1_r*s.xr[1][0] + s.B_omega_r*r
        yr_omega_theta = s.A_omega0_theta*s.xr[0][0] + s.A_omega1_theta*s.xr[1][0] + s.B_omega_theta*r

        eps_x = y_omega_x + yr_omega_x
        eps_r = y_omega_r + yr_omega_r
        eps_theta = y_omega_theta + yr_omega_theta

        z_x = np.dot(s.x.T,eps_x)
        z_r = np.dot(r,eps_r)
        z_theta = np.dot(phi.T,eps_theta)

        z_vec = np.concatenate((z_x.flatten(), z_r.flatten(), z_theta.flatten())).flatten()

        s.gains = s.gains + z_vec*s.Ts

        s.u = np.dot(s.gains[:s.n], s.x) + np.dot(s.gains[s.n:s.n + s.m], r) + np.dot(s.gains[s.n + s.m:s.n + s.m + s.q], phi)

        # Need these vectors later for plotting
        s.x_vec.append(s.x.flatten())
        s.xr_vec.append(s.xr.flatten())
        s.e_vec.append(e.flatten())
        s.r_vec.append(r)
        s.gains_vec.append(s.gains.flatten())

        s.t = s.t + s.Ts_time

    def attack(s, enc_u, k):
        if k >= 500 and k < 1500: # Adapted
            enc_u = mult(enc_u, int(np.ceil(np.abs(10 * math.sin(math.pi * s.t / 4)))), s.mod)
        # elif k >= 2000: # Trapped
        #     enc_u = mult(enc_u, 100, s.mod)
        return enc_u