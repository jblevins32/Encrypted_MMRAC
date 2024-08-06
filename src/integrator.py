from dyersmat import *

def integrator(k, par, enc_method, par_dot_vec, Ts_time, mod):
    if k != 1:
        if enc_method == 0:
            par = par + np.dot(par_dot_vec[k - 1].reshape((2, 1)), Ts_time)
        elif enc_method == 1:
            par = par + np.dot(par_dot_vec[k - 1].reshape((2, 1)), Ts_time)  # d7 # Ts might have to be different here than rest of code. IDK why.
        elif enc_method == 2:
            par = add(par, mat_mult(par_dot_vec[k - 2].reshape((2, 1)), Ts_time, mod), mod)  # need to input encrypted Ts
    return par