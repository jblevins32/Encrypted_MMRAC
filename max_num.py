from dyersmat import *
def max_num(s, max_vec, enc_value): # determines the max abs(plaintext) encrypted value in a matrix or scalar, accounting for encoding factor
    max_val = mat_dec(enc_value, s.kappa, s.p, 1)
    max_val = np.max(abs(max_val))
    return max_vec.append(max_val)