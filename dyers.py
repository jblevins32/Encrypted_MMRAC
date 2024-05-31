"""
:file: dyers.py
:author: Shane Kosieradzki
:created: 5/25/2023
"""

from eclib.randutils import *
from eclib.primeutils import *
from eclib.modutils import *
from math import floor
from prime_list import *

def keygen(bit_length, rho, rho_):
    p = get_prime_list(bit_length)
    nu = rho_ - rho
    kappa = get_prime_list(nu)

    return kappa, p


def pgen(bit_length, rho_, p):
    eta = bit_length ** 2 // rho_ - bit_length
    q = get_prime_list(eta)
    modulus = p * q

    return modulus, q


def encode(x, delta):
    m = floor(x / delta + 0.5)

    return int(m)


def decode(m, delta):
    x = m * delta

    return x


def encrypt(m, kappa, p, modulus):
    q = modulus

    r = q #get_rand(1, q-1)
    s = kappa #get_rand(0, kappa-1)
    c = (m + s * kappa + r * p) % modulus

    return c

def encrypt_special(m, kappa, p, modulus):
    q = modulus

    r = get_rand(1, q)
    s = get_rand(0, kappa)
    c = (m + s * kappa + r * p) % modulus

    return s,r,c


def decrypt(c, kappa, p):
    m = min_residue(min_residue(c, p), kappa)

    return m


def enc(x, kappa, p, modulus, delta):
    c = encrypt(encode(x, delta), kappa, p, modulus)

    return c

def enc_special(x, kappa, p, modulus, delta):
    s,r,c = encrypt_special(encode(x, delta), kappa, p, modulus)

    return s,r,c

def dec(c, kappa, p, delta):
    x = decode(decrypt(c, kappa, p), delta)

    return x


def add(c1, c2, modulus):
    c3 = (c1 + c2) % modulus

    return c3


def mult(c1, c2, modulus):
    c3 = (c1 * c2) % modulus

    return c3

def longmult(c1, modulus): # c1 is a vector of numbers to be multiplied together
    c2 = c1[0]
    for i in range(len(c1)):
        if i == 0:
            c2 = c2
        else:
            c2 = (c2 * c1[i]) % modulus

    return c2

def longadd(c1, modulus): # c1 is a vector of numbers to be multiplied together
    c2 = c1[0]
    for i in range(len(c1)):
        if i == 0:
            c2 = c2
        else:
            c2 = (c2 + c1[i]) % modulus

    return c2

if __name__ == '__main__':
    # Security Parameters #
    bit_length = 350
    rho = 1
    rho_ = 32
    delta = 0.1

    # Primes #
    kappa, p = keygen(bit_length, rho, rho_)
    modulus = pgen(bit_length, rho_, p)
