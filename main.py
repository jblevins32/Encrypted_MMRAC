import matplotlib.pyplot as plt

from Encrypter import Encrypter
from compare import *
from dyersmat import *

# Walkthrough by numbers with encode and encrypt to compare where it goes wrong.

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

    # Analyze the data
    if e.Encrypt == 2:
        write_matrices_to_csv([e.r_vec, e.x_vec, e.xr_vec, e.e_vec, e.par_vec, e.reg_vec, e.par_dot_vec_test, e.u_vec, e.enc_u_vec], ['r_vec', 'x_vec', 'xr_vec', 'e_vec', 'par_vec', 'reg_vec', 'par_dot_vec', 'u_vec', 'enc_u_vec'], 'enc_data.csv')
    elif e.Encrypt == 1:
        write_matrices_to_csv([e.r_vec, e.x_vec, e.xr_vec, e.e_vec, e.par_vec, e.reg_vec, e.par_dot_vec_test, e.u_vec], ['r_vec', 'x_vec', 'xr_vec', 'e_vec', 'par_vec', 'reg_vec', 'par_dot_vec_test', 'u_vec'], 'encode_data.csv')
    elif e.Encrypt == 0:
        write_matrices_to_csv([e.r_vec, e.x_vec, e.xr_vec, e.e_vec, e.par_vec, e.reg_vec, e.par_dot_vec, e.u_vec], ['r_vec', 'x_vec', 'xr_vec', 'e_vec', 'par_vec', 'reg_vec', 'par_dot_vec', 'u_vec'], 'un_enc_data.csv')

    # Plot the Results
    plt.close('all')

    plt.figure(1)
    plt.subplot(211)
    plt.plot(e.t, np.array(e.e_vec)[:, 0])
    plt.ylabel('Tracking Error')
    # plt.title(f'Gains: gam1={e.gam1}, gam2={e.gam2}')
    plt.subplot(212)
    plt.plot(e.t, np.array(e.par_vec)[:, 0], label='Par 0: Mass')
    plt.plot(e.t, np.array(e.par_vec)[:, 1], label='Par 1: Damping')
    plt.xlabel('Time (s)')
    plt.ylabel('Parameters')
    plt.legend(loc='center right')
    # plt.subplot(313)
    # plt.plot(e.t, np.array(e.x_vec)[:, 0])
    # plt.xlabel('Time (sec)')
    # plt.ylabel('Output')
    plt.savefig('info.png', dpi=300)  # Specify the filename and DPI (dots per inch)

    plt.figure(2)
    plt.plot(e.t, np.array(e.x_vec)[:, 0], label='Plant')
    plt.plot(e.t, np.array(e.xr_vec)[:, 0], label='Reference model')
    # plt.title('Plant vs Reference Model')
    plt.ylabel('Output: State 1')
    plt.xlabel('Time (s)')
    plt.legend(loc='lower right')
    plt.savefig('outputs.png', dpi=300)  # Specify the filename and DPI (dots per inch)
    plt.show()

    # Create a single figure with two subplots
    plt.figure(3)
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    # First subplot: Encrypted Plant Input
    enc_u_vec = np.array(e.enc_u_vec)
    axs[0].plot(e.t, enc_u_vec)
    # axs[0].set_title('Encrypted Plant Input')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Enc(u): Log Scale')

    # Second subplot: Decrypted Plant Input
    e.u_vec = np.array(e.u_vec)
    axs[1].plot(e.t, e.u_vec[:, 0])
    # axs[1].set_title('Decrypted Plant Input')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Dec(u)')

    # Adjust the space between subplots
    plt.tight_layout()

    # Save the combined plot
    plt.savefig('inputs.png', dpi=300)

    # Show the combined plot
    plt.show()

if __name__ == '__main__':
    main()