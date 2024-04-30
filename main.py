import matplotlib.pyplot as plt

from Encrypter import MRAC_Encrypter
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
    e = MRAC_Encrypter(enc_method=2)  # input enc_ada, overflowed ada, encode_ada, or ada here. Instantiating the class
    e.encrypt()

    # Analyze the data
    # if e.Encrypt == 2:
    #     write_matrices_to_csv([e.r_vec, e.x_vec, e.xr_vec, e.e_vec, e.par_vec, e.reg_vec, e.par_dot_vec_test, e.u_vec, e.enc_u_vec], ['r_vec', 'x_vec', 'xr_vec', 'e_vec', 'par_vec', 'reg_vec', 'par_dot_vec', 'u_vec', 'enc_u_vec'], 'enc_data.csv')
    # elif e.Encrypt == 1:
    #     write_matrices_to_csv([e.r_vec, e.x_vec, e.xr_vec, e.e_vec, e.par_vec, e.reg_vec, e.par_dot_vec_test, e.u_vec], ['r_vec', 'x_vec', 'xr_vec', 'e_vec', 'par_vec', 'reg_vec', 'par_dot_vec_test', 'u_vec'], 'encode_data.csv')
    # elif e.Encrypt == 0:
    #     write_matrices_to_csv([e.r_vec, e.x_vec, e.xr_vec, e.e_vec, e.par_vec, e.reg_vec, e.par_dot_vec, e.u_vec], ['r_vec', 'x_vec', 'xr_vec', 'e_vec', 'par_vec', 'reg_vec', 'par_dot_vec', 'u_vec'], 'un_enc_data.csv')

    # Plot the Results
    plt.rcParams.update({'font.size': 12})  # Adjust the font size as needed
    plt.close('all')

    # Error plot
    plt.figure(1, figsize=(8, 4.5))
    # len = 185 # for showing the overflow case
    len = 3000 # for the MMRAC case
    plt.plot(e.t[0:len], np.array(e.e_vec)[0:len, 0], label=r'$x_1 \; error$', color='blue')
    plt.plot(e.t[0:len], np.array(e.e_vec)[0:len, 1], label=r'$x_2 \; error$', color='red')
    plt.ylabel('Tracking Error (m)', fontsize=16)
    plt.xlabel('Time (s)', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.savefig('encrypt_error.eps', dpi=300, format='eps')  # Specify the filename and DPI (dots per inch)

    # Output plot
    plt.figure(2, figsize=(8, 5))
    plt.plot(e.t, np.array(e.x_vec)[:, 0], label=r'Plant $x_1$', color='blue')
    plt.plot(e.t, np.array(e.xr_vec)[:, 0], label=r'Ref Model $x_1$', color='red')
    plt.plot(e.t, np.array(e.x_vec)[:, 1], label=r'Plant $x_2$', color='blue', linestyle='--')
    plt.plot(e.t, np.array(e.xr_vec)[:, 1], label=r'Ref Model $x_2$', color='red', linestyle='--')
    plt.ylabel('Output (m)', fontsize=16)
    plt.xlabel('Time (s)', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.savefig('encrypt_outputs.eps', dpi=300, format='eps')  # Specify the filename and DPI (dots per inch)
    plt.show()

    # Gains plot
    plt.figure(3, figsize=(8, 5))
    gains_vec_array = np.array(e.gains_vec)
    plt.plot(e.t, gains_vec_array[:, 0], label=r'$K_{x_{1}}$')
    plt.plot(e.t, gains_vec_array[:, 1], label=r'$K_{x_{2}}$')
    plt.plot(e.t, gains_vec_array[:, 2], label=r'$K_{r}$')
    plt.plot(e.t, gains_vec_array[:, 3], label=r'$K_{\theta_{1}}$')
    plt.plot(e.t, gains_vec_array[:, 4], label=r'$K_{\theta_{2}}$')
    plt.plot(e.t, gains_vec_array[:, 5], label=r'$K_{\theta_{3}}$')
    plt.plot(e.t, gains_vec_array[:, 6], label=r'$K_{\theta_{4}}$')
    plt.plot(e.t, gains_vec_array[:, 7], label=r'$K_{\theta_{5}}$')
    plt.plot(e.t, gains_vec_array[:, 8], label=r'$K_{\theta_{6}}$')
    plt.ylabel('Gains', fontsize=16)
    plt.xlabel('Time (s)', fontsize=16)
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=3, fontsize=12) #, prop={'size': 10}
    plt.savefig('encrypt_gains.eps', dpi=300, format='eps')  # Specify the filename and DPI (dots per inch)
    plt.show()

    if e.Encrypt == 2 or 1:
        # Create a single figure with two subplots
        plt.figure(3)
        fig, axs = plt.subplots(2, 1, figsize=(8, 5))

        # First subplot: Encrypted Plant Input
        enc_u_vec = np.array(e.enc_u_vec)
        axs[0].plot(e.t, enc_u_vec, color='blue')
        # axs[0].set_title('Encrypted Plant Input')
        axs[0].set_ylabel(r'$\bar{\bar{u}}$ (deg)', fontsize=16)

        # Second subplot: Decrypted Plant Input
        e.u_vec = np.array(e.u_vec)
        axs[1].plot(e.t, e.u_vec[:, 0], color='blue')
        # axs[1].set_title('Decrypted Plant Input')
        axs[1].set_xlabel('Time (s)', fontsize=16)
        axs[1].set_ylabel(r'$u$ (deg)', fontsize=16)

        # Adjust the space between subplots
        plt.tight_layout()

        # Save the combined plot
        plt.savefig('encrypt_inputs.eps', dpi=300, format='eps')

        # Show the combined plot
        plt.show()

if __name__ == '__main__':
    main()