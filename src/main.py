import matplotlib.pyplot as plt
from Encrypter import MRAC_Encrypter
from compare import *
from dyersmat import *

def main():
    get_exec = False # Change if you want execution time plot
    if get_exec != True:
        e = MRAC_Encrypter(enc_method=2, bit_length = 1027)  # input enc_ada, overflowed ada, encode_ada, or ada here. Instantiating the class
        e.encrypt()
    else:
        e = MRAC_Encrypter(enc_method=2, bit_length = 1027)  # input enc_ada, overflowed ada, encode_ada, or ada here. Instantiating the class
        e2 = MRAC_Encrypter(enc_method=2, bit_length = 1127)
        e3 = MRAC_Encrypter(enc_method=2, bit_length = 1227)
        e4 = MRAC_Encrypter(enc_method=2, bit_length = 1327)
        e0 = MRAC_Encrypter(enc_method=0, bit_length = 1327)
        e.encrypt()
        e2.encrypt()
        e3.encrypt()
        e4.encrypt()
        e0.encrypt()

    # Plot the Results
    plt.rcParams.update({'font.size': 12})  # Adjust the font size as needed
    plt.close('all')

    # Error plot
    plt.figure(1, figsize=(8, 3))
    # len = 185 # for showing the overflow case
    len = 3000 # for the MMRAC case
    plt.plot(e.t[0:len], np.array(e.e_vec)[0:len, 0], label=r'$x_1$', color='blue')
    plt.plot(e.t[0:len], np.array(e.e_vec)[0:len, 1], label=r'$x_2$', color='red')
    plt.ylabel('Tracking Error (deg)', fontsize=16)
    plt.xlabel('Time (s)', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.savefig(r'C:\Users\jblevins32\PycharmProjects\Encrypted_Direct_MMRAC\figs\Scenario5_error.eps', dpi=300, format='eps')  # Specify the filename and DPI (dots per inch)

    # Output plot
    plt.figure(2, figsize=(8, 3))
    plt.plot(e.t, np.array(e.x_vec)[:, 0], label=r'Plant $x_1$', color='blue')
    plt.plot(e.t, np.array(e.xr_vec)[:, 0], label=r'Ref Model $x_1$', color='red')
    plt.plot(e.t, np.array(e.x_vec)[:, 1], label=r'Plant $x_2$', color='blue', linestyle='--')
    plt.plot(e.t, np.array(e.xr_vec)[:, 1], label=r'Ref Model $x_2$', color='red', linestyle='--')
    plt.ylabel('Output (deg)', fontsize=16)
    plt.xlabel('Time (s)', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.savefig(r'C:\Users\jblevins32\PycharmProjects\Encrypted_Direct_MMRAC\figs\Scenario5_outputs.eps', dpi=300, format='eps')  # Specify the filename and DPI (dots per inch)
    plt.show()

    # Gains plot
    plt.figure(3, figsize=(8, 5))
    gains_vec_array = np.array(e.gains_vec)
    plt.plot(e.t, gains_vec_array[:, 0], label=r'$K_{x_{1}}$', color='blue')
    plt.plot(e.t, gains_vec_array[:, 1], label=r'$K_{x_{2}}$', color='red')
    plt.plot(e.t, gains_vec_array[:, 2], label=r'$K_{r}$', color='green')
    plt.plot(e.t, gains_vec_array[:, 3], label=r'$K_{\theta_{1}}$', color='yellow')
    plt.plot(e.t, gains_vec_array[:, 4], label=r'$K_{\theta_{2}}$', color='magenta')
    plt.plot(e.t, gains_vec_array[:, 5], label=r'$K_{\theta_{3}}$', color='orange')
    plt.plot(e.t, gains_vec_array[:, 6], label=r'$K_{\theta_{4}}$', color='black')
    plt.plot(e.t, gains_vec_array[:, 7], label=r'$K_{\theta_{5}}$', color='grey')
    plt.plot(e.t, gains_vec_array[:, 8], label=r'$K_{\theta_{6}}$', color='cyan')
    plt.ylabel('Gains', fontsize=16)
    plt.xlabel('Time (s)', fontsize=16)
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=3, fontsize=12) #, prop={'size': 10}
    plt.savefig(r'C:\Users\jblevins32\PycharmProjects\Encrypted_Direct_MMRAC\figs\Scenario5_gains.eps', dpi=300, format='eps')  # Specify the filename and DPI (dots per inch)
    plt.show()

    # Execution time plot
    if get_exec == True:
        plt.figure(4, figsize=(8, 4))
        plt.plot(e0.t, np.array(e0.exec_time_vec)*1000, color='orange', label=r'Plaintext MRAC')
        plt.plot(e.t, np.array(e.exec_time_vec)*1000, color='blue', label=r'$\lambda = 1027$')
        plt.plot(e2.t, np.array(e2.exec_time_vec)*1000, color='red', label=r'$\lambda = 1127$')
        plt.plot(e3.t, np.array(e3.exec_time_vec)*1000, color='green', label=r'$\lambda = 1227$')
        plt.plot(e4.t, np.array(e4.exec_time_vec)*1000, color='purple', label=r'$\lambda = 1327$')
        plt.ylabel('Execution Time (ms)', fontsize=16)
        plt.xlabel('Time (s)', fontsize=16)
        plt.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=3, borderaxespad=0., frameon=False, fontsize=12)
        plt.tight_layout()
        plt.savefig(r'C:\Users\jblevins32\PycharmProjects\Encrypted_Direct_MMRAC\figs\execution_times.eps', dpi=300, format='eps')  # Specify the filename and DPI (dots per inch)
        plt.show()

    if e.Encrypt == 2 or 1:
        # Create a single figure with two subplots
        plt.figure(5)
        fig, axs = plt.subplots(2, 1, figsize=(8, 5))

        # First subplot: Encrypted Plant Input
        enc_u_vec = np.array(e.enc_u_vec)
        axs[0].plot(e.t, enc_u_vec, color='blue')
        # axs[0].set_title('Encrypted Plant Input')
        axs[0].set_ylabel(r'$\bar{\bar{u}}$', fontsize=16)

        # Second subplot: Decrypted Plant Input
        e.u_vec = np.array(e.u_vec)
        axs[1].plot(e.t, e.u_vec[:, 0], color='blue')
        # axs[1].set_title('Decrypted Plant Input')
        axs[1].set_xlabel('Time (s)', fontsize=16)
        axs[1].set_ylabel(r'$u$', fontsize=16)

        # Adjust the space between subplots
        plt.tight_layout()

        # Save the combined plot
        plt.savefig(r'C:\Users\jblevins32\PycharmProjects\Encrypted_Direct_MMRAC\figs\Scenario5_inputs.eps', dpi=300, format='eps')

        # Show the combined plot
        plt.show()

if __name__ == '__main__':
    main()