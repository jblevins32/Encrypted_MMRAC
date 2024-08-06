from dyers import *
from dyersmat import *
import csv
import pandas as pd
import numpy as np

bit_length = 64
rho = 32
rho_ = 64
delta = .1
kappa, p = keygen(bit_length, rho, rho_)
mod = pgen(bit_length, rho_, p)

# Read the Excel file into a Pandas DataFrame
df = pd.read_excel('unenc_data.xlsx')

# Convert the DataFrame to a NumPy array
matrix = df.to_numpy()

matrix_enc = mat_enc(matrix,kappa,p,mod,delta)

# Print the matrix
print(matrix_enc)

# Create a new DataFrame from the encrypted matrix
encrypted_df = pd.DataFrame(matrix_enc)

# Save the encrypted matrix to a new CSV file
encrypted_df.to_csv('encrypted_matrix_2.csv', index=False)