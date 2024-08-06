import random
from sympy import nextprime
import time

def get_prime_list(bit_length):
    flag = False
    file_path = r'C:\Users\jblevins32\PycharmProjects\Encrypted_Direct_MMRAC\src\primes.txt'  # Replace with your file path
    with open(file_path, 'r') as file:
        for line in file:
            if f"{bit_length}," in line:
                flag = True # check if the prime number was actually gotten or just the last prime in the list
                break

    # if prime is not in the list, add it
    if flag == False:
        print(f'Adding {int(bit_length)} to prime list')
        start_time = time.time()
        with open(file_path, 'a') as file:
            random_number = random.getrandbits(int(bit_length))
            prime_number = nextprime(random_number)
            file.write(f'\n{bit_length},{prime_number}')
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"It took {execution_time} seconds to add {bit_length} to prime list")

    prime = int(line.split(',')[1])
    return prime