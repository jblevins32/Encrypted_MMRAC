import csv

def write_matrices_to_csv(matrices, titles, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(titles)  # Write header row
        rows = zip(*matrices)  # Transpose matrices to get rows
        writer.writerows(rows)