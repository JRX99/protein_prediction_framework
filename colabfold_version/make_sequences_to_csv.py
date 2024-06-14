import random

# Define the set of amino acids including U (selenocysteine)
amino_acids = "ACDEFGHIKLMNPQRSTVWY"


def generate_random_protein_sequence(min_length, max_length):
    """
    Generate a random protein sequence with a variable length between min_length and max_length.

    Args:
    min_length (int): Minimum length of the protein sequence.
    max_length (int): Maximum length of the protein sequence.

    Returns:
    str: Randomly generated protein sequence.
    """
    sequence_length = random.randint(min_length, max_length)
    random_sequence = [random.choice(amino_acids) for _ in range(sequence_length)]
    return "".join(random_sequence)

# Example: Generate a random protein sequence with a length between 10 and 20
import csv
import random
import string



# Number of rows you want in the CSV file
num_rows = 50
name="sequences"
# Create a CSV file and write data to it
with open((name+'.csv'), 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    # Write the header row
    csv_writer.writerow(['id', 'sequence'])

    # Generate and write the data rows
    for i in range(1, num_rows + 1):
        id=name+"_"+str(i)
        sequence = "KPTENNEDFNIVAVASNFATTDLDADRGKLPGKKLPLEVLKEMEANARKAGCTRGCLICLSHIKCTPKMKKFIPGRCHTYEGDKESAQGGIGEAIVDIPEIPGFKDLEPMEQFIAQVDLCVDCTTGCLKGLANVQCSDLLKKWLPQRCATFASKIQGQVDKIKGAGGDGSSGSSGSSGSSGSSGSS"+generate_random_protein_sequence(15,20)
        csv_writer.writerow([id, sequence])

print("CSV file 'sequences.csv' has been generated.")