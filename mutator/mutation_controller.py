

import subprocess
from find_best_peptide  import find_best_peptide 


def broad_search(peptide, depth):
    if depth > 3:
        return
    
    # Define the command to run, including the script name and arguments
    command = ["python", "esmfold_mutate.py", peptide]

    # Run the command
    # Generate mutations for the current peptide
    subprocess.run(command)
    
    # Process the results for each mutation
    command = ["python", "esmfold_mutation_processor.py", peptide]
    subprocess.run(command)
    # Find the best peptides among the results
    best_peptides = find_best_peptide(original_sequence=peptide)
    # Continue the search with the best peptides
    for best_peptide in best_peptides:
        broad_search(best_peptide, depth + 1)

# Start the search with the root peptide
root_peptide = "your_root_peptide"
broad_search(root_peptide, 0)
