import random
import shutil
from transformers import AutoTokenizer, EsmForProteinFolding
import csv, re, os
import numpy as np
import torch
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from scipy.special import softmax
import evolutionary_algorithm_processor
from evolutionary_algorithm_processor import process, load_models

# Function to convert model outputs to PDB format
def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs

# Function to convert outputs to PAE (Predicted Aligned Error)
def convert_outputs_to_pae(output):
    pae = (output["aligned_confidence_probs"][0].cpu().numpy() * np.arange(64)).mean(-1) * 31
    mask = output["atom37_atom_exists"][0, :, 1] == 1
    mask = mask.cpu()
    pae = pae[mask, :][:, mask]
    return pae

# Initialize the ESMFold model and tokenizer
def init_esmfold():
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("/path/to/esmfold_model_directory", low_cpu_mem_usage=True)
    return tokenizer, model

# Make predictions using the ESMFold model
def make_esm_prediction(peptide, tokenizer, model):
    global output_directory
    
    model = model.cuda()
    model.esm = model.esm.half()
    
    torch.backends.cuda.matmul.allow_tf32 = True

    # Sequences and linkers
    luciferase = "KPTENNEDFNIVAVASNFATTDLDADRGKLPGKKLPLEVLKEMEANARKAGCTRGCLICLSHIKCTPKMKKFIPGRCHTYEGDKESAQGGIGEAIVDIPEIPGFKDLEPMEQFIAQVDLCVDCTTGCLKGLANVQCSDLLKKWLPQRCATFASKIQGQVDKIKGAGGD"
    linker1 = "GSSGSSGSSGSSGSSGSS"
    linker2 = "GGGGGGGGGGGGGGGGGG"
    
    sequence = luciferase + linker2 + peptide

    num_recycles = 3  # Number of recycles
    
    ID = peptide
    seqs = sequence.split(":")
    lengths = [len(s) for s in seqs]
    length = sum(lengths)
    u_seqs = list(set(seqs))
    mode = "mono" if len(seqs) == 1 else "homo" if len(u_seqs) == 1 else "hetero"
    
    torch.cuda.empty_cache()
    tokenized_input = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)['input_ids']
    tokenized_input = tokenized_input.cuda()

    with torch.no_grad():
        output = model(tokenized_input)
    
    pdb = convert_outputs_to_pdb(output)
    pae = convert_outputs_to_pae(output)
    
    scratchdir = os.getenv('SCRATCHDIR')
    if not scratchdir:
        raise EnvironmentError('SCRATCHDIR environment variable is not set')
    
    # Ensure output directory exists and has write permissions
    os.makedirs(output_directory, exist_ok=True)
    os.system(f"chmod a+w {output_directory}")

    # Save the PAE and PDB files
    np.savetxt(f"{output_directory}/{ID}.pae.txt", pae, "%.3f")
    for idx, pdb_string in enumerate(pdb):
        with open(f"{output_directory}/{ID}_{idx}.pdb", "w") as out:
            out.write(pdb_string)

# Generate a random peptide of a given length
def generate_random_peptide(length):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    return 'LYEYAFNAWYILFAH'

# Initialize the population for the evolutionary algorithm
def initialize_population(pop_size, peptide_length):
    return [generate_random_peptide(peptide_length) for _ in range(pop_size)]

# Fitness function to evaluate peptide performance
def fitness_function(peptide, tokenizer, model):
    global output_directory, peptide_scores
    
    if peptide in peptide_scores:
        return peptide_scores[peptide]
    else:
        # Remove output directory if it exists
        try:
            if os.path.isdir(output_directory):
                shutil.rmtree(output_directory)
            else:
                os.remove(output_directory)
        except Exception as e:
            print(f"Error while removing output directory: {e}")
        
        make_esm_prediction(peptide, tokenizer, model)
        criterion = process()
        
        output_directory = os.path.join(os.getenv('SCRATCHDIR'), 'temp')
        os.makedirs(output_directory, exist_ok=True)
        
        # Calculate fitness score
        pae_mean = criterion[0]
        max_dist = criterion[1]
        mean_dist = criterion[2]
        deformation = criterion[4]
        
        J = 0.4 * (15 / max_dist) + 0.3 * (12 / mean_dist) + 0.2 * (20 / pae_mean) + 0.1 * (15 / deformation)
        peptide_scores[peptide] = J
        return J

# Select parents using tournament selection
def select_parents(population, fitness_scores, num_parents, tournament_size=5):
    selected_parents = []
    for _ in range(num_parents):
        tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
        tournament_winner = max(tournament, key=lambda x: x[1])
        selected_parents.append(tournament_winner[0])
    return selected_parents

# Perform crossover between two parents to produce offspring
def crossover(parent1, parent2, mutation_rate=0.5):
    crossover_point = random.randint(1, len(parent1) - 1)
    if random.uniform(0, 1) < mutation_rate:
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    else:
        offspring1 = parent1
        offspring2 = parent2
    return offspring1, offspring2

# Mutate the peptide sequence with a given mutation rate
def mutate(peptide, mutation_rate):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    peptide_list = list(peptide)
    for i in range(len(peptide_list)):
        if random.uniform(0, 1) < mutation_rate:
            peptide_list[i] = random.choice(amino_acids)
    return ''.join(peptide_list)

# Create a new population from parents and offspring
def create_new_population(parents, offspring):
    return parents + offspring

# Run the evolutionary algorithm
def evolutionary_algorithm(pop_size, peptide_length, num_generations, num_parents, mutation_rate, crossover_rate, tokenizer, model):
    global results_directory
    population = initialize_population(pop_size, peptide_length)
    output_csv_path = os.path.join(results_directory, 'evolution_results.csv')
    
    with open(output_csv_path, mode='w', newline='') as csv_file:
        fieldnames1 = ['Generation', 'Best_Fitness', 'Average_Fitness']
        writer1 = csv.DictWriter(csv_file, fieldnames=fieldnames1)
        writer1.writeheader()

    for generation in range(num_generations):
        print(f"Generation: {generation}")
        fitness_scores = [fitness_function(peptide, tokenizer, model) for peptide in population]
        
        best_fitness = max(fitness_scores)
        average_fitness = sum(fitness_scores) / len(fitness_scores)
        
        print(f"Best Fitness: {best_fitness}")
        print(f"Average Fitness: {average_fitness}")
        
        with open(output_csv_path, mode='a', newline='') as csv_file:
            writer1.writerow({'Generation': generation, 'Best_Fitness': best_fitness, 'Average_Fitness': average_fitness})
        
        parents = select_parents(population, fitness_scores, num_parents)
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                offspring1, offspring2 = crossover(parents[i], parents[i + 1], crossover_rate)
                offspring.extend([offspring1, offspring2])
        
        mutated_offspring = [mutate(child, mutation_rate) for child in offspring]
        
        with open(os.path.join(results_directory, f'gen_{generation}.csv'), mode='w', newline='') as csv_file:
           
import random
import shutil
from transformers import AutoTokenizer, EsmForProteinFolding
import csv,random,re,os
import numpy as np
import torch
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from scipy.special import softmax
import evolutionary_algorithm_processor
from evolutionary_algorithm_processor import process,load_models













def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs

def convert_outputs_to_pae(output):
	pae = (output["aligned_confidence_probs"][0].cpu().numpy() * np.arange(64)).mean(-1) * 31
	mask = output["atom37_atom_exists"][0,:,1] == 1
	mask = mask.cpu()
	pae = pae[mask,:][:,mask]       
	return pae

def init_esmfold():
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("/storage/plzen4-ntis/home/jrx99/esmfold/", low_cpu_mem_usage=True)
    return tokenizer,model
def make_esm_prediction(peptide,tokenizer,model):
    global output_directory
    
    model = model.cuda()
    model.esm = model.esm.half()
    
    torch.backends.cuda.matmul.allow_tf32 = True

    luciferase="KPTENNEDFNIVAVASNFATTDLDADRGKLPGKKLPLEVLKEMEANARKAGCTRGCLICLSHIKCTPKMKKFIPGRCHTYEGDKESAQGGIGEAIVDIPEIPGFKDLEPMEQFIAQVDLCVDCTTGCLKGLANVQCSDLLKKWLPQRCATFASKIQGQVDKIKGAGGD"
    linker1="GSSGSSGSSGSSGSSGSS"
    linker2="GGGGGGGGGGGGGGGGGG"
    # Create a CSV file and write data to it

    
    sequence=luciferase+linker2+peptide

    num_recycles = 3 #@param ["0", "1", "2", "3", "6", "12", "24"] {type:"raw"}
        
    ID = peptide
    seqs = sequence.split(":")
    lengths = [len(s) for s in seqs]
    length = sum(lengths)        
    u_seqs = list(set(seqs))
    if len(seqs) == 1: mode = "mono"
    elif len(u_seqs) == 1: mode = "homo"
    else: mode = "hetero"
        
    torch.cuda.empty_cache()
    tokenized_input = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)['input_ids']
    tokenized_input = tokenized_input.cuda()
        # optimized for Tesla T4
        

        
    with torch.no_grad():
        output = model(tokenized_input)
        
    pdb = convert_outputs_to_pdb(output)
    pae = convert_outputs_to_pae(output)
    
    scratchdir = os.getenv('SCRATCHDIR')
    if not scratchdir:
        raise EnvironmentError('SCRATCHDIR environment variable is not set')
    
    # Define paths
        
    # Create the directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
        
    # Change permissions to allow writing
    os.system(f"chmod a+w {output_directory}")

    # Define the directory path
    directory_path = output_directory
        
        # Saving files
    np.savetxt(f"{directory_path}/{ID}.pae.txt", pae, "%.3f")
    for idx, pdb_string in enumerate(pdb):
        with open(f"{directory_path}/{ID}_{idx}.pdb", "w") as out:
            out.write(pdb_string)





def generate_random_peptide(length):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    return 'LYEYAFNAWYILFAH'

def initialize_population(pop_size, peptide_length):
    return [generate_random_peptide(peptide_length) for _ in range(pop_size)]



def fitness_function(peptide,tokenizer,model):
    global output_directory,peptide_scores
    if peptide in peptide_scores:
       return peptide_scores[peptide]
    else:
       try:
           # Remove the output directory and its contents
           if os.path.isdir(output_directory):
               shutil.rmtree(output_directory)
           else:
               os.remove(output_directory)
       except Exception as e:
           print(f"Error while removing output directory: {e}")


       make_esm_prediction(peptide,tokenizer,model)
       criterion = process()
       output_directory = os.path.join(scratchdir, 'temp')
        
       # Create the directory if it doesn't exist
    
       os.makedirs(output_directory, exist_ok=True)
       pae_mean=criterion[0]
       max_dist=criterion[1]
       mean_dist=criterion[2]
       deformation=criterion[4]

       J= 0.4*(15/max_dist)+0.3*(12/mean_dist)+0.2*(20/pae_mean)+0.1*(15/deformation)
       peptide_scores[peptide] = J
       return J

    
    
    





def select_parents(population, fitness_scores, num_parents, tournament_size=5):
    selected_parents = []
    for _ in range(num_parents):
        tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
        tournament_winner = max(tournament, key=lambda x: x[1])
        selected_parents.append(tournament_winner[0])
    return selected_parents


def crossover(parent1, parent2,mutation_rate=0.5):
    crossover_point = random.randint(1, len(parent1) - 1)#oprav rozdílnou délku peptidu
    if random.uniform(0, 1) < mutation_rate:
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    else:
        offspring1 = parent1
        offspring2 = parent2
    return offspring1, offspring2



def mutate(peptide, mutation_rate):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    peptide_list = list(peptide)
    for i in range(0,len(peptide_list)):
        if random.uniform(0, 1) < mutation_rate:
            peptide_list[i] = random.choice(amino_acids)
    return ''.join(peptide_list)



def create_new_population(parents, offspring):
    return parents + offspring

# Example usage



def evolutionary_algorithm(pop_size, peptide_length, num_generations, num_parents, mutation_rate,crossover_rate, tokenizer,model):
    global results_directory
    population = initialize_population(pop_size, peptide_length)
    output_csv_path = os.path.join(results_directory, 'evolution_results.csv')
    with open(output_csv_path, mode='w', newline='') as csv_file:
               fieldnames1 = ['Generation', 'Best_Fitness', 'Average_Fitness']
               writer1 = csv.DictWriter(csv_file, fieldnames=fieldnames1)
               writer1.writeheader()
    # Open the CSV file in write mode
    
    for generation in range(num_generations):
            print("Generation: "+ str(generation))
            fitness_scores = [fitness_function(peptide,tokenizer,model) for peptide in population]
            # Record the best and average fitness scores
            best_fitness = max(fitness_scores)
            print("BF: ",best_fitness)
            average_fitness = sum(fitness_scores) / len(fitness_scores)
            print("AF: ",average_fitness)
            with open(output_csv_path, mode='a', newline='') as csv_file:
               fieldnames1 = ['Generation', 'Best_Fitness', 'Average_Fitness']
               writer1 = csv.DictWriter(csv_file, fieldnames=fieldnames1)
               writer1.writerow({'Generation': generation, 'Best_Fitness': best_fitness, 'Average_Fitness': average_fitness})
            parents = select_parents(population, fitness_scores, num_parents)
            offspring = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    offspring1, offspring2 = crossover(parents[i], parents[i + 1],crossover_rate)
                    offspring.extend([offspring1, offspring2])
            mutated_offspring = [mutate(child, mutation_rate) for child in offspring]
            with open(os.path.join(results_directory, 'gen_'+str(generation)+'.csv'), mode='w', newline='') as csv_file:
               fieldnames2 = ['Peptide', 'Fitness']
               writer2 = csv.DictWriter(csv_file, fieldnames=fieldnames2)
               writer2.writeheader()

               for peptide, fitness in zip(population, fitness_scores):
                  writer2.writerow({'Peptide': peptide, 'Fitness': fitness})

            population = create_new_population(parents, mutated_offspring)
            
    final_population_path = os.path.join(results_directory, 'final_population.csv')
    with open(final_population_path, mode='w', newline='') as csv_file:
            fieldnames3 = ['Peptide', 'Fitness']
            writer3 = csv.DictWriter(csv_file, fieldnames=fieldnames3)
            writer3.writeheader()
            fitness_scores = [fitness_function(peptide, tokenizer, model) for peptide in population]
            for peptide, fitness in zip(population, fitness_scores):
                writer3.writerow({'Peptide': peptide, 'Fitness': fitness})

# Example usage


if __name__ == "__main__":

    # processor parameters
    mode = 'end'
    linker="GGGGGGGGGGGGGGGGGG"
    peptide_scores = {}
    # Example usage
    population_size = 2000
    peptide_length = 50
    num_generations = 50
    crossover_rate=0.5
    mutation_rate=0.05
    num_parents=400

    scratchdir = os.getenv('SCRATCHDIR')
    if not scratchdir:
        raise EnvironmentError('SCRATCHDIR environment variable is not set')
    
    # Define paths
    input_folder = os.path.join(scratchdir, 'input')
    output_directory = os.path.join(scratchdir, 'temp')
    results_directory = os.path.join(scratchdir, 'results')
    
    csv_file_path = os.path.join(output_directory, 'g-end.csv')
    pred_protein_template = load_models(os.path.join(input_folder,'pdb_template'),template=True)




    tokenizer,model=init_esmfold()
    evolutionary_algorithm(population_size, peptide_length, num_generations, num_parents, mutation_rate,crossover_rate,tokenizer,model)
    
