
from Bio import PDB
import numpy as np
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import zipfile
import os
import re
import glob
import shutil
import csv
import math
from scipy.spatial.transform import Rotation as R
import copy
import pandas as pd
import seaborn as sns
import json
import sys


def extract_protein_sequence_linker(input_folder):
    global mode
    linker_position=mode
    # linker_position="end"
    # Define a regular expression pattern to match folder names
    if mode=="start" and linker=="GSSGSSGSSGSSGSSGSS":
        folder_pattern = re.compile(r'.*_GSS_linker_start$')
    if mode=="end" and linker=="GSSGSSGSSGSSGSSGSS":
        folder_pattern = re.compile(r'.*_GSS_linker_end$')
    if mode=="start" and linker=="GGGGGGGGGGGGGGGGGG":
        folder_pattern = re.compile(r'.*_G_linker_start$')
    if mode=="end" and linker=="GGGGGGGGGGGGGGGGGG":
        folder_pattern = re.compile(r'.*_G_linker_end$')
    

    # Define a regular expression pattern to match file names with a number in the specified range
    pdb_file_pattern = re.compile(r'.+\.pdb$')

    # Walk through the input folder and its subdirectories
    for root, dirs, files in os.walk(input_folder):
        for folder in dirs:
            # Check if the folder name matches the pattern
            if folder_pattern.match(folder):
                folder_path = os.path.join(root, folder)

                # Walk through the matching folder and extract files
                for _, _, filenames in os.walk(folder_path):
                    for filename in filenames:
                        # Check if the file name matches the pattern and the number is in the specified range
                        match = pdb_file_pattern.match(filename)
                        if match:
                            sequence_number = int(match.group(1))
                            file_path=os.path.join(folder_path, filename)
                            chain_b_sequence=extract_sequence_from_pdb_linker(file_path,linker_position)
                            directory_path=os.path.join(root, chain_b_sequence)

                            file_to_transfer_pattern =re.compile(r'^sequences_%d_.*' % sequence_number)

                            if not os.path.exists(directory_path):
                                os.makedirs(directory_path)
                                print(f"Created directory: {directory_path}")
                                for _, _, filenames in os.walk(folder_path):
                                    for filename in filenames:

                                        # Check if the file name matches the pattern and the number is in the specified range
                                        match = file_to_transfer_pattern.match(filename)
                                        if match:
                                            file_path=os.path.join(folder_path, filename)
                                            new_file_path=os.path.join(directory_path, filename)
                                            shutil.copyfile(file_path, new_file_path)

def extract_sequence_from_pdb_linker(input_pdb,linker_position,template=False):
    parser = PDB.PDBParser(QUIET=True)
    global peptide,amino_acid_short
    global protein
    structure = parser.get_structure('protein', input_pdb)

    # Assume only one model in the structure
    model = structure[0]

    # Extract sequences from chain B
    amino_acids = ''
    for chain in model:
        for residue in chain:
            if PDB.is_aa(residue):
                amino_acids += amino_acid_short[residue.get_resname()]
    global linker,t
    sequences=amino_acids.split(linker)
    if template==True:
        peptide=sequences[0]
        return "".join(sequences[0])
    else:
        if linker_position=="start":
            peptide=sequences[0]
            protein=sequences[1]
            return "".join(sequences[0])
        if linker_position=="end":  
            peptide=sequences[1]
            protein=sequences[0]
            return "".join(sequences[1])
        if linker_position=="multimer":  
            protein=amino_acids[0:t]
            peptide=amino_acids[t:]
            return "".join(peptide)
        else: return 0

# Example usage:

#extract_protein_sequence_linker(input_folder)

def procrustes_analysis(Y, X, t):
    global mode, peptide_length,linker_length
    """
    Perform Procrustes analysis to find the optimal rigid transformation (rotation and translation)
    that aligns one set of 3D points (X) to another set of 3D points (Y).

    Args:
    X (numpy.ndarray): Numpy array of shape (N, 3) representing the first set of 3D points.
    Y (numpy.ndarray): Numpy array of shape (N, 3) representing the second set of 3D points.

    Returns:
    numpy.ndarray: The rotation matrix (3x3) and translation vector (3x1) as a tuple.
    """

    if mode == "start":
        X=X[10:t-15]
        Y=Y[peptide_length+linker_length+10:len(Y)-15]
    else:
        X=X[10:t-15]
        Y=Y[10:t-15]

    # Center both sets of points to their respective centroids
    centroid_X = np.mean(X, axis=0)
    centroid_Y = np.mean(Y, axis=0)
    X_centered = X - centroid_X
    Y_centered = Y - centroid_Y

    # Calculate the covariance matrix H
    H = np.dot(X_centered.T, Y_centered)

    # Use Singular Value Decomposition (SVD) to find the optimal rotation matrix
    U, s, Vt = np.linalg.svd(H)
    d=np.sign(np.linalg.det(np.dot(Vt.T,U.T)))
    R_optimal = np.dot(Vt.T,np.array([[1,0,0],[0,1,0],[0,0,d]]))
    R_optimal = np.dot(R_optimal, U.T)
    # Calculate the optimal translation vector
    t_optimal = centroid_Y - np.dot(R_optimal,centroid_X)
    return R_optimal, t_optimal

def get_pdb_file_paths(folder_path):
    pdb_paths = []

    # Iterate through all files and directories in the given folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file has a .pdb extension
            if file.endswith(".pdb"):
                # Construct the full path to the .pdb file and add it to the list
                pdb_path = os.path.join(root, file)
                pdb_paths.append(pdb_path)

    return pdb_paths

def error_between_proteins(protein1,protein2):
  distances=[]
  for x in range(0,len(protein1)):
    distances.append(math.sqrt((protein1[x][2]-protein2[x][2])**2+(protein1[x][1]-protein2[x][1])**2+(protein1[x][0]-protein2[x][0])**2))
  return np.mean(distances)

def check_missing_residues(pdb_file):
    # Parse the PDB file
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)

    # Extract residue numbers
    residue_numbers = []
    for model in structure:
        for chain in model:
            for residue in chain:
                residue_numbers.append(residue.id[1])

    # Check for missing residues
    missing_residues = []
    for residue_number in range(1,len(residue_numbers)-1):
        if residue_number not in residue_numbers:
            missing_residues.append(residue_number)

    return missing_residues

def load_models(path,template=False):
    """
    Load protein models from PDB files.

    Parameters:
    - path: The path to the directory containing PDB files.

    Returns:
    - models: Array of predicted protein models.
    """
    models = []  # Array to store predicted proteins
    pdb_paths = get_pdb_file_paths(path)
    pdb_parser = PDB.PDBParser(QUIET=True)
    
        
        # Iterate through PDB files
    for path in pdb_paths:
            if check_missing_residues(path)==[]:
                peptide_string=extract_sequence_from_pdb_linker(path,mode,template)
                global peptide_length
                peptide_length=len(peptide_string)



                prot = pdb_parser.get_structure('protein', path)
                residue_atoms = []
                residue_centroids = []
                
                # Iterate through the structure hierarchy
                for model in prot:
                    for chain in model:
                        for residue in chain:
                            n = 0
                            sum_coord = [0, 0, 0]
                            
                            # Iterate through atoms in a residue
                            for atom in residue:
                                residue_atoms.append(atom.get_coord())
                                sum_coord += atom.get_coord()
                                n += 1
                            residue_centroids.append(sum_coord / n)
                models.append(residue_centroids)
            else:
                    return "missing"
    return models
            
    
def model_variance(models):
    """
    Compute variance between protein models.

    Parameters:
    - models: Array of protein models.

    Returns:
    - model_variance: Matrix representing variance between models.
    """
    results = np.zeros((len(models), len(models)))
    # Iterate through pairs of models
    for i in range(len(models)):
        for j in range(i,len(models)):
            model_to_transform = copy.deepcopy(models[i])
            R_optimal, t_optimal = procrustes_analysis(np.array(models[j]), np.array(models[i]), t)
            
            # Apply Procrustes transformation to the model
            for x in range(len(model_to_transform)):
                model_to_transform[x] = np.dot(model_to_transform[x], R_optimal.T) + t_optimal
            
            # Compute distance between transformed model and the target model
             
            dist = error_between_proteins(model_to_transform[t:], models[j][t:])
            results[i, j] =dist
            results[j, i] =dist
    
    model_variance = results / (len(models[0]) - t)
    return model_variance

def distance_to_keypoint(model, keypoint):
    """
    Compute the mean distance from luc to keypoints in a protein model.

    Parameters:
    - model: Protein model.
    - keypoint: List of keypoint indices.

    Returns:
    - mean_distance: Mean distance to keypoints.
    """
    global peptide_length
    global linker_length,t
    distances = []
    if mode=="start":
        luc = model[peptide_length+linker_length:]
        peptide = model[0:peptide_length]
    elif mode == "end":
        luc = model[:t]
        peptide = model[t+linker_length:]
    else:
        luc = model[:t]
        peptide = model[t:]
    
    # Iterate through keypoints
    for x in keypoint:
        distance_to_residue = []
        
        # Iterate through residues in the peptide
        for y in range(len(peptide)):
            distance_to_residue.append(math.sqrt((luc[x][2]-peptide[y][2])**2+(luc[x][1] - peptide[y][1])**2 + (luc[x][0] - peptide[y][0])**2))
        distances.append(distance_to_residue)
        
    return distances

def compute_distances_to_keypoints(model,whole_peptide=False,whole_cavity=False):
    """
    Compute mean distances to predefined keypoints in a protein model.

    Parameters:
    - model: Protein model.

    Returns:
    - distances: List of mean distances to keypoints.
    """
    distances = []

    if whole_peptide==True:
        keypoints=np.arange(0,t)
        keypoints=[keypoints.tolist()]
    if whole_cavity==True:
        keypoints=[[13],[60],[61],[64],[65],[76],[77],[78],[117]]
    else:
        keypoints = [
                [65, 78],
                [13, 61],
                [13, 64],
                [61, 78]
                
        ]

    
    # Iterate through keypoints and compute distances
    for x in keypoints:
        distances.append(distance_to_keypoint(model, x))

    return distances

def compute_mean_criterion(models):
    """
    Compute and print maximum variance and mean distances to keypoints for a set of protein models.

    Parameters:
    - models: Array of protein models.
    """
    variance = model_variance(models)
    distances = []
    
    # Iterate through models and compute distances
    for model in models:
        distances.append(np.mean(compute_distances_to_keypoints(model), axis=1))
    mean_distance = np.mean(np.mean(distances, axis=0),axis=1)

    print("Maximum variance:", np.max(variance))
    print("Mean distance to keypoint 1:", mean_distance[0])
    print("Mean distance to keypoint 2:", mean_distance[1])
    print("Mean distance to keypoint 3:", mean_distance[2])
    print("Mean distance to keypoint 4:", mean_distance[3])
    return_list= [np.max(variance),mean_distance[0],mean_distance[1],mean_distance[2],mean_distance[3]]
    return return_list

def compute_minimum_distance_criterion(models):
    """
    Compute and print maximum variance and mean distances to keypoints for a set of protein models.

    Parameters:
    - models: Array of protein models.
    """
    
    distances = []
    
    # Iterate through models and compute distances
    for model in models:
        distances.append(np.array((compute_distances_to_keypoints(model,whole_cavity=True))))
    for x in range(len(models)):
        max=np.max(np.min(distances[x],axis=2))
        mean=np.mean(np.min(distances[x],axis=2))

    return max,mean

def find_same_charge_close_proximity(models):
    for model in models:
        criterion=0
        global peptide, protein_sequence
        positive_charge=["R","H","K"]
        negative_charge=["D","E"]
        
        indices_positive = []
        indices_negative = []
        for i in range(len(peptide)):
            if peptide[i] in positive_charge:
                indices_positive.append(i)
            if peptide[i] in negative_charge:
                indices_negative.append(i)
        print("[PEPTIDE] Number of pos residues: "+str(len(indices_positive))+", number of neg residues: "+ str(len(indices_negative)))
        
        if len(indices_positive)+len(indices_negative)>0:
            distances=np.array(compute_distances_to_keypoints(model,whole_peptide=True))
            distances=distances[0]
            if len(indices_positive)>0:
                for x in indices_positive:
                    sorted_indices = np.argsort(distances[:,x])  # Indices of the sorted array
                    top_3_indices = sorted_indices[:3]   # Indices of the top 3 minimum values
                    for y in top_3_indices:
                        
                        if protein_sequence[y] in positive_charge:
                            if distances[y][x]<4:
                                criterion+=1
                            else:
                                criterion+=0

            if len(indices_negative)>0:
                for x in indices_negative:
                    sorted_indices = np.argsort(distances[:,x])  # Indices of the sorted array
                    top_3_indices = sorted_indices[:3]   # Indices of the top 3 minimum values
                    for y in top_3_indices:
                        if protein_sequence[y] in negative_charge:
                            if distances[y][x]<4:
                             criterion+=1
                            else:
                                criterion+=0
                        

        return criterion

def compute_pae(file):
    # Example usage:
    
    pae_array  = np.loadtxt(file)
    if mode=="start":
        peptide_error=pae_array[:peptide_length,peptide_length+linker_length:]
    elif mode=="end":
        peptide_error=pae_array[t+linker_length:,:t]
    else:
        peptide_error=pae_array[t:,:t]
    print(peptide_error)


        # Mean of Predicted Values
    mean_predicted = np.mean(peptide_error)

        # Standard Deviation of Predicted Values
    std_predicted = np.std(peptide_error)

        # Range of Predicted Values
    range_predicted = np.max(peptide_error) - np.min(peptide_error)

        # Coefficient of Variation (CV)
    cv_predicted = std_predicted / mean_predicted

    return mean_predicted,std_predicted,range_predicted,cv_predicted

def luc_deformation(models):

    for model in models:
        global pred_protein_template,mode
        
        
        R_optimal, t_optimal = procrustes_analysis(model, pred_protein_template[0], t)

    # Apply the transformation matrix to the entire set2
        model = np.dot(model, R_optimal.T)
        model = model + t_optimal
        if mode=="end" or mode == "multimer":
            return error_between_proteins(model[10:t-15],pred_protein_template[0][10:t-15])
        elif mode=="start":
            global peptide_length,linker_length
            return error_between_proteins(model[peptide_length+linker_length+10:t-15],pred_protein_template[0][10:t-15])
    
if __name__ == "__main__":
    

    mode = "end" #start/end/multimer
    peptide=""
    protein=""
    protein_sequence="KPTENNEDFNIVAVASNFATTDLDADRGKLPGKKLPLEVLKEMEANARKAGCTRGCLICLSHIKCTPKMKKFIPGRCHTYEGDKESAQGGIGEAIVDIPEIPGFKDLEPMEQFIAQVDLCVDCTTGCLKGLANVQCSDLLKKWLPQRCATFASKIQGQVDKIKGAGGD"
    amino_acid_short= {

    "ALA":"A",
    "ARG":"R",
    "ASN":"N",
    "ASP":"D",
    "CYS":"C",
    "GLU":"E",
    "GLN":"Q",
    "GLY":"G",
    "HIS":"H",
    "ILE":"I",
    "LEU":"L",
    "LYS":"K",
    "MET":"M",
    "PHE":"F",
    "PRO":"P",
    "SER":"S",
    "THR":"T",
    "TRP":"W",
    "TYR":"Y",
    "VAL":"V"
    }
    arguments = sys.argv
    if len(arguments) > 1:
        original_sequence = arguments[1]
    t = len(protein_sequence)  # Global variable representing the starting point in the models
    peptide_length=0
    linker="GSSGSSGSSGSSGSSGSS" # linker="GSSGSSGSSGSSGSSGSS" linker="GGGGGGGGGGGGGGGGGG"
    linker_length=len(linker)
    input_directory = f"/storage/plzen4-ntis/home/jrx99/esm_mutations/{original_sequence}"
    output_directory = f"/storage/plzen4-ntis/home/jrx99/esm_mutations/{original_sequence}"
    csv_file_path = os.path.join(output_directory, f'{original_sequence}_mutation.csv')
    pred_protein_template = load_models('C:\\Users\\Juryx\\OneDrive\\pdb_template',template=True)
    
        



    for root, dirs, files in os.walk(input_directory):
            for folder in dirs:
                if mode in folder:
                    if mode=="start" and linker=="GSSGSSGSSGSSGSSGSS":
                        folder_pattern = re.compile(r'.*_GSS_linker_start$')
                    if mode=="end" and linker=="GSSGSSGSSGSSGSSGSS":
                        folder_pattern = re.compile(r'.*_GSS_linker_end$')
                    if mode=="start" and linker=="GGGGGGGGGGGGGGGGGG":
                        folder_pattern = re.compile(r'.*_G_linker_start$')
                    if mode=="end" and linker=="GGGGGGGGGGGGGGGGGG":
                        folder_pattern = re.compile(r'.*_G_linker_end$')
                    
                    if folder_pattern.match(folder):
                        with open(csv_file_path, mode='a', newline='') as csv_file:
                            criterion=[]
                            folder_path = os.path.join(root, folder)
                            print(folder_path)
                            
                            models = load_models(folder_path)
                            if models != "missing":
                                #criterion=compute_mean_criterion(models)
                                proximity_same_charge=find_same_charge_close_proximity(models)
                                max,mean=compute_minimum_distance_criterion(models)
                                
                                for file in os.listdir(folder_path):
                                    if file.endswith('.txt'):
                                        mean_predicted,std_predicted,range_predicted,cv_predicted=compute_pae(os.path.join(folder_path, file))
                                        criterion.append(mean_predicted)
                                        criterion.append(max)
                                        criterion.append(mean)
                                        criterion.append(proximity_same_charge)
                                        criterion.append(luc_deformation(models))
                                        criterion.append(folder.split("_")[0])
                                csv_writer = csv.writer(csv_file)
                                csv_writer.writerow(criterion)



  