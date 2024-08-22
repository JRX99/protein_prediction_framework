#!/usr/bin/env python
# coding: utf-8

"""
File Sorter for ESMFold Runs
This script processes files from large ESMFold runs according to the sequence in chain B of the PDB file.
The chain B sequence corresponds to a random peptide added to GLuc luciferase for complex predictions.
The output of this script is cvv file containing important information about binding
"""

from Bio import PDB
import numpy as np
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import zipfile, os, re, glob, shutil, csv, math, copy, pandas as pd, seaborn as sns, json
from scipy.spatial.transform import Rotation as R

# Dictionary mapping 3-letter amino acid codes to 1-letter codes
amino_acid_short = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E", 
    "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", 
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", 
    "TYR": "Y", "VAL": "V"
}

# Global variables
peptide = ""  # Placeholder for the peptide sequence
protein = ""  # Placeholder for the protein sequence
protein_sequence = "KPTENNEDFNIVAVASNFATTDLDADRGKLPGKKLPLEVLKEMEANARKAGCTRGCLICLSHIKCTPKMKKFIPGRCHTYEGDKESAQGGIGEAIVDIPEIPGFKDLEPMEQFIAQVDLCVDCTTGCLKGLANVQCSDLLKKWLPQRCATFASKIQGQVDKIKGAGGD"
t = len(protein_sequence)  # Length of the protein sequence, used as a reference point
peptide_length = 0  # Length of the peptide sequence
mode = 'end'  # Mode for linker position ('start' or 'end')
linker = "GGGGGGGGGGGGGGGGGG"  # Linker sequence
linker_length = len(linker)  # Length of the linker sequence

# Access SCRATCHDIR environment variable
scratchdir = os.getenv('SCRATCHDIR')
if not scratchdir:
    raise EnvironmentError('SCRATCHDIR environment variable is not set')

# Define paths
input_folder = os.path.join(scratchdir, 'input')
output_directory = os.path.join(scratchdir, 'temp')

def extract_protein_sequence_linker(input_folder):
    """
    Extract and sort protein sequences based on the presence of a linker sequence.

    Args:
    input_folder (str): The path to the input folder containing PDB files.

    Returns:
    None
    """
    linker_position = mode  # Get the mode (start/end) for linker position
    folder_pattern = None  # Initialize the folder pattern

    # Define regex patterns based on the linker and mode
    if mode == "start" and linker == "GSSGSSGSSGSSGSSGSS":
        folder_pattern = re.compile(r'.*_GSS_linker_start$')
    if mode == "end" and linker == "GSSGSSGSSGSSGSSGSS":
        folder_pattern = re.compile(r'.*_GSS_linker_end$')
    if mode == "start" and linker == "GGGGGGGGGGGGGGGGGG":
        folder_pattern = re.compile(r'.*_G_linker_start$')
    if mode == "end" and linker == "GGGGGGGGGGGGGGGGGG":
        folder_pattern = re.compile(r'.*_G_linker_end$')

    pdb_file_pattern = re.compile(r'.+\.pdb$')  # Pattern to match PDB files

    # Walk through the input folder and its subdirectories
    for root, dirs, files in os.walk(input_folder):
        for folder in dirs:
            # Check if the folder name matches the pattern
            if folder_pattern.match(folder):
                folder_path = os.path.join(root, folder)

                # Walk through the matching folder and extract files
                for _, _, filenames in os.walk(folder_path):
                    for filename in filenames:
                        # Check if the file name matches the pattern
                        match = pdb_file_pattern.match(filename)
                        if match:
                            file_path = os.path.join(folder_path, filename)
                            chain_b_sequence = extract_sequence_from_pdb_linker(file_path, linker_position)
                            directory_path = os.path.join(root, chain_b_sequence)

                            # Define a regex pattern to match files to transfer
                            file_to_transfer_pattern = re.compile(r'^sequences_\d+_.*')
                            if not os.path.exists(directory_path):
                                os.makedirs(directory_path)

                                # Transfer files matching the pattern
                                for _, _, filenames in os.walk(folder_path):
                                    for filename in filenames:
                                        match = file_to_transfer_pattern.match(filename)
                                        if match:
                                            file_path = os.path.join(folder_path, filename)
                                            new_file_path = os.path.join(directory_path, filename)
                                            shutil.copyfile(file_path, new_file_path)

def extract_sequence_from_pdb_linker(input_pdb, linker_position="end", template=False):
    """
    Extract peptide and protein sequences from a PDB file based on the linker position.

    Args:
    input_pdb (str): Path to the PDB file.
    linker_position (str): Position of the linker in the sequence (start or end).
    template (bool): If True, extracts the template sequence.

    Returns:
    str: Extracted sequence.
    """
    parser = PDB.PDBParser(QUIET=True)
    global peptide, protein, t
    linker = "GGGGGGGGGGGGGGGGGG"
    structure = parser.get_structure('protein', input_pdb)

    # Assume only one model in the structure
    model = structure[0]
    amino_acids = ''
    
    # Extract sequences from chain B
    for chain in model:
        for residue in chain:
            if PDB.is_aa(residue):
                amino_acids += amino_acid_short[residue.get_resname()]
    
    sequences = amino_acids.split(linker)

    if template:
        peptide = sequences[0]
        return "".join(sequences[0])
    else:
        if linker_position == "start":
            peptide = sequences[0]
            protein = sequences[1]
            return "".join(sequences[0])
        elif linker_position == "end":
            peptide = sequences[1]
            protein = sequences[0]
            return "".join(sequences[1])
        elif linker_position == "multimer":
            protein = amino_acids[:t]
            peptide = amino_acids[t:]
            return "".join(peptide)
        else:
            return 0

def procrustes_analysis(Y, X, t):
    """
    Perform Procrustes analysis to find the optimal rigid transformation (rotation and translation)
    that aligns one set of 3D points (X) to another set of 3D points (Y).

    Args:
    Y (numpy.ndarray): Numpy array of shape (N, 3) representing the second set of 3D points.
    X (numpy.ndarray): Numpy array of shape (N, 3) representing the first set of 3D points.

    Returns:
    tuple: The optimal rotation matrix (3x3) and translation vector (3x1).
    """
    if mode == "start":
        X = X[10:t-15]
        Y = Y[peptide_length + linker_length + 10:len(Y) - 15]
    else:
        X = X[10:t-15]
        Y = Y[10:t-15]

    # Center both sets of points to their respective centroids
    centroid_X = np.mean(X, axis=0)
    centroid_Y = np.mean(Y, axis=0)
    X_centered = X - centroid_X
    Y_centered = Y - centroid_Y

    # Calculate the covariance matrix H
    H = np.dot(X_centered.T, Y_centered)

    # Use Singular Value Decomposition (SVD) to find the optimal rotation matrix
    U, s, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.dot(Vt.T, U.T)))
    R_optimal = np.dot(Vt.T, np.array([[1, 0, 0], [0, 1, 0], [0, 0, d]]))
    R_optimal = np.dot(R_optimal, U.T)

    # Calculate the optimal translation vector
    t_optimal = centroid_Y - np.dot(R_optimal, centroid_X)
    return R_optimal, t_optimal


def get_pdb_file_paths(folder_path):
    """
    Retrieves all PDB file paths from the given directory.

    Parameters:
    - folder_path: The directory containing PDB files.

    Returns:
    - pdb_paths: A list of paths to the PDB files.
    """
    pdb_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pdb"):
                pdb_path = os.path.join(root, file)
                pdb_paths.append(pdb_path)
    return pdb_paths

def error_between_proteins(protein1, protein2):
    """
    Calculate the mean error (distance) between corresponding atoms of two proteins.

    Parameters:
    - protein1: List of atom coordinates of the first protein.
    - protein2: List of atom coordinates of the second protein.

    Returns:
    - Mean distance between corresponding atoms in both proteins.
    """
    distances = [
        math.sqrt(
            (protein1[x][2] - protein2[x][2])**2 +
            (protein1[x][1] - protein2[x][1])**2 +
            (protein1[x][0] - protein2[x][0])**2
        ) for x in range(len(protein1))
    ]
    return np.mean(distances)

def check_missing_residues(pdb_file):
    """
    Check for missing residues in a PDB file.

    Parameters:
    - pdb_file: The path to the PDB file.

    Returns:
    - missing_residues: A list of missing residue numbers.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    
    residue_numbers = [
        residue.id[1]
        for model in structure
        for chain in model
        for residue in chain
    ]
    
    missing_residues = [
        residue_number
        for residue_number in range(1, len(residue_numbers) - 1)
        if residue_number not in residue_numbers
    ]
    return missing_residues

def load_models(path, template=False):
    """
    Load protein models from PDB files in a directory.

    Parameters:
    - path: Directory containing PDB files.
    - template: Boolean flag for template sequence.

    Returns:
    - models: List of protein models represented by their residue centroids.
    """
    models = []
    pdb_paths = get_pdb_file_paths(path)
    pdb_parser = PDB.PDBParser(QUIET=True)
    
    for path in pdb_paths:
        if check_missing_residues(path) == []:
            peptide_string = extract_sequence_from_pdb_linker(path, mode, template)
            peptide_length = len(peptide_string)
            
            prot = pdb_parser.get_structure('protein', path)
            residue_centroids = []

            for model in prot:
                for chain in model:
                    for residue in chain:
                        sum_coord = np.sum([atom.get_coord() for atom in residue], axis=0)
                        n_atoms = len(list(residue))
                        residue_centroids.append(sum_coord / n_atoms)
            models.append(residue_centroids)
        else:
            return "missing"
    return models

def model_variance(models):
    """
    Compute the variance between protein models.

    Parameters:
    - models: List of protein models.

    Returns:
    - model_variance: Matrix of variance between the models.
    """
    results = np.zeros((len(models), len(models)))
    
    for i in range(len(models)):
        for j in range(i, len(models)):
            model_to_transform = copy.deepcopy(models[i])
            R_optimal, t_optimal = procrustes_analysis(np.array(models[j]), np.array(models[i]), t)
            
            for x in range(len(model_to_transform)):
                model_to_transform[x] = np.dot(model_to_transform[x], R_optimal.T) + t_optimal
            
            dist = error_between_proteins(model_to_transform[t:], models[j][t:])
            results[i, j] = dist
            results[j, i] = dist
    
    model_variance = results / (len(models[0]) - t)
    return model_variance

def distance_to_keypoint(model, keypoint):
    """
    Compute the mean distance from a peptide to keypoints in a protein model.

    Parameters:
    - model: Protein model.
    - keypoint: List of keypoint indices.

    Returns:
    - distances: List of distances between the peptide and the keypoints.
    """
    distances = []
    
    if mode == "start":
        luc = model[peptide_length + linker_length:]
        peptide = model[:peptide_length]
    elif mode == "end":
        luc = model[:t]
        peptide = model[t + linker_length:]
    else:
        luc = model[:t]
        peptide = model[t:]
    
    for x in keypoint:
        distance_to_residue = [
            math.sqrt(
                (luc[x][2] - peptide[y][2])**2 +
                (luc[x][1] - peptide[y][1])**2 +
                (luc[x][0] - peptide[y][0])**2
            ) for y in range(len(peptide))
        ]
        distances.append(distance_to_residue)
    
    return distances

import os
import numpy as np
import math
from Bio import PDB

def compute_distances_to_keypoints(model, whole_peptide=False, whole_cavity=False):
    """
    Compute mean distances to predefined keypoints in a protein model.

    Parameters:
    - model: Protein model represented as a list of coordinates.
    - whole_peptide: Boolean flag to consider all residues in peptide.
    - whole_cavity: Boolean flag to use predefined cavity keypoints.

    Returns:
    - distances: List of mean distances to keypoints.
    """
    distances = []
    protein_sequence = "KPTENNEDFNIVAVASNFATTDLDADRGKLPGKKLPLEVLKEMEANARKAGCTRGCLICLSHIKCTPKMKKFIPGRCHTYEGDKESAQGGIGEAIVDIPEIPGFKDLEPMEQFIAQVDLCVDCTTGCLKGLANVQCSDLLKKWLPQRCATFASKIQGQVDKIKGAGGD"
    t = len(protein_sequence)

    if whole_peptide:
        keypoints = [np.arange(0, t).tolist()]
    elif whole_cavity:
        keypoints = [[13], [60], [61], [64], [65], [76], [77], [78], [117]]
    else:
        keypoints = [
            [65, 78],
            [13, 61],
            [13, 64],
            [61, 78]
        ]

    for keypoint in keypoints:
        distances.append(distance_to_keypoint(model, keypoint))

    return distances

def compute_mean_criterion(models):
    """
    Compute maximum variance and mean distances to keypoints for a set of protein models.

    Parameters:
    - models: List of protein models.

    Returns:
    - return_list: List containing maximum variance and mean distances to keypoints.
    """
    variance = model_variance(models)
    distances = [np.mean(compute_distances_to_keypoints(model), axis=1) for model in models]
    mean_distance = np.mean(np.mean(distances, axis=0), axis=1)
    return [
        np.max(variance),
        mean_distance[0],
        mean_distance[1],
        mean_distance[2],
        mean_distance[3]
    ]

def compute_minimum_distance_criterion(models):
    """
    Compute the maximum and mean of minimum distances to keypoints for a set of protein models.

    Parameters:
    - models: List of protein models.

    Returns:
    - max_distance: Maximum of minimum distances.
    - mean_distance: Mean of minimum distances.
    """
    distances = [np.array(compute_distances_to_keypoints(model, whole_cavity=True)) for model in models]
    max_distance = np.max([np.min(dist, axis=2) for dist in distances])
    mean_distance = np.mean([np.min(dist, axis=2) for dist in distances])
    return max_distance, mean_distance

def find_same_charge_close_proximity(models):
    """
    Find residues of the same charge in close proximity in the models.

    Parameters:
    - models: List of protein models.

    Returns:
    - criterion: Count of same charge residues within 4 Å of each other.
    """
    criterion = 0
    protein_sequence = "KPTENNEDFNIVAVASNFATTDLDADRGKLPGKKLPLEVLKEMEANARKAGCTRGCLICLSHIKCTPKMKKFIPGRCHTYEGDKESAQGGIGEAIVDIPEIPGFKDLEPMEQFIAQVDLCVDCTTGCLKGLANVQCSDLLKKWLPQRCATFASKIQGQVDKIKGAGGD"
    positive_charge = ["R", "H", "K"]
    negative_charge = ["D", "E"]

    for model in models:
        indices_positive = [i for i, aa in enumerate(peptide) if aa in positive_charge]
        indices_negative = [i for i, aa in enumerate(peptide) if aa in negative_charge]

        if indices_positive or indices_negative:
            distances = np.array(compute_distances_to_keypoints(model, whole_peptide=True))[0]

            for x in indices_positive:
                top_3_indices = np.argsort(distances[:, x])[:3]
                criterion += sum(distances[y][x] < 4 for y in top_3_indices if protein_sequence[y] in positive_charge)

            for x in indices_negative:
                top_3_indices = np.argsort(distances[:, x])[:3]
                criterion += sum(distances[y][x] < 4 for y in top_3_indices if protein_sequence[y] in negative_charge)

    return criterion

def compute_pae(file):
    """
    Compute statistical metrics from a PAE file.

    Parameters:
    - file: Path to the PAE file.

    Returns:
    - mean_predicted: Mean of predicted values.
    - std_predicted: Standard deviation of predicted values.
    - range_predicted: Range of predicted values.
    - cv_predicted: Coefficient of variation.
    """
    pae_array = np.loadtxt(file)
    
    if mode == "start":
        peptide_error = pae_array[:peptide_length, peptide_length + linker_length:]
    elif mode == "end":
        peptide_error = pae_array[t + linker_length:, :t]
    else:
        peptide_error = pae_array[t:, :t]

    mean_predicted = np.mean(peptide_error)
    std_predicted = np.std(peptide_error)
    range_predicted = np.max(peptide_error) - np.min(peptide_error)
    cv_predicted = std_predicted / mean_predicted

    return mean_predicted, std_predicted, range_predicted, cv_predicted

def luc_deformation(models):
    """
    Compute deformation of protein models relative to a template.

    Parameters:
    - models: List of protein models.

    Returns:
    - error: Error between the transformed model and the template.
    """
    pred_protein_template = load_models(os.path.join(input_folder, 'pdb_template'), template=True)
    for model in models:
        R_optimal, t_optimal = procrustes_analysis(model, pred_protein_template[0], t)
        model = np.dot(model, R_optimal.T) + t_optimal

        if mode == "end" or mode == "multimer":
            return error_between_proteins(model[10:t - 15], pred_protein_template[0][10:t - 15])
        elif mode == "start":
            global peptide_length, linker_length
            return error_between_proteins(model[peptide_length + linker_length + 10:t - 15], pred_protein_template[0][10:t - 15])

def process_folder(root, folder, mode, linker):
    """
    Process a folder to compute criteria for protein models.

    Parameters:
    - root: Root directory.
    - folder: Folder name.
    - mode: Mode of processing.
    - linker: Linker string.

    Returns:
    - criterion: List of computed criteria.
    """
    folder_path = os.path.join(root, folder)
    models = load_models(folder_path)
    
    if models == "missing":
        return None
    
    criterion = []
    proximity_same_charge = find_same_charge_close_proximity(models)
    max_dist, mean_dist = compute_minimum_distance_criterion(models)

    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            mean_predicted, std_predicted, range_predicted, cv_predicted = compute_pae(os.path.join(folder_path, file))
            criterion.extend([
                mean_predicted,
                max_dist,
                mean_dist,
                proximity_same_charge,
                luc_deformation(models),
                folder.split("_")[0]
            ])
            return criterion

def process():
    """
    Main function to process protein models and compute criteria.

    Returns:
    - Result of process_folder function.
    """
    scratchdir = os.getenv('SCRATCHDIR')
    if not scratchdir:
        raise EnvironmentError('SCRATCHDIR environment variable is not set')
    
    input_folder = os.path.join(scratchdir, 'input')
    output_directory = os.path.join(scratchdir, 'temp')
    
    return process_folder(scratchdir, 'temp', mode, linker)


  


