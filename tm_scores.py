from tmtools import tm_align
from tmtools.io import get_structure, get_residue_data
import multiprocessing as mp
import itertools
import os
import numpy as np
import pandas as pd

def process_pair(pdb_pair):
    """Compute the TM score for a pair of PDB structures"""
    struct1 = get_structure(pdb_pair[0])
    struct2 = get_structure(pdb_pair[1])
    
    coords1, seq1 = get_residue_data(next(struct1.get_chains()))
    coords2, seq2 = get_residue_data(next(struct2.get_chains()))
    
    result = tm_align(coords1, coords2, seq1, seq2)
    return (pdb_pair[0], pdb_pair[1], result.tm_norm_chain1)

def compute_tm_matrix(pdb_files, num_processes=8):
    """Compute TM scores for all unique pairs and store in a matrix"""
    pdb_combinations = list(itertools.combinations(range(len(pdb_files)), 2))  # Index pairs
    pdb_index_to_file = {i: pdb_files[i] for i in range(len(pdb_files))}  # Mapping index to filenames
    
    # Run in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_pair, [(pdb_index_to_file[i], pdb_index_to_file[j]) for i, j in pdb_combinations])

    # Create a square matrix filled with NaNs initially
    n = len(pdb_files)
    tm_matrix = np.full((n, n), np.nan)

    # Fill the matrix with computed TM scores
    for pdb1, pdb2, tm_score in results:
        i = pdb_files.index(pdb1)
        j = pdb_files.index(pdb2)
        tm_matrix[i, j] = tm_score
        tm_matrix[j, i] = tm_score  # Symmetric property

    # Fill diagonal with 1.0 (TM score of identical structures)
    np.fill_diagonal(tm_matrix, 1.0)

    return pdb_files, tm_matrix

# Usage
if __name__ == "__main__":
    pdb_folder = "pdbs/" 
    pdb_files = [os.path.join(pdb_folder, f) for f in os.listdir(pdb_folder) if f.endswith('.pdb')]

    # Compute TM matrix using full paths
    pdb_files_full, tm_matrix = compute_tm_matrix(pdb_files)

    # Convert full paths to just the protein names (remove folder and .pdb extension)
    pdb_names = [os.path.splitext(os.path.basename(f))[0] for f in pdb_files_full]

    # Convert to pandas DataFrame for pretty printing
    df = pd.DataFrame(tm_matrix, index=pdb_names, columns=pdb_names)

    # Save to csv
    output_csv = "tm_score_matrix.csv"
    df.to_csv(output_csv)

    # Print matrix
    print(df)
