import os
import itertools
import multiprocessing as mp
import numpy as np
import pandas as pd
import time
from tmtools import tm_align
from tmtools.io import get_structure, get_residue_data

# Global variable (local to subprocesses)
residue_data_cache = {}

def init_worker(shared_dict):
    global residue_data_cache
    residue_data_cache = shared_dict

def preload_residue_data(pdb_files):
    shared_dict = mp.Manager().dict()
    for i, pdb in enumerate(pdb_files):
        try:
            struct = get_structure(pdb)
            coords, seq = get_residue_data(next(struct.get_chains()))
            shared_dict[i] = (coords, seq)
        except Exception as e:
            print(f"Error loading {pdb}: {e}")
            shared_dict[i] = (None, None)
    return shared_dict

def process_pair(i, j):
    try:
        coords1, seq1 = residue_data_cache[i]
        coords2, seq2 = residue_data_cache[j]
        if coords1 is None or coords2 is None:
            raise ValueError("Missing residue data")
        result = tm_align(coords1, coords2, seq1, seq2)
        tm_score = result.tm_norm_chain1
    except Exception as e:
        print(f"Error processing pair index {i}, {j}: {e}")
        tm_score = np.nan
    return i, j, tm_score

def save_matrix_to_csv(tm_matrix, pdb_names, output_csv):
    df_matrix = pd.DataFrame(tm_matrix, index=pdb_names, columns=pdb_names)
    df_matrix.to_csv(output_csv)

def main():
    pdb_folder = "training_pdbs/"
    output_csv = "tm_score_matrix.csv"
    save_every_n_chunks = 10 

    pdb_files = sorted([os.path.join(pdb_folder, f) for f in os.listdir(pdb_folder) if f.endswith('.pdb')])
    pdb_names = [os.path.basename(f) for f in pdb_files]
    num_pdbs = len(pdb_files)
    print(f"Found {num_pdbs} PDB files.")

    pdb_index_to_file = {i: pdb_files[i] for i in range(num_pdbs)}

    # Load existing matrix if available
    if os.path.exists(output_csv):
        print(f"Loading existing TM-score matrix from {output_csv}")
        df_existing = pd.read_csv(output_csv, index_col=0)
        df_existing = df_existing.reindex(index=pdb_names, columns=pdb_names).astype(float)
        tm_matrix = df_existing.to_numpy()  
        # print("Matrix dtype:", tm_matrix.dtype)
        # print("Any NaNs?", np.isnan(tm_matrix).any())
        # print("Number of NaNs:", np.isnan(tm_matrix).sum())
        # print("Sample row with missing values:", tm_matrix[0])
    else:
        tm_matrix = np.full((num_pdbs, num_pdbs), np.nan)

    # Compute remaining pairs only
    all_pairs_remaining = [(i, j) for i, j in itertools.combinations(range(num_pdbs), 2)
                           if np.isnan(tm_matrix[i, j])]
    total_pairs_remaining = len(all_pairs_remaining)
    print(f"Total new comparisons to perform: {total_pairs_remaining}")

    # Preload residue data
    print("Preloading residue data...")
    shared_cache = preload_residue_data(pdb_files)

    # Chunking
    chunk_size = 10000
    num_chunks = (total_pairs_remaining // chunk_size) + (1 if total_pairs_remaining % chunk_size else 0)
    print(f"Processing in {num_chunks} chunks of up to {chunk_size} pairs each.")

    for chunk_index in range(num_chunks):
        start = chunk_index * chunk_size
        end = min(start + chunk_size, total_pairs_remaining)
        current_chunk = all_pairs_remaining[start:end]

        chunk_start = time.perf_counter()
        with mp.Pool(processes=8, initializer=init_worker, initargs=(shared_cache,)) as pool:
            results = pool.starmap(process_pair, current_chunk)
        chunk_end = time.perf_counter()

        elapsed_ms = (chunk_end - chunk_start) * 1000
        print(f"Chunk {chunk_index+1}/{num_chunks} processed in {elapsed_ms:.2f} ms")

        # Update matrix
        for i, j, score in results:
            if not np.isnan(score):
                tm_matrix[i, j] = score
                tm_matrix[j, i] = score

        print(f"Updated matrix after chunk {chunk_index+1}")

        # Periodic saving
        if (chunk_index + 1) % save_every_n_chunks == 0:
            print(f"Saving matrix to {output_csv} after chunk {chunk_index+1}")
            np.fill_diagonal(tm_matrix, 1.0)
            save_matrix_to_csv(tm_matrix, pdb_names, output_csv)

    # Final save
    print("All chunks completed.")
    np.fill_diagonal(tm_matrix, 1.0)
    print(f"Saving final matrix to {output_csv}")
    save_matrix_to_csv(tm_matrix, pdb_names, output_csv)
    print("TM-score matrix computation completed.")

if __name__ == "__main__":
    main()
