import os
import pandas as pd
import numpy as np

# --- User-configurable paths ---
tm_score_csv = "tm_score_matrix.csv"
training_pdbs_folder = "training_pdbs"
embeddings_folder = "protbert_embeddings"
embedding_ext = ".csv"
output_csv = "filtered_tm_score_matrix.csv"

def get_basename(filename):
    return os.path.splitext(filename)[0]

def main():
    print(f"[INFO] Loading TM-score matrix from: {tm_score_csv}")
    df = pd.read_csv(tm_score_csv, index_col=0)

    matrix_pdbs = set(df.index)

    print(f"[INFO] Reading training PDBs from: {training_pdbs_folder}")
    pdb_files = {f for f in os.listdir(training_pdbs_folder) if f.endswith(".pdb")}
    pdb_set = set(pdb_files)

    print(f"[INFO] Reading embeddings from: {embeddings_folder}")
    embedding_files = {
        f for f in os.listdir(embeddings_folder) if f.endswith(embedding_ext)
    }
    embedding_basenames = {get_basename(f) for f in embedding_files}

    # Keep only PDBs that exist in both the pdb folder and the embeddings folder
    valid_pdbs = {
        pdb for pdb in matrix_pdbs
        if pdb in pdb_set and get_basename(pdb) in embedding_basenames
    }

    print(f"[INFO] Kept {len(valid_pdbs)} PDBs out of {len(matrix_pdbs)}")

    filtered_pdbs = sorted(valid_pdbs)
    filtered_df = df.loc[filtered_pdbs, filtered_pdbs]

    # --- Symmetry Fix ---
    print("[INFO] Enforcing TM-score symmetry where one side is NaN...")

    values = filtered_df.values
    nan_mask = np.isnan(values)

    # Fix upper triangle -> lower triangle
    i_upper, j_upper = np.triu_indices_from(values, k=1)
    for i, j in zip(i_upper, j_upper):
        if nan_mask[i, j] and not nan_mask[j, i]:
            values[i, j] = values[j, i]
        elif not nan_mask[i, j] and nan_mask[j, i]:
            values[j, i] = values[i, j]

    filtered_df.loc[:, :] = values  # Assign back to DataFrame


    filtered_df.to_csv(output_csv)
    print(f"[SAVE] Filtered matrix saved to: {output_csv}")

if __name__ == "__main__":
    main()
