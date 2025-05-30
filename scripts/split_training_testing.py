#!/usr/bin/env python3

import os
import shutil
import pandas as pd
import numpy as np

# ====== GLOBAL CONFIGURATION ======
PDB_DIR = "training_pdbs"
EMB_DIR = "protbert_embeddings"
TM_MATRIX_CSV = "tm_score_matrix.csv"
OUTPUT_DIR = "output"
TRAIN_RATIO = 0.8
SEED = 42
# ==================================


def prepare_output_dirs(base_dir):
    """Create train/test subdirectories for PDBs and embeddings."""
    train_pdb = os.path.join(base_dir, "train", "pdb")
    train_emb = os.path.join(base_dir, "train", "emb")
    test_pdb  = os.path.join(base_dir, "test",  "pdb")
    test_emb  = os.path.join(base_dir, "test",  "emb")
    for d in [train_pdb, train_emb, test_pdb, test_emb]:
        os.makedirs(d, exist_ok=True)
    return train_pdb, train_emb, test_pdb, test_emb


def get_ids_from_dir(directory, ext):
    """List files with given extension and return sorted base filenames."""
    files = [f for f in os.listdir(directory) if f.lower().endswith(ext)]
    return sorted(os.path.splitext(f)[0] for f in files)


def split_ids(ids, train_ratio, seed):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(ids))
    split_idx = int(len(ids) * train_ratio)
    train_ids = [ids[i] for i in perm[:split_idx]]
    test_ids  = [ids[i] for i in perm[split_idx:]]
    return train_ids, test_ids


def copy_files(ids, src_dir, dst_dir, extension):
    """Copy files with given IDs and extension from src to destination."""
    for uid in ids:
        src = os.path.join(src_dir, f"{uid}{extension}")
        dst = os.path.join(dst_dir, f"{uid}{extension}")
        if not os.path.isfile(src):
            raise FileNotFoundError(f"File not found: {src}")
        shutil.copy2(src, dst)


def main():
    # 1. Prepare directories
    train_pdb_dir, train_emb_dir, test_pdb_dir, test_emb_dir = prepare_output_dirs(OUTPUT_DIR)

    # 2. Collect base IDs from PDB directory
    base_ids = get_ids_from_dir(PDB_DIR, '.pdb')

    # 3. Split into train/test lists
    train_ids, test_ids = split_ids(base_ids, TRAIN_RATIO, SEED)

    # 4. Copy PDB and embedding files for each set
    copy_files(train_ids, PDB_DIR, train_pdb_dir, '.pdb')
    copy_files(train_ids, EMB_DIR, train_emb_dir, '.csv')
    copy_files(test_ids,  PDB_DIR, test_pdb_dir,  '.pdb')
    copy_files(test_ids,  EMB_DIR, test_emb_dir,  '.csv')

    # 5. Load TM-score matrix and normalize index/columns
    tm_df = pd.read_csv(TM_MATRIX_CSV, index_col=0)
    # Strip '.pdb' suffix from labels to match base_ids
    tm_df.index = tm_df.index.str.replace(r"\.pdb$", "", regex=True)
    tm_df.columns = tm_df.columns.str.replace(r"\.pdb$", "", regex=True)

    # 6. Subset submatrices
    train_tm = tm_df.loc[train_ids, train_ids].copy()
    test_tm  = tm_df.loc[test_ids,  test_ids].copy()

    # 7. Re-append '.pdb' extension to row/column names if desired
    train_tm.index = [f"{uid}.pdb" for uid in train_tm.index]
    train_tm.columns = [f"{uid}.pdb" for uid in train_tm.columns]
    test_tm.index  = [f"{uid}.pdb" for uid in test_tm.index]
    test_tm.columns  = [f"{uid}.pdb" for uid in test_tm.columns]

    # 8. Save TM-score submatrices
    train_tm.to_csv(os.path.join(OUTPUT_DIR, 'train', 'tm_matrix.csv'))
    test_tm.to_csv(os.path.join(OUTPUT_DIR, 'test',  'tm_matrix.csv'))

    print(f"Split complete: {len(train_ids)} training & {len(test_ids)} testing entries.")

if __name__ == '__main__':
    main()
