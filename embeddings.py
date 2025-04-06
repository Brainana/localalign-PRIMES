import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from Bio.PDB import PDBParser, PPBuilder

# Load ProtBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = AutoModel.from_pretrained("Rostlab/prot_bert")

def sequence_from_pdb(pdb_file, chain_id):
    """Extracts amino acid sequence from a PDB file for a given chain."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    for model in structure:
        for chain in model:
            if chain.get_id() == chain_id:
                ppb = PPBuilder()
                sequence = ''.join([str(pp.get_sequence()) for pp in ppb.build_peptides(chain)])
                return str(sequence)
    return None

def prottrans_embeddings(sequence):
    """Generates ProtBERT embeddings for a given protein sequence."""
    formatted_seq = " ".join(list(sequence))  # Format sequence for ProtBERT
    encoded_input = tokenizer(formatted_seq, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**encoded_input)

    embeddings = outputs.last_hidden_state.squeeze(0).numpy()  # Shape: [L, hidden_dim]
    return embeddings

def save_embeddings_as_csv(embeddings, output_file):
    """Saves per-residue embeddings as a CSV file."""
    df = pd.DataFrame(embeddings)
    df.to_csv(output_file, index=False)
    print(f"Embeddings saved to {output_file}")

def read_chain_ids(chain_file):
    """Reads chain IDs from a file and returns a dictionary mapping PDB files to chain IDs."""
    chain_dict = {}
    with open(chain_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                pdb_file, chain_id = parts
                chain_dict[pdb_file] = chain_id
    return chain_dict

def process_pdb(pdb_folder, output_folder, chain_file):
    """Processes all PDB files using chain IDs from a file, extracts embeddings, and saves them."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    chain_dict = read_chain_ids(chain_file)
    all_embeddings = {}

    for pdb_file, chain_id in chain_dict.items():
        pdb_path = os.path.join(pdb_folder, pdb_file)
        if not os.path.exists(pdb_path):
            print(f"Skipping {pdb_file} (File not found).")
            continue
        
        sequence = sequence_from_pdb(pdb_path, chain_id)
        if sequence:
            print(f"Processing {pdb_file} (Chain {chain_id})... Sequence Length: {len(sequence)}")
            embeddings = prottrans_embeddings(sequence)
            output_file = os.path.join(output_folder, f"{pdb_file[:-4]}.csv")
            save_embeddings_as_csv(embeddings, output_file)
            all_embeddings[pdb_file] = embeddings
        else:
            print(f"Skipping {pdb_file} (No valid sequence found for chain {chain_id}).")

if __name__ == "__main__":
    pdb_folder = "pdbs"  
    output_folder = "embeddings"  
    chain_file = "chain_ids.txt"  

    process_pdb(pdb_folder, output_folder, chain_file)
