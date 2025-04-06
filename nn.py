import os
import torch
import torch.nn as nn
import pandas as pd
from embeddings import sequence_from_pdb, prottrans_embeddings

# Define a simple encoder that processes per-residue embeddings.
class ProteinEncoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256):
        super(ProteinEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x, lengths):
        mask = torch.arange(x.size(1), device=x.device).expand(len(lengths), x.size(1)) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()
        sum_x = (x * mask).sum(dim=1)
        avg_x = sum_x / lengths.unsqueeze(1).float()
        return self.fc(avg_x)

# Define the twin network that shares the same encoder for both inputs.
class TwinNetwork(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256):
        super(TwinNetwork, self).__init__()
        self.encoder = ProteinEncoder(input_dim, hidden_dim)
    
    def forward(self, emb1, length1, emb2, length2):
        z1 = self.encoder(emb1, length1)
        z2 = self.encoder(emb2, length2)
        z1_norm = z1 / (z1.norm(dim=1, keepdim=True) + 1e-8)
        z2_norm = z2 / (z2.norm(dim=1, keepdim=True) + 1e-8)
        cosine_sim = (z1_norm * z2_norm).sum(dim=1)
        return cosine_sim

def load_chain_ids(chain_ids_file):
    """Read the chain ids file and return a dictionary mapping pdb filenames to chain ids."""
    chain_dict = {}
    with open(chain_ids_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pdb_name, chain_id = parts[0], parts[1]
                chain_dict[pdb_name] = chain_id
    return chain_dict

def load_embedding(pdb_file, chain_id):
    """Load a PDB file, extract the sequence using the given chain id, and compute the embedding."""
    sequence = sequence_from_pdb(pdb_file, chain_id)
    if not sequence:
        print(f"Warning: No sequence extracted from {pdb_file} using chain {chain_id}. Skipping.")
        return None, None
    emb = torch.tensor(prottrans_embeddings(sequence))
    length = torch.tensor([emb.shape[0]])
    emb = emb.unsqueeze(0)  # Add batch dimension.
    return emb, length

if __name__ == "__main__":
    pdb_folder = "pdbs"
    chain_ids_file = "chain_ids.txt"

    # Load chain ids mapping.
    chain_ids_map = load_chain_ids(chain_ids_file)

    # Get list of PDB files from the folder.
    pdb_files = [os.path.join(pdb_folder, f) for f in os.listdir(pdb_folder) if f.endswith('.pdb')]

    if not pdb_files:
        print("No PDB files found in the folder.")
        exit(1)

    # Define architecture parameters as used during training.
    input_dim = 1024
    hidden_dim = 256

    # Instantiate and load the pre-trained model.
    model = TwinNetwork(input_dim=input_dim, hidden_dim=hidden_dim)
    state_dict = torch.load("twin_network_model.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    # Precompute embeddings for all PDB files.
    embeddings = {}
    for pdb_file in pdb_files:
        pdb_basename = os.path.basename(pdb_file)  # Extract filename
        pdb_name = os.path.splitext(pdb_basename)[0]  # Remove .pdb extension
        chain_id = chain_ids_map.get(pdb_basename, None)
        if chain_id is None:
            print(f"Warning: Chain ID not found for {pdb_basename}. Skipping.")
            continue
        emb, length = load_embedding(pdb_file, chain_id)
        if emb is not None:
            embeddings[pdb_name] = (emb, length)  # Store by cleaned name

    if not embeddings:
        print("No valid embeddings computed. Exiting.")
        exit(1)

    # Create an empty similarity matrix.
    file_names = list(embeddings.keys())
    similarity_matrix = pd.DataFrame(index=file_names, columns=file_names)

    # Compute pairwise cosine similarity.
    with torch.no_grad():
        for i, name_i in enumerate(file_names):
            emb_i, len_i = embeddings[name_i]
            for j, name_j in enumerate(file_names):
                emb_j, len_j = embeddings[name_j]
                sim = model(emb_i, len_i, emb_j, len_j)
                similarity_matrix.loc[name_i, name_j] = sim.item()

    # Save the similarity matrix to a CSV file.
    similarity_matrix.to_csv("pairwise_cosine_similarity.csv", index=True)
    print("Pairwise cosine similarity matrix saved to 'pairwise_cosine_similarity.csv'.")
