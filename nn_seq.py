import os
import torch
import torch.nn as nn
import pandas as pd
from protbert_embeddings import sequence_from_pdb, prottrans_embeddings
_ = prottrans_embeddings("M") # make a call to preload the model

# choose GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# encoder for embeddings
class ProteinEncoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256):
        super(ProteinEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x, lengths):
        mask = torch.arange(x.size(1), device=x.device) \
                     .expand(len(lengths), x.size(1)) \
                     < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()
        sum_x = (x * mask).sum(dim=1)
        avg_x = sum_x / lengths.unsqueeze(1).float()
        return self.fc(avg_x)

# twin NN using same encoder for both input sequences
class TwinNetwork(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256):
        super(TwinNetwork, self).__init__()
        self.encoder = ProteinEncoder(input_dim, hidden_dim)
    
    def forward(self, emb1, emb2, mask1, mask2):
        len1 = mask1.sum(dim=1)
        len2 = mask2.sum(dim=1)
        z1 = self.encoder(emb1, len1)
        z2 = self.encoder(emb2, len2)
        z1_norm = z1 / (z1.norm(dim=1, keepdim=True) + 1e-8)
        z2_norm = z2 / (z2.norm(dim=1, keepdim=True) + 1e-8)
        return (z1_norm * z2_norm).sum(dim=1)

def load_chain_ids(chain_ids_file):
    chain_dict = {}
    with open(chain_ids_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pdb_name, chain_id = parts[0], parts[1]
                chain_dict[pdb_name] = chain_id
    return chain_dict

def load_embedding(pdb_file, chain_id):
    sequence = sequence_from_pdb(pdb_file, chain_id)
    if not sequence:
        print(f"Warning: No sequence extracted from {pdb_file} using chain {chain_id}. Skipping.")
        return None, None
    emb = torch.tensor(prottrans_embeddings(sequence), device=device)
    length = torch.tensor([emb.shape[0]], device=device, dtype=torch.long)
    emb = emb.unsqueeze(0)
    return emb, length

def load_embeddings_from_csv_folder(csv_dir: str, device: torch.device, expect_dim: int = 1024):
    embeddings = {}
    for fname in os.listdir(csv_dir):
        if not fname.lower().endswith('.csv'):
            continue
        name = os.path.splitext(fname)[0]
        path = os.path.join(csv_dir, fname)

        # 1) read CSV (no header assumed)
        df = pd.read_csv(path, header=None, skiprows=1, dtype=float)
        arr = df.values  # shape (L, D)
        L, D = arr.shape
        if D != expect_dim:
            raise ValueError(f"File {fname} has {D} dims, expected {expect_dim}")

        # 2) to torch.Tensor on device and add batch dim
        emb = torch.from_numpy(arr).float().to(device).unsqueeze(0)  # (1, L, D)
        length = torch.tensor([L], dtype=torch.long, device=device)

        embeddings[name] = (emb, length)

    print(f"Loaded {len(embeddings)} embeddings from '{csv_dir}'")
    return embeddings

if __name__ == "__main__":
    # pdb_folder = "testing_data/testing_pdbs"
    # chain_ids_file = "chain_mapping.txt"
    # chain_ids_map = load_chain_ids(chain_ids_file)
    # pdb_files = [os.path.join(pdb_folder, f)
    #              for f in os.listdir(pdb_folder) if f.endswith('.pdb')]

    input_dim = 1024
    hidden_dim = 256

    # instantiate and load pre-trained model
    model = TwinNetwork(input_dim=input_dim, hidden_dim=hidden_dim)
    state_dict = torch.load("tm_score_nn.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # # precompute embeddings
    # embeddings = {}
    # for pdb_file in pdb_files:
    #     basename = os.path.basename(pdb_file)
    #     name = os.path.splitext(basename)[0]
    #     chain_id = chain_ids_map.get(basename)
    #     if chain_id is None:
    #         print(f"Warning: Chain ID not found for {basename}. Skipping.")
    #         continue
    #     emb, length = load_embedding(pdb_file, chain_id)
    #     if emb is not None:
    #         embeddings[name] = (emb, length)

    # if not embeddings:
    #     print("No valid embeddings computed. Exiting.")
    #     exit(1)

    # print("All embeddings loaded.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_dir = "testing_data/testing_embeddings"
    embeddings = load_embeddings_from_csv_folder(csv_dir, device)

    # for n, p in model.encoder.named_parameters():
    #     print(n, p.norm().item())
                                                 
    # grab your embedding names and encode+normalize all at once
    model.eval()
    with torch.no_grad():
        encodings = []
        names = list(embeddings.keys())   # one-time list of names
        for name in names:
            emb, length = embeddings[name]
            z = model.encoder(emb, length)            
            z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
            encodings.append(z)

        # stack and do full matrix-multiply on GPU
        Z = torch.cat(encodings, dim=0)     # shape (n, hidden_dim)
        sim_matrix = (Z @ Z.T).cpu().numpy()

    # save df
    base_names = list(embeddings.keys())
    labels = [f"{n}.pdb" for n in base_names]
    df = pd.DataFrame(sim_matrix, index=labels, columns=labels)
    df.to_csv("pred_tm_score_matrix_seq.csv")
    print(f"Computed and saved similarity matrix for {len(names)} structures.")
