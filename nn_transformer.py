import os
import torch
import torch.nn as nn
import pandas as pd

# choose GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TransformerProteinEncoder(nn.Module):
    def __init__(self, D, hidden_dim, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(D, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1000, hidden_dim))  # Max sequence length of 1000
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection and attention
        self.attn_w = nn.Linear(hidden_dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(self, x, lengths):
        # x: [B, L, D], lengths: [B]
        batch_size, max_len = x.size(0), x.size(1)
        
        # Create padding mask for transformer
        padding_mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        
        # Project input to hidden dimension
        x = self.input_projection(x)  # [B, L, H]
        
        # Add positional encoding
        pos_enc = self.positional_encoding[:max_len, :].unsqueeze(0)  # [1, L, H]
        x = x + pos_enc
        
        # Pass through transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)  # [B, L, H]
        
        # Apply attention pooling
        scores = self.attn_w(x).squeeze(-1)  # [B, L]
        scores = scores.masked_fill(padding_mask, -1e4) 
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, L, 1]
        pooled = (x * alpha).sum(1)  # [B, H]
        
        return self.mlp(pooled)  # [B, H]

# twin NN using same encoder for both input sequences
class TwinNetwork(nn.Module):
    def __init__(self, input_dim=1027, hidden_dim=256, num_heads=8, num_layers=2, dropout=0.1):
        super(TwinNetwork, self).__init__()
        self.encoder = TransformerProteinEncoder(input_dim, hidden_dim, num_heads, num_layers, dropout)
    
    def forward(self, emb1, emb2, mask1, mask2):
        len1 = mask1.sum(dim=1)
        len2 = mask2.sum(dim=1)
        z1 = self.encoder(emb1, len1)
        z2 = self.encoder(emb2, len2)
        z1_norm = z1 / (z1.norm(dim=1, keepdim=True) + 1e-8)
        z2_norm = z2 / (z2.norm(dim=1, keepdim=True) + 1e-8)
        return (z1_norm * z2_norm).sum(dim=1)

def load_embeddings(csv_dir: str, device: torch.device, expect_dim: int = 1027):
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
    input_dim = 1027
    hidden_dim = 256

    # instantiate and load pre-trained model
    model = TwinNetwork(input_dim=input_dim, hidden_dim=hidden_dim)
    state_dict = torch.load("tm_score_transformer.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_dir = "testing_data/testing_embeddings"
    embeddings = load_embeddings(csv_dir, device)
                                                 
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

    # save predicted matrix
    base_names = list(embeddings.keys())
    labels = pd.Index([f"{n}.pdb" for n in base_names])
    df = pd.DataFrame(sim_matrix, index=labels, columns=labels)
    df.to_csv("pred_tm_score_matrix_transformer.csv")
    print(f"Computed and saved similarity matrix for {len(names)} structures.") 