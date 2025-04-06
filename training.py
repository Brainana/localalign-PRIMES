import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import itertools
from torch.nn.utils.rnn import pad_sequence
from nn import TwinNetwork  

def load_embeddings(embeddings_folder):
    """
    Load protein embeddings from individual CSV files in a folder.
    Assumes each file is named like 'protein1.pdb.csv'. The key will be 'protein1.pdb'.
    Each CSV is read, and its contents are converted to a tensor.
    Returns a dictionary with embeddings and a dictionary with their lengths.
    """
    embeddings = {}
    lengths = {}
    for file in os.listdir(embeddings_folder):
        if file.endswith(".csv"):
            pdb_name = file[:-4]  # Remove the ".csv" extension
            file_path = os.path.join(embeddings_folder, file)
            df = pd.read_csv(file_path)
            embedding_tensor = torch.tensor(df.values, dtype=torch.float32)
            embeddings[pdb_name] = embedding_tensor
            lengths[pdb_name] = embedding_tensor.size(0)
    return embeddings, lengths

def load_tm_scores(tm_score_csv):
    """
    Load the TM-score matrix from CSV and return a list of training pairs:
    Each pair is a tuple (pdb1, pdb2, tm_score)
    """
    df = pd.read_csv(tm_score_csv, index_col=0)
    pdb_files = df.index.tolist()
    training_pairs = []

    for i in range(len(pdb_files)):
        for j in range(i + 1, len(pdb_files)):  # use only upper triangle (unique pairs)
            pdb1, pdb2 = pdb_files[i], pdb_files[j]
            tm_score = df.iloc[i, j]
            if not pd.isna(tm_score):
                training_pairs.append((pdb1, pdb2, float(tm_score)))
    return pdb_files, training_pairs

def get_training_batch(training_pairs, embeddings, lengths, batch_size):
    """
    Sample a batch from the training pairs.
    Returns padded tensors for embeddings, their lengths, and a tensor for the TM score target.
    """
    indices = np.random.choice(len(training_pairs), batch_size, replace=False)
    emb1_list, emb2_list, target_list = [], [], []
    lengths1, lengths2 = [], []

    for idx in indices:
        pdb1, pdb2, tm_score = training_pairs[idx]
        if pdb1 in embeddings and pdb2 in embeddings:
            emb1_list.append(embeddings[pdb1])
            emb2_list.append(embeddings[pdb2])
            target_list.append(tm_score)
            lengths1.append(lengths[pdb1])
            lengths2.append(lengths[pdb2])
        else:
            print(f"Warning: Missing embedding for {pdb1 if pdb1 not in embeddings else pdb2}")

    # Pad sequences to the length of the longest sequence in the batch
    emb1_padded = pad_sequence(emb1_list, batch_first=True)
    emb2_padded = pad_sequence(emb2_list, batch_first=True)
    print(emb1_padded.shape)
    print(emb1_padded.shape)

    target = torch.tensor(target_list, dtype=torch.float32)
    lengths1 = torch.tensor(lengths1, dtype=torch.int64)
    lengths2 = torch.tensor(lengths2, dtype=torch.int64)

    return emb1_padded, lengths1, emb2_padded, lengths2, target

def train_model(embeddings_folder, tm_score_csv, num_epochs=10, batch_size=16, hidden_dim=256, learning_rate=1e-3):
    """
    Train the twin network using embeddings loaded from individual CSV files
    and their associated TM scores from a TM-score matrix CSV.
    """
    embeddings, lengths = load_embeddings(embeddings_folder)
    pdb_files, training_pairs = load_tm_scores(tm_score_csv)
    
    # Infer input dimension from one of the embeddings
    input_dim = next(iter(embeddings.values())).size(1)
    
    # Instantiate the model, loss function, and optimizer
    model = TwinNetwork(input_dim, hidden_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        emb1, lengths1, emb2, lengths2, target = get_training_batch(training_pairs, embeddings, lengths, batch_size)
        
        optimizer.zero_grad()
        predicted_tm = model(emb1, lengths1, emb2, lengths2)
        loss = criterion(predicted_tm, target)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}")
    
    return model

if __name__ == "__main__":
    embeddings_folder = "embeddings"  # Folder with individual protein embedding CSV files.
    tm_score_csv = "tm_score_matrix.csv"  # Path to your TM-score CSV.
    
    trained_model = train_model(embeddings_folder, tm_score_csv,
                                num_epochs=10,
                                batch_size=16,
                                hidden_dim=256,
                                learning_rate=1e-3)
    
    torch.save(trained_model.state_dict(), "twin_network_model.pth")
