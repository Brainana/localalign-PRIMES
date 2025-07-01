# Install dependencies (run this cell once in Colab)
!pip install transformers biopython google-cloud-storage bio --quiet

import os
import io
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from Bio.PDB.PDBParser import PDBParser
from Bio.Data.PDBData import protein_letters_3to1
from google.colab import auth
from google.cloud import storage

auth.authenticate_user()

# === CONFIGURATION ===
GCS_PROJECT = ''
GCS_BUCKET = ''
PDB_PREFIX = ''
EMBEDDINGS_PREFIX = ''
BATCH_SIZE = 4  # Adjust based on GPU memory
# =====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = AutoModel.from_pretrained("Rostlab/prot_bert", use_safetensors=True).to(device)
model.eval()

client = storage.Client(project=GCS_PROJECT)
bucket = client.bucket(GCS_BUCKET)

def extract_sequence_and_calpha_from_string(pdb_string):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', io.StringIO(pdb_string))
    for model in structure or []:
        for chain in model:
            sequence_builder = []
            calpha_coords = []
            for residue in chain:
                if "CA" in residue:
                    res_name = residue.get_resname()
                    if res_name in protein_letters_3to1:
                        sequence_builder.append(protein_letters_3to1[res_name])
                        calpha_coords.append(residue["CA"].get_coord())
            sequence = "".join(sequence_builder)
            if sequence:
                return sequence, np.array(calpha_coords)
            break
    return None, None

def batch_extract_sequences_and_calpha(blob_batch):
    results = []
    for blob in blob_batch:
        pdb_id = os.path.basename(blob.name)[:-4]
        pdb_string = blob.download_as_text()
        sequence, calpha_coords = extract_sequence_and_calpha_from_string(pdb_string)
        results.append((pdb_id, sequence, calpha_coords))
    return results

def batch_prottrans_embeddings(sequences):
    max_seq_len = 300
    # +2 for [CLS] and [SEP]
    max_length = max_seq_len + 2
    formatted_seqs = [" ".join(list(seq)) for seq in sequences]
    encoded_input = tokenizer(
        formatted_seqs,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=max_length
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        outputs = model(**encoded_input)
    return outputs.last_hidden_state.cpu().numpy()  # [batch, seq_len, hidden_dim]

def save_embeddings_to_gcs(embeddings, gcs_path):
    df = pd.DataFrame(embeddings)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    blob = bucket.blob(gcs_path)
    blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')

def process_pdbs_from_gcs_batched(pdb_prefix, embeddings_prefix, batch_size=4):
    batch_blobs = []
    
    for blob in bucket.list_blobs(prefix=pdb_prefix):
        if not blob.name.endswith('.pdb'):
            continue
        pdb_id = os.path.basename(blob.name)[:-4]
        embedding_blob_path = f"{embeddings_prefix}{pdb_id}.csv"
        if bucket.blob(embedding_blob_path).exists():
            print(f"Skipping {pdb_id} (embedding already exists)")
            continue
        batch_blobs.append(blob)
        if len(batch_blobs) == batch_size:
            process_batch(batch_blobs, embeddings_prefix)
            batch_blobs = []
    
    # Process any remaining blobs
    if batch_blobs:
        process_batch(batch_blobs, embeddings_prefix)

def process_batch(batch_blobs, embeddings_prefix):
    max_len = 300
    batch_info = batch_extract_sequences_and_calpha(batch_blobs)
    valid = [(pdb_id, seq, calpha) for pdb_id, seq, calpha in batch_info if seq and calpha is not None]
    if not valid:
        return
    pdb_ids, sequences, calpha_coords_list = zip(*valid)
    try:
        embeddings_batch = batch_prottrans_embeddings(sequences)
    except Exception as e:
        print(f"Batch embedding error: {e}")
        return
    for j, (pdb_id, seq, calpha_coords) in enumerate(valid):
        seq = seq[:max_len]
        calpha_coords = calpha_coords[:max_len]
        emb = embeddings_batch[j][1:len(seq)+1, :]  # [1:len(seq)+1] trims [CLS] and keeps up to len(seq)
        # Pad embeddings and coordinates to max_len
        pad_len = max_len - emb.shape[0]
        if pad_len > 0:
            emb = np.pad(emb, ((0, pad_len), (0, 0)), mode='constant')
            calpha_coords = np.pad(calpha_coords, ((0, pad_len), (0, 0)), mode='constant')
        if emb.shape[0] == calpha_coords.shape[0] == max_len:
            combined_embeddings = np.concatenate((emb, calpha_coords), axis=1)  # shape (300, 1027)
            embedding_blob_path = f"{embeddings_prefix}{pdb_id}.csv"
            save_embeddings_to_gcs(combined_embeddings, embedding_blob_path)
            print(f"Saved embeddings for {pdb_id} to {embedding_blob_path}")
        else:
            print(f"Skipping {pdb_id} due to embedding/coordinate mismatch.")

if __name__ == "__main__":
    process_pdbs_from_gcs_batched(PDB_PREFIX, EMBEDDINGS_PREFIX, BATCH_SIZE) 