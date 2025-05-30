#!/usr/bin/env python3
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from nn_seq import TwinNetwork 

# Paths & hyperparams
CACHE_FILE        = "protbert_embeddings_cache.pt"
TM_SCORES_CACHE   = "tm_scores_cache.pt"
CHECKPOINT_PATH   = "seq_train_checkpoint.pth"
NUM_SHARDS        = 10

def normalize_pdb_name(name: str) -> str:
    base = os.path.basename(name)
    for ext in (".csv", ".pdb"):
        if base.endswith(ext):
            base = base[:-len(ext)]
    return base

def load_embeddings(embeddings_folder: str):
    if os.path.exists(CACHE_FILE):
        embeddings, lengths = torch.load(CACHE_FILE)
        print(f"[CACHE] Loaded {len(embeddings)} embeddings.")
        return embeddings, lengths

    embeddings, lengths, failed = {}, {}, []
    def read_one(path):
        try:
            pdb = normalize_pdb_name(path)
            df  = pd.read_csv(path)
            if df.empty:
                raise ValueError("empty")
            tensor = torch.from_numpy(df.values).float()
            return pdb, tensor
        except Exception as e:
            failed.append((path, e))
            return None

    csvs = [os.path.join(embeddings_folder, f)
            for f in os.listdir(embeddings_folder) if f.endswith(".csv")]

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as ex:
        for result in ex.map(read_one, csvs):
            if result:
                pdb, tensor = result
                embeddings[pdb] = tensor
                lengths[pdb]    = tensor.size(0)

    torch.save((embeddings, lengths), CACHE_FILE)
    if failed:
        print(f"[WARN] {len(failed)} failures (showing up to 10):")
        for p, e in failed[:10]:
            print(f"  • {os.path.basename(p)}: {e}")
    return embeddings, lengths

def load_tm_scores(tm_score_csv: str, device: torch.device):
    if os.path.exists(TM_SCORES_CACHE):
        names, matrix = torch.load(TM_SCORES_CACHE, map_location='cpu')
    else:
        df = pd.read_csv(tm_score_csv, index_col=0)
        names = [normalize_pdb_name(n) for n in df.index]
        matrix = torch.from_numpy(df.values).float()
        torch.save((names, matrix), TM_SCORES_CACHE)

    matrix = matrix.to(device)
    mask   = ~torch.isnan(matrix)
    tri    = torch.triu(mask, diagonal=1)
    idxs   = tri.nonzero(as_tuple=False).cpu().numpy()
    scores = matrix[tri].cpu().numpy()

    pairs = [
        (names[i], names[j], float(s))
        for (i, j), s in zip(idxs, scores)
    ]
    print(f"[LOAD] {len(pairs)} TM‐score pairs.")
    return pairs

class TMScoreDataset(Dataset):
    def __init__(self, pairs, embeddings, lengths):
        self.pairs      = pairs
        self.embeddings = embeddings
        self.lengths    = lengths

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, tm = self.pairs[idx]
        if p1 not in self.embeddings or p2 not in self.embeddings:
            missing = p1 if p1 not in self.embeddings else p2
            raise KeyError(f"Missing embedding {missing}")
        return (
            self.embeddings[p1],
            self.lengths[p1],
            self.embeddings[p2],
            self.lengths[p2],
            tm,
        )

def collate_fn(batch):
    emb1_raw, emb2_raw, tms, lengths1, lengths2 = [], [], [], [], []
    for e1, l1, e2, l2, tm in batch:
        emb1_raw.append(e1)
        emb2_raw.append(e2)
        lengths1.append(l1)
        lengths2.append(l2)
        tms.append(tm)

    emb1_pad = pad_sequence(emb1_raw, batch_first=True)
    emb2_pad = pad_sequence(emb2_raw, batch_first=True)

    lens1 = torch.tensor(lengths1, dtype=torch.int64)
    lens2 = torch.tensor(lengths2, dtype=torch.int64)

    max_L1 = emb1_pad.size(1)
    max_L2 = emb2_pad.size(1)
    idx1 = torch.arange(max_L1).unsqueeze(0)
    idx2 = torch.arange(max_L2).unsqueeze(0)

    mask1 = idx1 < lens1.unsqueeze(1)
    mask2 = idx2 < lens2.unsqueeze(1)

    return emb1_pad, emb2_pad, mask1, mask2, torch.tensor(tms, dtype=torch.float32)

def train_model(
    embeddings_folder: str,
    tm_score_csv: str,
    num_epochs: int = 10,
    batch_size: int   = 256,
    hidden_dim: int   = 256,
    learning_rate: float = 1e-3,
    num_shards: int   = NUM_SHARDS,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Device: {device}")

    emb_dict, lengths = load_embeddings(embeddings_folder)
    pairs = load_tm_scores(tm_score_csv, device)
    input_dim = next(iter(emb_dict.values())).size(1)

    shard_size = (len(pairs) + num_shards - 1) // num_shards
    model     = TwinNetwork(input_dim, hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_epoch = 1
    if os.path.exists(CHECKPOINT_PATH):
        chk = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(chk["model_state_dict"])
        optimizer.load_state_dict(chk["optimizer_state_dict"])
        start_epoch = chk["epoch"] + 1
        print(f"[RESUME] from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"[EPOCH] {epoch}/{num_epochs}")
        model.train()

        for sh in range(num_shards):
            running_loss = 0.0
            start = sh * shard_size
            end   = min(start + shard_size, len(pairs))
            shard_pairs = pairs[start:end]
            print(f"  [SHARD] {sh+1}/{num_shards}, {len(shard_pairs)} pairs")

            ds = TMScoreDataset(shard_pairs, emb_dict, lengths)
            loader = DataLoader(
                ds, batch_size=batch_size, shuffle=True,
                num_workers=2, pin_memory=True,
                collate_fn=collate_fn
            )

            t0 = time.time()
            for batch_idx, batch in enumerate(loader, 1):
                emb1, emb2, mask1, mask2, tm = batch
                emb1 = emb1.to(device)
                emb2 = emb2.to(device)
                mask1 = mask1.to(device)
                mask2 = mask2.to(device)
                tm = tm.to(device)

                optimizer.zero_grad()
                preds = model(emb1, emb2, mask1, mask2)
                loss  = criterion(preds, tm)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if batch_idx % 50 == 0:
                    delta = time.time() - t0
                    avg_loss = running_loss / (batch_idx)
                    print(f"    [BATCH {batch_idx}] AvgLoss: {avg_loss:.4f} | {delta:.1f}s")
                    t0 = time.time()

        # avg_loss = running_loss / len(pairs)
        # print(f"[EPOCH {epoch}] Avg Loss: {avg_loss:.4f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, CHECKPOINT_PATH)
        print(f"[CHECKPOINT] Saved epoch {epoch}\n")

    return model

if __name__ == "__main__":
    model = train_model(
        embeddings_folder="training_data/training_embeddings",
        tm_score_csv   ="training_data/training_tm_score_matrix.csv",
        num_epochs=2, batch_size=256,
        hidden_dim=256, learning_rate=1e-3,
        num_shards=NUM_SHARDS,
    )
    torch.save(model.state_dict(), "tm_score_nn.pth")
    print("[SAVE] Model saved to 'tm_score_nn.pth'")
