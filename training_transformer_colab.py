#!/usr/bin/env python3
"""
Google Colab Optimized Training Script for TM-Score Prediction
This script is designed to run efficiently on Google Colab with GPU acceleration.
"""

import os
import torch
import time
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from torch.amp.autocast_mode import autocast
from torch.profiler import profile, record_function, ProfilerActivity
import gc
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    print("⚠️  Google Colab not available, running in local mode")
import zipfile
from pathlib import Path

# Import your model
from nn_transformer import TwinNetwork

# Colab-specific configurations
USE_DRIVE = True  # Set to True to mount Google Drive
DRIVE_PATH = "/content/drive/MyDrive/TM-PRIMES"  # Adjust to your Drive path
LOCAL_PATH = "/content/TM-PRIMES"  # Local Colab path

# Training hyperparameters optimized for Colab
CACHE_FILE = "embeddings_cache.pt"
TM_SCORES_CACHE = "tm_scores_cache.pt"
CHECKPOINT_PATH = "transformer_checkpoint.pth"
NUM_SHARDS = 5  # Reduced for Colab memory constraints

def setup_colab_environment():
    """Setup Colab environment with GPU optimizations"""
    print("🚀 Setting up Google Colab environment...")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Enable GPU optimizations
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        
        # Set memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        # Enable memory efficient attention if available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            print("✅ Using memory efficient attention")
    else:
        print("⚠️  No GPU detected, using CPU")
    
    # Mount Google Drive if requested and in Colab
    if USE_DRIVE and IN_COLAB:
        try:
            drive.mount('/content/drive')
            print(f"✅ Google Drive mounted at /content/drive")
            
            # Create project directory in Drive if it doesn't exist
            drive_project_path = Path(DRIVE_PATH)
            drive_project_path.mkdir(parents=True, exist_ok=True)
            
            # Copy files from Drive to local if they exist
            if drive_project_path.exists():
                print(f"📁 Project directory found at {DRIVE_PATH}")
                # We'll work locally and sync back to Drive
        except Exception as e:
            print(f"⚠️  Failed to mount Drive: {e}")
            print("   Continuing with local storage only")
    elif USE_DRIVE and not IN_COLAB:
        print("⚠️  Google Drive requested but not in Colab environment")
        print("   Continuing with local storage only")
    
    # Set up local working directory
    if IN_COLAB:
        local_project_path = Path(LOCAL_PATH)
        local_project_path.mkdir(parents=True, exist_ok=True)
        os.chdir(local_project_path)
        print(f"📁 Working directory: {os.getcwd()}")
    else:
        print(f"📁 Working directory: {os.getcwd()}")

def download_data_from_drive():
    """Download training data from Google Drive"""
    if not USE_DRIVE or not IN_COLAB:
        print("⚠️  Drive not mounted or not in Colab, using local data")
        return True  # Return True to indicate we should proceed with local data
    
    drive_data_path = Path(DRIVE_PATH)
    
    # Check if data exists in Drive
    required_files = [
        "training_data/training_embeddings",
        "training_data/training_tm_score_matrix.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not (drive_data_path / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files in Drive: {missing_files}")
        return False
    
    # Copy data to local
    print("📥 Copying data from Drive to local...")
    import shutil
    
    for file_path in required_files:
        src = drive_data_path / file_path
        dst = Path(file_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    
    print("✅ Data copied successfully")
    return True

def upload_results_to_drive():
    """Upload training results back to Google Drive"""
    if not USE_DRIVE or not IN_COLAB:
        print("⚠️  Drive not mounted or not in Colab, skipping upload")
        return
    
    print("📤 Uploading results to Drive...")
    import shutil
    
    drive_project_path = Path(DRIVE_PATH)
    local_project_path = Path(LOCAL_PATH)
    
    # Files to upload
    files_to_upload = [
        "transformer_checkpoint.pth",
        "tm_score_transformer.pth",
        "embeddings_cache.pt",
        "tm_scores_cache.pt"
    ]
    
    for file_name in files_to_upload:
        local_file = local_project_path / file_name
        drive_file = drive_project_path / file_name
        
        if local_file.exists():
            shutil.copy2(local_file, drive_file)
            print(f"   ✅ Uploaded {file_name}")
        else:
            print(f"   ⚠️  {file_name} not found locally")
    
    print("✅ Results uploaded to Drive")

def normalize_pdb_name(name: str) -> str:
    """Normalize PDB filename"""
    base = os.path.basename(name)
    for ext in (".csv", ".pdb"):
        if base.endswith(ext):
            base = base[:-len(ext)]
    return base

def load_embeddings(embeddings_folder: str):
    """Load embeddings directly from Drive with Colab-optimized caching"""
    if os.path.exists(CACHE_FILE):
        print(f"📂 Loading cached embeddings from {CACHE_FILE}")
        embeddings, lengths = torch.load(CACHE_FILE, map_location='cpu')
        print(f"✅ Loaded {len(embeddings)} embeddings from cache")
        return embeddings, lengths

    print(f"📂 Loading embeddings directly from Drive: {embeddings_folder}")
    embeddings, lengths, failed = {}, {}, []
    
    def read_one(path):
        try:
            pdb = normalize_pdb_name(path)
            df = pd.read_csv(path)
            if df.empty:
                raise ValueError("empty")
            tensor = torch.from_numpy(df.values).float()
            return pdb, tensor
        except Exception as e:
            failed.append((path, e))
            return None

    # List all CSV files in Drive folder
    drive_embeddings_path = Path(embeddings_folder)
    if not drive_embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings folder not found: {embeddings_folder}")
    
    csv_files = list(drive_embeddings_path.glob("*.csv"))
    print(f"📊 Found {len(csv_files)} CSV files in Drive")

    # Process files in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(csv_files), batch_size):
        batch_files = csv_files[i:i + batch_size]
        print(f"📊 Processing batch {i//batch_size + 1}/{(len(csv_files) + batch_size - 1)//batch_size}")
        
        for file_path in batch_files:
            result = read_one(file_path)
            if result:
                pdb, tensor = result
                embeddings[pdb] = tensor
                lengths[pdb] = tensor.size(0)
        
        # Progress update
        if (i + batch_size) % 500 == 0 or i + batch_size >= len(csv_files):
            print(f"   Processed {min(i + batch_size, len(csv_files))}/{len(csv_files)} files")

    torch.save((embeddings, lengths), CACHE_FILE)
    
    if failed:
        print(f"⚠️  {len(failed)} failures (showing up to 5):")
        for p, e in failed[:5]:
            print(f"   • {os.path.basename(p)}: {e}")
    
    print(f"✅ Loaded {len(embeddings)} embeddings directly from Drive")
    return embeddings, lengths

def load_tm_scores(tm_score_csv: str, device: torch.device, available_embeddings: set | None = None):
    """Load TM scores directly from Drive with caching and filtering for available embeddings"""
    if os.path.exists(TM_SCORES_CACHE):
        print(f"📂 Loading cached TM scores from {TM_SCORES_CACHE}")
        names, matrix = torch.load(TM_SCORES_CACHE, map_location='cpu')
    else:
        print(f"📂 Loading TM scores directly from Drive: {tm_score_csv}")
        df = pd.read_csv(tm_score_csv, index_col=0)
        names = [normalize_pdb_name(n) for n in df.index]
        matrix = torch.from_numpy(df.values).float()
        torch.save((names, matrix), TM_SCORES_CACHE)

    matrix = matrix.to(device)
    mask = ~torch.isnan(matrix)
    tri = torch.triu(mask, diagonal=1)
    idxs = tri.nonzero(as_tuple=False).cpu().numpy()
    scores = matrix[tri].cpu().numpy()

    pairs = [
        (names[i], names[j], float(s))
        for (i, j), s in zip(idxs, scores)
    ]
    
    # Filter pairs to only include proteins with embeddings
    if available_embeddings is not None:
        original_count = len(pairs)
        pairs = [
            (p1, p2, score) for p1, p2, score in pairs
            if p1 in available_embeddings and p2 in available_embeddings
        ]
        filtered_count = len(pairs)
        print(f"📊 Filtered TM-score pairs: {original_count} → {filtered_count} (removed {original_count - filtered_count})")
    
    print(f"✅ Loaded {len(pairs)} TM-score pairs")
    return pairs

class TMScoreDataset(Dataset):
    def __init__(self, pairs, embeddings, lengths):
        self.pairs = pairs
        self.embeddings = embeddings
        self.lengths = lengths

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
    """Collate function for DataLoader"""
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

def train_model_colab(
    embeddings_folder: str = f"{DRIVE_PATH}/training_data/training_embeddings",
    tm_score_csv: str = f"{DRIVE_PATH}/training_data/training_tm_score_matrix.csv",
    num_epochs: int = 5,
    batch_size: int = 128,  # Reduced for Colab memory
    hidden_dim: int = 256,
    learning_rate: float = 5e-4,  # Lowered learning rate for stability
    num_shards: int = NUM_SHARDS,
    num_heads: int = 8,
    num_layers: int = 2,
    dropout: float = 0.1,
    accumulation_steps: int = 2,  # Increased for smaller batches
    num_workers: int = 0,  # Set to 0 for direct Drive access
    clip_grad_norm: float = 1.0, # Add gradient clipping
):
    """Train model with Colab optimizations using direct Drive access"""
    
    # Setup environment
    setup_colab_environment()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Training on device: {device}")
    
    # Initialize mixed precision training
    try:
        scaler = GradScaler() if device.type == 'cuda' else None
        print(f"⚡ Mixed Precision: {'Enabled' if scaler else 'Disabled'}")
    except Exception as e:
        print(f"⚠️  Mixed precision failed to initialize: {e}")
        print("   Continuing with standard precision training")
        scaler = None

    # Load data directly from Drive
    print("📊 Loading training data directly from Drive...")
    emb_dict, lengths = load_embeddings(embeddings_folder)
    available_embeddings = set(emb_dict.keys())
    pairs = load_tm_scores(tm_score_csv, device, available_embeddings)
    input_dim = next(iter(emb_dict.values())).size(1)

    print(f"📈 Training configuration:")
    print(f"   • Input dimension: {input_dim}")
    print(f"   • Hidden dimension: {hidden_dim}")
    print(f"   • Number of pairs: {len(pairs)}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Accumulation steps: {accumulation_steps}")
    print(f"   • Effective batch size: {batch_size * accumulation_steps}")

    shard_size = (len(pairs) + num_shards - 1) // num_shards
    
    # Create model
    model = TwinNetwork(
        input_dim=input_dim, 
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    # Use DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        print(f"🔄 Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Load checkpoint if exists
    start_epoch = 1
    if os.path.exists(CHECKPOINT_PATH):
        print(f"📂 Loading checkpoint from {CHECKPOINT_PATH}")
        chk = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(chk["model_state_dict"])
        optimizer.load_state_dict(chk["optimizer_state_dict"])
        start_epoch = chk["epoch"] + 1
        print(f"🔄 Resuming from epoch {start_epoch}")

    # Training loop
    print(f"🚀 Starting training for {num_epochs} epochs...")
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n📅 EPOCH {epoch}/{num_epochs}")
        model.train()
        epoch_loss = 0.0
        total_batches = 0

        for sh in range(num_shards):
            start = sh * shard_size
            end = min(start + shard_size, len(pairs))
            shard_pairs = pairs[start:end]
            print(f"  📦 SHARD {sh+1}/{num_shards}, {len(shard_pairs)} pairs")

            ds = TMScoreDataset(shard_pairs, emb_dict, lengths)
            loader = DataLoader(
                ds, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True,
                persistent_workers=False,  # Disable for direct Drive access
                collate_fn=collate_fn
            )

            shard_loss = 0.0
            shard_batches = 0
            t0 = time.time()
            
            for batch_idx, batch in enumerate(loader, 1):
                emb1, emb2, mask1, mask2, tm = batch
                emb1 = emb1.to(device, non_blocking=True)
                emb2 = emb2.to(device, non_blocking=True)
                mask1 = mask1.to(device, non_blocking=True)
                mask2 = mask2.to(device, non_blocking=True)
                tm = tm.to(device, non_blocking=True)

                if torch.isnan(emb1).any() or torch.isnan(emb2).any() or torch.isnan(tm).any():
                    print('NaN detected in batch!')

                # Mixed precision training
                if scaler:
                    with autocast('cuda'):
                        preds = model(emb1, emb2, mask1, mask2)
                        loss = criterion(preds, tm) / accumulation_steps
                    
                    scaler.scale(loss).backward()
                    
                    if batch_idx % accumulation_steps == 0:
                        # Unscale the gradients before clipping
                        scaler.unscale_(optimizer)
                        
                        # Clip the gradients to prevent explosion
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                        
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    preds = model(emb1, emb2, mask1, mask2)
                    loss = criterion(preds, tm) / accumulation_steps
                    loss.backward()
                    
                    if batch_idx % accumulation_steps == 0:
                        # Clip the gradients to prevent explosion
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                        
                        optimizer.step()
                        optimizer.zero_grad()

                shard_loss += loss.item() * accumulation_steps
                shard_batches += 1
                
                # Progress updates
                if batch_idx % 20 == 0:
                    delta = time.time() - t0
                    avg_loss = shard_loss / shard_batches
                    print(f"    📊 Batch {batch_idx}: Loss={avg_loss:.4f}, Time={delta:.1f}s")
                    t0 = time.time()
                
                # Memory management for Colab
                if batch_idx % 100 == 0:
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()

            epoch_loss += shard_loss
            total_batches += shard_batches
            
            print(f"  ✅ Shard {sh+1} complete: Avg Loss = {shard_loss/shard_batches:.4f}")

        # Epoch summary
        avg_epoch_loss = epoch_loss / total_batches
        print(f"📊 EPOCH {epoch} SUMMARY: Average Loss = {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_epoch_loss,
        }, CHECKPOINT_PATH)
        print(f"💾 Checkpoint saved: {CHECKPOINT_PATH}")
        
        # Upload to Drive periodically
        if epoch % 2 == 0:
            upload_results_to_drive()

    # Final save
    torch.save(model.state_dict(), "tm_score_transformer.pth")
    print("💾 Final model saved: tm_score_transformer.pth")
    
    # Upload final results
    upload_results_to_drive()
    
    return model

if __name__ == "__main__":
    print("🎯 Google Colab TM-Score Training (Direct Drive Access)")
    print("=" * 50)
    
    model = train_model_colab(
        embeddings_folder=f"{DRIVE_PATH}/training_data/training_embeddings",
        tm_score_csv=f"{DRIVE_PATH}/training_data/training_tm_score_matrix.csv",
        num_epochs=5,  # Reduced for Colab
        batch_size=64,  # Optimized for Colab memory
        hidden_dim=256,
        learning_rate=5e-4,  # Lowered learning rate for stability
        num_shards=NUM_SHARDS,
        num_heads=8,
        num_layers=2,
        dropout=0.1,
        accumulation_steps=2,
        num_workers=0,  # Important: set to 0 for direct Drive access
        clip_grad_norm=0.5, # Add gradient clipping
    )
    
    print("\n🎉 Training complete!")
    print("📁 Results saved locally and uploaded to Google Drive") 