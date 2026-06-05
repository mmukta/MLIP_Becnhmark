import os
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

# ==========================================
# 1. Dataset Preparation & Tokenization
# ==========================================

# A simple character-level tokenizer for SMILES
class SmilesTokenizer:
    def __init__(self, smiles_list):
        chars = set(''.join(smiles_list))
        self.char_to_id = {c: i+4 for i, c in enumerate(sorted(chars))}
        self.char_to_id['<PAD>'] = 0
        self.char_to_id['<BOS>'] = 1
        self.char_to_id['<EOS>'] = 2
        self.char_to_id['<UNK>'] = 3
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
        
    def encode(self, smiles, max_len=100):
        tokens = [self.char_to_id.get(c, self.char_to_id['<UNK>']) for c in smiles]
        tokens = [self.char_to_id['<BOS>']] + tokens + [self.char_to_id['<EOS>']]
        if len(tokens) < max_len:
            tokens += [self.char_to_id['<PAD>']] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len-1] + [self.char_to_id['<EOS>']]
        return tokens

# Discretizer for Cell Parameters
class CellDiscretizer:
    def __init__(self, num_bins=100):
        self.num_bins = num_bins
        # Standard ranges for organic crystals
        self.ranges = {
            'a': (2.0, 40.0),
            'b': (2.0, 40.0),
            'c': (2.0, 40.0),
            'alpha': (40.0, 140.0),
            'beta': (40.0, 140.0),
            'gamma': (40.0, 140.0)
        }
        
    def encode(self, param_name, value):
        min_val, max_val = self.ranges[param_name]
        clamped = max(min(value, max_val), min_val)
        norm = (clamped - min_val) / (max_val - min_val)
        bin_idx = int(norm * (self.num_bins - 1))
        return bin_idx
        
    def decode(self, param_name, bin_idx):
        min_val, max_val = self.ranges[param_name]
        norm = bin_idx / (self.num_bins - 1)
        return min_val + norm * (max_val - min_val)

class TransformerCrystalDataset(Dataset):
    def __init__(self, df, tokenizer, discretizer, max_len=100):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.discretizer = discretizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Encoder Input
        smiles_tokens = torch.tensor(self.tokenizer.encode(row['smiles'], self.max_len), dtype=torch.long)
        
        # Target Sequence Offset map for Unified Vocab
        # 0: <PAD>, 1: <BOS>, 2: <EOS>
        # 3 to 534: Hall numbers (532 classes)
        # 535 to 634: cell bins (100 classes)
        hall_tok = int(row['hall']) - 1 + 3
        a_tok = self.discretizer.encode('a', row['a']) + 535
        b_tok = self.discretizer.encode('b', row['b']) + 535
        c_tok = self.discretizer.encode('c', row['c']) + 535
        alpha_tok = self.discretizer.encode('alpha', row['alpha']) + 535
        beta_tok = self.discretizer.encode('beta', row['beta']) + 535
        gamma_tok = self.discretizer.encode('gamma', row['gamma']) + 535
        
        # Sequence: [<BOS>, hall, a, b, c, alpha, beta, gamma]
        tgt_tokens = torch.tensor([1, hall_tok, a_tok, b_tok, c_tok, alpha_tok, beta_tok, gamma_tok], dtype=torch.long)
        return smiles_tokens, tgt_tokens

# ==========================================
# 2. Model Architecture
# ==========================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class AutoregressiveCrystalTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.d_model = d_model
        
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        src_mask = (src == 0)
        tgt_key_padding_mask = (tgt == 0)
        tgt_causal_mask = self.generate_square_subsequent_mask(tgt.size(1), src.device)
        
        src_emb = self.pos_enc(self.src_emb(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.tgt_emb(tgt) * math.sqrt(self.d_model))
        
        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_causal_mask,
            memory_key_padding_mask=src_mask
        )
        return self.fc_out(out)

# ==========================================
# 3. Training and Evaluation Loops
# ==========================================

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        logits = model(src, tgt_input)
        
        # Flatten logits and targets for CE loss
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        logits = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        total_loss += loss.item()
        
    return total_loss / len(loader)

def predict_top_k(model, smiles_string, tokenizer, discretizer, device, k=3):
    """Autoregressive generation to sample the top K crystal configurations."""
    model.eval()
    src_tokens = tokenizer.encode(smiles_string, 100)
    src = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    results = []
    with torch.no_grad():
        src_emb = model.pos_enc(model.src_emb(src) * math.sqrt(model.d_model))
        memory = model.transformer.encoder(src_emb, src_key_padding_mask=(src == 0))
        
        # Step 1: Predict Hall number
        tgt = torch.tensor([[1]], dtype=torch.long).to(device) # <BOS>
        tgt_emb = model.pos_enc(model.tgt_emb(tgt) * math.sqrt(model.d_model))
        out = model.transformer.decoder(tgt_emb, memory, memory_key_padding_mask=(src == 0))
        logits = model.fc_out(out[:, -1, :])
        
        # Only consider valid Hall tokens (3 to 534)
        hall_logits = logits[0, 3:535]
        hall_probs = F.softmax(hall_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(hall_probs, k)
        
        for prob, hall_idx in zip(top_k_probs, top_k_indices):
            # Start sequence for each top-K Hall number
            curr_tgt = torch.tensor([[1, hall_idx.item() + 3]], dtype=torch.long).to(device)
            seq_prob = prob.item()
            
            # Predict a, b, c, alpha, beta, gamma sequentially
            for step in range(6):
                tgt_emb = model.pos_enc(model.tgt_emb(curr_tgt) * math.sqrt(model.d_model))
                tgt_mask = model.generate_square_subsequent_mask(curr_tgt.size(1), device)
                out = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=(src == 0))
                
                step_logits = model.fc_out(out[:, -1, :])
                
                # Only consider valid cell bin tokens (535 to 634)
                bin_logits = step_logits[0, 535:635]
                best_bin_idx = torch.argmax(bin_logits).item()
                bin_prob = F.softmax(bin_logits, dim=-1)[best_bin_idx].item()
                
                seq_prob *= bin_prob
                next_tok = torch.tensor([[best_bin_idx + 535]], dtype=torch.long).to(device)
                curr_tgt = torch.cat([curr_tgt, next_tok], dim=1)
                
            # Decode tokens back into continuous numbers
            toks = curr_tgt[0].tolist()
            hall_pred = toks[1] - 3 + 1
            a_pred = discretizer.decode('a', toks[2] - 535)
            b_pred = discretizer.decode('b', toks[3] - 535)
            c_pred = discretizer.decode('c', toks[4] - 535)
            al_pred = discretizer.decode('alpha', toks[5] - 535)
            be_pred = discretizer.decode('beta', toks[6] - 535)
            ga_pred = discretizer.decode('gamma', toks[7] - 535)
            
            results.append({
                'hall_number': hall_pred,
                'probability': seq_prob,
                'cell_parameters': [round(p, 3) for p in [a_pred, b_pred, c_pred, al_pred, be_pred, ga_pred]]
            })
            
    results.sort(key=lambda x: x['probability'], reverse=True)
    return results

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    csv_file = "data.csv"
    
    if not os.path.exists(csv_file):
        print(f"{csv_file} not found. Creating a synthetic toy dataset...")
        df = pd.DataFrame({
            'smiles': ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CCO', 'C1=CC=CC=C1', 'C', 'O=C=O'] * 50,
            'hall': np.random.randint(1, 533, 250),
            'a': np.random.uniform(5, 15, 250),
            'b': np.random.uniform(5, 15, 250),
            'c': np.random.uniform(5, 15, 250),
            'alpha': np.random.uniform(80, 100, 250),
            'beta': np.random.uniform(80, 120, 250),
            'gamma': np.random.uniform(80, 100, 250)
        })
        df.to_csv(csv_file, index=False)

    print("Loading data...")
    df = pd.read_csv(csv_file)
    
    # Init Tokenizer & Discretizer
    tokenizer = SmilesTokenizer(df['smiles'].tolist())
    discretizer = CellDiscretizer(num_bins=100)
    
    # Train / Val Split
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    train_dataset = TransformerCrystalDataset(train_df, tokenizer, discretizer, max_len=100)
    val_dataset = TransformerCrystalDataset(val_df, tokenizer, discretizer, max_len=100)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Setup Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    src_vocab_size = tokenizer.vocab_size
    tgt_vocab_size = 3 + 532 + 100  # Pads/SOS/EOS + Halls + Bins = 635
    model = AutoregressiveCrystalTransformer(
        src_vocab_size=src_vocab_size, 
        tgt_vocab_size=tgt_vocab_size,
        d_model=128, nhead=4, num_layers=3
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train Loop
    epochs = 10
    print("\nStarting NLP Transformer Training...")
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d} | Train CE Loss: {train_loss:.4f} | Val CE Loss: {val_loss:.4f}")

    # Inference Test
    print("\nTraining Complete! Testing Autoregressive Inference...")
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    print(f"Sampling top 3 configurations for {test_smiles}:")
    
    predictions = predict_top_k(model, test_smiles, tokenizer, discretizer, device, k=3)
    
    for i, pred in enumerate(predictions, 1):
        print(f"Rank {i}: Hall {pred['hall_number']} | Transformer Authoregressive Prob: {pred['probability']:.2e} | "
              f"Cell [a,b,c,α,β,γ]: {pred['cell_parameters']}")
