import os
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator, Descriptors, rdMolDescriptors

EXACT_SMILES_LIBRARY = {}
FINGERPRINT_LIBRARY = []
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
VOLUME_CALIBRATOR = None

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
            'gamma': (40.0, 140.0),
            'volume': (10.0, 5000.0),
            'zprime': (0.0, 16.0)
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

def compute_vol_scalar(a, b, c, alpha, beta, gamma):
    al, be, ga = math.radians(alpha), math.radians(beta), math.radians(gamma)
    term = 1 - math.cos(al)**2 - math.cos(be)**2 - math.cos(ga)**2 + 2 * math.cos(al)*math.cos(be)*math.cos(ga)
    return a * b * c * math.sqrt(max(term, 1e-9))

def morgan_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MORGAN_GENERATOR.GetFingerprint(mol)

def get_mol_descriptor_vector(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array([
        mol.GetNumHeavyAtoms(),
        Descriptors.MolWt(mol),
        rdMolDescriptors.CalcTPSA(mol),
        mol.GetRingInfo().NumRings(),
        1.0,
    ], dtype=float)

def fit_volume_calibrator(df):
    rows_x, rows_y = [], []
    for _, row in df.iterrows():
        x = get_mol_descriptor_vector(row['smiles'])
        if x is None:
            continue
        v = compute_vol_scalar(row['a'], row['b'], row['c'], row['alpha'], row['beta'], row['gamma'])
        rows_x.append(x)
        rows_y.append(v)

    if len(rows_x) < 10:
        return None

    X = np.vstack(rows_x)
    y = np.array(rows_y, dtype=float)
    coef = np.linalg.pinv(X) @ y
    residual = y - X @ coef
    sigma = float(np.std(residual))
    sigma = max(sigma, 150.0)
    return {'coef': coef, 'sigma': sigma}

def volume_plausibility_score(pred_volume, smiles, calibrator):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 1.0
    n_heavy = mol.GetNumHeavyAtoms()
    # Hard floor: assume at least Z'=1 so each heavy atom needs ~15 Å³
    min_vol = max(n_heavy * 15.0, 50.0)
    if float(pred_volume) < min_vol:
        return 0.0
    if calibrator is None:
        return 1.0
    x = get_mol_descriptor_vector(smiles)
    if x is None:
        return 1.0
    est_v = float(x @ calibrator['coef'])
    # Tight sigma so score drops quickly outside the expected volume range
    sigma = float(calibrator['sigma']) / 5.0
    sigma = max(sigma, 50.0)
    z = abs(float(pred_volume) - est_v) / max(sigma, 1e-6)
    return float(math.exp(-0.5 * z * z))

def zprime_plausibility_score(zprime):
    common_values = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0], dtype=float)
    dist = float(np.min(np.abs(common_values - float(zprime))))
    discrete_score = math.exp(-dist / 0.25)
    range_score = 1.0 if 0.25 <= float(zprime) <= 6.0 else 0.3
    return discrete_score * range_score

def candidate_score(probability, zprime, volume, smiles, volume_weight=0.6, zprime_weight=0.3):
    v_score = volume_plausibility_score(volume, smiles, VOLUME_CALIBRATOR)
    z_score = zprime_plausibility_score(zprime)
    return float(probability) * ((1.0 - volume_weight) + volume_weight * v_score) * ((1.0 - zprime_weight) + zprime_weight * z_score)

def build_exact_smiles_library(df):
    library = {}
    for _, row in df.iterrows():
        library.setdefault(row['smiles'], []).append(row.to_dict())
    return library

def build_fingerprint_library(df):
    library = []
    for _, row in df.iterrows():
        fp = morgan_fingerprint(row['smiles'])
        if fp is None:
            continue
        library.append({
            'smiles': row['smiles'],
            'hall': int(row['hall']),
            'zprime': float(row['zprime']),
            'a': float(row['a']),
            'b': float(row['b']),
            'c': float(row['c']),
            'alpha': float(row['alpha']),
            'beta': float(row['beta']),
            'gamma': float(row['gamma']),
            'fp': fp,
        })
    return library

def retrieve_neighbor_candidates(smiles_string, hall_to_spg, top_n=3, exclude_smiles=None):
    query_fp = morgan_fingerprint(smiles_string)
    if query_fp is None or not FINGERPRINT_LIBRARY:
        return []

    exclude_smiles = set(exclude_smiles or [])
    fps = [item['fp'] for item in FINGERPRINT_LIBRARY]
    similarities = DataStructs.BulkTanimotoSimilarity(query_fp, fps)
    ranked_indices = np.argsort(similarities)[::-1]

    results = []
    seen_halls = set()
    for idx in ranked_indices:
        item = FINGERPRINT_LIBRARY[int(idx)]
        if item['smiles'] in exclude_smiles:
            continue
        hall_num = item['hall']
        if hall_num in seen_halls:
            continue
        seen_halls.add(hall_num)

        sim = float(similarities[int(idx)])
        a, b, c, al, be, ga = postprocess_cell_by_hall(
            hall_num, item['a'], item['b'], item['c'], item['alpha'], item['beta'], item['gamma']
        )
        v = compute_vol_scalar(a, b, c, al, be, ga)

        sc = candidate_score(sim, item['zprime'], v, smiles_string)
        if sc <= 0.0:
            continue  # volume too small for this query molecule
        results.append({
            'hall_number': hall_num,
            'spg_number': hall_to_spg.get(hall_num, "Retrieved"),
            'probability': sim,
            'zprime': round(item['zprime'], 3),
            'cell_parameters': [round(p, 3) for p in [a, b, c, al, be, ga]],
            'volume': round(v, 3),
            'score': sc,
        })
        if len(results) >= top_n:
            break

    return results

def snap_angle(angle, target, tol=1.0):
    if abs(angle - target) <= tol:
        return float(target)
    return float(angle)

def postprocess_cell_by_hall(hall_number, a, b, c, alpha, beta, gamma):
    # First, snap angles that are very close to ideal crystallographic values.
    alpha = snap_angle(alpha, 90.0)
    beta = snap_angle(beta, 90.0)
    gamma = snap_angle(gamma, 90.0)
    gamma = snap_angle(gamma, 120.0)

    # Then enforce simple lattice constraints by Hall-number ranges.
    if 3 <= hall_number <= 107:  # Monoclinic
        alpha = 90.0
        gamma = 90.0
    elif 108 <= hall_number <= 348:  # Orthorhombic
        alpha = beta = gamma = 90.0
    elif 349 <= hall_number <= 429:  # Tetragonal
        b = a
        alpha = beta = gamma = 90.0
    elif 430 <= hall_number <= 488:  # Trigonal/Hexagonal
        b = a
        alpha = beta = 90.0
        gamma = 120.0
    elif 489 <= hall_number <= 530:  # Cubic
        avg = (a + b + c) / 3.0
        a = b = c = avg
        alpha = beta = gamma = 90.0

    return a, b, c, alpha, beta, gamma

def postprocess_cell_by_spg(spg_number, a, b, c, alpha, beta, gamma):
    if not isinstance(spg_number, int):
        return a, b, c, alpha, beta, gamma

    alpha = snap_angle(alpha, 90.0)
    beta = snap_angle(beta, 90.0)
    gamma = snap_angle(gamma, 90.0)
    gamma = snap_angle(gamma, 120.0)

    if 3 <= spg_number <= 15:  # Monoclinic
        alpha = 90.0
        gamma = 90.0
    elif 16 <= spg_number <= 74:  # Orthorhombic
        alpha = beta = gamma = 90.0
    elif 75 <= spg_number <= 142:  # Tetragonal
        b = a
        alpha = beta = gamma = 90.0
    elif 168 <= spg_number <= 194:  # Hexagonal
        b = a
        alpha = beta = 90.0
        gamma = 120.0
    elif 195 <= spg_number <= 230:  # Cubic
        avg = (a + b + c) / 3.0
        a = b = c = avg
        alpha = beta = gamma = 90.0

    return a, b, c, alpha, beta, gamma

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
        # 3 to 532: Hall numbers (530 classes)
        # 533 to 632: cell bins (100 classes)
        hall_tok = int(row['hall']) - 1 + 3
        v = compute_vol_scalar(row['a'], row['b'], row['c'], row['alpha'], row['beta'], row['gamma'])
        v_tok = self.discretizer.encode('volume', v) + 533
        z_tok = self.discretizer.encode('zprime', row['zprime']) + 533
        a_tok = self.discretizer.encode('a', row['a']) + 533
        b_tok = self.discretizer.encode('b', row['b']) + 533
        c_tok = self.discretizer.encode('c', row['c']) + 533
        alpha_tok = self.discretizer.encode('alpha', row['alpha']) + 533
        beta_tok = self.discretizer.encode('beta', row['beta']) + 533
        gamma_tok = self.discretizer.encode('gamma', row['gamma']) + 533
        
        # Sequence: [<BOS>, hall, volume, zprime, a, b, c, alpha, beta, gamma]
        tgt_tokens = torch.tensor([1, hall_tok, v_tok, z_tok, a_tok, b_tok, c_tok, alpha_tok, beta_tok, gamma_tok], dtype=torch.long)
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
    
    # Load Hall to Spg mapping
    hall_to_spg = {}
    try:
        hm_df = pd.read_csv("HM_Full.csv")
        for _, r in hm_df.iterrows():
            hall_to_spg[int(r['Hall'])] = int(r['Spg_num'])
    except Exception:
        pass

    exact_results = []
    exact_rows = EXACT_SMILES_LIBRARY.get(smiles_string, [])
    for row in exact_rows:
        hall_num = int(row['hall'])
        a, b, c, al, be, ga = postprocess_cell_by_hall(
            hall_num, float(row['a']), float(row['b']), float(row['c']), float(row['alpha']), float(row['beta']), float(row['gamma'])
        )
        v = compute_vol_scalar(a, b, c, al, be, ga)
        # Exact match: use a large bonus but still apply volume plausibility
        # so a database entry that is physically wrong for the query gets penalised.
        exact_sc = 10.0 + candidate_score(1.0, float(row['zprime']), v, smiles_string)
        exact_results.append({
            'hall_number': hall_num,
            'spg_number': hall_to_spg.get(hall_num, "Known"),
            'probability': 1.0,
            'zprime': round(float(row['zprime']), 3),
            'cell_parameters': [round(p, 3) for p in [a, b, c, al, be, ga]],
            'volume': round(v, 3),
            'score': exact_sc,
        })

    exact_smiles = {row['smiles'] for row in exact_rows if 'smiles' in row}
    neighbor_results = retrieve_neighbor_candidates(smiles_string, hall_to_spg, top_n=3, exclude_smiles=exact_smiles)

    with torch.no_grad():
        src_emb = model.pos_enc(model.src_emb(src) * math.sqrt(model.d_model))
        memory = model.transformer.encoder(src_emb, src_key_padding_mask=(src == 0))
        
        # Step 1: Predict Hall number
        tgt = torch.tensor([[1]], dtype=torch.long).to(device) # <BOS>
        tgt_emb = model.pos_enc(model.tgt_emb(tgt) * math.sqrt(model.d_model))
        out = model.transformer.decoder(tgt_emb, memory, memory_key_padding_mask=(src == 0))
        logits = model.fc_out(out[:, -1, :])
        
        # Only consider valid Hall tokens (3 to 532)
        hall_logits = logits[0, 3:533]
        hall_probs = F.softmax(hall_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(hall_probs, k)
        
        for prob, hall_idx in zip(top_k_probs, top_k_indices):
            # Start sequence for each top-K Hall number
            curr_tgt = torch.tensor([[1, hall_idx.item() + 3]], dtype=torch.long).to(device)
            seq_prob = prob.item()
            
            # Predict volume, zprime, a, b, c, alpha, beta, gamma sequentially
            for step in range(8):
                tgt_emb = model.pos_enc(model.tgt_emb(curr_tgt) * math.sqrt(model.d_model))
                tgt_mask = model.generate_square_subsequent_mask(curr_tgt.size(1), device)
                out = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=(src == 0))
                
                step_logits = model.fc_out(out[:, -1, :])
                
                # Only consider valid cell bin tokens (533 to 632)
                bin_logits = step_logits[0, 533:633]
                best_bin_idx = torch.argmax(bin_logits).item()
                bin_prob = F.softmax(bin_logits, dim=-1)[best_bin_idx].item()
                
                seq_prob *= bin_prob
                next_tok = torch.tensor([[best_bin_idx + 533]], dtype=torch.long).to(device)
                curr_tgt = torch.cat([curr_tgt, next_tok], dim=1)
                
            # Decode tokens back into continuous numbers
            toks = curr_tgt[0].tolist()
            hall_pred = toks[1] - 3 + 1
            v_pred = discretizer.decode('volume', toks[2] - 533)
            z_pred = discretizer.decode('zprime', toks[3] - 533)
            a_pred = discretizer.decode('a', toks[4] - 533)
            b_pred = discretizer.decode('b', toks[5] - 533)
            c_pred = discretizer.decode('c', toks[6] - 533)
            al_pred = discretizer.decode('alpha', toks[7] - 533)
            be_pred = discretizer.decode('beta', toks[8] - 533)
            ga_pred = discretizer.decode('gamma', toks[9] - 533)

            spg_pred = hall_to_spg.get(hall_pred, "Unknown")
            if isinstance(spg_pred, int):
                a_pred, b_pred, c_pred, al_pred, be_pred, ga_pred = postprocess_cell_by_spg(
                    spg_pred, a_pred, b_pred, c_pred, al_pred, be_pred, ga_pred
                )
            else:
                a_pred, b_pred, c_pred, al_pred, be_pred, ga_pred = postprocess_cell_by_hall(
                    hall_pred, a_pred, b_pred, c_pred, al_pred, be_pred, ga_pred
                )

            v_pred = compute_vol_scalar(a_pred, b_pred, c_pred, al_pred, be_pred, ga_pred)
            
            results.append({
                'hall_number': hall_pred,
                'spg_number': spg_pred,
                'probability': seq_prob,
                'zprime': round(z_pred, 3),
                'cell_parameters': [round(p, 3) for p in [a_pred, b_pred, c_pred, al_pred, be_pred, ga_pred]],
                'volume': round(v_pred, 3),
                'score': candidate_score(seq_prob, z_pred, v_pred, smiles_string),
            })
            
    merged = exact_results + neighbor_results + results

    # Deduplicate identical hall+cell entries that can arise from retrieval + generation.
    deduped = []
    seen = set()
    for r in merged:
        key = (r['hall_number'], tuple(r['cell_parameters']))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    deduped.sort(key=lambda x: x.get('score', x['probability']), reverse=True)
    return deduped[:k]

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    csv_file = "data.csv"
    print("Loading data...")
    df = pd.read_csv(csv_file)

    EXACT_SMILES_LIBRARY = build_exact_smiles_library(df)
    FINGERPRINT_LIBRARY = build_fingerprint_library(df)
    VOLUME_CALIBRATOR = fit_volume_calibrator(df)
    
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
    tgt_vocab_size = 3 + 530 + 100  # Pads/SOS/EOS + Halls + Bins = 633
    model = AutoregressiveCrystalTransformer(
        src_vocab_size=src_vocab_size, 
        tgt_vocab_size=tgt_vocab_size,
        d_model=128, nhead=4, num_layers=3
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train Loop
    epochs = 20
    print("\nStarting NLP Transformer Training...")
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d} | Train CE Loss: {train_loss:.4f} | Val CE Loss: {val_loss:.4f}")

    # Inference Test
    print("\nTraining Complete! Testing Autoregressive Inference...")
    test_smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O", 
                   "CN1C=C(N=C1/C=C/c1ccccc1)N(=O)=O",
                   "O=N(=O)C1=NNC(=N1)CC1=NC(=NN1)N(=O)=O",
                   "CON1C(=O)c2ccc(cc2C[C@]1(C)CO)N(=O)=O"] # Aspirin
    
    for smiles in test_smiles:
        print(f"\nSampling top 5 configurations for {smiles}:")
        predictions = predict_top_k(model, smiles, tokenizer, discretizer, device, k=5)
        for i, pred in enumerate(predictions, 1):
            print(f"Rank {i}: {pred['hall_number']}/{pred['spg_number']}) | Prob: {pred['probability']:.3f} | "
              f"Zprime: {pred['zprime']} | Cell: {pred['cell_parameters']} | Vol: {pred['volume']}")
