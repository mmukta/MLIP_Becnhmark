import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool, global_max_pool
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.model_selection import train_test_split
import numpy as np

VOLUME_CALIBRATOR = None
EXACT_SMILES_LIBRARY = {}
FINGERPRINT_LIBRARY = []
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# ==========================================
# 1. Dataset Preparation
# ==========================================
import math

def get_atom_features(atom):
    """Return a compact but richer atom feature vector."""
    return [
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        atom.GetTotalNumHs(),
        int(atom.GetHybridization()),
    ]

def get_bond_features(bond):
    """Return bond feature vector."""
    bond_type = bond.GetBondType()
    bond_type_id = {
        Chem.rdchem.BondType.SINGLE: 1,
        Chem.rdchem.BondType.DOUBLE: 2,
        Chem.rdchem.BondType.TRIPLE: 3,
        Chem.rdchem.BondType.AROMATIC: 4,
    }.get(bond_type, 0)
    return [
        bond_type_id,
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
        int(bond.GetStereo()),
    ]

def enforce_symmetry(cell_pred, hall_indices):
    """
    Differentiably projects raw cell predictions onto their exact symmetry constraints.
    cell_pred shape: [Batch, Mixtures, 7] or [Batch, 7]
    hall_indices shape: [Batch]
    """
    hall_numbers = hall_indices + 1
    if cell_pred.dim() == 3:
        hall_numbers = hall_numbers.unsqueeze(-1)
        
    device = cell_pred.device
    
    # Enforce positivity on lengths and volume using softplus
    # Also strictly normalize angles to fall within physically plausible reasonable ranges [40, 140] to avoid exploding gradients
    
    # We must clone cell_pred explicitly first so we don't accidentally pull NaNs backwards from gradients
    cp = cell_pred.clone()
    
    a = F.softplus(cp[..., 0]) + 1.0 # Minimum cell length 1A
    b = F.softplus(cp[..., 1]) + 1.0
    c = F.softplus(cp[..., 2]) + 1.0
    alpha = 40.0 + 100.0 * torch.sigmoid(cp[..., 3])
    beta = 40.0 + 100.0 * torch.sigmoid(cp[..., 4])
    gamma = 40.0 + 100.0 * torch.sigmoid(cp[..., 5])
    
    # 2. MONOCLINIC (Hall 3-107)
    monoclinic_mask = (hall_numbers >= 3) & (hall_numbers <= 107)
    alpha = torch.where(monoclinic_mask, torch.tensor(90.0, device=device), alpha)
    gamma = torch.where(monoclinic_mask, torch.tensor(90.0, device=device), gamma)
    
    # 3. ORTHORHOMBIC (Hall 108-348)
    ortho_mask = (hall_numbers >= 108) & (hall_numbers <= 348)
    alpha = torch.where(ortho_mask, torch.tensor(90.0, device=device), alpha)
    beta = torch.where(ortho_mask, torch.tensor(90.0, device=device), beta)
    gamma = torch.where(ortho_mask, torch.tensor(90.0, device=device), gamma)
    
    # 4. TETRAGONAL (Hall 349-429)
    tetra_mask = (hall_numbers >= 349) & (hall_numbers <= 429)
    b = torch.where(tetra_mask, a, b)
    alpha = torch.where(tetra_mask, torch.tensor(90.0, device=device), alpha)
    beta = torch.where(tetra_mask, torch.tensor(90.0, device=device), beta)
    gamma = torch.where(tetra_mask, torch.tensor(90.0, device=device), gamma)
    
    # 5. TRIGONAL / HEXAGONAL (Hall 430-488)
    hex_mask = (hall_numbers >= 430) & (hall_numbers <= 488)
    b = torch.where(hex_mask, a, b)
    alpha = torch.where(hex_mask, torch.tensor(90.0, device=device), alpha)
    beta = torch.where(hex_mask, torch.tensor(90.0, device=device), beta)
    gamma = torch.where(hex_mask, torch.tensor(120.0, device=device), gamma)
    
    # 6. CUBIC (Hall 489-530)
    cubic_mask = (hall_numbers >= 489) & (hall_numbers <= 530)
    avg_length = (a + b + c) / 3.0
    a = torch.where(cubic_mask, avg_length, a)
    b = torch.where(cubic_mask, avg_length, b)
    c = torch.where(cubic_mask, avg_length, c)
    alpha = torch.where(cubic_mask, torch.tensor(90.0, device=device), alpha)
    beta = torch.where(cubic_mask, torch.tensor(90.0, device=device), beta)
    gamma = torch.where(cubic_mask, torch.tensor(90.0, device=device), gamma)
    
    if cell_pred.shape[-1] == 8:
        # Instead of regressing the volume separately (which breaks if the regression scales are mismatched),
        # ALWAYS calculate the volume explicitly based on the predicted lengths and angles.
        al_rad = alpha * torch.pi / 180.0
        be_rad = beta * torch.pi / 180.0
        ga_rad = gamma * torch.pi / 180.0
        term = 1 - torch.cos(al_rad)**2 - torch.cos(be_rad)**2 - torch.cos(ga_rad)**2 + 2 * torch.cos(al_rad)*torch.cos(be_rad)*torch.cos(ga_rad)
        term = torch.clamp(term, min=1e-9)
        v = a * b * c * torch.sqrt(term)
        
        # We explicitly rely on the softplussed network output for Zprime
        zprime = F.softplus(cp[..., 7])
        
        return torch.stack([a, b, c, alpha, beta, gamma, v, zprime], dim=-1)
    return torch.stack([a, b, c, alpha, beta, gamma], dim=-1)

def compute_vol_scalar(a, b, c, alpha, beta, gamma):
    al, be, ga = math.radians(alpha), math.radians(beta), math.radians(gamma)
    term = 1 - math.cos(al)**2 - math.cos(be)**2 - math.cos(ga)**2 + 2 * math.cos(al)*math.cos(be)*math.cos(ga)
    return a * b * c * math.sqrt(max(term, 1e-9))

def smiles_to_graph(smiles, hall=None, cell_params=None, zprime=None):
    """Converts a SMILES string to a PyTorch Geometric Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features: richer local chemistry than just atomic number / degree.
    x = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(x, dtype=torch.float)

    # Edge indices and edge attributes
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]]) # Undirected graph
        bond_features = get_bond_features(bond)
        edge_attr.extend([bond_features, bond_features])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float)
    else:
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Global descriptors provide a molecule-level signal that complements the GNN.
    ring_info = mol.GetRingInfo()
    global_features = torch.tensor([
        mol.GetNumAtoms() / 100.0,
        mol.GetNumHeavyAtoms() / 100.0,
        mol.GetNumBonds() / 100.0,
        ring_info.NumRings() / 10.0,
        rdMolDescriptors.CalcNumAromaticRings(mol) / 10.0,
        Descriptors.MolWt(mol) / 500.0,
        rdMolDescriptors.CalcTPSA(mol) / 200.0,
        rdMolDescriptors.CalcFractionCSP3(mol),
    ], dtype=torch.float)
    data.global_x = global_features

    if hall is not None:
        data.hall_y = torch.tensor([hall - 1], dtype=torch.long) # Shift 1-530 to 0-529
    if cell_params is not None:
        data.cell_y = torch.tensor([cell_params], dtype=torch.float)
    if zprime is not None:
        data.zprime_y = torch.tensor([zprime], dtype=torch.float)

    return data

class CrystalDataset(Dataset):
    def __init__(self, df):
        super().__init__(None, None, None)
        self.df = df.reset_index(drop=True)
        # Process graphs
        self.graphs = []
        print(f"Processing {len(self.df)} molecules...")
        valid_indices = []

        # Compute inverse-frequency class weights for Hall classification.
        hall_counts = self.df['hall'].value_counts()
        self.class_weights = torch.ones(530, dtype=torch.float)
        for h_val, count in hall_counts.items():
            if 1 <= h_val <= 530:
                self.class_weights[h_val - 1] = len(self.df) / (len(hall_counts) * (count + 10))

        for idx, row in self.df.iterrows():
            v = compute_vol_scalar(row['a'], row['b'], row['c'], row['alpha'], row['beta'], row['gamma'])
            zprime = row['zprime']
            # We predict 8 dimension distribution now: [a, b, c, alpha, beta, gamma, V, Zprime]
            cell = [row['a'], row['b'], row['c'], row['alpha'], row['beta'], row['gamma'], v, zprime]
            graph = smiles_to_graph(row['smiles'], row['hall'], cell, zprime)
            if graph is not None:
                self.graphs.append(graph)
                valid_indices.append(idx)
        print(f"Successfully processed {len(self.graphs)} valid graphs.")

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]


# ==========================================
# 2. Model Architecture
# ==========================================
class HierarchicalCrystalMDN(nn.Module):
    def __init__(self, node_feat_dim=7, global_feat_dim=8, hidden_dim=128, num_hall_classes=530, num_mixtures=5):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.global_feat_dim = global_feat_dim
        self.graph_emb_dim = hidden_dim * 3 + global_feat_dim

        # GNN Backbone
        self.conv1 = GINConv(nn.Sequential(nn.Linear(node_feat_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv3 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))

        # Hall Number Classification Head
        self.hall_classifier = nn.Sequential(
            nn.Linear(self.graph_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_hall_classes)
        )

        # Hall Number Embedding
        self.hall_embedding = nn.Embedding(num_hall_classes, hidden_dim)

        # MDN Head
        self.mdn_shared = nn.Sequential(
            nn.Linear(self.graph_emb_dim + hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mdn_pi = nn.Linear(hidden_dim, num_mixtures)
        self.mdn_mu = nn.Linear(hidden_dim, num_mixtures * 8)
        self.mdn_sigma = nn.Linear(hidden_dim, num_mixtures * 8)

        nn.init.xavier_uniform_(self.mdn_pi.weight, gain=0.5)
        nn.init.zeros_(self.mdn_pi.bias)
        nn.init.xavier_uniform_(self.mdn_mu.weight, gain=0.2)
        nn.init.zeros_(self.mdn_mu.bias)
        nn.init.xavier_uniform_(self.mdn_sigma.weight, gain=0.2)
        nn.init.constant_(self.mdn_sigma.bias, -1.0)

    def encode_graph(self, x, edge_index, batch, global_x=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        pooled = torch.cat([
            global_mean_pool(x, batch),
            global_max_pool(x, batch),
            global_add_pool(x, batch),
        ], dim=1)

        if global_x is None:
            global_x = pooled.new_zeros((pooled.size(0), self.global_feat_dim))
        elif global_x.dim() == 1:
            global_x = global_x.view(-1, self.global_feat_dim)
        elif global_x.size(-1) != self.global_feat_dim:
            global_x = global_x.reshape(-1, self.global_feat_dim)

        return torch.cat([pooled, global_x.to(pooled.dtype)], dim=1)

    def forward(self, x, edge_index, batch, condition_hall_idx=None, global_x=None):
        mol_emb = self.encode_graph(x, edge_index, batch, global_x=global_x)

        # Predict Hall
        hall_logits = self.hall_classifier(mol_emb)
        hall_logits = torch.clamp(hall_logits, min=-10.0, max=10.0)

        if condition_hall_idx is None:
            condition_hall_idx = hall_logits.argmax(dim=-1)

        # Regress Cell Parameters via MDN
        hall_emb = self.hall_embedding(condition_hall_idx)
        combined_emb = torch.cat([mol_emb, hall_emb], dim=1)

        mdn_hidden = self.mdn_shared(combined_emb)
        
        # Clamp logits BEFORE softmax/exp to safely prevent ANY NaNs ever forming
        pi_logits = torch.clamp(self.mdn_pi(mdn_hidden), min=-8.0, max=8.0)
        pi = F.softmax(pi_logits, dim=-1)
        
        # Keep raw means in a non-saturated band to avoid symmetry projection collapse.
        raw_mu = (5.0 * torch.tanh(self.mdn_mu(mdn_hidden) / 5.0)).view(-1, self.num_mixtures, 8)
        mu = enforce_symmetry(raw_mu, condition_hall_idx)
        
        sigma_raw = self.mdn_sigma(mdn_hidden).view(-1, self.num_mixtures, 8)
        sigma = F.softplus(sigma_raw) + 1e-3
        sigma = torch.clamp(sigma, min=1e-3, max=3.0)

        return hall_logits, (pi, mu, sigma)

def mdn_loss_fn(pi, mu, sigma, target):
    """
    Computes a stable regression loss on the mixture expectation.
    target shape: [Batch, 8]
    """
    # Target scaling for stability.
    scale_factors = torch.tensor([10.0, 10.0, 10.0, 100.0, 100.0, 100.0, 1000.0, 1.0], device=mu.device)
    target_scaled = target / scale_factors
    mu_scaled = mu / scale_factors

    # Mixture expectation gives a stable point prediction while still using the MDN head.
    expected_scaled = (pi.unsqueeze(-1) * mu_scaled).sum(dim=1)
    regression_loss = F.smooth_l1_loss(expected_scaled, target_scaled)

    # Encourage non-degenerate mixture weights without letting the entropy term dominate.
    entropy_loss = (pi * torch.log(pi + 1e-8)).sum(dim=1).mean()

    return regression_loss + 0.01 * entropy_loss


# ==========================================
# 3. Training and Evaluation Loops
# ==========================================
def train(model, loader, optimizer, device, class_weights, alpha=1.0):
    model.train()
    total_loss, total_hall_acc = 0, 0
    hall_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        hall_logits, (pi, mu, sigma) = model(batch.x, batch.edge_index, batch.batch, condition_hall_idx=batch.hall_y, global_x=batch.global_x)

        loss_hall = hall_criterion(hall_logits, batch.hall_y)
        loss_cell = mdn_loss_fn(pi, mu, sigma, batch.cell_y)

        loss = loss_hall + alpha * loss_cell
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item()
        total_hall_acc += (hall_logits.argmax(dim=-1) == batch.hall_y).sum().item()

    return total_loss / len(loader), total_hall_acc / len(loader.dataset)

@torch.no_grad()
def topk_hall_accuracy(logits, targets, k):
    topk = torch.topk(logits, k=k, dim=-1).indices
    correct = (topk == targets.unsqueeze(1)).any(dim=1).float()
    return correct.mean().item()

@torch.no_grad()
def evaluate(model, loader, device, class_weights, alpha=1.0):
    model.eval()
    total_loss, total_hall_acc = 0, 0
    total_top3_acc = 0.0
    total_top5_acc = 0.0
    total_samples = 0
    hall_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    for batch in loader:
        batch = batch.to(device)
        hall_logits, (pi, mu, sigma) = model(batch.x, batch.edge_index, batch.batch, condition_hall_idx=batch.hall_y, global_x=batch.global_x)

        loss_hall = hall_criterion(hall_logits, batch.hall_y)
        loss_cell = mdn_loss_fn(pi, mu, sigma, batch.cell_y)
        loss = loss_hall + alpha * loss_cell

        total_loss += loss.item()
        total_hall_acc += (hall_logits.argmax(dim=-1) == batch.hall_y).sum().item()
        batch_size = batch.hall_y.shape[0]
        total_top3_acc += topk_hall_accuracy(hall_logits, batch.hall_y, k=3) * batch_size
        total_top5_acc += topk_hall_accuracy(hall_logits, batch.hall_y, k=5) * batch_size
        total_samples += batch_size

    return (
        total_loss / len(loader),
        total_hall_acc / len(loader.dataset),
        total_top3_acc / max(total_samples, 1),
        total_top5_acc / max(total_samples, 1),
    )

def zprime_plausibility_score(zprime):
    """Score Zprime plausibility against common crystallographic values."""
    common_values = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0], dtype=float)
    dist = float(np.min(np.abs(common_values - float(zprime))))
    # Sharp preference near common values, but not a hard cutoff.
    discrete_score = math.exp(-dist / 0.25)
    # Mild range penalty outside a typical organic-crystal window.
    range_score = 1.0 if 0.25 <= float(zprime) <= 6.0 else 0.3
    return discrete_score * range_score

def get_mol_descriptor_vector(smiles):
    """Descriptor vector used for a lightweight volume prior model."""
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

def morgan_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MORGAN_GENERATOR.GetFingerprint(mol)

def fit_volume_calibrator(df):
    """Fit a tiny linear model: volume ~ f(molecular descriptors)."""
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
    # Least-squares linear fit with pseudo-inverse for stability.
    coef = np.linalg.pinv(X) @ y
    residual = y - X @ coef
    sigma = float(np.std(residual))
    sigma = max(sigma, 150.0)
    return {'coef': coef, 'sigma': sigma}

def volume_plausibility_score(pred_volume, smiles, calibrator):
    """Return a plausibility score for predicted volume given molecular size."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        n_heavy = mol.GetNumHeavyAtoms()
        min_vol = max(n_heavy * 15.0, 50.0)
        if float(pred_volume) < min_vol:
            return 0.0
    if calibrator is None:
        return 1.0
    x = get_mol_descriptor_vector(smiles)
    if x is None:
        return 1.0
    est_v = float(x @ calibrator['coef'])
    sigma = float(calibrator['sigma']) / 5.0
    sigma = max(sigma, 50.0)
    z = abs(float(pred_volume) - est_v) / max(sigma, 1e-6)
    return float(math.exp(-0.5 * z * z))

def calibrated_rank_score(combined_prob, zprime_score, zprime_weight=0.3):
    """
    Softly calibrate ranking with Z' plausibility.
    zprime_weight=0.0 -> pure model probability ranking.
    zprime_weight=1.0 -> full multiplicative Z' penalty.
    """
    zprime_weight = float(np.clip(zprime_weight, 0.0, 1.0))
    soft_factor = (1.0 - zprime_weight) + zprime_weight * float(zprime_score)
    return float(combined_prob) * soft_factor

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

def retrieve_neighbor_candidates(smiles_string, top_n=5, exclude_smiles=None):
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
        v = compute_vol_scalar(item['a'], item['b'], item['c'], item['alpha'], item['beta'], item['gamma'])
        results.append({
            'hall_number': hall_num,
            'spg_number': 'Retrieved',
            'hall_prob': sim,
            'mix_prob': sim,
            'combined_prob': sim,
            'zprime_score': round(zprime_plausibility_score(item['zprime']), 4),
            'volume_score': 1.0,
            'combined_score': sim * 10.0,
            'cell_parameters': [round(item['a'], 3), round(item['b'], 3), round(item['c'], 3), round(item['alpha'], 3), round(item['beta'], 3), round(item['gamma'], 3)],
            'volume': round(v, 3),
            'zprime': round(item['zprime'], 3),
        })
        if len(results) >= top_n:
            break

    return results

def predict_top_k(model, smiles_string, device, k_hall=10, k_mixtures=2, zprime_weight=0.3, volume_weight=0.25):
    """Predicts a ranked list of (Hall, Cell) solutions for a test SMILES."""
    model.eval()
    data = smiles_to_graph(smiles_string)
    if data is None:
        return "Invalid SMILES"

    exact_results = []
    for row in EXACT_SMILES_LIBRARY.get(smiles_string, []):
        hall_num = int(row['hall'])
        v = compute_vol_scalar(row['a'], row['b'], row['c'], row['alpha'], row['beta'], row['gamma'])
        exact_results.append({
            'hall_number': hall_num,
            'spg_number': 'Known',
            'hall_prob': 1.0,
            'mix_prob': 1.0,
            'combined_prob': 1.0,
            'zprime_score': 1.0,
            'volume_score': 1.0,
            'combined_score': 1e6,
            'cell_parameters': [round(row['a'], 3), round(row['b'], 3), round(row['c'], 3), round(row['alpha'], 3), round(row['beta'], 3), round(row['gamma'], 3)],
            'volume': round(v, 3),
            'zprime': round(float(row['zprime']), 3),
        })

    exact_smiles = {row['smiles'] for row in EXACT_SMILES_LIBRARY.get(smiles_string, [])}
    neighbor_results = retrieve_neighbor_candidates(smiles_string, top_n=5, exclude_smiles=exact_smiles)

    data = data.to(device)
    batch_vector = torch.zeros(data.x.size(0), dtype=torch.long).to(device)

    with torch.no_grad():
        mol_emb = model.encode_graph(data.x, data.edge_index, batch_vector, global_x=data.global_x)

        hall_logits = model.hall_classifier(mol_emb)
        hall_probs = F.softmax(hall_logits, dim=-1)[0]

        top_k_probs, top_k_indices = torch.topk(hall_probs, k_hall)

        results = []
        
        # Load Hall to Spg mapping
        hall_to_spg = {}
        try:
            hm_df = pd.read_csv("HM_Full.csv")
            for _, r in hm_df.iterrows():
                hall_to_spg[int(r['Hall'])] = int(r['Spg_num'])
        except Exception:
            pass

        for prob, hall_idx in zip(top_k_probs, top_k_indices):
            hall_emb = model.hall_embedding(hall_idx.unsqueeze(0))
            combined_emb = torch.cat([mol_emb, hall_emb], dim=1)

            mdn_hidden = model.mdn_shared(combined_emb)
            pi_logits = torch.clamp(model.mdn_pi(mdn_hidden), min=-8.0, max=8.0)
            pi = F.softmax(pi_logits, dim=-1)[0]
            raw_mu = (5.0 * torch.tanh(model.mdn_mu(mdn_hidden) / 5.0)).view(-1, model.num_mixtures, 8)
            mu = enforce_symmetry(raw_mu, hall_idx.unsqueeze(0))[0]

            # Get the top mixtures for this Hall setting
            top_pi_probs, top_pi_indices = torch.topk(pi, k_mixtures)

            for mix_prob, mix_idx in zip(top_pi_probs, top_pi_indices):
                cell_pred = mu[mix_idx]

                combined_prob = prob.item() * mix_prob.item()
                zprime_value = cell_pred[7].item()
                zprime_score = zprime_plausibility_score(zprime_value)
                volume_value = cell_pred[6].item()
                volume_score = volume_plausibility_score(volume_value, smiles_string, VOLUME_CALIBRATOR)
                combined_score = calibrated_rank_score(combined_prob, zprime_score, zprime_weight=zprime_weight)
                combined_score *= (1.0 - volume_weight) + volume_weight * volume_score

                h_num = hall_idx.item() + 1
                results.append({
                    'hall_number': h_num,
                    'spg_number': hall_to_spg.get(h_num, "Unknown"),
                    'hall_prob': prob.item(),
                    'mix_prob': mix_prob.item(),
                    'combined_prob': combined_prob,
                    'zprime_score': round(zprime_score, 4),
                    'volume_score': round(volume_score, 4),
                    'combined_score': combined_score,
                    'cell_parameters': [round(p, 3) for p in cell_pred[:6].tolist()],
                    'volume': round(cell_pred[6].item(), 3),
                    'zprime': round(zprime_value, 3)
                })

    results = exact_results + neighbor_results + results

    # Sort by calibrated physically informed score, then by raw probability.
    results.sort(key=lambda x: (x['combined_score'], x['combined_prob']), reverse=True)
    return results

@torch.no_grad()
def print_embedding_sanity_check(model, smiles_list, device):
    """Print pairwise cosine similarity of molecular embeddings for a small SMILES set."""
    model.eval()
    embeddings = []
    labels = []
    for smiles in smiles_list:
        data = smiles_to_graph(smiles)
        if data is None:
            continue
        data = data.to(device)
        batch_vector = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        mol_emb = model.encode_graph(data.x, data.edge_index, batch_vector, global_x=data.global_x)
        embeddings.append(mol_emb.squeeze(0))
        labels.append(smiles)

    if len(embeddings) < 2:
        print("Embedding sanity check skipped: need at least two valid SMILES.")
        return

    emb = torch.stack(embeddings, dim=0)
    emb = F.normalize(emb, dim=1)
    sim = emb @ emb.T
    print("Embedding cosine similarity matrix:")
    print("SMILES:")
    for label in labels:
        print(f"  - {label}")
    print(sim.cpu().numpy())

def build_exact_smiles_library(df):
    library = {}
    for _, row in df.iterrows():
        library.setdefault(row['smiles'], []).append(row.to_dict())
    return library


# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    csv_file = "data.csv"

    # 1. Create dummy data
    if not os.path.exists(csv_file):
        print(f"Creating a synthetic toy dataset compatible with symmetry constraints...")
        n_samples = 250
        smiles_list = ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CCO', 'C1=CC=CC=C1', 'C', 'O=C=O'] * (n_samples // 5)
        halls = np.random.randint(1, 531, n_samples)
    
        # Simple hard-coded constraints for fake MDN generation (matches GNN symmetry)
        a, b, c = np.random.uniform(5, 15, n_samples), np.random.uniform(5, 15, n_samples), np.random.uniform(5, 15, n_samples)
        al, be, ga = np.random.uniform(80, 100, n_samples), np.random.uniform(80, 120, n_samples), np.random.uniform(80, 100, n_samples)
        z_primes = np.random.choice([0.5, 1.0, 1.5, 2.0, 3.0, 4.0], n_samples)
    
        for i, h in enumerate(halls):
            if h >= 108 and h <= 348: # Ortho
                al[i] = be[i] = ga[i] = 90.0
            elif h >= 3 and h <= 107: # Mono
                al[i] = ga[i] = 90.0
            elif h >= 349 and h <= 429: # Tetra
                b[i] = a[i]
                al[i] = be[i] = ga[i] = 90.0
            elif h >= 430 and h <= 488: # Hex/Trig
                b[i] = a[i]
                al[i] = be[i] = 90.0
                ga[i] = 120.0
            elif h >= 489 and h <= 530: # Cubic
                b[i] = c[i] = a[i]
                al[i] = be[i] = ga[i] = 90.0

        df = pd.DataFrame({'smiles': smiles_list, 'hall': halls, 'zprime': z_primes, 'a': a, 'b': b, 'c': c, 'alpha': al, 'beta': be, 'gamma': ga})
        df.to_csv(csv_file, index=False)

    # 2. Load Data
    print("Loading raw data...")
    df = pd.read_csv(csv_file)

    EXACT_SMILES_LIBRARY = build_exact_smiles_library(df)
    FINGERPRINT_LIBRARY = build_fingerprint_library(df)

    VOLUME_CALIBRATOR = fit_volume_calibrator(df)
    if VOLUME_CALIBRATOR is not None:
        print(f"Volume calibrator ready (residual sigma={VOLUME_CALIBRATOR['sigma']:.2f})")
    else:
        print("Volume calibrator unavailable; skipping volume prior in ranking.")

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    train_dataset = CrystalDataset(train_df)
    val_dataset = CrystalDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 3. Setup Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    node_feat_dim = train_dataset.graphs[0].x.size(1)
    model = HierarchicalCrystalMDN(node_feat_dim=node_feat_dim, num_mixtures=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-5,
    )

    # Quick encoder sanity check: embeddings for a few different molecules should not collapse.
    sanity_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "c1ccccc1",
        "CCO",
        "O=C=O",
    ]
    print_embedding_sanity_check(model, sanity_smiles, device)

    # 4. Training Loop
    epochs = 50#0
    hall_warmup_epochs = 3
    cell_loss_weight = 0.05
    print("\nStarting Training...")
    class_weights = train_dataset.class_weights
    best_val_loss = float('inf')
    best_top5 = 0.0
    best_loss_ckpt = "mdn_best_by_val_loss.pt"
    best_top5_ckpt = "mdn_best_by_top5.pt"
    early_stop_patience = 30
    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
        alpha = 0.0 if epoch <= hall_warmup_epochs else cell_loss_weight
        train_loss, train_acc = train(model, train_loader, optimizer, device, class_weights, alpha=alpha)
        val_loss, val_acc, val_top3, val_top5 = evaluate(model, val_loader, device, class_weights, alpha=alpha)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improve = 0
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_top5': val_top5,
                },
                best_loss_ckpt,
            )
        else:
            epochs_without_improve += 1

        if val_top5 > best_top5:
            best_top5 = val_top5
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_top5': val_top5,
                },
                best_top5_ckpt,
            )

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch:03d} | Alpha: {alpha:.2f} | Train Loss: {train_loss:.4f}, Hall: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Hall: {val_acc:.4f}, Top3: {val_top3:.4f}, Top5: {val_top5:.4f}, LR: {current_lr:.6f}"
        )

        if epochs_without_improve >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch} (no val loss improvement in {early_stop_patience} epochs)")
            break

    # Load best checkpoint for inference: prefer ranking quality (top-5), then val loss fallback.
    if os.path.exists(best_top5_ckpt):
        ckpt = torch.load(best_top5_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded checkpoint for inference: {best_top5_ckpt} (epoch {ckpt['epoch']}, val_top5={ckpt['val_top5']:.4f})")
    elif os.path.exists(best_loss_ckpt):
        ckpt = torch.load(best_loss_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded checkpoint for inference: {best_loss_ckpt} (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

    # 5. Inference Demonstration
    print("\nTraining Complete! Testing Inference...")
    test_smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O", 
                   "CN1C=C(N=C1/C=C/c1ccccc1)N(=O)=O",
                   "O=N(=O)C1=NNC(=N1)CC1=NC(=NN1)N(=O)=O",
                   "CON1C(=O)c2ccc(cc2C[C@]1(C)CO)N(=O)=O"] # Aspirin

    for smiles in test_smiles:
        print(f"\nPredicting top 5 polymorphs for {smiles}:")
        # Extract the top k overall combinations
        predictions = predict_top_k(model, smiles, device, k_hall=5, k_mixtures=2)[:5]
        for i, pred in enumerate(predictions, 1):
            print(
                f"Rank {i}: {pred['hall_number']}/{pred['spg_number']}) | Prob: {pred['mix_prob']:.3f} | "
                f"Score: {pred['combined_score']:.4f} | "
                f"Zprime: {pred['zprime']} | Cell: {pred['cell_parameters']} | Vol: {pred['volume']}"
            )
