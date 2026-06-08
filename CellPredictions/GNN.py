import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from rdkit import Chem
from sklearn.model_selection import train_test_split
import numpy as np

# ==========================================
# 1. Dataset Preparation
# ==========================================
def compute_volume(cell_params):
    """Computes unit cell volume from a tensor of shape [..., 6] (a,b,c,alpha,beta,gamma)."""
    a, b, c = cell_params[..., 0], cell_params[..., 1], cell_params[..., 2]
    alpha = cell_params[..., 3] * torch.pi / 180.0
    beta = cell_params[..., 4] * torch.pi / 180.0
    gamma = cell_params[..., 5] * torch.pi / 180.0
    
    term = 1 - torch.cos(alpha)**2 - torch.cos(beta)**2 - torch.cos(gamma)**2 + 2 * torch.cos(alpha)*torch.cos(beta)*torch.cos(gamma)
    term = torch.clamp(term, min=1e-9)
    return a * b * c * torch.sqrt(term)

def enforce_symmetry(cell_pred, hall_indices):
    """
    Differentiably projects raw 6D cell predictions onto their exact symmetry constraints.
    hall_indices is 0-531 (which is Hall Number 1-532 offset by 1).
    """
    hall_numbers = hall_indices + 1
    
    # Extract raw predictions
    a, b, c, alpha, beta, gamma = [cell_pred[:, i] for i in range(6)]
    
    # Enforce positivity on cell lengths and constrain free angles to [30, 150] degrees
    new_a = F.softplus(a) + 1.0
    new_b = F.softplus(b) + 1.0
    new_c = F.softplus(c) + 1.0
    new_alpha = 30.0 + 120.0 * torch.sigmoid(alpha)
    new_beta = 30.0 + 120.0 * torch.sigmoid(beta)
    new_gamma = 30.0 + 120.0 * torch.sigmoid(gamma)
    
    # 1. TRICLINIC (Hall 1-2): Space Group 1-2 -> No constraints
    
    # 2. MONOCLINIC (Hall 3-107): Space Groups 3-15
    # Standard setting is b-unique: alpha = gamma = 90
    # (Note: Hall numbers specifically define unique axes. You can expand this logic 
    # if you have c-unique or a-unique settings in your augmented dataset).
    monoclinic_mask = (hall_numbers >= 3) & (hall_numbers <= 107)
    new_alpha = torch.where(monoclinic_mask, torch.tensor(90.0, device=cell_pred.device), new_alpha)
    new_gamma = torch.where(monoclinic_mask, torch.tensor(90.0, device=cell_pred.device), new_gamma)
    
    # 3. ORTHORHOMBIC (Hall 108-348): Space Groups 16-74
    # alpha = beta = gamma = 90
    ortho_mask = (hall_numbers >= 108) & (hall_numbers <= 348)
    new_alpha = torch.where(ortho_mask, torch.tensor(90.0, device=cell_pred.device), new_alpha)
    new_beta = torch.where(ortho_mask, torch.tensor(90.0, device=cell_pred.device), new_beta)
    new_gamma = torch.where(ortho_mask, torch.tensor(90.0, device=cell_pred.device), new_gamma)
    
    # 4. TETRAGONAL (Hall 349-429): Space Groups 75-142
    # a = b, alpha = beta = gamma = 90
    tetra_mask = (hall_numbers >= 349) & (hall_numbers <= 429)
    new_b = torch.where(tetra_mask, new_a, new_b) # Force b to equal a
    new_alpha = torch.where(tetra_mask, torch.tensor(90.0, device=cell_pred.device), new_alpha)
    new_beta = torch.where(tetra_mask, torch.tensor(90.0, device=cell_pred.device), new_beta)
    new_gamma = torch.where(tetra_mask, torch.tensor(90.0, device=cell_pred.device), new_gamma)
    
    # 5. TRIGONAL / HEXAGONAL (Hall 430-488): Space Groups 143-194
    # Hexagonal setting: a = b, alpha = beta = 90, gamma = 120
    hex_mask = (hall_numbers >= 430) & (hall_numbers <= 488)
    new_b = torch.where(hex_mask, new_a, new_b)
    new_alpha = torch.where(hex_mask, torch.tensor(90.0, device=cell_pred.device), new_alpha)
    new_beta = torch.where(hex_mask, torch.tensor(90.0, device=cell_pred.device), new_beta)
    new_gamma = torch.where(hex_mask, torch.tensor(120.0, device=cell_pred.device), new_gamma)
    
    # 6. CUBIC (Hall 489-530): Space Groups 195-230
    # a = b = c, alpha = beta = gamma = 90
    cubic_mask = (hall_numbers >= 489) & (hall_numbers <= 530)
    # Optional: Average a, b, c predictions for smoother gradients, or just overwrite with a
    avg_length = (new_a + new_b + new_c) / 3.0
    new_a = torch.where(cubic_mask, avg_length, new_a)
    new_b = torch.where(cubic_mask, avg_length, new_b)
    new_c = torch.where(cubic_mask, avg_length, new_c)
    new_alpha = torch.where(cubic_mask, torch.tensor(90.0, device=cell_pred.device), new_alpha)
    new_beta = torch.where(cubic_mask, torch.tensor(90.0, device=cell_pred.device), new_beta)
    new_gamma = torch.where(cubic_mask, torch.tensor(90.0, device=cell_pred.device), new_gamma)
    
    return torch.stack([new_a, new_b, new_c, new_alpha, new_beta, new_gamma], dim=1)

def smiles_to_graph(smiles, hall=None, cell_params=None, zprime=None):
    """Converts a SMILES string to a PyTorch Geometric Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features: [Atomic Number, Degree]
    x = [[atom.GetAtomicNum(), atom.GetDegree()] for atom in mol.GetAtoms()]
    x = torch.tensor(x, dtype=torch.float)

    # Edge indices
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]]) # Undirected graph

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)

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
        
        # Calculate class weights for Hall numbers to handle heavy dataset imbalance (e.g. P21/c, P-1)
        # Weight = total_samples / (num_classes * frequency)
        hall_counts = self.df['hall'].value_counts()
        self.class_weights = torch.ones(530, dtype=torch.float)
        for h_val, count in hall_counts.items():
            if 1 <= h_val <= 530:
                # Add a smoothing factor (+10) so perfectly rare classes don't get infinite weight
                self.class_weights[h_val - 1] = len(self.df) / (len(hall_counts) * (count + 10))
                
        for idx, row in self.df.iterrows():
            cell = [row['a'], row['b'], row['c'], row['alpha'], row['beta'], row['gamma']]
            zprime = row['zprime']
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
class HierarchicalCrystalGNN(nn.Module):
    def __init__(self, node_feat_dim=2, hidden_dim=128, num_hall_classes=530):
        super().__init__()

        # GNN Backbone
        self.conv1 = GINConv(nn.Sequential(nn.Linear(node_feat_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv3 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))

        # Hall Number Classification Head
        self.hall_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_hall_classes)
        )

        # Hall Number Embedding
        self.hall_embedding = nn.Embedding(num_hall_classes, hidden_dim)

        # Cell Parameter Regression Head
        self.cell_regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 6) # [a, b, c, alpha, beta, gamma]
        )
        
        # Zprime Regression Head
        self.zprime_regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    # Override previous mistake
    def forward(self, x, edge_index, batch, condition_hall_idx=None):
        # Global Molecular Embedding
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        mol_emb = global_mean_pool(x, batch)

        # Predict Hall
        hall_logits = self.hall_classifier(mol_emb)

        if condition_hall_idx is None:
            condition_hall_idx = hall_logits.argmax(dim=-1)

        # Regress Cell Parameters
        hall_emb = self.hall_embedding(condition_hall_idx)
        combined_emb = torch.cat([mol_emb, hall_emb], dim=1)
        raw_cell_pred = self.cell_regressor(combined_emb)
        
        # Regress Zprime (must be strictly positive)
        zprime_pred = F.softplus(self.zprime_regressor(combined_emb)).squeeze(-1)
        
        # Apply strict crystallographic symmetry
        cell_pred = enforce_symmetry(raw_cell_pred, condition_hall_idx)

        return hall_logits, cell_pred, zprime_pred


# ==========================================
# 3. Training and Evaluation Loops
# ==========================================
def train(model, loader, optimizer, device, class_weights, alpha=1.0):
    model.train()
    total_loss, total_hall_acc = 0, 0
    hall_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Switch cell regression to L1 (MAE) or Huber Loss to avoid MSE gradient explosion
    cell_criterion = nn.L1Loss() 

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        hall_logits, cell_pred, zprime_pred = model(batch.x, batch.edge_index, batch.batch, condition_hall_idx=batch.hall_y)

        loss_hall = hall_criterion(hall_logits, batch.hall_y)
        loss_cell = cell_criterion(cell_pred, batch.cell_y)
        
        # Volumetric constraint loss (Switched to L1 / MAE)
        pred_vol = compute_volume(cell_pred)
        true_vol = compute_volume(batch.cell_y)
        
        # Prevent huge absolute values of Volume dominating L1 Loss completely
        loss_vol = nn.L1Loss()(pred_vol / 100.0, true_vol / 100.0) 
        
        # Zprime loss
        loss_zprime = nn.L1Loss()(zprime_pred, batch.zprime_y.squeeze(-1))
        
        # Balance out the multi-task learning dynamically
        loss = loss_hall + (alpha * loss_cell) + (1.0 * loss_vol) + (2.0 * loss_zprime)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_hall_acc += (hall_logits.argmax(dim=-1) == batch.hall_y).sum().item()

    return total_loss / len(loader), total_hall_acc / len(loader.dataset)

@torch.no_grad()
def topk_hall_accuracy(logits, targets, k):
    """Compute top-k accuracy for Hall classification."""
    topk = torch.topk(logits, k=k, dim=-1).indices
    correct = (topk == targets.unsqueeze(1)).any(dim=1).float()
    return correct.mean().item()

@torch.no_grad()
def evaluate(model, loader, device, class_weights, alpha=1.0):
    model.eval()
    total_loss, total_hall_acc = 0, 0
    total_cell_mae = 0
    total_top3_acc = 0.0
    total_top5_acc = 0.0
    total_samples = 0
    hall_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    cell_criterion = nn.L1Loss()
    mae_criterion = nn.L1Loss()

    for batch in loader:
        batch = batch.to(device)
        hall_logits, cell_pred, zprime_pred = model(batch.x, batch.edge_index, batch.batch, condition_hall_idx=batch.hall_y)

        loss_hall = hall_criterion(hall_logits, batch.hall_y)
        loss_cell = cell_criterion(cell_pred, batch.cell_y)
        
        # Volumetric constraint loss
        pred_vol = compute_volume(cell_pred)
        true_vol = compute_volume(batch.cell_y)
        loss_vol = nn.L1Loss()(pred_vol / 100.0, true_vol / 100.0)
        
        # Zprime loss
        loss_zprime = nn.L1Loss()(zprime_pred, batch.zprime_y.squeeze(-1))
        
        loss = loss_hall + (alpha * loss_cell) + (1.0 * loss_vol) + (2.0 * loss_zprime)

        total_loss += loss.item()
        total_hall_acc += (hall_logits.argmax(dim=-1) == batch.hall_y).sum().item()
        total_cell_mae += mae_criterion(cell_pred, batch.cell_y).item() * batch.num_graphs
        batch_size = batch.hall_y.shape[0]
        total_top3_acc += topk_hall_accuracy(hall_logits, batch.hall_y, k=3) * batch_size
        total_top5_acc += topk_hall_accuracy(hall_logits, batch.hall_y, k=5) * batch_size
        total_samples += batch_size

    return (
        total_loss / len(loader),
        total_hall_acc / len(loader.dataset),
        total_cell_mae / len(loader.dataset),
        total_top3_acc / max(total_samples, 1),
        total_top5_acc / max(total_samples, 1),
    )

def predict_top_k(model, smiles_string, device, k=5):
    """Predicts a ranked list of (Hall, Cell) solutions for a test SMILES."""
    model.eval()
    data = smiles_to_graph(smiles_string)
    if data is None:
        return "Invalid SMILES"

    data = data.to(device)
    batch_vector = torch.zeros(data.x.size(0), dtype=torch.long).to(device)

    with torch.no_grad():
        x = F.relu(model.conv1(data.x, data.edge_index))
        x = F.relu(model.conv2(x, data.edge_index))
        x = F.relu(model.conv3(x, data.edge_index))
        mol_emb = global_mean_pool(x, batch_vector)

        hall_logits = model.hall_classifier(mol_emb)
        hall_probs = F.softmax(hall_logits, dim=-1)[0]

        top_k_probs, top_k_indices = torch.topk(hall_probs, k)

        results = []
        
        # Load Hall to Spg mapping for inference
        hall_to_spg = {}
        try:
            hm_df = pd.read_csv("HM_Full.csv")
            for _, r in hm_df.iterrows():
                hall_to_spg[int(r['Hall'])] = int(r['Spg_num'])
        except Exception:
            pass # Fallback if file isn't around

        for prob, hall_idx in zip(top_k_probs, top_k_indices):
            hall_emb = model.hall_embedding(hall_idx.unsqueeze(0))
            combined_emb = torch.cat([mol_emb, hall_emb], dim=1)
            raw_cell_pred = model.cell_regressor(combined_emb)
            zprime_pred = F.softplus(model.zprime_regressor(combined_emb)).item()
            
            # CRITICAL FIX: Apply symmetry constraints during inference too!
            cell_pred = enforce_symmetry(raw_cell_pred, hall_idx.unsqueeze(0))[0]
            
            h_num = hall_idx.item() + 1
            spg_num = hall_to_spg.get(h_num, "Unknown")

            results.append({
                'hall_number': h_num,
                'spg_number': spg_num,
                'probability': prob.item(),
                'cell_parameters': [round(p, 3) for p in cell_pred.tolist()],
                'zprime': round(zprime_pred, 3)
            })
    return results


# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    csv_file = "data.csv"

    # 1. Create matching dummy data (For testing the script)
    if not os.path.exists(csv_file):
        print(f"Creating a synthetic toy dataset compatible with symmetry constraints...")
        n_samples = 250
        smiles_list = ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CCO', 'C1=CC=CC=C1', 'C', 'O=C=O'] * (n_samples // 5)
        halls = np.random.randint(1, 531, n_samples)
        z_primes = np.random.choice([0.5, 1.0, 1.5, 2.0, 3.0, 4.0], n_samples)

        a_raw = np.random.uniform(5, 15, n_samples)
        b_raw = np.random.uniform(5, 15, n_samples)
        c_raw = np.random.uniform(5, 15, n_samples)
        al_raw = np.random.uniform(80, 100, n_samples)
        be_raw = np.random.uniform(80, 120, n_samples)
        ga_raw = np.random.uniform(80, 100, n_samples)

        # We can utilize our differentiable symmetry enforcer to clean the dummy data
        raw_cells = torch.tensor(np.stack([a_raw, b_raw, c_raw, al_raw, be_raw, ga_raw], axis=1), dtype=torch.float)
        hall_tensor = torch.tensor(halls - 1, dtype=torch.long)
        sym_cells = enforce_symmetry(raw_cells, hall_tensor).numpy()

        df = pd.DataFrame({
            'smiles': smiles_list,
            'hall': halls,
            'zprime': z_primes,
            'a': sym_cells[:, 0],
            'b': sym_cells[:, 1],
            'c': sym_cells[:, 2],
            'alpha': sym_cells[:, 3],
            'beta': sym_cells[:, 4],
            'gamma': sym_cells[:, 5]
        })
        df.to_csv(csv_file, index=False)

    # 2. Load Data
    print("Loading raw data...")
    df = pd.read_csv(csv_file)

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    train_dataset = CrystalDataset(train_df)
    val_dataset = CrystalDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 3. Setup Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = HierarchicalCrystalGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-5,
    )

    # 4. Training Loop
    epochs = 50
    print("\nStarting Training...")
    # Retrieve the inverse frequency weights attached to the graph generation
    class_weights = train_dataset.class_weights
    
    best_val_mae = float('inf')
    best_top5 = 0.0
    best_mae_ckpt = "best_by_val_mae.pt"
    best_top5_ckpt = "best_by_top5.pt"
    early_stop_patience = 12
    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, device, class_weights)
        val_loss, val_acc, val_mae, val_top3, val_top5 = evaluate(model, val_loader, device, class_weights)
        scheduler.step(val_mae)

        # Checkpoint: best validation MAE
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_without_improve = 0
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_mae': val_mae,
                    'val_top5': val_top5,
                },
                best_mae_ckpt,
            )
        else:
            epochs_without_improve += 1

        # Checkpoint: best top-5 Hall accuracy
        if val_top5 > best_top5:
            best_top5 = val_top5
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_mae': val_mae,
                    'val_top5': val_top5,
                },
                best_top5_ckpt,
            )

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}, Hall Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Hall Acc: {val_acc:.4f}, Top3: {val_top3:.4f}, Top5: {val_top5:.4f}, "
              f"Cell MAE: {val_mae:.4f}, LR: {current_lr:.6f}")

        if epochs_without_improve >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch} (no val MAE improvement in {early_stop_patience} epochs)")
            break

    # 5. Inference Demonstration
    print("\nTraining Complete! Testing Inference...")
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O" # Aspirin
    print(f"Predicting top 5 solutions for {test_smiles}:")
    predictions = predict_top_k(model, test_smiles, device, k=5)

    for i, pred in enumerate(predictions, 1):
        print(f"Rank {i}: Hall {pred['hall_number']} (Spg: {pred['spg_number']}) (Prob: {pred['probability']:.3f}) | "
              f"Zprime: {pred['zprime']} | Cell [a,b,c,α,β,γ]: {pred['cell_parameters']}")
