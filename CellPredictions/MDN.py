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
def smiles_to_graph(smiles, hall=None, cell_params=None):
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
        data.hall_y = torch.tensor([hall - 1], dtype=torch.long) # Shift 1-532 to 0-531
    if cell_params is not None:
        data.cell_y = torch.tensor([cell_params], dtype=torch.float)

    return data

class CrystalDataset(Dataset):
    def __init__(self, df):
        super().__init__(None, None, None)
        self.df = df.reset_index(drop=True)
        # Process graphs
        self.graphs = []
        print(f"Processing {len(self.df)} molecules...")
        valid_indices = []
        for idx, row in self.df.iterrows():
            cell = [row['a'], row['b'], row['c'], row['alpha'], row['beta'], row['gamma']]
            graph = smiles_to_graph(row['smiles'], row['hall'], cell)
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
    def __init__(self, node_feat_dim=2, hidden_dim=128, num_hall_classes=532, num_mixtures=5):
        super().__init__()
        self.num_mixtures = num_mixtures

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

        # MDN Head
        self.mdn_shared = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        self.mdn_pi = nn.Linear(hidden_dim, num_mixtures)
        self.mdn_mu = nn.Linear(hidden_dim, num_mixtures * 6)
        self.mdn_sigma = nn.Linear(hidden_dim, num_mixtures * 6)

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

        # Regress Cell Parameters via MDN
        hall_emb = self.hall_embedding(condition_hall_idx)
        combined_emb = torch.cat([mol_emb, hall_emb], dim=1)

        mdn_hidden = self.mdn_shared(combined_emb)
        pi = F.softmax(self.mdn_pi(mdn_hidden), dim=-1)
        mu = self.mdn_mu(mdn_hidden).view(-1, self.num_mixtures, 6)
        
        # Add a minimum clamp to prevent log(0) and clamp max to prevent exp overflow
        sigma_logits = self.mdn_sigma(mdn_hidden).clamp(min=-20, max=20)
        sigma = torch.exp(sigma_logits).view(-1, self.num_mixtures, 6)

        return hall_logits, (pi, mu, sigma)

def mdn_loss_fn(pi, mu, sigma, target):
    """
    Computes Negative Log-Likelihood for Mixture Density Network.
    target shape: [Batch, 6] -> Must duplicate to shape [Batch, num_mixtures, 6]
    """
    target = target.unsqueeze(1).expand_as(mu)
    m = torch.distributions.Normal(mu, sigma)
    log_prob = m.log_prob(target).sum(dim=2)
    log_pi = torch.log(pi + 1e-8)
    weighted_log_probs = log_prob + log_pi
    loss = -torch.logsumexp(weighted_log_probs, dim=1).mean()
    return loss


# ==========================================
# 3. Training and Evaluation Loops
# ==========================================
def train(model, loader, optimizer, device, alpha=1.0):
    model.train()
    total_loss, total_hall_acc = 0, 0
    hall_criterion = nn.CrossEntropyLoss()

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        hall_logits, (pi, mu, sigma) = model(batch.x, batch.edge_index, batch.batch, condition_hall_idx=batch.hall_y)

        loss_hall = hall_criterion(hall_logits, batch.hall_y)
        loss_cell = mdn_loss_fn(pi, mu, sigma, batch.cell_y)

        loss = loss_hall + alpha * loss_cell
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_hall_acc += (hall_logits.argmax(dim=-1) == batch.hall_y).sum().item()

    return total_loss / len(loader), total_hall_acc / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device, alpha=1.0):
    model.eval()
    total_loss, total_hall_acc = 0, 0
    hall_criterion = nn.CrossEntropyLoss()

    for batch in loader:
        batch = batch.to(device)
        hall_logits, (pi, mu, sigma) = model(batch.x, batch.edge_index, batch.batch, condition_hall_idx=batch.hall_y)

        loss_hall = hall_criterion(hall_logits, batch.hall_y)
        loss_cell = mdn_loss_fn(pi, mu, sigma, batch.cell_y)
        loss = loss_hall + alpha * loss_cell

        total_loss += loss.item()
        total_hall_acc += (hall_logits.argmax(dim=-1) == batch.hall_y).sum().item()

    return total_loss / len(loader), total_hall_acc / len(loader.dataset)

def predict_top_k(model, smiles_string, device, k_hall=3, k_mixtures=2):
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

        top_k_probs, top_k_indices = torch.topk(hall_probs, k_hall)

        results = []
        for prob, hall_idx in zip(top_k_probs, top_k_indices):
            hall_emb = model.hall_embedding(hall_idx.unsqueeze(0))
            combined_emb = torch.cat([mol_emb, hall_emb], dim=1)

            mdn_hidden = model.mdn_shared(combined_emb)
            pi = F.softmax(model.mdn_pi(mdn_hidden), dim=-1)[0]
            mu = model.mdn_mu(mdn_hidden).view(-1, model.num_mixtures, 6)[0]

            # Get the top mixtures for this Hall setting
            top_pi_probs, top_pi_indices = torch.topk(pi, k_mixtures)

            for mix_prob, mix_idx in zip(top_pi_probs, top_pi_indices):
                cell_pred = mu[mix_idx]
                combined_prob = prob.item() * mix_prob.item()

                results.append({
                    'hall_number': hall_idx.item() + 1,
                    'hall_prob': prob.item(),
                    'mix_prob': mix_prob.item(),
                    'combined_prob': combined_prob,
                    'cell_parameters': [round(p, 3) for p in cell_pred.tolist()]
                })

    # Sort everything by combined probability
    results.sort(key=lambda x: x['combined_prob'], reverse=True)
    return results


# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    csv_file = "data.csv"

    # 1. Create dummy data if CSV doesn't exist
    if not os.path.exists(csv_file):
        print(f"{csv_file} not found. Creating a synthetic toy dataset to demonstrate...")
        df = pd.DataFrame({
            'smiles': ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CCO', 'C1=CC=CC=C1', 'C', 'O=C=O'] * 50,
            'hall': np.random.randint(1, 533, 250), # Random Hall numbers
            'a': np.random.uniform(5, 15, 250),
            'b': np.random.uniform(5, 15, 250),
            'c': np.random.uniform(5, 15, 250),
            'alpha': np.random.uniform(80, 100, 250),
            'beta': np.random.uniform(80, 120, 250),
            'gamma': np.random.uniform(80, 100, 250)
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
    model = HierarchicalCrystalMDN(num_mixtures=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 4. Training Loop
    epochs = 10
    print("\nStarting Training...")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}, Hall Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Hall Acc: {val_acc:.4f}")

    # 5. Inference Demonstration
    print("\nTraining Complete! Testing Inference...")
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O" # Aspirin
    print(f"Predicting top 3 polymorphs for {test_smiles}:")
    
    # Extract the top 3 overall combinations
    predictions = predict_top_k(model, test_smiles, device, k_hall=3, k_mixtures=2)[:3]

    for i, pred in enumerate(predictions, 1):
        print(f"Rank {i}: Hall {pred['hall_number']} | Polymorph prob: {pred['mix_prob']:.3f} | Combined Prob: {pred['combined_prob']:.3f} | "
              f"Cell [a,b,c,α,β,γ]: {pred['cell_parameters']}")
