import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_networkx
import networkx as nx

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

#check if GPU is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    # Get the name of each GPU
    for i in range(num_gpus):
        device_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {device_name}")
else:
    print("No CUDA GPUs available.")

# Load dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = PygGraphPropPredDataset(name="ogbg-molpcba")
split_idx = dataset.get_idx_split()

#Split data into training and testing datasets
train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)

# Get data information
for graph in train_loader.take(1):
    print(graph)

# Data Visualization
# Select multiple graphs (first three graphs in this example)
graphs = [dataset[i] for i in range(6)]

num_graphs = len(graphs)
num_cols = num_graphs // 2 if num_graphs % 2 == 0 else (num_graphs // 2) + 1

# Create a figure with subplots to display multiple graphs
fig, axes = plt.subplots(2, num_cols, figsize=(15, 5))

# Ensure axes is a 2D array even for odd numbers of graphs
axes = axes.flatten()

# Define node and edge attributes for all graphs
node_colors = "lightblue"
edge_colors = "gray"
node_size = 200
edge_width = 2.0

for i, data in enumerate(graphs):
    # Convert each graph to a NetworkX graph
    G = to_networkx(data, to_undirected=True)
    
    # Create a spring layout for node positioning
    pos = nx.spring_layout(G, seed=42)
    
    # Draw the graph on the corresponding subplot axis
    ax = axes[i]
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, node_size=node_size,
            width=edge_width, font_size=8, alpha=0.8, ax=ax)
    
    ax.set_title(f"Graph {i+1}")

plt.tight_layout()
plt.show()

# Model definition
# class GNN(torch.nn.Module):
class GNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim):
        super().__init__()
        self.atom_encoder = AtomEncoder(node_dim)
        self.bond_encoder = BondEncoder(edge_dim)
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, 128)
        
    def forward(self, x, edge_index, edge_attr, batch):
        # Encode atom and bond features
        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)
        
        # GCN layers
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        
        # Pool node embeddings into graph-level embeddings
        graph_emb = global_mean_pool(x, batch)  # Shape: [num_graphs, hidden_dim]
        
        return self.fc(graph_emb)  # Shape: [num_graphs, output_dim]

# Initialize model
node_dim = dataset.num_node_features
edge_dim = dataset.num_edge_features
hidden_dim = 128
output_dim = 1  # Binary classification
model = GNN(node_dim, edge_dim, hidden_dim, output_dim).to(device)

# Loss and optimizer
criterion = torch.nn.BCEWithLogitsLoss()  # For binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Evaluator
evaluator = Evaluator(name="ogbg-molpcba")

# Training function
def train(model, loader):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(out, batch.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Evaluation function
def evaluate(model, loader):
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            y_pred.append(out)
            y_true.append(batch.y)
    y_pred = torch.nan_to_num(torch.cat(y_pred, dim=0), nan=0.0)
    y_true = torch.nan_to_num(torch.cat(y_true, dim=0), nan=0.0)
    
    return evaluator.eval({"y_true": y_true.cpu().numpy(), "y_pred": y_pred.cpu().numpy()})

# Training loop
for epoch in range(3):
    train_loss = train(model, train_loader)
    test_result = evaluate(model, test_loader)
    print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, AP: {test_result['ap']:.4f}")

