import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import random
import os
import logging
from pathlib import Path
from sklearn.neighbors import kneighbors_graph
from typing import Tuple, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeterministicSeeder:
    @staticmethod
    def set_seed(seed: int = 42):
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        logger.info(f"Deterministic mode enabled with seed: {seed}")

class GeneDataProcessor:
    @staticmethod
    def load_and_preprocess(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        logger.info(f"Loading data from {file_path}")
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        df = pd.read_csv(file_path)
        gene_names = df.iloc[:, 0].astype(str).values
        numeric_df = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        if numeric_df.shape[0] < numeric_df.shape[1]:
            logger.info("Transposing data (samples were in rows)")
            gene_names = numeric_df.columns.values
            expression_matrix = numeric_df.values.T
        else:
            expression_matrix = numeric_df.values
        expression_matrix = np.nan_to_num(expression_matrix, nan=0.0)
        mean = expression_matrix.mean(axis=1, keepdims=True)
        std = expression_matrix.std(axis=1, keepdims=True) + 1e-8
        normalized_matrix = (expression_matrix - mean) / std
        logger.info(f"Processed {len(gene_names)} genes across {expression_matrix.shape[1]} samples")
        return normalized_matrix, gene_names
    
    @staticmethod
    def build_graph(expression_matrix: np.ndarray, n_neighbors: int = 12) -> torch.Tensor:
        logger.info(f"Building k-NN graph with {n_neighbors} neighbors")
        adj = kneighbors_graph(expression_matrix, n_neighbors=n_neighbors, include_self=False)
        edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
        logger.info(f"Graph created with {edge_index.size(1)} edges")
        return edge_index

class RobustGeneGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, concat=True, add_self_loops=False, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, add_self_loops=False, dropout=dropout))
        self.convs.append(GATConv(hidden_dim * heads, 64, heads=1, concat=False, add_self_loops=False, dropout=dropout))
        self.edge_predictor = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1), nn.Sigmoid())
    
    def forward(self, x, edge_index, return_attn: bool = False):
        attn_weights = []
        for i, conv in enumerate(self.convs):
            if return_attn:
                x, (ei, alpha) = conv(x, edge_index, return_attention_weights=True)
                attn_weights.append((ei, alpha))
            else:
                x = conv(x, edge_index)
            x = F.elu(x)
        return (x, attn_weights) if return_attn else x
    
    def predict_edges(self, embeddings: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        edge_features = torch.cat([embeddings[edges[0]], embeddings[edges[1]]], dim=-1)
        return self.edge_predictor(edge_features).squeeze()

class GeneNetworkTrainer:
    def __init__(self, model: RobustGeneGNN, device: torch.device, lr: float = 0.0005, weight_decay: float = 1e-5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.history = {'loss': [], 'edge_pred_loss': [], 'consistency_loss': []}
    
    def train_epoch(self, data: Data, noise_level: float = 0.05) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        emb1 = self.model(data.x + torch.randn_like(data.x) * noise_level, data.edge_index)
        emb2 = self.model(data.x + torch.randn_like(data.x) * noise_level, data.edge_index)
        num_genes = data.x.size(0)
        neg_edges = torch.randint(0, num_genes, data.edge_index.size(), device=self.device)
        all_edges = torch.cat([data.edge_index, neg_edges], dim=1)
        targets = torch.cat([torch.ones(data.edge_index.size(1)), torch.zeros(neg_edges.size(1))]).to(self.device)
        pred1 = self.model.predict_edges(emb1, all_edges)
        pred2 = self.model.predict_edges(emb2, all_edges)
        edge_loss = F.binary_cross_entropy(pred1, targets)
        consistency_loss = F.mse_loss(pred1, pred2)
        total_loss = edge_loss + 0.5 * consistency_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return {'loss': total_loss.item(), 'edge_pred_loss': edge_loss.item(), 'consistency_loss': consistency_loss.item()}
    
    def train(self, data: Data, epochs: int = 300, log_interval: int = 50):
        logger.info(f"Starting training for {epochs} epochs")
        for epoch in range(epochs):
            metrics = self.train_epoch(data)
            for key, value in metrics.items():
                self.history[key].append(value)
            if epoch % log_interval == 0:
                logger.info(f"Epoch {epoch:3d} | Loss: {metrics['loss']:.4f}")