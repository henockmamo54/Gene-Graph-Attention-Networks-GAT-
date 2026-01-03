import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import random
import os
import requests
from sklearn.neighbors import kneighbors_graph
from typing import Tuple, Dict, List, Optional
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DeterministicSeeder:
    """Handles all deterministic seeding operations"""
    
    @staticmethod
    def set_seed(seed: int = 42):
        """Set all random seeds for reproducibility"""
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
    """Handles gene expression data loading and preprocessing"""
    
    @staticmethod
    def load_and_preprocess(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess gene expression data
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Tuple of (normalized_matrix, gene_names)
        """
        logger.info(f"Loading data from {file_path}")
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        gene_names = df.iloc[:, 0].astype(str).values
        numeric_df = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        
        # Auto-detect orientation
        if numeric_df.shape[0] < numeric_df.shape[1]:
            logger.info("Transposing data (samples were in rows)")
            gene_names = numeric_df.columns.values
            expression_matrix = numeric_df.values.T
        else:
            expression_matrix = numeric_df.values
        
        # Clean and normalize
        expression_matrix = np.nan_to_num(expression_matrix, nan=0.0)
        mean = expression_matrix.mean(axis=1, keepdims=True)
        std = expression_matrix.std(axis=1, keepdims=True) + 1e-8
        normalized_matrix = (expression_matrix - mean) / std
        
        logger.info(f"Processed {len(gene_names)} genes across {expression_matrix.shape[1]} samples")
        return normalized_matrix, gene_names
    
    @staticmethod
    def build_graph(expression_matrix: np.ndarray, n_neighbors: int = 12) -> torch.Tensor:
        """Build k-nearest neighbor graph from expression data"""
        logger.info(f"Building k-NN graph with {n_neighbors} neighbors")
        adj = kneighbors_graph(expression_matrix, n_neighbors=n_neighbors, include_self=False)
        edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
        logger.info(f"Graph created with {edge_index.size(1)} edges")
        return edge_index


class RobustGeneGNN(nn.Module):
    """Graph Attention Network for gene interaction modeling"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 3, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(
            GATConv(input_dim, hidden_dim, heads=heads, 
                   concat=True, add_self_loops=False, dropout=dropout)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=heads,
                       concat=True, add_self_loops=False, dropout=dropout)
            )
        
        # Output layer
        self.convs.append(
            GATConv(hidden_dim * heads, 64, heads=1,
                   concat=False, add_self_loops=False, dropout=dropout)
        )
        
        # Edge prediction head
        self.edge_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                return_attn: bool = False):
        """Forward pass through the network"""
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
        """Predict edge probabilities from node embeddings"""
        edge_features = torch.cat([embeddings[edges[0]], embeddings[edges[1]]], dim=-1)
        return self.edge_predictor(edge_features).squeeze()


class GeneAnnotator:
    """Handles gene annotation using MyGene.info API"""
    
    @staticmethod
    def fetch_pathways(gene_list: List[str], chunk_size: int = 1000) -> Dict[str, str]:
        """
        Fetch biological pathways for genes
        
        Args:
            gene_list: List of gene symbols
            chunk_size: Number of genes per API request
            
        Returns:
            Dictionary mapping gene symbols to pathway terms
        """
        logger.info(f"Querying MyGene.info for {len(gene_list)} genes")
        url = 'https://mygene.info/v3/query'
        mapping = {}
        
        for i in range(0, len(gene_list), chunk_size):
            chunk = list(gene_list)[i:i+chunk_size]
            params = {
                'q': ','.join(chunk),
                'scopes': 'symbol',
                'fields': 'go.BP.term',
                'species': 'human'
            }
            
            try:
                response = requests.post(url, data=params, timeout=30)
                response.raise_for_status()
                results = response.json()
                
                for item in results:
                    gene = item.get('query')
                    go_data = item.get('go', {}).get('BP', [])
                    
                    if isinstance(go_data, dict):
                        go_data = [go_data]
                    
                    term = go_data[0].get('term', 'Unknown') if go_data else 'Unknown'
                    mapping[gene] = term
                    
            except Exception as e:
                logger.warning(f"Error fetching chunk {i//chunk_size}: {e}")
                continue
        
        logger.info(f"Retrieved annotations for {len(mapping)} genes")
        return mapping


class NetworkAnalyzer:
    """Analyzes trained gene networks"""
    
    @staticmethod
    def compute_hub_scores(model: RobustGeneGNN, data: Data, 
                          gene_names: np.ndarray, device: torch.device) -> pd.DataFrame:
        """Compute gene hub scores based on attention weights"""
        model.eval()
        with torch.no_grad():
            _, attn_weights = model(data.x.to(device), data.edge_index.to(device), 
                                   return_attn=True)
            edge_index, alpha = attn_weights[0]
            avg_alpha = alpha.mean(dim=1).cpu().numpy()
            edge_index = edge_index.cpu().numpy()
            
            # Aggregate influence scores
            influence_scores = np.zeros(len(gene_names))
            counts = np.zeros(len(gene_names))
            
            for i in range(len(avg_alpha)):
                source_idx = edge_index[1, i]
                if source_idx < len(gene_names):
                    influence_scores[source_idx] += avg_alpha[i]
                    counts[source_idx] += 1
            
            # Average scores
            influence_scores = np.divide(
                influence_scores, counts, 
                out=np.zeros_like(influence_scores), 
                where=counts != 0
            )
            # influence_scores = influence_scores
            
            hub_df = pd.DataFrame({
                'Gene': gene_names,
                'Influence_Score': influence_scores,
                'Num_Connections': counts
            }).sort_values(by='Influence_Score', ascending=False)
            
        return hub_df
    
    @staticmethod
    def extract_network_edges(model: RobustGeneGNN, data: Data,
                            gene_names: np.ndarray, device: torch.device) -> pd.DataFrame:
        """Extract network edges with attention weights"""
        model.eval()
        with torch.no_grad():
            _, attn_weights = model(data.x.to(device), data.edge_index.to(device),
                                   return_attn=True)
            edge_index, alpha = attn_weights[0]
            avg_alpha = alpha.mean(dim=1).cpu().numpy()
            edge_index = edge_index.cpu().numpy()
            
            edges = []
            for i in range(len(avg_alpha)):
                src, tgt = edge_index[1, i], edge_index[0, i]
                if src < len(gene_names) and tgt < len(gene_names):
                    edges.append({
                        'Source': gene_names[src],
                        'Target': gene_names[tgt],
                        'Weight': avg_alpha[i]
                    })
            
            return pd.DataFrame(edges).sort_values(by='Weight', ascending=False)


class GeneNetworkTrainer:
    """Handles model training"""
    
    def __init__(self, model: RobustGeneGNN, device: torch.device,
                 lr: float = 0.0005, weight_decay: float = 1e-5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, 
                                         weight_decay=weight_decay)
        self.history = {'loss': [], 'edge_pred_loss': [], 'consistency_loss': []}
    
    def train_epoch(self, data: Data, noise_level: float = 0.05) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass with augmentation
        emb1 = self.model(data.x + torch.randn_like(data.x) * noise_level, data.edge_index)
        emb2 = self.model(data.x + torch.randn_like(data.x) * noise_level, data.edge_index)
        
        # Generate negative samples
        num_genes = data.x.size(0)
        neg_edges = torch.randint(0, num_genes, data.edge_index.size(), device=self.device)
        
        all_edges = torch.cat([data.edge_index, neg_edges], dim=1)
        targets = torch.cat([
            torch.ones(data.edge_index.size(1)),
            torch.zeros(neg_edges.size(1))
        ]).to(self.device)
        
        # Compute losses
        pred1 = self.model.predict_edges(emb1, all_edges)
        pred2 = self.model.predict_edges(emb2, all_edges)
        
        edge_loss = F.binary_cross_entropy(pred1, targets)
        consistency_loss = F.mse_loss(pred1, pred2)
        total_loss = edge_loss + 0.5 * consistency_loss
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'loss': total_loss.item(),
            'edge_pred_loss': edge_loss.item(),
            'consistency_loss': consistency_loss.item()
        }
    
    def train(self, data: Data, epochs: int = 300, log_interval: int = 50):
        """Full training loop"""
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            metrics = self.train_epoch(data)
            
            for key, value in metrics.items():
                self.history[key].append(value)
            
            if epoch % log_interval == 0:
                logger.info(f"Epoch {epoch:3d} | Loss: {metrics['loss']:.4f} | "
                          f"Edge: {metrics['edge_pred_loss']:.4f} | "
                          f"Consistency: {metrics['consistency_loss']:.4f}")


class Visualizer:
    """Creates visualization plots for analysis results"""
    
    @staticmethod
    def plot_regulator_analysis(hub_df: pd.DataFrame, top_n: int, save_path: str = 'regulator_analysis.png'):
        """
        Visualizes the 'Master Regulator' landscape.
        X-axis: Complexity (Connections), Y-axis: Importance (Summed Influence)
        """
        plt.figure(figsize=(10, 8))
        
        # Color by Influence Score (Sum)
        scatter = plt.scatter(
            hub_df['Num_Connections'], 
            hub_df['Influence_Score'],
            c=hub_df['Influence_Score'], 
            cmap='magma', 
            alpha=0.6, 
            s=100,
            edgecolors='w'
        )
        
        # Label top N regulators from CONFIG
        top_regulators = hub_df.head(top_n)
        for i, row in top_regulators.iterrows():
            plt.annotate(
                row['Gene'], 
                (row['Num_Connections'], row['Influence_Score']),
                xytext=(5, 5), 
                textcoords='offset points', 
                fontsize=9, 
                fontweight='bold'
            )

        plt.xlabel('Degree (Number of Connections)', fontsize=12, fontweight='bold')
        plt.ylabel('Total Influence Score (Sum)', fontsize=12, fontweight='bold')
        plt.title(f'Master Regulator Identification (Top {top_n} Labeled)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Influence Magnitude')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
    @staticmethod
    def plot_training_history(history: Dict[str, List[float]], 
                             save_path: str = 'training_history.png'):
        """Plot training loss curves"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Total loss
        axes[0].plot(history['loss'], linewidth=2, color='#e74c3c')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Total Training Loss', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Edge prediction loss
        axes[1].plot(history['edge_pred_loss'], linewidth=2, color='#3498db')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Edge Prediction Loss', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Consistency loss
        axes[2].plot(history['consistency_loss'], linewidth=2, color='#2ecc71')
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Loss', fontsize=12)
        axes[2].set_title('Consistency Loss', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history saved to {save_path}")
        plt.close()
    
    @staticmethod
    def plot_hub_distribution(hub_df: pd.DataFrame, top_n: int = 30,
                            save_path: str = 'hub_distribution.png'):
        """Plot top gene hubs"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Top N hubs bar plot
        top_hubs = hub_df.head(top_n)
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_hubs)))
        
        ax1.barh(range(len(top_hubs)), top_hubs['Influence_Score'].values, color=colors)
        ax1.set_yticks(range(len(top_hubs)))
        ax1.set_yticklabels(top_hubs['Gene'].values, fontsize=9)
        ax1.set_xlabel('Influence Score', fontsize=12, fontweight='bold')
        ax1.set_title(f'Top {top_n} Gene Hubs by Influence', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # Influence score distribution
        ax2.hist(hub_df['Influence_Score'], bins=50, color='#3498db', 
                alpha=0.7, edgecolor='black')
        ax2.axvline(top_hubs['Influence_Score'].min(), color='red', 
                   linestyle='--', linewidth=2, label=f'Top {top_n} threshold')
        ax2.set_xlabel('Influence Score', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Distribution of Gene Influence Scores', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Hub distribution saved to {save_path}")
        plt.close()
    
    @staticmethod
    def plot_network_statistics(edges_df: pd.DataFrame, hub_df: pd.DataFrame,
                               top_pathways: int = 10,
                               save_path: str = 'network_statistics.png'):
        """Plot comprehensive network statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Edge weight distribution
        ax = axes[0, 0]
        ax.hist(edges_df['Weight'], bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Edge Weight', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Edge Weight Distribution', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Connection degree distribution
        ax = axes[0, 1]
        ax.hist(hub_df['Num_Connections'], bins=30, color='#e67e22', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Number of Connections', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Gene Connectivity Distribution', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Influence vs Connections scatter
        ax = axes[1, 0]
        scatter = ax.scatter(hub_df['Num_Connections'], hub_df['Influence_Score'],
                           alpha=0.6, c=hub_df['Influence_Score'], cmap='viridis', s=50)
        ax.set_xlabel('Number of Connections', fontsize=11, fontweight='bold')
        ax.set_ylabel('Influence Score', fontsize=11, fontweight='bold')
        ax.set_title('Influence Score vs Connectivity', fontsize=12, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Influence Score')
        ax.grid(alpha=0.3)
        
        # Top pathways if available
        ax = axes[1, 1]
        if 'Source_Function' in edges_df.columns:
            all_functions = pd.concat([edges_df['Source_Function'], edges_df['Target_Function']])
            filtered = all_functions[~all_functions.isin(['Unknown', 'Other', 'unknown'])]
            
            if not filtered.empty:
                pathway_counts = filtered.value_counts().head(top_pathways)
                colors = plt.cm.Set3(np.linspace(0, 1, len(pathway_counts)))
                ax.barh(range(len(pathway_counts)), pathway_counts.values, color=colors)
                ax.set_yticks(range(len(pathway_counts)))
                ax.set_yticklabels([label[:40] + '...' if len(label) > 40 else label 
                                   for label in pathway_counts.index], fontsize=9)
                ax.set_xlabel('Number of Interactions', fontsize=11, fontweight='bold')
                ax.set_title(f'Top {top_pathways} Biological Pathways', fontsize=12, fontweight='bold')
                ax.invert_yaxis()
                ax.grid(axis='x', alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No pathway data available', 
                       ha='center', va='center', fontsize=12)
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'No pathway data available', 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Network statistics saved to {save_path}")
        plt.close()
    
    @staticmethod
    def plot_attention_heatmap(model: RobustGeneGNN, data: Data, 
                              gene_names: np.ndarray, device: torch.device,
                              top_n: int = 30, save_path: str = 'attention_heatmap.png'):
        """Plot attention weights heatmap for top hub genes"""
        model.eval()
        with torch.no_grad():
            _, attn_weights = model(data.x.to(device), data.edge_index.to(device), 
                                   return_attn=True)
            edge_index, alpha = attn_weights[0]
            avg_alpha = alpha.mean(dim=1).cpu().numpy()
            edge_index = edge_index.cpu().numpy()
            
            # Get top hub genes
            influence_scores = np.zeros(len(gene_names))
            counts = np.zeros(len(gene_names))
            
            for i in range(len(avg_alpha)):
                source_idx = edge_index[1, i]
                if source_idx < len(gene_names):
                    influence_scores[source_idx] += avg_alpha[i]
                    counts[source_idx] += 1
            
            influence_scores = np.divide(influence_scores, counts, 
                                        out=np.zeros_like(influence_scores), 
                                        where=counts != 0)
            # influence_scores = influence_scores
            
            top_indices = np.argsort(influence_scores)[-top_n:][::-1]
            top_genes = gene_names[top_indices]
            
            # Build attention matrix for top genes
            attn_matrix = np.zeros((top_n, top_n))
            gene_to_idx = {gene: i for i, gene in enumerate(top_genes)}
            
            for i in range(len(avg_alpha)):
                src, tgt = edge_index[1, i], edge_index[0, i]
                if src < len(gene_names) and tgt < len(gene_names):
                    src_gene = gene_names[src]
                    tgt_gene = gene_names[tgt]
                    if src_gene in gene_to_idx and tgt_gene in gene_to_idx:
                        attn_matrix[gene_to_idx[src_gene], gene_to_idx[tgt_gene]] = avg_alpha[i]
            
            # Plot heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(attn_matrix, xticklabels=top_genes, yticklabels=top_genes,
                       cmap='YlOrRd', cbar_kws={'label': 'Attention Weight'},
                       square=True, linewidths=0.5, ax=ax)
            ax.set_title(f'Attention Weights Among Top {top_n} Hub Genes', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Target Gene', fontsize=12, fontweight='bold')
            ax.set_ylabel('Source Gene', fontsize=12, fontweight='bold')
            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.yticks(rotation=0, fontsize=9)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention heatmap saved to {save_path}")
            plt.close()
    
    @staticmethod
    def create_summary_plot(history: Dict[str, List[float]], hub_df: pd.DataFrame,
                          edges_df: pd.DataFrame, model: RobustGeneGNN, data: Data,
                          gene_names: np.ndarray, device: torch.device,
                          top_pathways: int = 8,
                          save_path: str = 'analysis_summary.png'):
        """Create comprehensive summary visualization"""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        # Training loss
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(history['loss'], linewidth=2, color='#e74c3c', label='Total Loss')
        ax1.set_xlabel('Epoch', fontsize=10)
        ax1.set_ylabel('Loss', fontsize=10)
        ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Component losses
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(history['edge_pred_loss'], linewidth=2, color='#3498db', label='Edge Pred')
        ax2.plot(history['consistency_loss'], linewidth=2, color='#2ecc71', label='Consistency')
        ax2.set_xlabel('Epoch', fontsize=10)
        ax2.set_ylabel('Loss', fontsize=10)
        ax2.set_title('Loss Components', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Edge weight distribution
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(edges_df['Weight'], bins=40, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Edge Weight', fontsize=10)
        ax3.set_ylabel('Frequency', fontsize=10)
        ax3.set_title('Edge Weight Distribution', fontsize=12, fontweight='bold')
        ax3.grid(alpha=0.3)
        
        # Top 15 hubs
        ax4 = fig.add_subplot(gs[1, :2])
        top_hubs = hub_df.head(15)
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_hubs)))
        ax4.barh(range(len(top_hubs)), top_hubs['Influence_Score'].values, color=colors)
        ax4.set_yticks(range(len(top_hubs)))
        ax4.set_yticklabels(top_hubs['Gene'].values, fontsize=10)
        ax4.set_xlabel('Influence Score', fontsize=10)
        ax4.set_title('Top 15 Gene Hubs', fontsize=12, fontweight='bold')
        ax4.invert_yaxis()
        ax4.grid(axis='x', alpha=0.3)
        
        # Influence score distribution
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.hist(hub_df['Influence_Score'], bins=40, color='#3498db', alpha=0.7, edgecolor='black')
        ax5.set_xlabel('Influence Score', fontsize=10)
        ax5.set_ylabel('Frequency', fontsize=10)
        ax5.set_title('Score Distribution', fontsize=12, fontweight='bold')
        ax5.grid(alpha=0.3)
        
        # Connectivity analysis
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.scatter(hub_df['Num_Connections'], hub_df['Influence_Score'],
                   alpha=0.5, c=hub_df['Influence_Score'], cmap='viridis', s=30)
        ax6.set_xlabel('Connections', fontsize=10)
        ax6.set_ylabel('Influence Score', fontsize=10)
        ax6.set_title('Influence vs Connectivity', fontsize=12, fontweight='bold')
        ax6.grid(alpha=0.3)
        
        # Degree distribution
        ax7 = fig.add_subplot(gs[2, 1])
        ax7.hist(hub_df['Num_Connections'], bins=30, color='#e67e22', alpha=0.7, edgecolor='black')
        ax7.set_xlabel('Number of Connections', fontsize=10)
        ax7.set_ylabel('Frequency', fontsize=10)
        ax7.set_title('Connectivity Distribution', fontsize=12, fontweight='bold')
        ax7.grid(alpha=0.3)
        
        # Top pathways
        ax8 = fig.add_subplot(gs[2, 2])
        if 'Source_Function' in edges_df.columns:
            all_functions = pd.concat([edges_df['Source_Function'], edges_df['Target_Function']])
            filtered = all_functions[~all_functions.isin(['Unknown', 'Other', 'unknown'])]
            
            if not filtered.empty:
                pathway_counts = filtered.value_counts().head(top_pathways)
                colors_pw = plt.cm.Set3(np.linspace(0, 1, len(pathway_counts)))
                ax8.barh(range(len(pathway_counts)), pathway_counts.values, color=colors_pw)
                ax8.set_yticks(range(len(pathway_counts)))
                ax8.set_yticklabels([label[:25] + '...' if len(label) > 25 else label 
                                    for label in pathway_counts.index], fontsize=8)
                ax8.set_xlabel('Interactions', fontsize=10)
                ax8.set_title(f'Top {top_pathways} Pathways', fontsize=12, fontweight='bold')
                ax8.invert_yaxis()
                ax8.grid(axis='x', alpha=0.3)
            else:
                ax8.text(0.5, 0.5, 'No pathway\ndata available', 
                        ha='center', va='center', fontsize=10)
                ax8.axis('off')
        else:
            ax8.text(0.5, 0.5, 'No pathway\ndata available', 
                    ha='center', va='center', fontsize=10)
            ax8.axis('off')
        
        fig.suptitle('Gene Network Analysis Summary', fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Summary plot saved to {save_path}")
        plt.close()


class Reporter:
    """Generates analysis reports"""

    
    @staticmethod
    def print_hub_report(hub_df: pd.DataFrame, top_n: int = 30):
        """Print top gene hubs sorted by influence (desc) then connections (desc)"""
        # Ensure proper sorting before display
        hub_df['Influence_Score'] = hub_df['Influence_Score'].round(5)
        sorted_hub_df = hub_df.sort_values(
            by=['Influence_Score', 'Num_Connections'], 
            ascending=[False, False]
        ).reset_index(drop=True)
        
        print("\n" + "="*70)
        print("TOP GENE HUBS BY INFLUENCE SCORE")
        print("="*70)
        print(f"{'Rank':<6}{'Gene':<20}{'Influence':<15}{'Connections'}")
        print("-"*70)
        
        for i, (_, row) in enumerate(sorted_hub_df.head(top_n).iterrows(), 1):
            print(f"{i:<6}{row['Gene']:<20}{row['Influence_Score']:<15.4f}"
                  f"{int(row['Num_Connections'])}")
    
    @staticmethod
    def print_pathway_report(network_df: pd.DataFrame, top_n: int = 10):
        """Print top pathways"""
        all_functions = pd.concat([
            network_df['Source_Function'],
            network_df['Target_Function']
        ])
        filtered = all_functions[~all_functions.isin(['Unknown', 'Other', 'unknown'])]
        
        if filtered.empty:
            logger.warning("No specific pathways found in network")
            return
        
        pathway_counts = filtered.value_counts()
        
        print("\n" + "="*70)
        print(f"TOP {top_n} BIOLOGICAL PATHWAYS")
        print("="*70)
        
        for i, (pathway, count) in enumerate(pathway_counts.head(top_n).items(), 1):
            print(f"{i:2d}. {pathway} ({count} interactions)")

    @staticmethod
    def export_subnetworks(input_filename: str = "gene_network_cytoscape.csv",
                          top_n_pathways: int = 3,
                          output_dir: str = "subnetworks") -> List[Dict[str, any]]:
        """
        Export subnetworks for top N pathways
        
        Args:
            input_filename: Path to the full network CSV file
            top_n_pathways: Number of top pathways to export subnetworks for
            output_dir: Directory to save subnetwork files
            
        Returns:
            List of dictionaries containing pathway info and statistics
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load the full network
        df = pd.read_csv(input_filename)
        
        # Identify top pathways (excluding Unknowns)
        all_functions = pd.concat([df['Source_Function'], df['Target_Function']])
        filtered = all_functions[~all_functions.isin(['Unknown', 'Other', 'unknown'])]
        
        if filtered.empty:
            logger.warning("No valid pathways found for subnetwork export")
            return []
        
        pathway_counts = filtered.value_counts().head(top_n_pathways)
        
        results = []
        logger.info(f"\nExporting subnetworks for top {top_n_pathways} pathways...")
        print("\n" + "="*70)
        print(f"EXPORTING TOP {top_n_pathways} PATHWAY SUBNETWORKS")
        print("="*70)
        
        for rank, (pathway, count) in enumerate(pathway_counts.items(), 1):
            # Filter edges where either Source or Target belongs to this pathway
            sub_df = df[(df['Source_Function'] == pathway) | (df['Target_Function'] == pathway)]
            
            # Create safe filename
            safe_pathway_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' 
                                       for c in pathway)[:50]
            output_filename = f"{output_dir}/pathway_{rank}_{safe_pathway_name}.csv"
            
            # Save subnetwork
            sub_df.to_csv(output_filename, index=False)
            
            # Collect statistics
            unique_genes = set(sub_df['Source'].unique()) | set(sub_df['Target'].unique())
            avg_weight = sub_df['Weight'].mean()
            
            result = {
                'rank': rank,
                'pathway': pathway,
                'total_interactions': count,
                'edges_in_subnetwork': len(sub_df),
                'unique_genes': len(unique_genes),
                'avg_edge_weight': avg_weight,
                'filename': output_filename
            }
            results.append(result)
            
            # Print summary
            print(f"\nRank {rank}: {pathway}")
            print(f"  Total interactions in network: {count}")
            print(f"  Edges in subnetwork: {len(sub_df)}")
            print(f"  Unique genes involved: {len(unique_genes)}")
            print(f"  Average edge weight: {avg_weight:.4f}")
            print(f"  Saved to: {output_filename}")
        
        print("\n" + "="*70)
        logger.info(f"Successfully exported {len(results)} subnetworks to '{output_dir}/' directory")
        
        # Create summary CSV
        summary_df = pd.DataFrame(results)
        summary_filename = f"{output_dir}/subnetwork_summary.csv"
        summary_df.to_csv(summary_filename, index=False)
        logger.info(f"Subnetwork summary saved to {summary_filename}")
        
        return results


def main():
    """Main execution pipeline"""
    
    # Configuration
    CONFIG = {
        'seed': 42,
        'data_path': 'ovarian2016Updated_allGeneNames.csv',
        'n_neighbors': 32,
        'hidden_dim': 128,
        'num_layers': 4,
        'heads': 8,
        'dropout': 0.2,
        'lr': 0.001,
        'epochs': 300,
        'top_n_hubs': 30,
        'top_n_edges': 150,
        'top_n_pathways': 15  # Added parameter for pathway display
    }
    
    # Initialize
    DeterministicSeeder.set_seed(CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Load and preprocess data
        expr_matrix, gene_names = GeneDataProcessor.load_and_preprocess(CONFIG['data_path'])
        edge_index = GeneDataProcessor.build_graph(expr_matrix, CONFIG['n_neighbors'])
        
        # Create PyG data object
        data = Data(
            x=torch.tensor(expr_matrix, dtype=torch.float32),
            edge_index=edge_index
        ).to(device)
        
        # Initialize model
        model = RobustGeneGNN(
            input_dim=expr_matrix.shape[1],
            hidden_dim=CONFIG['hidden_dim'],
            num_layers=CONFIG['num_layers'],
            heads=CONFIG['heads'],
            dropout=CONFIG['dropout']
        )
        
        # Train model
        trainer = GeneNetworkTrainer(model, device, lr=CONFIG['lr'])
        trainer.train(data, epochs=CONFIG['epochs'])
        
        # Analyze results
        hub_df = NetworkAnalyzer.compute_hub_scores(model, data, gene_names, device)
        Reporter.print_hub_report(hub_df, top_n=CONFIG['top_n_hubs'])
        
        # Export hub report
        hub_df.to_csv('gene_hubs.csv', index=False)
        logger.info("Saved gene hubs to gene_hubs.csv")
        
        # Extract network edges
        edges_df = NetworkAnalyzer.extract_network_edges(model, data, gene_names, device)
        
        # Annotate with pathways
        annotations = GeneAnnotator.fetch_pathways(gene_names)
        edges_df['Source_Function'] = edges_df['Source'].map(annotations).fillna('Unknown')
        edges_df['Target_Function'] = edges_df['Target'].map(annotations).fillna('Unknown')
        
        # Export network
        top_edges = edges_df.head(CONFIG['top_n_edges'])
        top_edges.to_csv('gene_network_cytoscape.csv', index=False)
        logger.info(f"Saved top {len(top_edges)} edges to gene_network_cytoscape.csv")
        
        # Report pathways
        Reporter.print_pathway_report(top_edges, top_n=CONFIG['top_n_pathways'])

        # Export subnetworks for top pathways
        subnetwork_results = Reporter.export_subnetworks(
            input_filename='gene_network_cytoscape.csv',
            top_n_pathways=CONFIG['top_n_pathways'],
            output_dir='subnetworks'
        )
        
        # Generate all visualizations
        logger.info("\nGenerating visualizations...")
        
        # Training history plot
        Visualizer.plot_training_history(trainer.history)

        # New Regulator Analysis plot using CONFIG
        Visualizer.plot_regulator_analysis(hub_df, top_n=CONFIG['top_n_hubs'])
        
        # Hub distribution plot
        Visualizer.plot_hub_distribution(hub_df, top_n=CONFIG['top_n_hubs'])
        
        # Network statistics plot
        Visualizer.plot_network_statistics(top_edges, hub_df, top_pathways=CONFIG['top_n_pathways'])
        
        # Attention heatmap
        Visualizer.plot_attention_heatmap(model, data, gene_names, device, top_n=20)
        
        # Comprehensive summary plot
        Visualizer.create_summary_plot(
            trainer.history, hub_df, top_edges, model, data, gene_names, device,
            top_pathways=CONFIG['top_n_pathways']
        )
       
        
        logger.info("\nAnalysis complete!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()