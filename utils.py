import torch
import numpy as np
import pandas as pd
import requests
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, List

logger = logging.getLogger(__name__)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class GeneAnnotator:
    @staticmethod
    def fetch_pathways(gene_list: List[str], chunk_size: int = 1000) -> Dict[str, str]:
        logger.info(f"Querying MyGene.info for {len(gene_list)} genes")
        url = 'https://mygene.info/v3/query'
        mapping = {}
        for i in range(0, len(gene_list), chunk_size):
            chunk = list(gene_list)[i:i+chunk_size]
            params = {'q': ','.join(chunk), 'scopes': 'symbol', 'fields': 'go.BP.term', 'species': 'human'}
            try:
                response = requests.post(url, data=params, timeout=30)
                response.raise_for_status()
                results = response.json()
                for item in results:
                    gene = item.get('query')
                    go_data = item.get('go', {}).get('BP', [])
                    if isinstance(go_data, dict): go_data = [go_data]
                    term = go_data[0].get('term', 'Unknown') if go_data else 'Unknown'
                    mapping[gene] = term
            except Exception as e:
                logger.warning(f"Error fetching chunk {i//chunk_size}: {e}")
                continue
        return mapping

class NetworkAnalyzer:
    @staticmethod
    def compute_hub_scores(model, data, gene_names, device) -> pd.DataFrame:
        model.eval()
        with torch.no_grad():
            _, attn_weights = model(data.x.to(device), data.edge_index.to(device), return_attn=True)
            edge_index, alpha = attn_weights[0]
            avg_alpha = alpha.mean(dim=1).cpu().numpy()
            edge_index = edge_index.cpu().numpy()
            influence_scores = np.zeros(len(gene_names)); counts = np.zeros(len(gene_names))
            for i in range(len(avg_alpha)):
                source_idx = edge_index[1, i]
                if source_idx < len(gene_names):
                    influence_scores[source_idx] += avg_alpha[i]
                    counts[source_idx] += 1
            # influence_scores = np.divide(influence_scores, counts, out=np.zeros_like(influence_scores), where=counts != 0)
            influence_scores
            return pd.DataFrame({'Gene': gene_names, 'Influence_Score': influence_scores, 'Num_Connections': counts}).sort_values(by='Influence_Score', ascending=False)

    @staticmethod
    def extract_network_edges(model, data, gene_names, device) -> pd.DataFrame:
        model.eval()
        with torch.no_grad():
            _, attn_weights = model(data.x.to(device), data.edge_index.to(device), return_attn=True)
            edge_index, alpha = attn_weights[0]
            avg_alpha = alpha.mean(dim=1).cpu().numpy(); edge_index = edge_index.cpu().numpy()
            edges = []
            for i in range(len(avg_alpha)):
                src, tgt = edge_index[1, i], edge_index[0, i]
                if src < len(gene_names) and tgt < len(gene_names):
                    edges.append({'Source': gene_names[src], 'Target': gene_names[tgt], 'Weight': avg_alpha[i]})
            return pd.DataFrame(edges).sort_values(by='Weight', ascending=False)


class Visualizer:
    """Creates visualization plots for analysis results"""
    
    @staticmethod
    def plot_regulator_analysis(hub_df: pd.DataFrame, top_n: int, save_path: str = 'output/regulator_analysis.png'):
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
                             save_path: str = 'output/training_history.png'):
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
                            save_path: str = 'output/hub_distribution.png'):
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
                               save_path: str = 'output/network_statistics.png'):
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
    def plot_attention_heatmap(model, data,gene_names, device,
                              top_n: int = 30, save_path: str = 'output/attention_heatmap.png'):
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
                          edges_df: pd.DataFrame, model, data,
                          gene_names: np.ndarray, device: torch.device,
                          top_pathways: int = 8,
                          save_path: str = 'output/analysis_summary.png'):
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
    @staticmethod
    def print_hub_report(hub_df, top_n=30):
        print("\n" + "="*70 + "\nTOP GENE HUBS\n" + "="*70)
        for i, row in hub_df.head(top_n).iterrows():
            print(f"{row['Gene']:<20} | Influence: {row['Influence_Score']:.4f} | Degree: {int(row['Num_Connections'])}")

    @staticmethod
    def print_pathway_report(network_df, top_n=10):
        all_f = pd.concat([network_df['Source_Function'], network_df['Target_Function']])
        counts = all_f[~all_f.isin(['Unknown', 'Other'])].value_counts()
        print("\n" + "="*70 + "\nTOP PATHWAYS\n" + "="*70)
        for p, c in counts.head(top_n).items(): print(f"{p}: {c} interactions")

    @staticmethod
    def export_subnetworks(input_filename, top_n_pathways=3, output_dir="output/subnetworks"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(input_filename)
        all_f = pd.concat([df['Source_Function'], df['Target_Function']])
        filtered = all_f[~all_f.isin(['Unknown', 'Other', 'unknown'])]
        if filtered.empty: return []
        pathway_counts = filtered.value_counts().head(top_n_pathways)
        results = []
        for rank, (pathway, count) in enumerate(pathway_counts.items(), 1):
            sub_df = df[(df['Source_Function'] == pathway) | (df['Target_Function'] == pathway)]
            safe_name = "".join(c if c.isalnum() else '_' for c in pathway)[:50]
            out_file = f"{output_dir}/pathway_{rank}_{safe_name}.csv"
            sub_df.to_csv(out_file, index=False)
            results.append({'rank': rank, 'pathway': pathway, 'filename': out_file})
        pd.DataFrame(results).to_csv(f"{output_dir}/subnetwork_summary.csv", index=False)
        return results