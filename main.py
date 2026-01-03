import torch
import logging
from pathlib import Path
from torch_geometric.data import Data
from gat import DeterministicSeeder, GeneDataProcessor, RobustGeneGNN, GeneNetworkTrainer
from utils import GeneAnnotator, NetworkAnalyzer, Visualizer, Reporter

# Setup output directories
Path("output/subnetworks").mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)

def main():
    CONFIG = {
        'seed': 42,
        'data_path': 'data/ovarian2016Updated_allGeneNames.csv',
        'n_neighbors': 32,
        'hidden_dim': 128,
        'num_layers': 4,
        'heads': 8,
        'dropout': 0.2,
        'lr': 0.001,
        'epochs': 300,
        'top_n_hubs': 30,
        'top_n_edges': 150,
        'top_n_pathways': 15
    }
    
    DeterministicSeeder.set_seed(CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # 1. Data Prep
        expr_matrix, gene_names = GeneDataProcessor.load_and_preprocess(CONFIG['data_path'])
        edge_index = GeneDataProcessor.build_graph(expr_matrix, CONFIG['n_neighbors'])
        data = Data(x=torch.tensor(expr_matrix, dtype=torch.float32), edge_index=edge_index).to(device)
        
        # 2. Training
        model = RobustGeneGNN(input_dim=expr_matrix.shape[1], hidden_dim=CONFIG['hidden_dim'], 
                              num_layers=CONFIG['num_layers'], heads=CONFIG['heads'], dropout=CONFIG['dropout'])
        trainer = GeneNetworkTrainer(model, device, lr=CONFIG['lr'])
        trainer.train(data, epochs=CONFIG['epochs'])
        
        # 3. Analysis & Annotations
        hub_df = NetworkAnalyzer.compute_hub_scores(model, data, gene_names, device)
        Reporter.print_hub_report(hub_df, top_n=CONFIG['top_n_hubs'])
        hub_df.to_csv('output/gene_hubs.csv', index=False)
        
        edges_df = NetworkAnalyzer.extract_network_edges(model, data, gene_names, device)
        annotations = GeneAnnotator.fetch_pathways(gene_names)
        edges_df['Source_Function'] = edges_df['Source'].map(annotations).fillna('Unknown')
        edges_df['Target_Function'] = edges_df['Target'].map(annotations).fillna('Unknown')
        
        top_edges = edges_df.head(CONFIG['top_n_edges'])
        top_edges.to_csv('output/gene_network_cytoscape.csv', index=False)
        
        # 4. Reports & Subnetworks
        Reporter.print_pathway_report(top_edges, top_n=CONFIG['top_n_pathways'])
        Reporter.export_subnetworks(input_filename='output/gene_network_cytoscape.csv', 
                                    top_n_pathways=CONFIG['top_n_pathways'], 
                                    output_dir='output/subnetworks')
        
        # 5. Visualizations
        logger.info("Generating plots in output/ folder...")
        Visualizer.plot_training_history(trainer.history)
        Visualizer.plot_regulator_analysis(hub_df, top_n=CONFIG['top_n_hubs'])
        Visualizer.plot_hub_distribution(hub_df, top_n=CONFIG['top_n_hubs'])
        Visualizer.plot_network_statistics(top_edges, hub_df, top_pathways=CONFIG['top_n_pathways'])
        Visualizer.plot_attention_heatmap(model, data, gene_names, device, top_n=20)
        Visualizer.create_summary_plot(trainer.history, hub_df, top_edges, model, data, gene_names, device, 
                                       top_pathways=CONFIG['top_n_pathways'])
        
        logger.info("\nAnalysis complete! All results saved to 'output/' folder.")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()