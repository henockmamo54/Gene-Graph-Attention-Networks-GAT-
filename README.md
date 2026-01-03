#  Gene Graph Attention Networks (GAT) Analysis Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/) 

An advanced bioinformatics framework that utilizes **Graph Attention Networks (GAT)** to model gene expression data. This pipeline transforms static expression matrices into dynamic interaction networks, identifying master regulators and functional gene modules through deep representation learning.

---

##  Key Features

* **Graph-Based Representation**: Automatically converts expression data into a $k$-Nearest Neighbors ($k$-NN) graph to capture local and global gene relationships.
* **Robust GAT Architecture**: Implements a multi-layer Graph Attention Network that learns to weigh the importance of gene interactions dynamically via attention coefficients.
* **Noise-Resilient Training**: Employs **Consistency Loss** and stochastic perturbations during training to ensure gene embeddings remain stable despite biological noise.
* **Automated Annotation**: Seamlessly integrates with the **MyGene.info API** to provide real-time Gene Ontology (GO) and pathway mapping for discovered interactions.
* **Subnetwork Extraction**: Automatically isolates and exports high-traffic biological pathways into Cytoscape-compatible CSV formats for downstream analysis.

---

##  Project Structure

```text
├── main.py           # Central execution engine and pipeline logic
├── gat.py            # Model architecture (RobustGeneGNN, Trainer, and Seeder)
├── utils.py          # Helper logic (GeneAnnotator, NetworkAnalyzer, Visualizer)
├── data/             # Input directory for gene expression CSV datasets
└── output/           # Generated results (Reports, CSVs, and PNG visualizations)
    └── subnetworks/  # Extracted pathway-specific edge lists for Cytoscape
```

##  Installation

### 1. Requirements

Ensure you have Python 3.8+ installed. You can install the necessary dependencies using the following command:

```bash
pip install torch torch-geometric pandas numpy scikit-learn matplotlib seaborn requests
```

### 2. Data Preparation

Place your expression CSV (e.g., `ovarian_data.csv`) inside the `data/` folder.

*   **Format**: The pipeline handles automatic transposition if your samples and genes are swapped.
*   **Preprocessing**: Includes automatic normalization (Z-score) and missing value imputation.

##  Usage

To execute the full analytical pipeline:

```bash
python main.py
```

### Configuration

You can fine-tune the analysis by modifying the `CONFIG` dictionary in `main.py`:

*   `n_neighbors`: Controls the density of the initial k-NN graph.
*   `epochs`: Training duration for the GNN (default: 300).
*   `top_n_hubs`: Number of top-ranked genes to report in the final summary.

##  Analytical Outputs

The pipeline populates the `output/` directory with publication-ready assets:

### Data Reports:

*   `gene_hubs.csv`: Ranked list of genes based on Influence Score.
*   `gene_network_cytoscape.csv`: Interaction edges formatted for direct import into Cytoscape.

### Visualizations:

*   `regulator_analysis.png`: Plots influence scores against node degrees to find "Master Regulators."
*   `attention_heatmap.png`: High-resolution interaction strengths between top genes.
*   `training_history.png`: Convergence plots for loss and consistency.

##  Scientific Application

By applying GAT to gene expression, this tool moves beyond simple correlation. The Attention Mechanism allows the model to filter out spurious co-expression and focus on "edges" that are consistently predictive of the gene's functional neighborhood. 
