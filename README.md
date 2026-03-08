# Graph Neural Networks — TP3

> MVA Master's Program — Graphs in Machine Learning (2025–2026)  
> **Author:** Adam Mounir

## Overview

This project explores **Graph Neural Networks (GNNs)** through two main tasks:

1. **Node Classification on PPI Graphs** — Design and train a GNN model that achieves >93% micro F1-score on the Protein-Protein Interaction (PPI) dataset.
2. **Conv2D as Message Passing** — Implement a standard 2D convolution as a Message Passing Neural Network (MPNN), demonstrating the theoretical connection between CNNs and GNNs.

## Project Structure

```
├── TP3.ipynb               # Main notebook with all questions, code, and explanations
├── class_model_gnn.py      # GAT-based model architecture (StudentModel)
├── model.pth               # Trained model weights
├── message_passing.py      # Conv2D reimplemented as message passing (3 functions)
├── raw/                    # Raw PPI dataset files
├── processed/              # Preprocessed PyTorch Geometric dataset
└── README.md
```

## Part 1 — PPI Node Classification

### Task

The [PPI dataset](https://cs.stanford.edu/~jure/pubs/pathways-psb18.pdf) consists of 24 protein-protein interaction graphs (20 train / 2 val / 2 test). Each node represents a protein with 50 features, and the goal is to predict 121 binary labels per node (multi-label classification).

### Approach

The model uses **Graph Attention Networks (GAT)** ([Veličković et al., 2018](https://arxiv.org/abs/1710.10903)) with the following design choices:

- **3 GATConv layers** with multi-head attention (4, 4, and 6 heads)
- **Residual / skip connections** via linear projections
- **Layer normalization** after each attention layer
- **ELU** activation function
- **BCEWithLogitsLoss** for multi-label classification

### Results

| Metric | Score |
|--------|-------|
| Validation F1 (micro) | **~95.5%** |
| Target | ≥ 93% |

## Part 2 — Conv2D as Message Passing

This section demonstrates that a standard `torch.nn.Conv2d` operation can be exactly replicated using PyTorch Geometric's `MessagePassing` framework.

### Key Idea

Each pixel in an image becomes a **node** in a graph. Edges connect each pixel to its neighbors within the convolution kernel window. Edge attributes encode the relative kernel position. The message function applies the corresponding weight slice to the source node features, and sum aggregation replicates the convolution.

### Implemented Functions (`message_passing.py`)

| Function | Description |
|----------|-------------|
| `image_to_graph()` | Converts a `(C, H, W)` image tensor to a PyG `Data` object |
| `graph_to_image()` | Reshapes node features `(N, C)` back to `(C, H, W)` |
| `Conv2dMessagePassing` | `MessagePassing` subclass that exactly simulates `Conv2d` (including bias) |

### Verification

Both roundtrip identity (`image → graph → image`) and numerical equivalence with `torch.nn.Conv2d` are validated with assertions.

## Setup

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- PyTorch Geometric ≥ 2.0
- scikit-learn, matplotlib, numpy

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio
pip install torch-geometric scikit-learn matplotlib numpy
```

### Running

Open `TP3.ipynb` in Jupyter or VS Code and run all cells sequentially. The trained model weights are already saved in `model.pth`.

## License

Academic project — MVA Master's Program, ENS Paris-Saclay.
