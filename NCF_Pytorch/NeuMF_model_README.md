# NeuMF Model Implementation

## Overview
This directory contains the implementation of the **Neural Matrix Factorization (NeuMF)** model, which is the most sophisticated model in the Neural Collaborative Filtering (NCF) framework. NeuMF combines both GMF (element-wise product) and MLP (concatenation + feed-forward) approaches to leverage the strengths of both methods.

## Files

### 1. `NeuMF_model.py`
Contains the core NeuMF model implementation and training function.

#### Key Components:

**NeuMF Class:**
- **Purpose**: Implements the Neural Matrix Factorization model that combines GMF and MLP
- **Architecture**: 
  - **GMF Branch**:
    - User embeddings: `nn.Embedding(num_users, latent_dim)`
    - Item embeddings: `nn.Embedding(num_items, latent_dim)`
    - Element-wise product: `user_emb * item_emb`
  - **MLP Branch**:
    - User embeddings: `nn.Embedding(num_users, layers[0]//2)`
    - Item embeddings: `nn.Embedding(num_items, layers[0]//2)`
    - Concatenation: `torch.concat([user_emb, item_emb], dim=-1)`
    - MLP layers: Configurable fully connected layers with ReLU
  - **Fusion Layer**: Concatenates GMF and MLP outputs
  - **Output Layer**: `nn.Linear(final_dim, 1)` with sigmoid activation

**Forward Pass:**
1. **GMF Branch**: Get embeddings → element-wise product
2. **MLP Branch**: Get embeddings → concatenate → MLP layers
3. **Fusion**: Concatenate GMF and MLP outputs
4. **Prediction**: Final linear layer + sigmoid

**Weight Initialization:**
- Embeddings: Normal distribution (std=0.01)
- Linear layers: Xavier uniform initialization
- Biases: Zero initialization

**train_NeuMF_model Function:**
- **Purpose**: Training loop for the NeuMF model
- **Features**:
  - Multiple optimizer support (Adam, Adagrad, RMSprop, SGD)
  - Binary Cross-Entropy loss
  - Real-time evaluation with Hit Rate and NDCG metrics
  - Model checkpointing (saves best model)
  - Progress tracking with timing

#### Parameters:
- `num_users`: Number of unique users in the dataset
- `num_items`: Number of unique items in the dataset
- `latent_dim`: GMF embedding dimension (typically 10)
- `layers`: MLP layer architecture (e.g., [32, 16, 8])

### 2. `NeuMF_model_train.py`
Training script that orchestrates the NeuMF model training process.

#### Configuration:
```python
configurations = {
    "train_data": "NCF_Pytorch/train_data.csv",
    "test_data": "NCF_Pytorch/test_data.csv", 
    "test_negative_data": "NCF_Pytorch/test_negative_data.csv",
    "dataset": "ml-1m",
    "regs": [0, 0],           # L1, L2 regularization
    "layers": [32, 16, 8],    # MLP layer architecture
    "lr": 0.001,              # Learning rate
    "batch_size": 256,        # Batch size
    "epochs": 3,              # Number of training epochs
    "learner": "adam",        # Optimizer
    "num_factors": 10,        # GMF latent dimensions
    "num_neg": 2,             # Negative samples per positive
    "out": True,              # Save model flag
    "topK": 10                # Evaluation top-K
}
```

#### Workflow:
1. **Data Loading**: Creates train and test dataset objects
2. **Model Initialization**: Instantiates NeuMF model with dataset dimensions and configs
3. **DataLoader Setup**: Creates PyTorch DataLoaders for batching
4. **Training**: Calls the training function with all components

## Usage

### Training the Model:
```bash
python NeuMF_model_train.py
```

### Using the Model:
```python
from NeuMF_model import NeuMF

# Initialize model
model = NeuMF(num_users=6040, num_items=3706, latent_dim=10, layers=[32, 16, 8])

# Forward pass
user_ids = torch.tensor([1, 2, 3])
item_ids = torch.tensor([10, 20, 30])
predictions = model(user_ids, item_ids)
```

## Model Architecture

```
Input: (user_id, item_id)
    ↓
┌─────────────────┬─────────────────┐
│   GMF Branch    │   MLP Branch    │
│                 │                 │
│ User Emb (10)   │ User Emb (16)   │
│ Item Emb (10)   │ Item Emb (16)   │
│                 │                 │
│ Element-wise    │ Concatenate     │
│ Product (10)    │ (32)            │
│                 │                 │
│                 │ MLP Layer 1     │
│                 │ (32→16) + ReLU  │
│                 │                 │
│                 │ MLP Layer 2     │
│                 │ (16→8) + ReLU   │
└─────────────────┴─────────────────┘
    ↓                       ↓
    └───── Concatenate ─────┘
            ↓
    Final Layer (18→1)
            ↓
    Sigmoid Activation
            ↓
    Output: [0, 1]
```

## Key Features

1. **Hybrid Architecture**: Combines GMF and MLP approaches
2. **Dual Embeddings**: Separate embeddings for GMF and MLP branches
3. **Fusion Learning**: Learns optimal combination of both approaches
4. **Advanced Initialization**: Proper weight initialization for stable training
5. **Binary Classification**: Predicts interaction probability (0-1)

## Architecture Details

### GMF Branch:
- **Embedding Size**: `latent_dim` (typically 10)
- **Interaction**: Element-wise product
- **Output Size**: `latent_dim`

### MLP Branch:
- **Embedding Size**: `layers[0]//2` each (typically 16 each)
- **Concatenated Size**: `layers[0]` (typically 32)
- **Hidden Layers**: Configurable (e.g., 32→16→8)
- **Output Size**: `layers[-1]` (typically 8)

### Fusion Layer:
- **Input Size**: `latent_dim + layers[-1]` (typically 10+8=18)
- **Output Size**: 1
- **Activation**: Sigmoid

## Dependencies

- PyTorch
- NumPy
- Pandas
- Custom modules: `ml_1m_dataset`, `NCF_evaluation`

## Output

The training process outputs:
- Model checkpoints: `{dataset}_NeuMF_{num_factors}.pth`
- Training metrics: Hit Rate, NDCG, Loss per epoch
- Best model identification and saving

## Performance

Typical performance on MovieLens-1M:
- Hit Rate @10: ~0.60-0.65
- NDCG @10: ~0.35-0.40
- Training time: ~70-85 seconds per epoch (CPU)

## Advantages over GMF and MLP

1. **Best of Both Worlds**: Combines linear (GMF) and non-linear (MLP) interactions
2. **Complementary Learning**: GMF learns multiplicative patterns, MLP learns additive patterns
3. **Higher Performance**: Typically achieves better results than individual models
4. **Flexible Architecture**: Can be tuned for different datasets
5. **Robust Learning**: More stable training due to dual pathways

## Pretrained Model Support

The model includes commented code for loading pretrained GMF and MLP models:
- Can initialize NeuMF with pretrained GMF embeddings
- Can initialize NeuMF with pretrained MLP embeddings
- Supports strict and non-strict loading modes

## Notes

- The model uses binary cross-entropy loss for implicit feedback
- Negative sampling is handled in the dataset class
- Evaluation uses the standard leave-one-out protocol
- Proper weight initialization is crucial for training stability
- The MLP first layer size must be even (for user+item embedding concatenation)
- At least 2 layers are required in the MLP branch
- The model is more complex than GMF/MLP but typically performs better
