# GMF Model Implementation

## Overview
This directory contains the implementation of the **Generalized Matrix Factorization (GMF)** model, which is one of the three core models in the Neural Collaborative Filtering (NCF) framework. GMF uses element-wise product of user and item embeddings to predict user-item interactions.

## Files

### 1. `GMF_model.py`
Contains the core GMF model implementation and training function.

#### Key Components:

**GMF Class:**
- **Purpose**: Implements the Generalized Matrix Factorization model
- **Architecture**: 
  - User embeddings: `nn.Embedding(num_users, latent_dim)`
  - Item embeddings: `nn.Embedding(num_items, latent_dim)`
  - Output layer: `nn.Linear(latent_dim, 1)` with sigmoid activation
- **Forward Pass**: 
  1. Get user and item embeddings
  2. Compute element-wise product: `user_latent * item_latent`
  3. Pass through linear layer and sigmoid activation
- **Initialization**: Normal distribution with std=0.01

**train_GMF_model Function:**
- **Purpose**: Training loop for the GMF model
- **Features**:
  - Multiple optimizer support (Adam, Adagrad, RMSprop, SGD)
  - Binary Cross-Entropy loss
  - Real-time evaluation with Hit Rate and NDCG metrics
  - Model checkpointing (saves best model)
  - Progress tracking with timing

#### Parameters:
- `num_users`: Number of unique users in the dataset
- `num_items`: Number of unique items in the dataset
- `latent_dim`: Embedding dimension (typically 10)
- `reg`: Regularization parameters (L1, L2)

### 2. `GMF_Model_train.py`
Training script that orchestrates the GMF model training process.

#### Configuration:
```python
configurations = {
    "train_data": "NCF_Pytorch/train_data.csv",
    "test_data": "NCF_Pytorch/test_data.csv", 
    "test_negative_data": "NCF_Pytorch/test_negative_data.csv",
    "dataset": "ml-1m",
    "regs": [0, 0],           # L1, L2 regularization
    "lr": 0.001,              # Learning rate
    "batch_size": 256,        # Batch size
    "epochs": 3,              # Number of training epochs
    "learner": "adam",        # Optimizer
    "num_factors": 10,        # Latent dimensions
    "num_neg": 2,             # Negative samples per positive
    "out": True,              # Save model flag
    "topK": 10                # Evaluation top-K
}
```

#### Workflow:
1. **Data Loading**: Creates train and test dataset objects
2. **Model Initialization**: Instantiates GMF model with dataset dimensions
3. **DataLoader Setup**: Creates PyTorch DataLoaders for batching
4. **Training**: Calls the training function with all components

## Usage

### Training the Model:
```bash
python GMF_Model_train.py
```

### Using the Model:
```python
from GMF_model import GMF

# Initialize model
model = GMF(num_users=6040, num_items=3706, latent_dim=10, reg=[0,0])

# Forward pass
user_ids = torch.tensor([1, 2, 3])
item_ids = torch.tensor([10, 20, 30])
predictions = model(user_ids, item_ids)
```

## Model Architecture

```
Input: (user_id, item_id)
    ↓
User Embedding: [num_users, latent_dim]
    ↓
Item Embedding: [num_items, latent_dim]
    ↓
Element-wise Product: user_emb * item_emb
    ↓
Linear Layer: [latent_dim] → [1]
    ↓
Sigmoid Activation
    ↓
Output: [0, 1] (interaction probability)
```

## Key Features

1. **Element-wise Interaction**: Uses Hadamard product instead of concatenation
2. **Binary Classification**: Predicts interaction probability (0-1)
3. **Embedding-based**: Learns dense representations for users and items
4. **Efficient Training**: Supports multiple optimizers and loss functions
5. **Evaluation**: Real-time Hit Rate and NDCG computation

## Dependencies

- PyTorch
- NumPy
- Pandas
- Custom modules: `ml_1m_dataset`, `NCF_evaluation`

## Output

The training process outputs:
- Model checkpoints: `{dataset}_GMF_{num_factors}.pth`
- Training metrics: Hit Rate, NDCG, Loss per epoch
- Best model identification and saving

## Performance

Typical performance on MovieLens-1M:
- Hit Rate @10: ~0.58
- NDCG @10: ~0.33
- Training time: ~70-85 seconds per epoch (CPU)

## Notes

- The model uses binary cross-entropy loss for implicit feedback
- Negative sampling is handled in the dataset class
- Evaluation uses the standard leave-one-out protocol
- Model weights are initialized with normal distribution (std=0.01)
