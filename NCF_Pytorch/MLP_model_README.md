# MLP Model Implementation

## Overview
This directory contains the implementation of the **Multi-Layer Perceptron (MLP)** model, which is the second core model in the Neural Collaborative Filtering (NCF) framework. MLP uses concatenation of user and item embeddings followed by multiple fully connected layers to learn non-linear interactions.

## Files

### 1. `MLP_model.py`
Contains the core MLP model implementation and training function.

#### Key Components:

**NCF_mlp Class:**
- **Purpose**: Implements the Multi-Layer Perceptron model for collaborative filtering
- **Architecture**: 
  - User embeddings: `nn.Embedding(num_users, layers[0]//2)`
  - Item embeddings: `nn.Embedding(num_items, layers[0]//2)`
  - MLP layers: Configurable number of fully connected layers with ReLU activation
  - Output layer: `nn.Linear(layers[-1], 1)` with sigmoid activation
- **Forward Pass**: 
  1. Get user and item embeddings (each of size `layers[0]//2`)
  2. Concatenate embeddings: `torch.concat([user_emb, item_emb], dim=-1)`
  3. Pass through MLP layers with ReLU activation
  4. Final linear layer and sigmoid activation
- **Initialization**: Normal distribution with std=0.01 for embeddings

**NCF_mlp_train Function:**
- **Purpose**: Training loop for the MLP model
- **Features**:
  - Multiple optimizer support (Adam, Adagrad, RMSprop, SGD)
  - Binary Cross-Entropy loss
  - Real-time evaluation with Hit Rate and NDCG metrics
  - Model checkpointing (saves best model)
  - Progress tracking with timing

#### Parameters:
- `num_users`: Number of unique users in the dataset
- `num_items`: Number of unique items in the dataset
- `layers`: List defining MLP architecture (e.g., [32, 16, 8])
  - First element: Total input size (user_emb + item_emb)
  - Subsequent elements: Hidden layer sizes

### 2. `MLP_model_train.py`
Training script that orchestrates the MLP model training process.

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
    "layers": [32, 16, 8],    # MLP layer architecture
    "num_factors": 10,        # Not used in MLP (for compatibility)
    "num_neg": 2,             # Negative samples per positive
    "out": True,              # Save model flag
    "topK": 10                # Evaluation top-K
}
```

#### Workflow:
1. **Data Loading**: Creates train and test dataset objects
2. **Model Initialization**: Instantiates MLP model with dataset dimensions and layer config
3. **DataLoader Setup**: Creates PyTorch DataLoaders for batching
4. **Training**: Calls the training function with all components

## Usage

### Training the Model:
```bash
python MLP_model_train.py
```

### Using the Model:
```python
from MLP_model import NCF_mlp

# Initialize model with custom architecture
model = NCF_mlp(num_users=6040, num_items=3706, layers=[32, 16, 8])

# Forward pass
user_ids = torch.tensor([1, 2, 3])
item_ids = torch.tensor([10, 20, 30])
predictions = model(user_ids, item_ids)
```

## Model Architecture

```
Input: (user_id, item_id)
    ↓
User Embedding: [num_users, layers[0]//2]
    ↓
Item Embedding: [num_items, layers[0]//2]
    ↓
Concatenation: [layers[0]] = user_emb + item_emb
    ↓
MLP Layer 1: [layers[0]] → [layers[1]] + ReLU
    ↓
MLP Layer 2: [layers[1]] → [layers[2]] + ReLU
    ↓
... (additional layers)
    ↓
Output Layer: [layers[-1]] → [1]
    ↓
Sigmoid Activation
    ↓
Output: [0, 1] (interaction probability)
```

## Key Features

1. **Concatenation-based**: Uses concatenation instead of element-wise product
2. **Deep Learning**: Multiple fully connected layers for complex pattern learning
3. **Non-linear Activation**: ReLU activation between layers for non-linearity
4. **Configurable Architecture**: Flexible layer configuration
5. **Binary Classification**: Predicts interaction probability (0-1)

## Architecture Details

### Default Configuration: [32, 16, 8]
- **Input Layer**: 32 neurons (16 user + 16 item embeddings)
- **Hidden Layer 1**: 16 neurons with ReLU
- **Hidden Layer 2**: 8 neurons with ReLU
- **Output Layer**: 1 neuron with sigmoid

### Layer Construction:
```python
mlp_module = []
for i in range(1, self.num_layers):
    mlp_module.append(nn.Linear(self.layers[i-1], self.layers[i]))
    mlp_module.append(nn.ReLU())
self.MLP = nn.Sequential(*mlp_module)
```

## Dependencies

- PyTorch
- NumPy
- Pandas
- Custom modules: `ml_1m_dataset`, `NCF_evaluation`

## Output

The training process outputs:
- Model checkpoints: `{dataset}_MLP_{num_factors}.pth`
- Training metrics: Hit Rate, NDCG, Loss per epoch
- Best model identification and saving

## Performance

Typical performance on MovieLens-1M:
- Hit Rate @10: ~0.55-0.58
- NDCG @10: ~0.31-0.33
- Training time: ~70-85 seconds per epoch (CPU)

## Advantages over GMF

1. **Non-linear Interactions**: Can learn complex user-item interaction patterns
2. **Feature Learning**: Automatically learns relevant feature combinations
3. **Flexibility**: Configurable architecture for different datasets
4. **Deep Representation**: Multiple layers allow for hierarchical feature learning

## Notes

- The model uses binary cross-entropy loss for implicit feedback
- Negative sampling is handled in the dataset class
- Evaluation uses the standard leave-one-out protocol
- Model weights are initialized with normal distribution (std=0.01)
- The first layer size must be even (for user+item embedding concatenation)
- At least 2 layers are required (input + output)
