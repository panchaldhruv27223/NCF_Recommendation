# MovieLens-1M Dataset Implementation

## Overview
This file contains the PyTorch dataset implementation for the MovieLens-1M dataset, providing data loading and preprocessing functionality for the Neural Collaborative Filtering (NCF) models. It handles both training data generation with negative sampling and test data preparation for evaluation.

## File: `ml_1m_dataset.py`

### Key Components:

#### 1. NCFTrainDataset Class
**Purpose**: Handles training data generation with negative sampling for implicit feedback learning.

**Key Features:**
- **Positive Instance Generation**: Creates positive user-item pairs from training data
- **Negative Sampling**: Generates negative samples for each positive interaction
- **Dataset Statistics**: Automatically calculates number of users and items
- **PyTorch Dataset**: Implements `__len__` and `__getitem__` for DataLoader compatibility

**Initialization Parameters:**
- `train_csv`: Path to training CSV file
- `num_negatives`: Number of negative samples per positive (default: 4)
- `num_users`: Optional user count override
- `num_items`: Optional item count override

**Data Processing:**
1. **Load Training Data**: Reads CSV file with columns [UserID, ItemID, Rating, Timestamp]
2. **Create User-Item Set**: Converts to set for efficient negative sampling
3. **Calculate Dimensions**: Determines unique user and item counts
4. **Generate Instances**: Creates positive and negative training instances

**Training Instance Generation:**
```python
def _get_train_instances(self):
    user_input, item_input, labels = [], [], []
    
    for (u, i) in self.user_item_set:
        # Positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        
        # Negative instances
        for _ in range(self.num_negatives):
            j = np.random.randint(self.num_items)
            while (u, j) in self.user_item_set:
                j = np.random.randint(self.num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    
    return user_input, item_input, labels
```

#### 2. NCFTestDataset Class
**Purpose**: Handles test data loading for model evaluation with negative items.

**Key Features:**
- **Test Ratings**: Loads positive test interactions
- **Negative Items**: Loads pre-sampled negative items for each test case
- **Data Validation**: Ensures test ratings and negatives have same length
- **PyTorch Dataset**: Compatible with DataLoader for batch processing

**Initialization Parameters:**
- `test_csv`: Path to test ratings CSV file
- `test_negative_csv`: Path to test negative items CSV file

**Data Structure:**
- **Test Ratings**: [UserID, ItemID, Rating, Timestamp]
- **Test Negatives**: [UserID, ItemID, NegativeItems] where NegativeItems is a list of 99 negative item IDs

**Data Processing:**
1. **Load Test Data**: Reads positive test interactions
2. **Load Negatives**: Reads pre-sampled negative items
3. **Parse Negatives**: Converts string representation to integer list
4. **Validation**: Ensures data consistency

## Usage Examples

### Training Dataset:
```python
from ml_1m_dataset import NCFTrainDataset
from torch.utils.data import DataLoader

# Create training dataset
train_dataset = NCFTrainDataset(
    train_csv="train_data.csv",
    num_negatives=4
)

# Create DataLoader
train_loader = DataLoader(
    train_dataset, 
    batch_size=256, 
    shuffle=True
)

# Access dataset properties
print(f"Number of users: {train_dataset.num_users}")
print(f"Number of items: {train_dataset.num_items}")
print(f"Dataset size: {len(train_dataset)}")

# Iterate through batches
for users, items, labels in train_loader:
    print(f"Batch shape - Users: {users.shape}, Items: {items.shape}, Labels: {labels.shape}")
    break
```

### Test Dataset:
```python
from ml_1m_dataset import NCFTestDataset

# Create test dataset
test_dataset = NCFTestDataset(
    test_csv="test_data.csv",
    test_negative_csv="test_negative_data.csv"
)

# Access single test case
user, pos_item, neg_items = test_dataset[0]
print(f"User: {user}, Positive Item: {pos_item}")
print(f"Negative Items: {neg_items} (shape: {neg_items.shape})")
```

## Data Format

### Training Data CSV:
```csv
UserID,ItemID,Rating,Timestamp
1,1193,5,978300760
1,661,3,978302109
1,914,3,978301968
...
```

### Test Data CSV:
```csv
UserID,ItemID,Rating,Timestamp
1,661,3,978302109
2,1193,5,978300760
...
```

### Test Negative Data CSV:
```csv
UserID,ItemID,NegativeItems
1,661,"[1064, 174, 2791, 2791, ...]"
2,1193,"[864, 582, 1426, ...]"
...
```

## Key Features

1. **Automatic Negative Sampling**: Generates negative samples during training
2. **Efficient Data Structures**: Uses sets for fast negative sampling
3. **PyTorch Integration**: Full compatibility with PyTorch DataLoader
4. **Data Validation**: Ensures data consistency and proper formatting
5. **Flexible Configuration**: Configurable negative sampling ratio
6. **Memory Efficient**: Generates data on-the-fly during training

## Negative Sampling Strategy

### Training Negatives:
- **Random Sampling**: Uniformly samples from all items
- **Collision Avoidance**: Ensures negative items are not in user's positive set
- **Configurable Ratio**: Default 4 negatives per positive (1:4 ratio)

### Test Negatives:
- **Pre-sampled**: Uses pre-generated negative items (typically 99 per test case)
- **Consistent Evaluation**: Same negatives used across all models for fair comparison
- **Standard Protocol**: Follows standard evaluation protocol for collaborative filtering

## Dataset Statistics

### MovieLens-1M Dataset:
- **Users**: 6,040 unique users
- **Items**: 3,706 unique movies
- **Training Interactions**: 994,169 positive interactions
- **Test Interactions**: 6,040 positive interactions (1 per user)
- **Negative Samples**: 99 per test case

### Training Data Size:
- **Positive Instances**: 994,169
- **Negative Instances**: 994,169 × num_negatives
- **Total Training Instances**: 994,169 × (1 + num_negatives)

## Dependencies

- PyTorch
- Pandas
- NumPy
- Pathlib (for file operations)

## Notes

- The dataset assumes 0-based indexing for users and items
- Negative sampling is done with replacement
- Test negatives are pre-sampled and stored in CSV format
- The dataset automatically handles data type conversion to PyTorch tensors
- Memory usage scales with the number of negative samples per positive
