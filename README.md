# NCF_Recommendation

A PyTorch implementation of Neural Collaborative Filtering (NCF) for implicit feedback recommendation on the MovieLens 1M-style data. The project includes:

- Generalized Matrix Factorization (GMF)
- Multi-Layer Perceptron (MLP)
- Evaluation utilities with Hit Rate (HR) and NDCG@K

Pretrained example checkpoints for MovieLens-1M settings are provided at the repo root and under `NCF_Pytorch/`.

## Repository structure

- `NCF_Pytorch/`
  - `GMF_model.py`: GMF model and training loop
  - `MLP_model.py`: MLP-based NCF model and training loop
  - `GMF_Model_train.py`: script to train GMF
  - `MLP_model_train.py`: script to train MLP
  - `NeuMF_model_train.py`: training script scaffold
  - `ml_1m_dataset.py`: Dataset loaders for train/test with negative sampling
  - `NCF_evaluation.py`: Evaluation with HR and NDCG@K
  - `train_data.csv`, `test_data.csv`, `test_negative_data.csv`: data files
  - `ml-1m_GMF_10.pth`: trained GMF checkpoint
- `ml-1m_GMF_10.pth`, `ml-1m_MLP_10.pth`: checkpoints at repo root
- `Docs/NCF.pdf`: Reference/notes
- `requirements.txt`, `pyproject.toml`: Environment and packaging metadata

## Environment

- Python 3.12.3 (see `pyproject.toml`)
- Core libs: `torch`, `numpy`, `pandas`, `scikit-learn`, `tqdm`
- Dev (optional): `pytest`, `ruff`, `black`, `mypy`

You can install with pip using the provided `requirements.txt` for dev tools, or use uv/pip to resolve from `pyproject.toml`.

```bash
# Using pip (recommended minimal runtime deps)
pip install torch numpy pandas scikit-learn tqdm

# Or install dev/format/test tools (optional)
pip install -r requirements.txt
```

Note: CUDA is optional; code defaults to CPU. Install the appropriate `torch` build if using GPU.

## Data format

All paths in scripts are relative to the repository root and point to files in `NCF_Pytorch/`.

- `train_data.csv` (implicit interactions) must contain columns:
  - `UserID`, `ItemID`
  - All (user, item) rows are treated as positive interactions. Negatives are generated on-the-fly per user via `num_neg`.
- `test_data.csv` (one positive per user) must contain columns:
  - `UserID`, `ItemID`
- `test_negative_data.csv` must contain at least three columns where the third column is a stringified list of negative item IDs for the corresponding test user, e.g.:
  - Row format: `UserID,ItemID,"[12,45,78, ...]"`

The dataset loader infers `num_users` and `num_items` from the max IDs in `train_data.csv` and adds 1 (zero-based IDs expected).

## Training

Each training script defines a `configurations` dict controlling learning rate, batch size, epochs, optimizer, negative sampling, and `topK` for evaluation.

Common config keys:
- `train_data`, `test_data`, `test_negative_data`: CSV paths
- `dataset`: tag used to name saved checkpoints
- `lr`, `batch_size`, `epochs`, `learner`
- `num_factors` (GMF latent dim) or `layers` (MLP layer sizes)
- `num_neg`: number of negatives per positive during training
- `out`: whether to save the best checkpoint
- `topK`: cutoff for HR/NDCG

### GMF

Runs GMF with element-wise product of user/item embeddings followed by a sigmoid output.

```bash
python -m NCF_Pytorch.GMF_Model_train | cat
```

- Saves best checkpoint to `ml-1m_GMF_10.pth` by default (name is `{dataset}_GMF_{num_factors}.pth`).

### MLP

Runs the MLP-based NCF with concatenated embeddings through configurable dense layers.

```bash
python -m NCF_Pytorch.MLP_model_train | cat
```

- Saves best checkpoint to `ml-1m_MLP_10.pth` by default (name is `{dataset}_MLP_{num_factors}.pth`).

### NeuMF




## Evaluation

Evaluation runs during training using `NCF_Pytorch/NCF_evaluation.py`:
- Computes HR@K and NDCG@K per user by ranking the true positive item against provided negatives
- Averages metrics across all test users and prints per-epoch results

Example output snippet during training:
```
Epoch 2 [12.3s]:
Hit Rate: 0.6789, NDCG: 0.4123, loss: 0.4567
End. Best Iteration 2: HR: 0.6789, NDCG:0.4123
```

## Using pretrained checkpoints

- GMF checkpoints: `ml-1m_GMF_10.pth` (root) and `NCF_Pytorch/ml-1m_GMF_10.pth`
- MLP checkpoint: `ml-1m_MLP_10.pth` (root)

To load a checkpoint manually:

```python
import torch
from NCF_Pytorch.GMF_model import GMF

num_users, num_items, latent_dim = 6041, 3707, 10  # set based on your data
model = GMF(num_users, num_items, latent_dim=latent_dim, reg=[0, 0])
model.load_state_dict(torch.load('ml-1m_GMF_10.pth', map_location='cpu'))
model.eval()
```

## Notes

- User and item IDs should be contiguous zero-based indices. If your raw data uses different IDs, map them first.
- `shuffle=False` is used in the example DataLoaders; you may enable shuffling for stochasticity.
- Loss is `BCELoss` on implicit labels (1 for observed, 0 for sampled negatives).

## License

This project is licensed under the terms of the LICENSE file included in the repository. 