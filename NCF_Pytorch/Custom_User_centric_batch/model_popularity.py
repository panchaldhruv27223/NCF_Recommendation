import os, sys, time
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../")))
from NCF_Pytorch.ml_1m_dataset import NCFTestDataset
from NCF_Pytorch.NCF_evaluation import NCFEvaluator


model = torch.load("/home/dhruv/Documents/NCF/NCF_Recommendation/NeuMF_Models/NeuMF_User_centric_El2n_model_3epoch/ml-1m_NeuMF_Batch_1024_epoch_3_10.pth", weights_only=False)

model.eval()
device = 'cpu'

mf_weight = model.item_embeddings_mf.weight.data.to(device)
mlp_weight = model.item_embeddings_mlp.weight.data.to(device)

assert mf_weight.shape[0] == mlp_weight.shape[0], "num_items mismatch"
num_items = mf_weight.shape[0]

mf_np = mf_weight.cpu().numpy()
mlp_np = mlp_weight.cpu().numpy()

combined_first_10 = (mf_np[:, :10] + mlp_np[:, :10]) / 2.0

remaining_6 = mlp_np[:, 10:16]

final_embeddings = np.concatenate([combined_first_10, remaining_6], axis=1)

print("Final embedding shape:", final_embeddings.shape)

df = pd.DataFrame(final_embeddings)
df.index.name = "item_index"


df.to_csv("item_embeddings_16.csv")