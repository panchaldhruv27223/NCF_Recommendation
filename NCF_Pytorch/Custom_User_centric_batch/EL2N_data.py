import warnings, os, sys
import logging
from time import time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset

# print(os.getcwd())
# print(sys.executable)

class NcfEl2nTrainDataset(Dataset):
    def __init__(self, train_csv):
        
        self.train_df = pd.read_csv(train_csv)

        self.num_items = max(0, max(self.train_df["item"])+1)
        self.num_users = max(0, max(self.train_df["user"])+1)
        
        self.users, self.items, self.labels = self.train_df["user"], self.train_df["item"], self.train_df["label"]


    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.tensor(self.users[idx], dtype=torch.long), torch.tensor(self.items[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)
    
    

if __name__ == "__main__":
    print("Calling from ml-1m Dataset")