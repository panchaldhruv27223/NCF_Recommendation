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

class NCFTrainDataset(Dataset):
    def __init__(self, train_csv, num_negatives=4, num_users=0 ,num_items=0):
        
        self.train_df = pd.read_csv(train_csv)
        self.user_item_set = set(zip(self.train_df["UserID"],self.train_df["ItemID"]))
        
        self.num_items = max(num_items, max(self.train_df["ItemID"])+1)
        
        self.num_users = max(num_users, max(self.train_df["UserID"])+1)
        
        self.num_negatives = num_negatives
        
        self.users, self.items, self.labels = self._get_train_instances()
        
    def _get_train_instances(self):
         
        user_input, item_input, labels = [], [], []
         
        for (u, i) in self.user_item_set:
            # positive instance
            user_input.append(u)
            item_input.append(i)
            labels.append(1)
            
            # negative instances
            
            for _ in range(self.num_negatives):
            
                j = np.random.randint(self.num_items)
            
                while (u, j) in self.user_item_set:
            
                    j = np.random.randint(self.num_items)
            
                user_input.append(u)
                item_input.append(j)
                labels.append(0)
                
        return user_input, item_input, labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.tensor(self.users[idx], dtype=torch.long), torch.tensor(self.items[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)
    

class NCFTestDataset(Dataset):
    def __init__(self, test_csv, test_negative_csv):
        
        self.test_ratings = pd.read_csv(test_csv)
        self.test_negatives = pd.read_csv(test_negative_csv)
        
        assert len(self.test_ratings) == len(self.test_negatives)
        
    def __len__(self):
        return len(self.test_ratings)
    
    def __getitem__(self, idx):
        user = self.test_ratings.iloc[idx, 0]
        item = self.test_ratings.iloc[idx, 1]
        negatives = list(map(int, self.test_negatives.iloc[idx, 2].strip("[]").split(",")) )
        # print(negatives)
        return torch.tensor(user, dtype=torch.long), torch.tensor(item, dtype=torch.long), torch.tensor(negatives, dtype=torch.long)
    
    

if __name__ == "__main__":
    print("Calling from ml-1m Dataset")