import warnings, os, sys
import logging
from time import time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset, DataLoader
from user_batch_sampler import UserBatchSampler, build_user_index_map

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
        
        self.user_id = self.users
        
        
    def _get_train_instances(self):
         
        user_input, item_input, labels = [], [], []


        ## this is per interaction
         
        # for (u, i) in self.user_item_set:
        #     # positive instance
        #     user_input.append(u)
        #     item_input.append(i)
        #     labels.append(1)
            
        #     # negative instances
        #     for _ in range(self.num_negatives):
            
        #         j = np.random.randint(self.num_items)
            
        #         while (u, j) in self.user_item_set:
            
        #             j = np.random.randint(self.num_items)
            
        #         user_input.append(u)
        #         item_input.append(j)
        #         labels.append(0)
        
        
        
        ## we want per user num_neg items 
        
        user_pos_items = {}
        
        
        for u, i in self.user_item_set:
            if u not in user_pos_items:
                user_pos_items[u] = []
                
            user_pos_items[u].append(i)
        
        for u, pos in user_pos_items.items():
            
            counter = 0
            
            for i in pos:
                counter += 1
                user_input.append(u)
                item_input.append(i)
                labels.append(1)
            
            if self.num_negatives != -1:
                
                for _ in range(self.num_negatives):
                    
                    j = np.random.randint(self.num_items)
                    
                    while (u,j) in self.user_item_set:
                        j = np.random.randint(self.num_items)
                    
                    user_input.append(u)
                    item_input.append(j)
                    labels.append(0)
            else:
                num_neg = counter
                for _ in range(num_neg):
                    
                    j = np.random.randint(self.num_items)
                    
                    while (u,j) in self.user_item_set:
                        j = np.random.randint(self.num_items)
                    
                    user_input.append(u)
                    item_input.append(j)
                    labels.append(0)
            
        return user_input, item_input, labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.tensor(self.users[idx], dtype=torch.long), torch.tensor(self.items[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)
    
    def get_user_centric_dataloader(self, shuffle_users=True, batch_size= 256,shuffle_within_user=True, drop_last=False, num_workers=0, pin_memory=False):
        
        user_to_indices = build_user_index_map(self.user_id)
        
        batch_sampler = UserBatchSampler(
            user_to_indices=user_to_indices,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle_users=shuffle_users,
            shuffle_within_user= shuffle_within_user
        )
        
        return DataLoader(self, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=pin_memory)
        

if __name__ == "__main__":
    print("Calling from User Centric Dataset")