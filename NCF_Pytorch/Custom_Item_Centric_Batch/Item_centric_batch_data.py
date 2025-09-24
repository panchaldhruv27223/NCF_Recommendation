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
from item_centric_batch_sampler import ItemBatchSampler, build_item_index_map


class NCFTrainDataset(Dataset):
    def __init__(self, train_csv, num_negatives=4, num_users=0 ,num_items=0):
        
        self.train_df = pd.read_csv(train_csv)
        self.user_item_set = set(zip(self.train_df["UserID"],self.train_df["ItemID"]))
        
        self.num_items = max(num_items, max(self.train_df["ItemID"])+1)
        
        self.num_users = max(num_users, max(self.train_df["UserID"])+1)
        
        self.num_negatives = num_negatives
        
        self.users, self.items, self.labels = self._get_train_instances()
        
        self.user_id = self.users
        
        self.item_id = self.items
        
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
        
        
        
        ## we want per user num_neg items     (It is like Group by User)
        
        # user_pos_items = {}
        
        
        # for u, i in self.user_item_set:
        #     if u not in user_pos_items:
        #         user_pos_items[u] = []
                
        #     user_pos_items[u].append(i)
        
        # for u, pos in user_pos_items.items():
            
        #     counter = 0
            
        #     for i in pos:
        #         counter += 1
        #         user_input.append(u)
        #         item_input.append(i)
        #         labels.append(1)
            
        #     if self.num_negatives != -1:
                
        #         for _ in range(self.num_negatives):
                    
        #             j = np.random.randint(self.num_items)
                    
        #             while (u,j) in self.user_item_set:
        #                 j = np.random.randint(self.num_items)
                    
        #             user_input.append(u)
        #             item_input.append(j)
        #             labels.append(0)
        #     else:
        #         num_neg = counter
        #         for _ in range(num_neg):
                    
        #             j = np.random.randint(self.num_items)
                    
        #             while (u,j) in self.user_item_set:
        #                 j = np.random.randint(self.num_items)
                    
        #             user_input.append(u)
        #             item_input.append(j)
        #             labels.append(0)
            
        # return user_input, item_input, labels
        
        
        # Let's Group by Item, means Per item now we want equal number of interactions.(Positive interactions is equal to negative interactions)
        
        item_pos_users = {}
        
        for u, i in self.user_item_set:
            
            if i not in item_pos_users:
                item_pos_users[i] = []
            
            item_pos_users[i].append(u)
            
        for i, pos_user in item_pos_users.items():
            count_pos = len(pos_user)
            
            for u in pos_user:
                user_input.append(u)
                item_input.append(i)
                labels.append(1)
                
                
            for _ in range(count_pos):
                
                u = np.random.randint(self.num_users)
                
                while (u,i) in self.user_item_set:
                    u = np.random.randint(self.num_users)
            
                user_input.append(u)
                item_input.append(i)
                labels.append(0)
    
        return user_input, item_input, labels
    
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.tensor(self.users[idx], dtype=torch.long), torch.tensor(self.items[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)
    
    def get_item_centric_dataloader(self, shuffle_items=True, batch_size= 256,shuffle_within_item=True, drop_last=False, num_workers=0, pin_memory=False):
        
        item_to_indices = build_item_index_map(self.item_id)
        
        batch_sampler = ItemBatchSampler(
            item_to_indices=item_to_indices,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle_items=shuffle_items,
            shuffle_within_item= shuffle_within_item
        )
        
        return DataLoader(self, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=pin_memory)
        

if __name__ == "__main__":
    print("Calling from Item Centric Dataset")