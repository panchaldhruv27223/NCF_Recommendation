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
from user_batch_sampler import UserBatchSampler, build_user_index_map, FixedRatioBatchSampler

# print(os.getcwd())
# print(sys.executable)

class NCFTrainDataset(Dataset):
    def __init__(self, train_csv, num_negatives=4, num_users=0 ,num_items=0, pos_percent=0.5):
        
        self.train_df = pd.read_csv(train_csv)
        self.train_df = self.train_df.sort_values(['UserID', 'Timestamp'])

        self.user_item_set = set(zip(self.train_df["UserID"],self.train_df["ItemID"]))
        
        self.num_items = int(max(num_items, max(self.train_df["ItemID"])+1))
        
        self.num_users = int(max(num_users, max(self.train_df["UserID"])+1))
        
        self.num_negatives = num_negatives
        
        self.pos_percent = pos_percent
        
        if pos_percent:
            self.users, self.items, self.labels = self._get_all_instances()
            
            
        else:
            self.users, self.items, self.labels = self._get_train_instances()
        
        self.user_id = self.users
        self.item_id = self.items
        
    
    def _get_all_instances(self):
        
        user_input = []
        item_input = []
        labels = []
        
        user_pos_items = {}
        
        
        for u, i in self.user_item_set:
            if u not in user_pos_items:
                user_pos_items[u] = []
                
            user_pos_items[u].append(i)
        
        for u, pos in user_pos_items.items():
            
            for i in pos:
                user_input.append(u)
                item_input.append(i)
                labels.append(1)
                
            for j in range(1, self.num_items):
                if (u,j) not in self.user_item_set:
                    user_input.append(u)
                    item_input.append(j)
                    labels.append(0)
        
            
        return user_input, item_input, labels

        
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
        # if isinstance(idx, tuple) and len(idx) == 3:
        #     user, item, label = idx
        #     return torch.tensor(user, dtype=torch.long), \
        #         torch.tensor(item, dtype=torch.long), \
        #         torch.tensor(label, dtype=torch.float32)
        # else:
            # fallback to indexing by integer
            # return torch.tensor(self.users[idx], dtype=torch.long), \
            #     torch.tensor(self.items[idx], dtype=torch.long), \
            #     torch.tensor(self.labels[idx], dtype=torch.float32)

    
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
    
    
    def get_fixed_ratio_neg_batch_per_user(self, batch_size=256, num_workers=0, pin_memory=False, shuffle_users=True, shuffle_within_user=True):
        
        batch_sampler = FixedRatioBatchSampler(
                user_ids=self.user_id,
                item_ids=self.item_id,
                labels = self.labels,
                user_item_set = self.user_item_set,
                num_items = self.num_items,
                batch_size=batch_size,
                pos_percent= self.pos_percent,
                shuffle_users=shuffle_users, 
                shuffle_within_user=shuffle_within_user
            )

        # return DataLoader(self, batch_sampler=batch_sampler,collate_fn=self.fixed_ratio_collate_fn,num_workers=num_workers, pin_memory=pin_memory)
        return DataLoader(self, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=pin_memory)
        

    def fixed_ratio_collate_fn(self, batch):

        # users = []
        # items = []
        # labels = []

        # for entry in batch:
        #     print(entry)
        #     if isinstance(entry, int):
        #         # Positive sample, get user/item from dataset
        #         user, item = self.users[entry], self.items[entry]
        #         label = 1
        #     elif isinstance(entry, tuple) and entry[0] == 'neg':
        #         _, user, item = entry
        #         label = 0
        #     else:
        #         raise ValueError("Unknown batch entry format")
            
        #     users.append(user)
        #     items.append(item)
        #     labels.append(label)

        # return torch.tensor(users, dtype=torch.long), \
        #     torch.tensor(items, dtype=torch.long), \
        #     torch.tensor(labels, dtype=torch.float32)
        
        users, items, labels = zip(*batch)
        # print(users, items, labels)
        
        return torch.tensor(users, dtype=torch.long), \
           torch.tensor(items, dtype=torch.long), \
           torch.tensor(labels, dtype=torch.float32)

        

if __name__ == "__main__":
    print("Calling from User Centric Dataset")