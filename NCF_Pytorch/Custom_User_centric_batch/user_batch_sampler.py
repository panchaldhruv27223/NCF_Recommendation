import random
from torch.utils.data import Sampler

def build_user_index_map(user_ids):
    user_to_indices = {}
    
    for idx, u in enumerate(user_ids):
        if u not in user_to_indices:
            user_to_indices[u] = []
            
        user_to_indices[u].append(idx)
        
    return user_to_indices

class UserBatchSampler(Sampler):
    
    def __init__(self, user_to_indices, batch_size=256, drop_last = False, shuffle_users = True, shuffle_within_user = True):
        self.user_to_indices = user_to_indices
        self.batch_size = batch_size
        self.users = list(user_to_indices.keys())
        self.drop_last = drop_last
        self.shuffle_users = shuffle_users
        self.shuffle_within_user = shuffle_within_user
        
    def __iter__(self):
        users = self.users.copy()
        
        if self.shuffle_users:
            random.shuffle(users)
            
        for user in users:
            indices = self.user_to_indices[user].copy()
            
            if self.shuffle_within_user:
                random.shuffle(indices)
            
            for  i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]
                
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                
                yield batch
            
            
    def __len__(self):
        return len(self.users)