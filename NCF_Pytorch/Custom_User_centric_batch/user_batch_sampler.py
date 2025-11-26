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
            
            if self.batch_size == -1:
                yield indices
                
            else:
                ## This is For creating multiple bacth from the single batch for users who's interaction are more than the batch size
                for  i in range(0, len(indices), self.batch_size):
                    batch = indices[i:i+self.batch_size]
                    
                    if self.drop_last and len(batch) < self.batch_size:
                        continue
                
                    yield batch
                    
    def __len__(self):
        return len(self.users)
    

                
                
# class FixedRatioBatchSampler(Sampler):
#     def __init__(self, user_ids, item_ids, user_item_set, num_items, batch_size=256, pos_percent=0.5):
        
#         self.batch_size = batch_size
#         self.pos_percent = pos_percent
#         self.user_item_set = user_item_set
#         self.num_items = num_items
        
#         # Store positives per user
#         self.user_pos = {}
#         for idx, u in enumerate(user_ids):
#             if u not in self.user_pos:
#                 self.user_pos[u] = []
#             self.user_pos[u].append(idx)
        
#         self.users = list(self.user_pos.keys())
    
#     def __iter__(self):
#         users = self.users.copy()
        
#         batch = []
        
#         for u in users:
            
#             pos_indices = self.user_pos[u]
#             # Take most recent positives for this user
#             num_pos = int(self.batch_size * self.pos_percent)
#             num_pos = min(num_pos, len(pos_indices))
#             batch.extend(pos_indices[-num_pos:])
            
#             # Dynamically sample negatives for this user
#             num_neg = self.batch_size - len(batch)
            
#             neg_samples = []
            
#             while len(neg_samples) < num_neg:
#                 neg_item = random.randint(0, self.num_items - 1)
#                 if (u, neg_item) not in self.user_item_set:
#                     neg_samples.append(neg_item)
            
#             # Store negative interactions as (user, item)
            
#             for item in neg_samples:
#                 batch.append(('neg', u, item))  
            
#             if len(batch) == self.batch_size:
#                 yield batch
#                 batch = []
        
#         if len(batch) > 0:
#             yield batch
    
#     def __len__(self):
#         return len(self.users)



class FixedRatioBatchSampler(Sampler):
    """
    Batch sampler that ensures each batch has a fixed size and maintains
    a ratio of positive and negative interactions per user.
    """

    def __init__(self, user_ids, item_ids, labels, user_item_set, user_item_label_set, num_items,
                 batch_size=256, pos_percent=0.5, shuffle_users=True, shuffle_within_user=True):

        self.batch_size = batch_size
        self.pos_percent = pos_percent
        self.neg_percent = 1 - pos_percent
        self.user_item_set = user_item_set
        self.user_item_label_set = user_item_label_set
        self.num_items = num_items
        self.shuffle_users = shuffle_users
        self.shuffle_within_user = shuffle_within_user

        # Store positives per user
        self.user_pos = {}
        self.user_neg = {}
        
        for idx, (u, i, l) in enumerate(self.user_item_label_set):
            # print(u,i,l)
            if u not in self.user_pos:
                self.user_pos[u] = []
                
            if u not in self.user_neg:
                self.user_neg[u] = []

            if int(l) == 1:
                self.user_pos[u].append(idx)
            if int(l) == 0:
                self.user_neg[u].append(idx)
                

        self.users = list(self.user_pos.keys())

    def __iter__(self):
        users = self.users.copy()
        if self.shuffle_users:
            random.shuffle(users)

        batch = []

        for u in users:
            pos_items = self.user_pos[u].copy()

            if self.shuffle_within_user:
                random.shuffle(pos_items)

            
            
            num_pos = int(self.batch_size * self.pos_percent)
            
            num_pos = min(num_pos, len(pos_items))
            
            # print(f"Number of positve we want to take: {num_pos}")
            # print(f"Number of positves user have : {len(pos_items)}")
            
            batch.extend(pos_items[len(pos_items)-num_pos:])

            num_neg = int(self.batch_size * self.neg_percent)
            neg_items = self.user_neg[u].copy()
            
            num_neg = min(num_neg, len(neg_items))

            
            random.shuffle(neg_items)
            # print(neg_items)
            # print(num_neg)

            if num_neg != 0:
                # print(num_neg)
                batch.extend(neg_items[:num_neg])
            #     break
            # else:
            #     # print(num_neg)
            #     continue

            yield batch
            batch = []
            
            # if len(batch) == self.batch_size:
            #     yield batch
            #     batch = []
            # else:
            #     print("Error")
            #     print(f"Shape of Batch is :{len(batch)}")
        
        if len(batch) > 0:
            yield batch
            
    def __len__(self):
        # batches = 0
        # for u in self.users:
        #     num_pos = min(int(self.batch_size * self.pos_percent), len(self.user_pos[u]))
        #     num_neg = min(int(self.batch_size * self.neg_percent), len(self.user_neg[u]))
        #     taken = num_pos + num_neg
        #     if taken > 0:
        #         batches += 1   # because you yield once per user
                
        # return batches
        
        return len(set(self.users))