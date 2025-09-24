import random
from torch.utils.data import Sampler


def build_item_index_map(item_ids):
    item_to_indices = {}
    for idx, i in enumerate(item_ids):
        if i not in item_to_indices:
            item_to_indices[i] = []
        item_to_indices[i].append(idx)
    return item_to_indices


class ItemBatchSampler:
    
    def __init__(self, item_to_indices, batch_size=256, drop_last = False, shuffle_items = True, shuffle_within_item = True):
        
        self.item_to_indices = item_to_indices
        self.batch_size = batch_size
        self.items = list(item_to_indices.keys())
        self.drop_last = drop_last
        self.shuffle_items = shuffle_items
        self.shuffle_within_item = shuffle_within_item

    def __iter__(self):
        
        items = self.items.copy()
        
        if self.shuffle_items:
            random.shuffle(items)
            
        for item in items:
            indices = self.item_to_indices[item].copy()
            
            if self.shuffle_within_item:
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
        
        if self.batch_size != -1:
            
            count = 0
            
            for indices in self.item_to_indices.values():
                count += (len(indices) // self.batch_size)
                
                if not self.drop_last and len(indices) % self.batch_size != 0:
                    count += 1
            
            return count

        return len(self.item_to_indices.values())
    
    
if __name__ == "__main__":
    print("Calling from item based batch sampler")