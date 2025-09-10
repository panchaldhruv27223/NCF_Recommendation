import heapq
import math
import torch

class NCFEvaluator:
    def __init__(self, model, test_negative_dataset, top_k=10, device='cpu'):
        self.model = model
        self.test_dataset = test_negative_dataset
        self.top_k = top_k
        self.device = device

    def evaluate(self):
        ndcg = []
        hit = []
        self.model.eval()
        
        with torch.no_grad():
            for idx in range(len(self.test_dataset)):
                
                user, pos_item, neg_items = self.test_dataset[idx]
                # print(user)
                # print(pos_item)
                # print(neg_items)
                
                
                # print(user.dtype)
                # print(pos_item.dtype)
                # print(neg_items.dtype)
                
                users = user.repeat(len(neg_items) + 1).to(self.device)
                items = torch.cat((torch.tensor([pos_item]), neg_items), dim=0).to(self.device)
                
                ## get prediction scores
                scores = self.model(users, items).cpu().numpy()
                
                ## Top k rank
                map_item_score = {item : score for item, score in zip(items.tolist(), scores.tolist())}
                
                ranklist= heapq.nlargest(self.top_k, map_item_score, key= map_item_score.get)
                
                # Metrics
                hr_val = self.getHitRate(ranklist, pos_item)
                ndcg_val = self.getNDCG(ranklist, pos_item)
                
                hit.append(hr_val)
                ndcg.append(ndcg_val)
                
        return (hit, ndcg)
    
    def getHitRate(self, ranklist, get_item):
        for item in ranklist:
            if item == get_item:
                return 1
        return 0
    
    def getNDCG(self, ranklist, get_item):
        for i, item in enumerate(ranklist):
            
            if item == get_item:
                return math.log(2) / math.log(i+2)
            
        return 0
    
    
if __name__ == "__main__":
    print("Calling from NCF Evaluation.")