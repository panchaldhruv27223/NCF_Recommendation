import torch 
import torch.nn as nn
import torch.optim as optim
import os, sys, time

class GMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim, reg):
        super(GMF, self).__init__()
        
        ## Users and items embeddings
        
        self.user_embeddings = nn.Embedding(num_embeddings=num_users, embedding_dim=latent_dim)
        self.item_embeddings = nn.Embedding(num_embeddings=num_items, embedding_dim=latent_dim)
        
        self.output = nn.Linear(latent_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
        ### set the initializer 
        nn.init.normal_(self.user_embeddings.weight, mean=0, std = 0.01)
        nn.init.normal_(self.item_embeddings.weight, mean=0, std = 0.01)
        
        
    def forward(self, user, item):
        user_latent = self.user_embeddings(user)
        item_latent = self.item_embeddings(item)
        
        elementwise_product  = user_latent * item_latent
        
        out = self.output(elementwise_product )
        logits = self.sigmoid(out)
        logits = logits.squeeze()
        # print(logits)
        return logits 
    

def train_GMF_model(model, train_loader, test_negative_dataset, config, NCFEvaluation, device="cpu"):
    
    if config["learner"].lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        
    elif config["learner"].lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=config["lr"])
    elif config["learner"].lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=config["lr"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config["lr"])
        
    criterion = nn.BCELoss()
    
    evaluator = NCFEvaluation(model, test_negative_dataset, top_k = config["topK"])
    
    best_hr, best_ndcg, best_epoch = 0, 0, -1
    
    for epoch in range(config["epochs"]):
        t1 = time.time()
        
        model.train()
        
        total_loss = 0 
        
        for users, items, labels in train_loader:
            users = users.to(device)
            items = items.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(users, items)
            
            ## making the input and output shape and datatype same
            
            outputs = outputs.to(torch.float32)
            labels = labels.to(torch.float32)
            
            # print(labels.shape)
            # print(outputs.shape)
            # print(labels.dtype)
            # print(outputs.dtype)
            # print(outputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        t2 = time.time()
        hits, ndcgs = evaluator.evaluate()
        hits = sum(hits)/len(hits)
        ndcgs = sum(ndcgs)/len(ndcgs)
        
        avg_loss = total_loss / len(train_loader)
        
        print(f"Epoch {epoch} [{t2-t1:.1f}s]: \n"
              f"Hit Rate: {hits:.4f}, NDCG: {ndcgs:.4f}, loss: {avg_loss:.4f}")
        
        if hits > best_hr:
            best_hr, best_ndcg, best_epoch = hits, ndcgs, epoch
            
            if config["out"]:
                torch.save(model.state_dict(),
                           f"{config["dataset"]}_GMF_{config["num_factors"]}.pth")
            print(f"End. Best Iteration {epoch}: HR: {best_hr:.4f}, NDCG:{best_ndcg:.4f}")
            
    print(f"The Best GMF Model is Saved from epoch {best_epoch}")
    
    return best_hr, best_ndcg
    
    
if __name__ == "__main__":
    print("Calling From GMF Model.")