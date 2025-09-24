import torch 
import torch.nn as nn
import torch.optim as optim
import os, sys, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))
# print(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))
# from NCF_Pytorch.logger import setup_logger

    
class GMF(nn.Module):
    def __init__(self, num_users=6040, num_items=3706, latent_dim=10, reg=[0,0], train_logger= None):
        super(GMF, self).__init__()
        
        ## Users and items embeddings
        self.train_loager = train_logger
        self.user_embeddings = nn.Embedding(num_embeddings=num_users, embedding_dim=latent_dim)
        self.item_embeddings = nn.Embedding(num_embeddings=num_items, embedding_dim=latent_dim)
        
        self.output = nn.Linear(latent_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
        ### set the initializer 
        nn.init.normal_(self.user_embeddings.weight, mean=0, std = 0.01)
        nn.init.normal_(self.item_embeddings.weight, mean=0, std = 0.01)
        
        
    def forward(self, user, item):
        # self.train_loager.info(f"Users shape: {user.shape}")
        # self.train_loager.info(f"Items shape: {item.shape}")
        
        user_latent = self.user_embeddings(user)
        item_latent = self.item_embeddings(item)
        
        # self.train_loager.info(f"In Latent Dimensions the shape of user latent: {user_latent.shape}")
        # self.train_loager.info(f"In Latent Dimensions the shape of item latent: {item_latent.shape}")
        
        elementwise_product  = user_latent * item_latent
        # self.train_loager.info(f"Point wise multiplication between user and item: {elementwise_product.shape}")
        
        out = self.output(elementwise_product )
        # self.train_loager.info(f"Output of hidden layer : {out.shape}")

        logits = self.sigmoid(out).view(-1)
        # self.train_loager.info(f"Output of hidden layer after applying sigmoid : {logits.shape}")

        # logits = logits.squeeze()
        # print(logits)

        return logits 
    

        train_logger.info(f"Time taken to complete epoch no {epoch}: {t2-t1}")
def train_GMF_model(model, train_loader, test_negative_dataset, config, NCFEvaluation, train_logger=None, test_logger =None, device="cpu"):
    
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
    test_logger.info(f"GMF Model Evaluation is defined with topk : {config["topK"]}")
    
    best_hr, best_ndcg, best_epoch = 0, 0, -1
    
    for epoch in range(config["epochs"]):
        t1 = time.time()
        
        model.train()
        
        total_loss = 0 
        
        for users, items, labels in train_loader:
            users = users.to(device)
            items = items.to(device)
            labels = labels.to(device)
            
            # unique_users, counts = torch.unique(users, return_counts=True)
            # train_logger.info(f"Number Of Uniques Users In The Batch, {len(unique_users)}")
            
            # for u, c in zip(unique_users.cpu().tolist(), counts.cpu().tolist()):
            #     train_logger.info(f"User {u} appears {c} times in this batch")
            
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
        train_logger.info(f"Time taken to complete epoch no {epoch}: {t2-t1}")
                
                
        hits, ndcgs = evaluator.evaluate()
        hits = sum(hits)/len(hits)
        ndcgs = sum(ndcgs)/len(ndcgs)
        
        avg_loss = total_loss / len(train_loader)
        
        # print(f"Epoch {epoch} [{t2-t1:.1f}s]: \n"
            #   f"Hit Rate: {hits:.4f}, NDCG: {ndcgs:.4f}, loss: {avg_loss:.4f}")
        
        test_logger.info(f"Epoch {epoch} [{t2-t1:.1f}s]: \n"
              f"Hit Rate: {hits:.4f}, NDCG: {ndcgs:.4f}, loss: {avg_loss:.4f}")
        
        
        if hits > best_hr:
            
            best_hr, best_ndcg, best_epoch = hits, ndcgs, epoch
            
            test_logger.info(f"till now best hr: {best_hr}, best ndcgs: {best_ndcg} and at epoch {best_epoch}")
            
            if config["out"]:
                
                os.makedirs(config['out_path'], exist_ok=True)
                
                torch.save(model,
                           f"{config['out_path']}/{config["dataset"]}_GMF_Batch_{config["batch_size"]}_epoch_{config["epochs"]}_{config["num_factors"]}.pth")

                torch.save(model.state_dict(),
                           f"{config['out_path']}/model_dict_{config["dataset"]}_GMF_Batch_{config["batch_size"]}_epoch_{config["epochs"]}_{config["num_factors"]}.pth")
            # print(f"End. Best Iteration {epoch}: HR: {best_hr:.4f}, NDCG:{best_ndcg:.4f}")
    

    test_logger.info(f"The Best GMF Model is Saved from epoch {best_epoch}")
    test_logger.info("The Best GMF Model is Saved at: \n"+
                     f"{config['out_path']}/{config["dataset"]}_GMF_Batch_{config["batch_size"]}_epoch_{config["epochs"]}_{config["num_factors"]}.pth")
    

    # print(f"The Best GMF Model is Saved from epoch {best_epoch}")
    
    return best_hr, best_ndcg
    
    
if __name__ == "__main__":
    print("Calling From GMF Model.")