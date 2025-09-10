import torch 
import torch.nn as nn
import torch.optim as optim
import time, os, sys

class NCF_mlp(nn.Module):
    def __init__(self, num_users, num_items, layers=[32,16,8]):
        
        super(NCF_mlp, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        
        self.num_layers = len(layers)
        self.layers = layers
        
        assert self.num_layers >= 2   ## we need atleast one input and one hidden layer
        
        self.users_embeddings = nn.Embedding(self.num_users, self.layers[0]//2)
        
        self.items_embeddings = nn.Embedding(self.num_items, self.layers[0]//2)
        
        nn.init.normal_(self.users_embeddings.weight, std = 0.01)
        nn.init.normal_(self.items_embeddings.weight, std = 0.01)
        
        mlp_module = []
        
        for i in range(1, self.num_layers):
            mlp_module.append(nn.Linear(self.layers[i-1], self.layers[i]))
            mlp_module.append(nn.ReLU())
        
        self.MLP = nn.Sequential(*mlp_module)
        
        self.output_layer = nn.Linear(self.layers[-1], 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, user, item):
        user_emb = self.users_embeddings(user)
        item_emb = self.items_embeddings(item)
        # print(user_emb.shape)
        # print(item_emb.shape)
        combined_emb = torch.concat([user_emb,item_emb], dim=-1)
        
        # print(combined_emb.shape)
        # print(combined_emb)
        
        mlp_output = self.MLP(combined_emb)
        logits = self.output_layer(mlp_output)
        # print(logits.shape)
        out = self.sigmoid(logits)
        out = out.squeeze()
        return out
    

def NCF_mlp_train(model, train_loader, test_negative_data_object, NCF_evaluation, config, device="cpu"):
    if config["learner"].lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    elif config["learner"].lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=config["lr"])
    elif config["learner"].lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=config["lr"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config["lr"])

    criterion = nn.BCELoss()

    evaluator = NCF_evaluation(model, test_negative_data_object, top_k = config["topK"])

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
                           f"{config["dataset"]}_MLP_{config["num_factors"]}.pth")
            print(f"End. Best Iteration {epoch}: HR: {best_hr:.4f}, NDCG:{best_ndcg:.4f}")

    print(f"The Best MLP Model is Saved from epoch {best_epoch}")

    return best_hr, best_ndcg


if __name__ == "__main__":
    print("Calling from MLP Model.")