import torch 
import torch.nn as nn
import torch.optim as optim
import os, sys, time

class NeuMF(nn.Module):
    """
    NeuMF: Neural Matrix Factorization
    Combines GMF (element-wise product) and MLP (concatenation + feed-forward).
    """
    def __init__(self, num_users, num_items, latent_dim, layers=[32,16,8]):
        super(NeuMF, self).__init__()
        assert len(layers) >= 2
        assert layers[0]%2 == 0
        
        # ---------- GMF embeddings ----------
        self.user_embeddings_mf = nn.Embedding(num_embeddings=num_users, embedding_dim=latent_dim)
        self.item_embeddings_mf = nn.Embedding(num_embeddings=num_items, embedding_dim=latent_dim)

        # ---------- MLP embeddings ----------
        self.user_embeddings_mlp = nn.Embedding(num_embeddings=num_users, embedding_dim=layers[0]//2)
        self.item_embeddings_mlp = nn.Embedding(num_embeddings=num_items, embedding_dim=layers[0]//2)
        
        # ---------- MLP hidden layers ----------
        mlp_module = []
        for input, output in zip(layers[:-1], layers[1:]):
            mlp_module.append(nn.Linear(input, output))
            mlp_module.append(nn.ReLU())    
        self.mlp = nn.Sequential(*mlp_module)
        ## now lets create final layer 
        ## it take input of both MF and MLP
        final_dim = latent_dim + layers[-1]
        self.predict_layer = nn.Linear(final_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()
        
    def _init_weights(self):
        """Initialize embeddings and linear layers"""
        nn.init.normal_(self.item_embeddings_mf.weight, std=0.01)
        nn.init.normal_(self.user_embeddings_mf.weight, std=0.01)
        nn.init.normal_(self.user_embeddings_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embeddings_mlp.weight, std=0.01)
        
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        nn.init.xavier_uniform_(self.predict_layer.weight)
        if self.predict_layer.bias is not None:
            nn.init.zeros_(self.predict_layer.bias)
        
    def forward(self, user, item):
        """
        Args:
            user
            item
        Returns:
            probs: Tensor of shape (batch,), probabilities after sigmoid
        """

        ## GMF Branch 
        mf_u = self.user_embeddings_mf(user)
        mf_i = self.item_embeddings_mf(item)
        mf_point_wise_multiplication = mf_u*mf_i
        
        ## MLP Branch
        mlp_u = self.user_embeddings_mlp(user)
        mlp_i = self.item_embeddings_mlp(item)
        mlp_input = torch.concat([mlp_u, mlp_i], dim=-1)
        mlp_output = self.mlp(mlp_input)
        
        ## Now concat both MF and MLP output
        merged_input = torch.concat([mf_point_wise_multiplication, mlp_output], dim=-1)
        logits = self.predict_layer(merged_input)
        out_prob = self.sigmoid(logits)
        out_prob = out_prob.squeeze()
        return out_prob

def train_NeuMF_model(model, train_loader, test_negative_dataset, config, NCFEvaluation, device="cpu"):
    
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
            outputs = outputs.to(torch.float32)
            labels = labels.to(torch.float32)

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
                           f"{config["dataset"]}_NeuMF_{config["num_factors"]}.pth")
            print(f"End. Best Iteration {epoch}: HR: {best_hr:.4f}, NDCG:{best_ndcg:.4f}")
            
    print(f"The Best NeuMF Model is Saved from epoch {best_epoch}")
    
    return best_hr, best_ndcg

if __name__ == "__main__":
    print("Calling From NeuMF Model.")
    
## next try to use pretrained 
    # def load_pretrained(self, gm_model=None, mlp_model=None, strict: bool = True):
    #     """
    #     Load pretrained GMF/MLP weights into NeuMF.
    #     gm_model: GMF instance with user_embedding, item_embedding
    #     mlp_model: MLP instance with user_embedding, item_embedding, mlp layers
    #     """
    #     if gm_model is not None:
    #         try:
    #             self.mf_user_emb.weight.data.copy_(gm_model.user_embedding.weight.data)
    #             self.mf_item_emb.weight.data.copy_(gm_model.item_embedding.weight.data)
    #         except Exception as e:
    #             if strict:
    #                 raise RuntimeError(f"GMF weight loading failed: {e}")

    #     if mlp_model is not None:
    #         try:
    #             self.mlp_user_emb.weight.data.copy_(mlp_model.user_embedding.weight.data)
    #             self.mlp_item_emb.weight.data.copy_(mlp_model.item_embedding.weight.data)

    #             # copy over linear layers (if shapes match)
    #             mlp_layers_src = [m for m in mlp_model.mlp if isinstance(m, nn.Linear)]
    #             mlp_layers_dst = [m for m in self.mlp if isinstance(m, nn.Linear)]
    #             for src, dst in zip(mlp_layers_src, mlp_layers_dst):
    #                 if src.weight.shape == dst.weight.shape:
    #                     dst.weight.data.copy_(src.weight.data)
    #                     if src.bias is not None and dst.bias is not None:
    #                         dst.bias.data.copy_(src.bias.data)
    #         except Exception as e:
    #             if strict:
    #                 raise RuntimeError(f"MLP weight loading failed: {e}")