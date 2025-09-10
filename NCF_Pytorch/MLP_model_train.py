import os, sys, time
import torch
from torch.utils.data import DataLoader

from MLP_model import NCF_mlp, NCF_mlp_train
from ml_1m_dataset import NCFTrainDataset, NCFTestDataset
from NCF_evaluation import NCFEvaluator

configurations = {
    "train_data" : r"NCF_Pytorch\train_data.csv",
    "test_data" : r"NCF_Pytorch\test_data.csv",
    "test_negative_data" : r"NCF_Pytorch\test_negative_data.csv",
    'dataset': 'ml-1m',   ## name of dataset
    'regs': [0, 0],       ## Regularigaion L1, L2
    'lr': 0.001,          ## Learning Rate
    'batch_size': 256,    ## Batch Size
    'epochs': 3,          ## Training Epochs
    'learner': 'adam',    ## Optimizer
    'layers': [32, 16, 8], ## 3 layer with this number of Nurons
    'num_factors': 10,    ## we used it as latent Dimensions
    'num_neg': 2,         ## per User no of negative items
    'out': True,          ## Save best model or not
    'topK': 10            ## Used in Evaluation.
}

# print('Configurations: ')
# for key, value in configurations.items():
#   print(f'{key} : {value}')

train_data_object = NCFTrainDataset(train_csv=configurations["train_data"], num_negatives=configurations["num_neg"])

test_data_object = NCFTestDataset(test_csv=configurations["test_data"], test_negative_csv=configurations["test_negative_data"])

num_users = train_data_object.num_users
num_items = train_data_object.num_items

print(f"Number of users: {num_users}, Number of items: {num_items}")

Model = NCF_mlp(num_users=num_users, num_items=num_items, layers= configurations["layers"])

print(f"Model: {Model}")

train_data_loader = DataLoader(train_data_object, configurations["batch_size"], shuffle=False)
test_data_loader = DataLoader(test_data_object, configurations["batch_size"], shuffle=False)

print("Data has been loaded.")

## Finally Train MLP Model

NCF_mlp_train(Model, train_loader=train_data_loader, test_negative_data_object=test_data_object, NCF_evaluation=NCFEvaluator, config=configurations, device="cpu")

if __name__ == "__main__":
    print("Calling from MLP Training.")