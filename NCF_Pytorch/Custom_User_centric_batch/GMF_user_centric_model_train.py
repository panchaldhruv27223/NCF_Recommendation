import os, sys, time
import torch
from torch.utils.data import DataLoader
from pathlib import Path
# print(os.path.dirname(__file__))
# print(os.path.join(os.path.dirname(__file__), "../../"))
# print(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../")))
from NCF_Pytorch.GMF_model import GMF, train_GMF_model
# from NCF_Pytorch.ml_1m_dataset import NCFTrainDataset, NCFTestDataset
from NCF_Pytorch.ml_1m_dataset import NCFTestDataset
from User_centric_batch_data import NCFTrainDataset
from NCF_Pytorch.NCF_evaluation import NCFEvaluator
from NCF_Pytorch.logger import setup_logger
# print(os.getcwd())

def main(learner = 'adam', epochs = 3, batch_size= 256, num_factors = 10, num_neg = 2, topK= 10, pos_percent=0.5, shuffle = False, shuffle_users=True, shuffle_within_user=True, output_folder_path="", output_folder_path_log = ""):
    
    configurations = {
        "train_data" : Path(os.getcwd()) / "NCF_Pytorch" / "train_data.csv",
        "test_data" : Path(os.getcwd()) / "NCF_Pytorch" / "test_data.csv",
        "test_negative_data" : Path(os.getcwd()) / "NCF_Pytorch" / "test_negative_data.csv",
        'dataset': 'ml-1m',   ## name of dataset
        'regs': [0, 0],       ## Regularigaion L1, L2
        'lr': 0.001,          ## Learning Rate
        'batch_size': batch_size,    ## Batch Size
        'epochs': epochs,          ## Training Epochs
        'learner': learner,    ## Optimizer
        'num_factors': num_factors,    ## we used it as latent Dimensions
        'num_neg': num_neg,         ## per User no of negative items
        'out': True,          ## Save best model or not
        'out_path' : Path(os.getcwd()) / f"GMF_Models/{output_folder_path}/",
        'topK' : topK,            ## Used in Evaluation.
        'shuffle' : shuffle,
        'shuffle_users' : shuffle_users,
        'shuffle_within_user':shuffle_within_user,
        'pos_percent' : pos_percent
    }
    
    # print('Configurations: ')
    # for key, value in configurations.items():
    #   print(f'{key} : {value}')

    train_logger, train_logger_path = setup_logger(output_folder_path_log, "traning", config=configurations)

    train_logger.info("Starting GMF_User_Centric traning... ")

    eval_logger, eval_logger_path = setup_logger(output_folder_path_log, "evaluation", config=configurations)

    eval_logger.info("Starting GMF_User_Centric Evaluation")

    train_data_object = NCFTrainDataset(train_csv=configurations["train_data"], num_negatives=configurations["num_neg"], pos_percent =configurations["pos_percent"])

    test_data_object = NCFTestDataset(test_csv=configurations["test_data"], test_negative_csv=configurations["test_negative_data"])

    num_users = train_data_object.num_users
    num_items = train_data_object.num_items

    train_logger.info(f"Total number of users are : {num_users}")
    
    train_logger.info(f"Total number of items are : {num_items}")


    Model = GMF(num_users=num_users, num_items=num_items, latent_dim=configurations["num_factors"], reg=configurations["regs"], train_logger = train_logger)

    train_logger.info(f"GMF Model loaded")

    # train_data_loader = DataLoader(train_data_object, configurations["batch_size"], shuffle=configurations["shuffle"])
    # train_data_loader = train_data_object.get_user_centric_dataloader(shuffle_users= configurations['shuffle_users'], batch_size= configurations['batch_size'], shuffle_within_user=configurations['shuffle_within_user'], num_workers=os.cpu_count() //2, pin_memory=True)
    train_data_loader = train_data_object.get_fixed_ratio_neg_batch_per_user(batch_size=configurations["batch_size"], shuffle_users=configurations["shuffle_users"], shuffle_within_user=configurations["shuffle_within_user"])
  
    # testing the batch.
    
    # for batch_idx, batch in enumerate(train_data_loader):
    #     print("Batch index:", batch_idx)
    #     users, items, labels = batch 
        
    #     # print(f"Users: {len(users.tolist())}")
    #     uniques_users, user_counts = torch.unique(users, return_counts=True)
    #     print(f"For user {uniques_users} and occured {user_counts}")
        
    #     # print(users)
    #     # print(f"items: {items.tolist()}")
    #     # print(f"labels: {labels.values_count()}")
    #     uniques_ele, counts = torch.unique(labels, return_counts=True)
    #     print(f"uniques elements are : {uniques_ele}")
    #     print(f"count of that elements : {counts}")
        
    #     if batch_idx == 2:
    #         break

    # Finally Train GMF Model
    train_logger.info(f"GMF Model passed for training...")
    eval_logger.info("GMF Model Passed for Evaluation...")

    train_GMF_model(Model, train_loader=train_data_loader, test_negative_dataset=test_data_object, config=configurations, NCFEvaluation=NCFEvaluator, train_logger = train_logger, test_logger = eval_logger, device="cpu")

if __name__ == "__main__":
    # print("Calling from GMF User Centric Model Training.")
    # print("Per batch one unique user, and inside the batch the ration of positve and negative is going to maintain.")
    
    
    for i in [0.80, 0.60, 0.40, 0.20, 0.05]:
        main(
            learner = 'adam', 
            epochs = 50, 
            batch_size = 256, 
            num_factors = 10, 
            num_neg = -1, 
            topK= 10, 
            pos_percent=i, 
            shuffle=False, 
            shuffle_users=False, 
            shuffle_within_user=False, 
            output_folder_path=f"GMF_User_centric_pos_neg_ratio_{i}_{1-i}", 
            output_folder_path_log=f"GMF_User_centric_pos_neg_ratio_{i}_{1-i}"
        )