import os, sys, time
import torch
from torch.utils.data import DataLoader
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../")))
from NCF_Pytorch.MLP_model import NCF_mlp, NCF_mlp_train
from NCF_Pytorch.ml_1m_dataset import NCFTestDataset
from User_centric_batch_data import NCFTrainDataset
from NCF_Pytorch.NCF_evaluation import NCFEvaluator
from NCF_Pytorch.logger import setup_logger


def main(learner = 'adam', layers= [32, 16, 8],epochs = 3, batch_size= 256, num_factors = 10, num_neg = 2, topK= 10, shuffle=False, shuffle_users=True, shuffle_within_user=True, pos_percent=0.5,output_folder_path="", output_folder_path_log = ""):

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
        'layers': layers, ## 3 layer with this number of Nurons
        'num_factors': num_factors,    ## we used it as latent Dimensions
        'num_neg': num_neg,         ## per User no of negative items
        'out': True,          ## Save best model or not
        'out_path' : Path(os.getcwd()) / f"MLP_Models/{output_folder_path}/",
        'topK': topK,            ## Used in Evaluation.
        'shuffle' : shuffle,
        'shuffle_users' : shuffle_users,
        'shuffle_within_user':shuffle_within_user,
        "pos_percent" : pos_percent
    }

    # print('Configurations: ')
    # for key, value in configurations.items():
    #   print(f'{key} : {value}')
    
    
    train_logger, train_logger_path = setup_logger(output_folder_path_log, "traning", config=configurations)

    train_logger.info("Starting MLP User Centric traning... ")

    eval_logger, eval_logger_path = setup_logger(output_folder_path_log, "evaluation", config=configurations)

    eval_logger.info("Starting MLP User Centric Evaluation")
    

    train_data_object = NCFTrainDataset(train_csv=configurations["train_data"], num_negatives=configurations["num_neg"], pos_percent=configurations["pos_percent"])

    test_data_object = NCFTestDataset(test_csv=configurations["test_data"], test_negative_csv=configurations["test_negative_data"])

    num_users = train_data_object.num_users
    num_items = train_data_object.num_items

    # print(f"Number of users: {num_users}, Number of items: {num_items}")
    train_logger.info(f"Total number of users are : {num_users}")
    train_logger.info(f"Total number of items are : {num_items}")


    Model = NCF_mlp(num_users=num_users, num_items=num_items, layers= configurations["layers"], train_logger = train_logger)

    train_logger.info("MLP User Centric Model loaded.")
    eval_logger.info("MLP User Centric Model loaded.")

    # print(f"Model: {Model}")

    # train_data_loader = DataLoader(train_data_object, configurations["batch_size"], shuffle=configurations["shuffle"])
    # test_data_loader = DataLoader(test_data_object, configurations["batch_size"], shuffle=configurations["shuffle"])
    # train_data_loader = train_data_object.get_user_centric_dataloader(shuffle_users= configurations['shuffle_users'], batch_size= configurations['batch_size'], shuffle_within_user=configurations['shuffle_within_user'], num_workers=os.cpu_count()//2, pin_memory=False)
    train_data_loader = train_data_object.get_fixed_ratio_neg_batch_per_user(batch_size=configurations["batch_size"], shuffle_users=configurations["shuffle_users"], shuffle_within_user=configurations["shuffle_within_user"])

    # print("Data has been loaded.")

    ## Finally Train MLP Model
    train_logger.info(f"MLP User Centric Model passed for training...")
    eval_logger.info("MLP User Centric Model Passed for Evaluation...")

    NCF_mlp_train(Model, train_loader=train_data_loader, test_negative_data_object=test_data_object, NCF_evaluation=NCFEvaluator, config=configurations, train_logger = train_logger, test_logger = eval_logger, device="cpu")

if __name__ == "__main__":
    print("Calling from MLP User Centric Training.")
    
    
    for i in [0.80, 0.60, 0.40, 0.20, 0.05]:
        main(
            learner = 'adam', 
            layers= [32, 16, 8],
            epochs = 50, 
            batch_size = 256, 
            num_factors = 10, 
            num_neg = -1, 
            topK= 10, 
            pos_percent=i, 
            shuffle=False, 
            shuffle_users=False, 
            shuffle_within_user=False, 
            output_folder_path=f"MLP_User_centric_pos_neg_ratio_{i}_{1-i}", 
            output_folder_path_log=f"MLp_User_centric_pos_neg_ratio_{i}_{1-i}"
        )