import os, sys, time
import torch
from torch.utils.data import DataLoader
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../")))
from NCF_Pytorch.NeuMF_model import NeuMF, train_NeuMF_model
from NCF_Pytorch.ml_1m_dataset import NCFTestDataset
from User_centric_batch_data import NCFTrainDataset
from NCF_Pytorch.NCF_evaluation import NCFEvaluator
from NCF_Pytorch.logger import setup_logger


def main(learner = 'adam', layers= [32, 16, 8], epochs = 3, batch_size= 256, num_factors = 10, num_neg = 2, topK= 10, shuffle=False, shuffle_users=True, shuffle_within_user=True, output_folder_path="", output_folder_path_log = ""):

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
        'layers': layers,
        'num_factors': num_factors,    ## we used it as latent Dimensions
        'num_neg': num_neg,         ## per User no of negative items
        'out': True,          ## Save best model or not
        'out_path' : Path(os.getcwd()) / f"NeuMF_Models/{output_folder_path}/",
        'topK': topK,           ## Used in Evaluation.
        'shuffle' : shuffle,
        'shuffle_users' : shuffle_users,
        'shuffle_within_user':shuffle_within_user
    }
    
    train_logger, train_logger_path = setup_logger(output_folder_path_log, "traning", config=configurations)

    train_logger.info("Starting NeuMF User Centric traning... ")

    eval_logger, eval_logger_path = setup_logger(output_folder_path_log, "evaluation", config=configurations)

    eval_logger.info("Starting NeuMF User Centric Evaluation")

    train_data_object = NCFTrainDataset(train_csv=configurations["train_data"], num_negatives=configurations["num_neg"])

    test_data_object = NCFTestDataset(test_csv=configurations["test_data"], test_negative_csv=configurations["test_negative_data"])

    num_users = train_data_object.num_users
    num_items = train_data_object.num_items
    
    # print(f"Number of users: {num_users}, Number of items: {num_items}")
    train_logger.info(f"Total number of users are : {num_users}")
    train_logger.info(f"Total number of items are : {num_items}")

    Model = NeuMF(num_users=num_users, num_items=num_items, latent_dim=configurations["num_factors"], layers=configurations["layers"], train_logger = train_logger)
    
    train_logger.info("NeuMF User Centric Model loaded.")
    eval_logger.info("NeuMF User Centric Model loaded.")

    # train_data_loader = DataLoader(train_data_object, configurations["batch_size"], shuffle=configurations["shuffle"])
    # test_data_loader = DataLoader(test_data_object, configurations["batch_size"], shuffle=configurations["shuffle"])
    train_data_loader = train_data_object.get_user_centric_dataloader(shuffle_users= configurations['shuffle_users'], batch_size= configurations['batch_size'], shuffle_within_user=configurations['shuffle_within_user'], num_workers=os.cpu_count()//2, pin_memory=False)
    

    ## Finally Train NeuMF Model
    train_logger.info(f"NeuMF User Centric Model passed for training...")
    eval_logger.info("NeuMF User Centric Model Passed for Evaluation...")
    
    train_NeuMF_model(Model, train_loader=train_data_loader, test_negative_dataset=test_data_object, config=configurations, NCFEvaluation=NCFEvaluator, train_logger = train_logger, test_logger = eval_logger, device="cpu")

if __name__ == "__main__":
    print("Calling from NeuMF User Centric Training.")
    
    
    main(learner = 'adam', layers= [32, 16, 8], epochs = 10, batch_size = -1, num_factors = 10, num_neg = -1, topK= 10, shuffle=False, shuffle_users=True, shuffle_within_user=True, output_folder_path="NeuMF_User_centric_pos_neg_eq", output_folder_path_log = "NeuMF_User_centric_pos_neg_eq")