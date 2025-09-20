import os, sys, time
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from GMF_model import GMF, train_GMF_model
from ml_1m_dataset import NCFTrainDataset, NCFTestDataset
from NCF_evaluation import NCFEvaluator
# print(os.getcwd())
from logger import setup_logger

def main(learner = 'adam', epochs = 3, batch_size= 256, num_factors = 10, num_neg = 2, topK= 10, shuffle = False, output_folder_path=""):
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
        'shuffle' : shuffle
    }
    
    # print('Configurations: ')
    # for key, value in configurations.items():
    #   print(f'{key} : {value}')

    train_logger, train_logger_path = setup_logger("GMF", "traning", config=configurations)

    train_logger.info("Starting GMF traning... ")

    eval_logger, eval_logger_path = setup_logger("GMF", "evaluation", config=configurations)

    eval_logger.info("Starting GMF Evaluation")

    train_data_object = NCFTrainDataset(train_csv=configurations["train_data"], num_negatives=configurations["num_neg"])

    test_data_object = NCFTestDataset(test_csv=configurations["test_data"], test_negative_csv=configurations["test_negative_data"])

    num_users = train_data_object.num_users
    num_items = train_data_object.num_items

    train_logger.info(f"Total number of users are : {num_users}")
    
    train_logger.info(f"Total number of items are : {num_items}")


    Model = GMF(num_users=num_users, num_items=num_items, latent_dim=configurations["num_factors"], reg=configurations["regs"], train_logger = train_logger)

    train_logger.info(f"GMF Model loaded")

    train_data_loader = DataLoader(train_data_object, configurations["batch_size"], shuffle=configurations["shuffle"])

    test_data_loader = DataLoader(test_data_object, configurations["batch_size"], shuffle=configurations["shuffle"])

    ## Finally Train GMF Model
    # train_logger.info(f"GMF Model passed for training...")
    # eval_logger.info("GMF Model Passed for Evaluation...")

    train_GMF_model(Model, train_loader=train_data_loader, test_negative_dataset=test_data_object, config=configurations, NCFEvaluation=NCFEvaluator, train_logger = train_logger, test_logger = eval_logger, device="cpu")

if __name__ == "__main__":
    print("Calling from GMF Training.")
    # train_logger, train_logger_path = setup_logger("GMF", "traning", config=configurations)
    # train_logger.info("Starting GMF traning... ")
    # eval_logger, eval_logger_path = setup_logger("GMF", "evaluation", config=configurations)
    
    # for i in [10, 20, 30]:
    #     for j in [128, 256, 512]:
    #         main(learner = 'adam', epochs = i, batch_size = j, num_factors = 10, num_neg = 2, topK= 10, shuffle=True, output_folder_path="With_shuffle")

    # for i in [10, 20, 30]:
    #     for j in [128, 256, 512]:
    #         main(learner = 'adam', epochs = i, batch_size = j, num_factors = 10, num_neg = 2, topK= 10, shuffle=False, output_folder_path="Without_shuffle")
    
    # for i in range(3,128):
    #     main(learner = 'adam', epochs = 10, batch_size = 128, num_factors = 10, num_neg = i, topK= 10, shuffle=False, output_folder_path=f"Without_shuffle_num_neg_{i}")