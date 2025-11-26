import os, sys, time
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../")))
from NCF_Pytorch.NeuMF_model import NeuMF, train_NeuMF_model
from NCF_Pytorch.ml_1m_dataset import NCFTestDataset
from User_centric_batch_data import NCFTrainDataset
from NCF_Pytorch.NCF_evaluation import NCFEvaluator
from NCF_Pytorch.logger import setup_logger
import time
from codecarbon import EmissionsTracker


def main(train_data="",learner = 'adam', layers= [32, 16, 8], epochs = 3, batch_size= 256, num_factors = 10, num_neg = 2, topK= 10, pos_percent=0.5, shuffle=False, shuffle_users=True, shuffle_within_user=True, output_folder_path="", output_folder_path_log = ""):

    configurations = {
        "train_data" : Path(os.getcwd()) / "NCF_Pytorch" / "train_data.csv" if train_data =="" else train_data,
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
        'shuffle_within_user':shuffle_within_user,
        'pos_percent' : pos_percent
    }
    
    train_logger, train_logger_path = setup_logger(output_folder_path_log, "traning", config=configurations)

    # train_logger.info("Starting NeuMF User Centric traning... ")

    eval_logger, eval_logger_path = setup_logger(output_folder_path_log, "evaluation", config=configurations)

    # eval_logger.info("Starting NeuMF User Centric Evaluation")

    train_data_object = NCFTrainDataset(train_csv=configurations["train_data"], num_negatives=configurations["num_neg"], pos_percent =configurations["pos_percent"])

    test_data_object = NCFTestDataset(test_csv=configurations["test_data"], test_negative_csv=configurations["test_negative_data"])

    num_users = train_data_object.num_users
    num_items = train_data_object.num_items
    
    # # print(f"Number of users: {num_users}, Number of items: {num_items}")
    # train_logger.info(f"Total number of users are : {num_users}")
    # train_logger.info(f"Total number of items are : {num_items}")

    Model = NeuMF(num_users=num_users, num_items=num_items, latent_dim=configurations["num_factors"], layers=configurations["layers"], train_logger = train_logger)
    
    # train_logger.info("NeuMF User Centric Model loaded.")
    # eval_logger.info("NeuMF User Centric Model loaded.")

    # train_data_loader = DataLoader(train_data_object, configurations["batch_size"], shuffle=configurations["shuffle"])
    # test_data_loader = DataLoader(test_data_object, configurations["batch_size"], shuffle=configurations["shuffle"])
    # train_data_loader = train_data_object.get_user_centric_dataloader(shuffle_users= configurations['shuffle_users'], batch_size= configurations['batch_size'], shuffle_within_user=configurations['shuffle_within_user'], num_workers=os.cpu_count()//2, pin_memory=False)
    
    train_data_loader = train_data_object.get_fixed_ratio_neg_batch_per_user(batch_size=configurations["batch_size"], shuffle_users=configurations["shuffle_users"], shuffle_within_user=configurations["shuffle_within_user"])
    
    # print(train_data_loader)
    
    data_loader_iter = iter(train_data_loader)
    
    # print(data_loader_iter)
    
    # user, item, label = next(data_loader_iter)
    
    # print(user)
    
    
    out_csv = os.path.join(configurations["out_path"], "El2n_15_data_batch_info.csv")
    
    
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    def to_1d_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        arr = np.asarray(x)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        return arr.reshape(-1)
  
  
    user_summary = {}

    for batch_idx, batch in enumerate(tqdm(data_loader_iter, desc="Building user summary")):

        users_t, items_t, labels_t = batch
        # print(users_t, items_t, labels_t)
        # print(len(users_t))
        
        users = to_1d_numpy(users_t).astype(int)
        items = to_1d_numpy(items_t).astype(int)
        labels = to_1d_numpy(labels_t).astype(int)

        # print(users, items, labels)
        
        # print(np.unique(labels)).issubset({0., 1.})
        
        
        # if not set(np.unique(labels)).issubset({0, 1}):
        #     labels_bin = (labels >= 1).astype(int)
        # else:
        #     labels_bin = labels.astype(int)
        
        # print(labels_bin)
        
        u = np.unique(users)
        u = u[0]
        
        # print(u[0])
        # break
        total = int(len(users))
        pos = int(labels.sum())
        # print(total)
        # print(pos)
        
        neg = total - pos

        user_summary[u] = [total, pos, neg]

        # break

    rows = []
    for u, (total, pos, neg) in user_summary.items():
        rows.append({"user_id": int(u), "total_items": total, "pos_count": pos, "neg_count": neg, "pos_rate": pos / total if total>0 else None})

    df = pd.DataFrame(rows).sort_values("user_id")
    df.to_csv(out_csv, index=False)
    print(f"Saved user summary to: {out_csv}")
    print(df.head(10))

    ## Finally Train NeuMF Model
    # train_logger.info(f"NeuMF User Centric Model passed for training...")
    # eval_logger.info("NeuMF User Centric Model Passed for Evaluation...")
    
    # train_NeuMF_model(Model, train_loader=train_data_loader, test_negative_dataset=test_data_object, config=configurations, NCFEvaluation=NCFEvaluator, train_logger = train_logger, test_logger = eval_logger, device="cpu")
    
    
    

if __name__ == "__main__":
    
    print("Calling from NeuMF User Centric Training.")
    
    # for i in [0.80, 0.60, 0.40, 0.20, 0.05]:
    # for i in [0.80, 0.40, 0.20, 0.05]:
    #     main(
    #         train_data="/home/dhruv/Documents/NCF/NCF_Recommendation/NCF_Pytorch/Custom_User_centric_batch/El2n_positive.csv",
    #         learner = 'adam', 
    #         layers= [32, 16, 8],
    #         epochs = 50, 
    #         batch_size = 1024, 
    #         num_factors = 10, 
    #         num_neg = -1, 
    #         topK= 10, 
    #         pos_percent=i, 
    #         shuffle=False, 
    #         shuffle_users=False, 
    #         shuffle_within_user=False, 
    #         output_folder_path=f"NeuMF_EL2N_User_centric_pos_neg_ratio_{i}_{1-i}", 
    #         output_folder_path_log=f"NeuMF_EL2N_User_centric_pos_neg_ratio_{i}_{1-i}"
    #     )
    
    # tracker = EmissionsTracker(log_level="error")
    # tracker.start()
    # start = time.time()

    main(
            train_data="/home/dhruv/Documents/NCF/NCF_Recommendation/NCF_Pytorch/Custom_User_centric_batch/El2n_15_data.csv",
            learner = 'adam', 
            layers= [32, 16, 8],
            epochs = 50, 
            batch_size = 1024, 
            num_factors = 10, 
            num_neg = -1, 
            topK= 10, 
            pos_percent=0.60, 
            shuffle=False, 
            shuffle_users=False, 
            shuffle_within_user=False, 
            output_folder_path=f"El2n_15_data_batch_distribution_info", 
            output_folder_path_log=f"El2n_15_data_batch_distribution_info"
        )
    
    
    
    # main(
    #         train_data="/home/dhruv/Documents/NCF/NCF_Recommendation/NCF_Pytorch/Custom_User_centric_batch/El2n_15_data.csv",
    #         learner = 'adam', 
    #         layers= [32, 16, 8],
    #         epochs = 50, 
    #         batch_size = 1024, 
    #         num_factors = 10, 
    #         num_neg = -1, 
    #         topK= 10, 
    #         pos_percent=0.60, 
    #         shuffle=False, 
    #         shuffle_users=False, 
    #         shuffle_within_user=False, 
    #         output_folder_path=f"final_NeuMF_El2n_15_User_centric_pos_neg_ratio_{0.60}_{1-0.60}", 
    #         output_folder_path_log=f"final_NeuMF_El2n_15_User_centric_pos_neg_ratio_{0.60}_{1-0.60}"
    #     )
    
    
    # main(
    #         train_data="/home/dhruv/Documents/NCF/NCF_Recommendation/NCF_Pytorch/Custom_User_centric_batch/El2n_25_data.csv",
    #         learner = 'adam', 
    #         layers= [32, 16, 8],
    #         epochs = 50, 
    #         batch_size = 1024, 
    #         num_factors = 10, 
    #         num_neg = -1, 
    #         topK= 10, 
    #         pos_percent=0.60, 
    #         shuffle=False, 
    #         shuffle_users=False, 
    #         shuffle_within_user=False, 
    #         output_folder_path=f"final_NeuMF_El2n_25_User_centric_pos_neg_ratio_{0.60}_{1-0.60}", 
    #         output_folder_path_log=f"final_NeuMF_El2n_25_User_centric_pos_neg_ratio_{0.60}_{1-0.60}"
    #     )
    
    # main(
    #         train_data="/home/dhruv/Documents/NCF/NCF_Recommendation/NCF_Pytorch/Custom_User_centric_batch/El2n_40_data.csv",
    #         learner = 'adam', 
    #         layers= [32, 16, 8],
    #         epochs = 50, 
    #         batch_size = 1024, 
    #         num_factors = 10, 
    #         num_neg = -1, 
    #         topK= 10, 
    #         pos_percent=0.60, 
    #         shuffle=False, 
    #         shuffle_users=False, 
    #         shuffle_within_user=False, 
    #         output_folder_path=f"final_NeuMF_El2n_40_User_centric_pos_neg_ratio_{0.60}_{1-0.60}", 
    #         output_folder_path_log=f"final_NeuMF_El2n_40_User_centric_pos_neg_ratio_{0.60}_{1-0.60}"
    #     )
        
        
    # end = time.time()
    # emissions = tracker.stop()
    # output_file = "/home/dhruv/Documents/NCF/NCF_Recommendation/NCF_Pytorch/Custom_User_centric_batch/time_co2_info.txt"
    
    # elapsed_time = end - start
    
    # with open(output_file, "a") as f:
    #     f.write(f"Run Summary:\n")
    #     f.write(f"El2n_pos_75_neg_15_data: \n")
    #     f.write(f"Execution Time: {elapsed_time:.2f} seconds\n")
    #     f.write(f"CO₂ Emitted: {emissions:.6f} kg\n")
    #     f.write("-" * 40 + "\n")

    # print(f"Execution time: {end - start:.2f}s")
    # print(f"CO₂ emitted: {emissions:.6f} kg")