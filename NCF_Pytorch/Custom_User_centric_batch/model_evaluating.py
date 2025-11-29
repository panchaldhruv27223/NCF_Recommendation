import os, sys, time
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../")))
# from NCF_Pytorch.NeuMF_model import NeuMF, train_NeuMF_model
from NCF_Pytorch.ml_1m_dataset import NCFTestDataset
# from User_centric_batch_data import NCFTrainDataset
from NCF_Pytorch.NCF_evaluation import NCFEvaluator


def evaluate_trained_model(model_path, config, device='cpu'):

    os.makedirs(config["results_path"], exist_ok=True)
    
    os.makedirs(config["plots_path"], exist_ok=True)
    
    topk = config["topK"]
    dataset_name = config["dataset"]
    neg_ratio = config["neg_ratio"]
    model_name = config.get("model_name", "NeuMF")



    model = torch.load(model_path, map_location=device, weights_only=False)
    
    model.to(device)
    
    model.eval()
    
    # print(model)
    print("Model Loaded Perfectly.")
    
    test_data_object = NCFTestDataset(test_csv=config["test_data"], test_negative_csv=config["test_negative_data"])
    # print(test_data_object)
    
    evaluator = NCFEvaluator(model, test_data_object, top_k=config["topK"], device=device)

    hits, ndcgs, precision, recall  = evaluator.evaluate()
    
    metrics = {
        "HitRate": hits,
        "nDCG": ndcgs,
        "Precision": precision,
        "Recall": recall
    }
    
    print(metrics)
    
    json_path = os.path.join(config["results_path"], "evaluation_metrics.json")
    
    csv_path = os.path.join(config["results_path"], "evaluation_metrics.csv")

    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    pd.DataFrame([metrics]).to_csv(csv_path, index=False)


    # for metric_name, value in metrics.items():
    #     plt.figure(figsize=(5, 4))
    #     plt.bar(metric_name, value, color='steelblue')
    #     plt.ylim(0, 1)
    #     plt.title(f"{metric_name} Score", fontsize=14)
    #     plt.ylabel("Score")
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(config["plots_path"], f"{metric_name}_score.png"))
    #     plt.close()

    # # Combined plot
    # plt.figure(figsize=(8, 5))
    # plt.bar(list(metrics.keys()), list(metrics.values()), color=['royalblue', 'seagreen', 'darkorange', 'crimson'])
    # plt.ylim(0, 1)
    # plt.title("NeuMF Evaluation Metrics", fontsize=16)
    # plt.ylabel("Score")
    # plt.grid(axis='y', linestyle='--', alpha=0.6)
    # plt.tight_layout()
    # plt.savefig(os.path.join(config["plots_path"], "combined_metrics.png"))
    # plt.close()

    # === Individual Metric Plots ===
    # for metric_name, value in metrics.items():
    #     plt.figure(figsize=(5, 4))
    #     display_name = f"{metric_name}@{topk}" if "@K" not in metric_name else metric_name
    #     plt.bar(display_name, value, color='steelblue')
    #     plt.ylim(0, 1)
    #     plt.title(f"{model_name} - {display_name} ({dataset_name})", fontsize=14, pad=12)
    #     plt.ylabel("Score", fontsize=12)
    #     plt.grid(axis='y', linestyle='--', alpha=0.4)

    #     # Add numerical label on top
    #     plt.text(display_name, value + 0.02, f"{value:.4f}", ha='center', fontsize=11, weight='bold')

    #     plt.tight_layout()
    #     filename = f"{dataset_name}_{display_name}_score.png".replace('@', 'At')
    #     plt.savefig(os.path.join(config["plots_path"], filename))
    #     plt.close()

    # === Combined Plot ===
    plt.figure(figsize=(8, 5))
    metric_labels = [f"{k}@{topk}" if "@K" not in k else k for k in metrics.keys()]
    values = list(metrics.values())

    bars = plt.bar(metric_labels, values, color=['royalblue', 'seagreen', 'darkorange', 'crimson'])
    plt.ylim(0, 1)
    plt.title(f"{model_name} Evaluation Metrics on {dataset_name} (TopK-{topk}) NR: ({neg_ratio})", fontsize=16, pad=12)
    plt.ylabel("Score", fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Add numeric values above bars
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.4f}",
                 ha='center', fontsize=11, weight='bold')

    plt.tight_layout()
    combined_path = os.path.join(config["plots_path"], f"{dataset_name}_combined_metrics_Top{topk}.png")
    plt.savefig(combined_path)
    plt.close()
    

if __name__ == "__main__":
    
    # config = {
    #     "dataset": "MovieLens1M",
    #     "model_name": "NeuMF",
    #     "test_data" : Path(os.getcwd()) / "NCF_Pytorch" / "test_data.csv",
    #     "test_negative_data" : Path(os.getcwd()) / "NCF_Pytorch" / "test_negative_data.csv",
    #     "topK": 20,
    #     "neg_ratio": 0.95,
    #     "results_path": "evaluation_results/NeuMF_EL2N_User_centric_pos_neg_ratio_0.05_0.95_topk_20/",
    #     "plots_path": "evaluation_plots/NeuMF_EL2N_User_centric_pos_neg_ratio_0.05_0.95_topk_20/"
    # }

    # model_path = "/home/dhruv/Documents/NCF/NCF_Recommendation/NeuMF_Models/NeuMF_EL2N_User_centric_pos_neg_ratio_0.05_0.95/ml-1m_NeuMF_Batch_1024_epoch_50_10.pth"


    # device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # evaluate_trained_model(model_path, config, device)

    # config = {
    #     "dataset": "MovieLens1M",
    #     "model_name": "NeuMF",
    #     "test_data" : Path(os.getcwd()) / "NCF_Pytorch" / "test_data.csv",
    #     "test_negative_data" : Path(os.getcwd()) / "NCF_Pytorch" / "test_negative_data.csv",
    #     "topK": 20,
    #     "neg_ratio": 0.8,
    #     "results_path": "evaluation_results/NeuMF_EL2N_User_centric_pos_neg_ratio_0.2_0.8_topk_20/",
    #     "plots_path": "evaluation_plots/NeuMF_EL2N_User_centric_pos_neg_ratio_0.2_0.8_topk_20/"
    # }

    # model_path = "/home/dhruv/Documents/NCF/NCF_Recommendation/NeuMF_Models/NeuMF_EL2N_User_centric_pos_neg_ratio_0.2_0.8/ml-1m_NeuMF_Batch_1024_epoch_50_10.pth"


    # device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # evaluate_trained_model(model_path, config, device)


    # config = {
    #     "dataset": "MovieLens1M",
    #     "model_name": "NeuMF",
    #     "test_data" : Path(os.getcwd()) / "NCF_Pytorch" / "test_data.csv",
    #     "test_negative_data" : Path(os.getcwd()) / "NCF_Pytorch" / "test_negative_data.csv",
    #     "topK": 20,
    #     "neg_ratio": 0.6,
    #     "results_path": "evaluation_results/NeuMF_EL2N_User_centric_pos_neg_ratio_0.4_0.6_topk_20/",
    #     "plots_path": "evaluation_plots/NeuMF_EL2N_User_centric_pos_neg_ratio_0.4_0.6_topk_20/"
    # }

    # model_path = "/home/dhruv/Documents/NCF/NCF_Recommendation/NeuMF_Models/NeuMF_EL2N_User_centric_pos_neg_ratio_0.4_0.6/ml-1m_NeuMF_Batch_1024_epoch_50_10.pth"


    # device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # evaluate_trained_model(model_path, config, device)


    # config = {
    #     "dataset": "MovieLens1M",
    #     "model_name": "NeuMF",
    #     "test_data" : Path(os.getcwd()) / "NCF_Pytorch" / "test_data.csv",
    #     "test_negative_data" : Path(os.getcwd()) / "NCF_Pytorch" / "test_negative_data.csv",
    #     "topK": 20,
    #     "neg_ratio": 0.4,
    #     "results_path": "evaluation_results/NeuMF_EL2N_User_centric_pos_neg_ratio_0.6_0.4_topk_20/",
    #     "plots_path": "evaluation_plots/NeuMF_EL2N_User_centric_pos_neg_ratio_0.6_0.4_topk_20/"
    # }

    # model_path = "/home/dhruv/Documents/NCF/NCF_Recommendation/NeuMF_Models/NeuMF_EL2N_User_centric_pos_neg_ratio_0.6_0.4/ml-1m_NeuMF_Batch_1024_epoch_50_10.pth"


    # device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # evaluate_trained_model(model_path, config, device)


    # config = {
    #     "dataset": "MovieLens1M",
    #     "model_name": "NeuMF",
    #     "test_data" : Path(os.getcwd()) / "NCF_Pytorch" / "test_data.csv",
    #     "test_negative_data" : Path(os.getcwd()) / "NCF_Pytorch" / "test_negative_data.csv",
    #     "topK": 20,
    #     "neg_ratio": 0.2,
    #     "results_path": "evaluation_results/NeuMF_EL2N_User_centric_pos_neg_ratio_0.8_0.2_topk_20/",
    #     "plots_path": "evaluation_plots/NeuMF_EL2N_User_centric_pos_neg_ratio_0.8_0.2_topk_20/"
    # }

    # model_path = "/home/dhruv/Documents/NCF/NCF_Recommendation/NeuMF_Models/NeuMF_EL2N_User_centric_pos_neg_ratio_0.8_0.19999999999999996/ml-1m_NeuMF_Batch_1024_epoch_50_10.pth"


    # device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # evaluate_trained_model(model_path, config, device)


    # config = {
    #     "dataset": "MovieLens1M",
    #     "model_name": "NeuMF",
    #     "test_data" : Path(os.getcwd()) / "NCF_Pytorch" / "test_data.csv",
    #     "test_negative_data" : Path(os.getcwd()) / "NCF_Pytorch" / "test_negative_data.csv",
    #     "topK": 5,
    #     "neg_ratio": 0.4,
    #     "results_path": "evaluation_results/NeuMF_EL2N_40_User_centric_pos_neg_ratio_0.6_0.4_topk_5/",
    #     "plots_path": "evaluation_plots/NeuMF_EL2N_40_User_centric_pos_neg_ratio_0.6_0.4_topk_5/"
    # }

    # model_path = "/home/dhruv/Documents/NCF/NCF_Recommendation/NeuMF_Models/final_NeuMF_El2n_40_User_centric_pos_neg_ratio_0.6_0.4/ml-1m_NeuMF_Batch_1024_epoch_50_10.pth"


    # device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # evaluate_trained_model(model_path, config, device)
    
    # config = {
    #     "dataset": "MovieLens1M",
    #     "model_name": "NeuMF",
    #     "test_data" : Path(os.getcwd()) / "NCF_Pytorch" / "test_data.csv",
    #     "test_negative_data" : Path(os.getcwd()) / "NCF_Pytorch" / "test_negative_data.csv",
    #     "topK": 10,
    #     "neg_ratio": 0.4,
    #     "results_path": "evaluation_results/NeuMF_EL2N_40_User_centric_pos_neg_ratio_0.6_0.4_topk_10/",
    #     "plots_path": "evaluation_plots/NeuMF_EL2N_40_User_centric_pos_neg_ratio_0.6_0.4_topk_10/"
    # }

    # model_path = "/home/dhruv/Documents/NCF/NCF_Recommendation/NeuMF_Models/final_NeuMF_El2n_40_User_centric_pos_neg_ratio_0.6_0.4/ml-1m_NeuMF_Batch_1024_epoch_50_10.pth"


    # device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # evaluate_trained_model(model_path, config, device)
    
    # config = {
    #     "dataset": "MovieLens1M",
    #     "model_name": "NeuMF",
    #     "test_data" : Path(os.getcwd()) / "NCF_Pytorch" / "test_data.csv",
    #     "test_negative_data" : Path(os.getcwd()) / "NCF_Pytorch" / "test_negative_data.csv",
    #     "topK": 20,
    #     "neg_ratio": 0.4,
    #     "results_path": "evaluation_results/NeuMF_EL2N_40_User_centric_pos_neg_ratio_0.6_0.4_topk_20/",
    #     "plots_path": "evaluation_plots/NeuMF_EL2N_40_User_centric_pos_neg_ratio_0.6_0.4_topk_20/"
    # }

    # model_path = "/home/dhruv/Documents/NCF/NCF_Recommendation/NeuMF_Models/final_NeuMF_El2n_40_User_centric_pos_neg_ratio_0.6_0.4/ml-1m_NeuMF_Batch_1024_epoch_50_10.pth"


    # device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # evaluate_trained_model(model_path, config, device)

    config = {
        "dataset": "MovieLens1M",
        "model_name": "NeuMF",
        "test_data" : Path(os.getcwd()) / "NCF_Pytorch" / "test_data.csv",
        "test_negative_data" : Path(os.getcwd()) / "NCF_Pytorch" / "test_negative_data.csv",
        "topK": 50,
        "neg_ratio": 0.4,
        "results_path": "evaluation_results/EL2N_pos_75_neg_75_User_centric_pos_neg_ratio_0.6_0.4_50_epoch_top_k_50/",
        "plots_path": "evaluation_plots/EL2N_pos_75_neg_75_User_centric_pos_neg_ratio_0.6_0.4_50_epoch_top_k_50/"
    }

    model_path = "/home/dhruv/Documents/NCF/NCF_Recommendation/NeuMF_Models/EL2N_pos_75_neg_75_User_centric_pos_neg_ratio_0.6_0.4/ml-1m_NeuMF_Batch_1024_epoch_50_10.pth"


    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    evaluate_trained_model(model_path, config, device)
    
    config = {
        "dataset": "MovieLens1M",
        "model_name": "NeuMF",
        "test_data" : Path(os.getcwd()) / "NCF_Pytorch" / "test_data.csv",
        "test_negative_data" : Path(os.getcwd()) / "NCF_Pytorch" / "test_negative_data.csv",
        "topK": 20,
        "neg_ratio": 0.4,
        "results_path": "evaluation_results/EL2N_pos_75_neg_75_User_centric_pos_neg_ratio_0.6_0.4_50_epoch_top_k_20/",
        "plots_path": "evaluation_plots/EL2N_pos_75_neg_75_User_centric_pos_neg_ratio_0.6_0.4_50_epoch_top_k_20/"
    }

    model_path = "/home/dhruv/Documents/NCF/NCF_Recommendation/NeuMF_Models/EL2N_pos_75_neg_75_User_centric_pos_neg_ratio_0.6_0.4/ml-1m_NeuMF_Batch_1024_epoch_50_10.pth"


    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    evaluate_trained_model(model_path, config, device)
    
    config = {
        "dataset": "MovieLens1M",
        "model_name": "NeuMF",
        "test_data" : Path(os.getcwd()) / "NCF_Pytorch" / "test_data.csv",
        "test_negative_data" : Path(os.getcwd()) / "NCF_Pytorch" / "test_negative_data.csv",
        "topK": 10,
        "neg_ratio": 0.4,
        "results_path": "evaluation_results/EL2N_pos_75_neg_75_User_centric_pos_neg_ratio_0.6_0.4_50_epoch_top_k_10/",
        "plots_path": "evaluation_plots/EL2N_pos_75_neg_75_User_centric_pos_neg_ratio_0.6_0.4_50_epoch_top_k_10/"
    }

    model_path = "/home/dhruv/Documents/NCF/NCF_Recommendation/NeuMF_Models/EL2N_pos_75_neg_75_User_centric_pos_neg_ratio_0.6_0.4/ml-1m_NeuMF_Batch_1024_epoch_50_10.pth"


    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    evaluate_trained_model(model_path, config, device)
    
    config = {
        "dataset": "MovieLens1M",
        "model_name": "NeuMF",
        "test_data" : Path(os.getcwd()) / "NCF_Pytorch" / "test_data.csv",
        "test_negative_data" : Path(os.getcwd()) / "NCF_Pytorch" / "test_negative_data.csv",
        "topK": 5,
        "neg_ratio": 0.4,
        "results_path": "evaluation_results/EL2N_pos_75_neg_75_User_centric_pos_neg_ratio_0.6_0.4_50_epoch_top_k_5/",
        "plots_path": "evaluation_plots/EL2N_pos_75_neg_75_User_centric_pos_neg_ratio_0.6_0.4_50_epoch_top_k_5/"
    }

    model_path = "/home/dhruv/Documents/NCF/NCF_Recommendation/NeuMF_Models/EL2N_pos_75_neg_75_User_centric_pos_neg_ratio_0.6_0.4/ml-1m_NeuMF_Batch_1024_epoch_50_10.pth"


    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    evaluate_trained_model(model_path, config, device)