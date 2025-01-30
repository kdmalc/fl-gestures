import torch
import torch.nn as nn
from itertools import product
import pandas as pd
from datetime import datetime

from CNN_hyperparamtuning_funcs import *
from moments_engr import *
from DNN_FT_funcs import *

# Define the search space
hyperparameter_space = {
    "learning_rate": [0.0001, 0.001, 0.01, 0.1],
    "batch_size": [32, 64, 128],
    "num_epochs": [30, 50, 70],
    "optimizer": ["adam", "sgd"],
    "weight_decay": [0.0, 1e-4],
    "dropout_rate": [0.0, 0.3, 0.5]
}

architecture_space = {
    "num_conv_layers": [2, 3],
    "conv_layer_sizes": [[16, 32, 64], [32, 64, 128]],  # Being longer doesnt hurt, the last one might just not get used
    "kernel_size": [3, 5],
    "stride": [1, 2],
    "padding": [1],
    "maxpool": [1, 2, 2],  # Want to sample 2 more often hence the duplicate
    "use_batchnorm": [True, False]
}

##################################################################
##################################################################
##################################################################

# Run hyperparameter tuning
def run_hyperparameter_tuning(train_dataset, intra_test_loader, cross_test_loader, num_configs_to_test=20, save_path="ELEC573_Proj\\results\\hyperparam_tuning"):
    configs = sample_config(hyperparameter_space, architecture_space, num_configs_to_test)
    results = []

    for config_idx, config in enumerate(configs):
        print(f"Config {config_idx}/{num_configs_to_test}")
        result = evaluate_config(config, train_dataset, intra_test_loader, cross_test_loader)
        results.append(result)

    # Sort results by intra_test_acc
    results = sorted(results, key=lambda x: x["intra_test_acc"], reverse=True)

    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Save to CSV
    df = pd.DataFrame([{"intra_test_acc": r["intra_test_acc"], "cross_test_acc": r["cross_test_acc"], "train_acc": r["train_acc"], **r["config"]} for r in results])
    df.to_csv(os.path.join(save_path, f"{timestamp}_hyperparameter_results.csv"), index=False)

    # Save to TXT
    ## This is hard to parse... I already have the csv so probably don't need to save this stuff
    #with open(os.path.join(save_path, f"{timestamp}_hyperparameter_results.txt"), "w") as f:
    #    for r in results:
    #        f.write(f"Config: {r['config']}, train_acc: {r['train_acc']:.4f}, intra_test_acc: {r['intra_test_acc']:.4f}, cross_test_acc: {r['cross_test_acc']:.4f}\n")

    return results

##################################################################
##################################################################
##################################################################

train_dataset, intra_test_loader, cross_test_loader = load_and_prepare_data()

# Example usage
results = run_hyperparameter_tuning(train_dataset, intra_test_loader, cross_test_loader, num_configs_to_test=100)
# print("Top configurations:", results[:5])
