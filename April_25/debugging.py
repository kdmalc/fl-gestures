import numpy as np
#from sklearn.preprocessing import LabelEncoder
#import matplotlib.pyplot as plt
#import pandas as pd

from global_seed import set_seed
set_seed()

from agglo_model_clust import *
from cluster_acc_viz_funcs import *
from DNN_FT_funcs import *
from DNN_AMC_funcs import *
from full_study_funcs import * 
from revamped_model_classes import *
from hyperparam_tuned_configs import *

import os
cwd = os.getcwd()
print("Current Working Directory: ", cwd)


MODEL_STR = "DynamicMomonaNet"
MY_CONFIG = DynamicMomonaNet_config
NUM_MONTE_CARLO_RUNS = 5
SAVE_FIGS = False
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# Train/test user split: 24/8
## Is this part of config now? The split number?
## No it is hardcoded below...
NUM_PRETRAIN_USERS = 24
NUM_FT_USERS = 8
# ^ This is actually handled in a separate file that saves the user split to a json file...


expdef_df = load_expdef_gestures(feateng_method=MY_CONFIG["feature_engr"])
# Load the CLUSTERING RESULTS DATA
with open('C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\results\\clustering_models\\20250210_2137\\20250210_2137_DynamicMomonaNet_agglo_merge_res.pkl', 'rb') as f:
    merge_log = pickle.load(f)
    intra_cluster_performance = pickle.load(f)
    cross_cluster_performance = pickle.load(f)
    nested_clus_model_dict = pickle.load(f)
full_path = os.path.join("C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\models\\20250211_2010", 'pretrained_DynamicMomonaNet_model.pth')
# Load the pretrained model
pretrained_generic_model = select_model(MODEL_STR, MY_CONFIG)
pretrained_generic_model.load_state_dict(torch.load(full_path))

import json
# Load the fixed user splits
with open("ELEC573_Proj//24_8_user_splits.json", "r") as f:
    splits = json.load(f)
all_participants = splits["all_users"]
test_participants = splits["test_users"]

#"finetune_strategy": "progressive_unfreeze",
#    - "full": Train the entire model.
#    - "freeze_cnn": Freeze CNN, train LSTM and dense layers.
#    - "freeze_all_but_final_dense": Freeze CNN + LSTM, train only dense layers.
#    - "progressive_unfreeze": Start with frozen CNN/LSTM, progressively unfreeze.
#"progressive_unfreezing_schedule": 5,
#"ft_weight_decay": 0.0,
#"ft_learning_rate": 0.001,
#"ft_batch_size": 10,
#"num_ft_epochs": 100,

progunfreezingFT_config = copy.deepcopy(MY_CONFIG)

fullFT_config = copy.deepcopy(MY_CONFIG)
fullFT_config["finetune_strategy"] = "full"

freezeCNNFT_config = copy.deepcopy(MY_CONFIG)
freezeCNNFT_config["finetune_strategy"] = "freeze_cnn"

freezeallbutdenseFT_config = copy.deepcopy(MY_CONFIG)
freezeallbutdenseFT_config["finetune_strategy"] = "freeze_all_but_final_dense"


NUM_FT_TRIALS = 1
shared_trial_data_splits_lst = create_shared_trial_data_splits(expdef_df, all_participants, test_participants, num_monte_carlo_runs=NUM_MONTE_CARLO_RUNS, num_train_gesture_trials=8, num_ft_gesture_trials=NUM_FT_TRIALS)
progunfreezingFT_data_dict_1_1 = finetuning_run(shared_trial_data_splits_lst, progunfreezingFT_config, pretrained_generic_model, nested_clus_model_dict)
plot_model_acc_boxplots(progunfreezingFT_data_dict_1_1, my_title=f"{MODEL_STR}: One-shot FT: Prog Unfr", save_fig=False, plot_save_name=f"Final_{MODEL_STR}_Acc_1TA_1TT", 
                            data_keys=['centralized_acc_data', 'pretrained_cluster_acc_data', 'ft_centralized_acc_data', 'ft_cluster_acc_data'],
                            labels=['Generic Centralized', 'Pretrained Cluster', 'Fine-Tuned Centralized', 'Fine-Tuned Cluster'])
    