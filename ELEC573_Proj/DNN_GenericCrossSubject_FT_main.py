# THIS TRAINS A SINGLE GENERIC CNN AND TESTS ON WITHHELD USERS
## NO CLUSTERING!!!

import numpy as np
from moments_engr import *
from DNN_FT_funcs import *
np.random.seed(42) 
import os
cwd = os.getcwd()
print("Current Working Directory: ", cwd)
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M")


FINETUNE = True
LOG_AND_VISUALIZE = True
MODEL_STR = "ELEC573Net" #"DynamicMomonaNet"
FEATENG = "moments"  # "moments" "FS" None
if FEATENG is not None and MODEL_STR=="DynamicMomonaNet":
    NUM_CHANNELS = 1
    SEQ_LEN = 80
elif FEATENG is None and MODEL_STR=="DynamicMomonaNet":
    NUM_CHANNELS = 16
    SEQ_LEN = 64
elif FEATENG is not None and MODEL_STR=="ELEC573Net":
    NUM_CHANNELS = 80
    SEQ_LEN = 1 
elif FEATENG is None and MODEL_STR=="ELEC573Net":
    NUM_CHANNELS = 16
    SEQ_LEN = 64  # I think this will break with ELEC573Net... not integrated AFAIK

# BEST PARAMS FOR GENERIC MODEL
MY_CONFIG = {
    "feature_engr": FEATENG, 
    "learning_rate": 0.0001,
    "batch_size": 32,
    "optimizer": "adam",
    "weight_decay": 1e-4,
    "num_channels": NUM_CHANNELS,  #(int): Number of input channels.
    "sequence_length": SEQ_LEN,  # Not used for this one AFAIK
    "time_steps": None,  # Not used for this one AFAIK
    "num_classes": 10, #(int): Number of output classes.
    "num_epochs": 50,  # To use earlystopping or not... I guess I can check both...
    "conv_layers": [(32, 5, 2), (64, 5, 2), (128, 5, 2)], 
    "fc_layers": [128],  #(list of int): List of integers specifying the sizes of fully connected layers.
    "use_batchnorm": True,
    "padding": 1,
    "pooling_layers": [None],  # In ELEC573Net pooling is on by default, see below
    "maxpool": 1,  # Just adding/passing this through 
    "cnn_dropout": 0.0,
    "fc_dropout": 0.3, 
    # DEFAULT PARAMS FROM fine_tune_model() IN OLD CODE: num_epochs=20, lr=0.00001, use_weight_decay=True, weight_decay=0.05)
    "ft_learning_rate": 0.00001,
    "ft_weight_decay": 0.05,  # This is really high...
    "num_ft_epochs": 20,
    "finetune_strategy": "progressive_unfreeze",
    "progressive_unfreezing_schedule": 5,
    # UNSURE ABOUT HOW THIS WORKS
    "ft_batch_size": 32,  # How was this using 32 as its batch_size if there's only 30 total finetuning gestures...
    # METADATA
    "results_save_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\results\\{timestamp}",  # \\hyperparam_tuning
    "models_save_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\models\\{timestamp}",  # \\hyperparam_tuning
    "perf_log_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\results\\performance_logs", 
    "timestamp": timestamp,
    "verbose": False,
    "log_each_pid_results": False, 
    "save_ft_models": False,  # Not even applicable here
    #"use_earlystopping": True,
    #"lr_scheduler_gamma": 1.0,
    "lr_scheduler_patience": 4, 
    "lr_scheduler_factor": 0.1, 
    "earlystopping_patience": 6,
    "earlystopping_min_delta": 0.01
}

#from hyperparam_tuned_configs import *
#config = DynamicMomonaNet_config
_ = {
    "feature_engr": FEATENG, 
    "weight_decay": 0.0,
    "verbose": False,
    "use_dense_cnn_lstm": True,
    "timestamp": timestamp,
    "time_steps": None,
    "sequence_length": SEQ_LEN,
    "save_ft_models": False,
    "pooling_layers": [True, True, True, True],
    "optimizer": "sgd",
    "num_ft_epochs": 100,
    "num_epochs": 100,
    "num_classes": 10,
    "num_channels": NUM_CHANNELS,
    "results_save_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\results\\{timestamp}",  # \\hyperparam_tuning
    "models_save_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\models\\{timestamp}",  # \\hyperparam_tuning
    "perf_log_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\results\\performance_logs", 
    "lstm_num_layers": 1,
    "lstm_hidden_size": 8,
    "lstm_dropout": 0.8,
    "log_each_pid_results": False,
    "learning_rate": 0.001,
    "ft_weight_decay": 0.0,
    "ft_learning_rate": 0.01,
    "ft_batch_size": 10,
    "finetune_strategy": "progressive_unfreeze",
    "progressive_unfreezing_schedule": 5,
    "fc_layers": [128, 64],
    "fc_dropout": 0.3,
    "dense_cnnlstm_dropout": 0.3,
    "conv_layers": [
        [64, 3, 1],
        [128, 3, 1]],
    "cnn_dropout": 0.3,
    "batch_size": 16,
    "added_dense_ft_hidden_size": 64, 
    #"use_earlystopping": True,
    #"lr_scheduler_gamma": 1.0,
    "lr_scheduler_patience": 4, 
    "lr_scheduler_factor": 0.1, 
    "earlystopping_patience": 6,
    "earlystopping_min_delta": 0.01
}

expdef_df = load_expdef_gestures(feateng_method=MY_CONFIG["feature_engr"])
all_participants = list(expdef_df['Participant'].unique())
# Shuffle the participants
#np.random.shuffle(all_participants)
# Split into two groups
#train_participants = all_participants[:24]  # First 24 participants
test_participants = all_participants[24:]  # Remaining 8 participants

# Prepare data
data_splits = prepare_data(
    expdef_df, 'feature', 'Gesture_Encoded', 
    all_participants, test_participants, 
    training_trials_per_gesture=8, finetuning_trials_per_gesture=3,
)

# Train base model
results = main_training_pipeline(
    data_splits, 
    all_participants=all_participants, 
    test_participants=test_participants,
    model_type=MODEL_STR,
    config=MY_CONFIG)

#full_path = os.path.join(cwd, 'ELEC573_Proj', 'models', 'generic_CNN_model.pth')
full_path = MY_CONFIG['models_save_dir']
os.makedirs(full_path, exist_ok=True)  # Ensure the directory exists
print("Full Path:", full_path)
# TODO: Could just use my existing save_model() func here...
torch.save(results["model"].state_dict(), f"{full_path}\\pretrained_{MODEL_STR}_model.pth")

# TODO: This only finetunes on one person lol should at least finetune over a couple
## Well I guess if this is just for debugging not performance this is fine
if FINETUNE:
    # Fine-tune on first test participant's data
    first_test_participant_data = data_splits['novel_trainFT']
    # Select just the first given participant ID from this dataset, probably for both train_data and labels

    participant_id = first_test_participant_data['participant_ids'][0]
    # Filter based on participant_id
    indices = [i for i, pid in enumerate(first_test_participant_data['participant_ids']) if pid == participant_id]
    ft_dataset = GestureDataset([first_test_participant_data['feature'][i] for i in indices], [first_test_participant_data['labels'][i] for i in indices])
    ft_loader = DataLoader(ft_dataset, batch_size=MY_CONFIG["batch_size"], shuffle=True)

    indices = [i for i, datasplit_pid in enumerate(first_test_participant_data['participant_ids']) if datasplit_pid == participant_id]
    # ^ These indices should be the same as the above indices I think...
    intra_test_dataset = GestureDataset([first_test_participant_data['feature'][i] for i in indices], [first_test_participant_data['labels'][i] for i in indices])
    intra_test_loader = DataLoader(intra_test_dataset, batch_size=MY_CONFIG["batch_size"], shuffle=True)

    finetuned_model, frozen_original_model, train_loss_log, test_loss_log = fine_tune_model(
        results['model'], ft_loader, MY_CONFIG, MY_CONFIG["timestamp"], test_loader=intra_test_loader, pid=participant_id
    )

if LOG_AND_VISUALIZE:
    # Does this visualize ONLY heatmaps, and not train/test loss curves? I only really care about the latter...
    #visualize_model_acc_heatmap(results)

    visualize_train_test_loss_curves(results, MY_CONFIG)
    visualize_train_test_loss_curves(None, MY_CONFIG, train_loss_log, test_loss_log, my_title="1subj FT Train Test Curves", ft=True)

    #log_file = log_performance(results, MY_CONFIG)
    #print(f"Detailed performance log saved to: {log_file}")
