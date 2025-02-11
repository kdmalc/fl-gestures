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
NUM_CHANNELS = 16
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
FINETUNE = False
LOG_AND_VISUALIZE = False

MODEL_STR = "DynamicMomonaNet"

#from hyperparam_tuned_configs import *
#config = DynamicMomonaNet_config

config = {
    "feature_engr": None, 
    "weight_decay": 0.0,
    "verbose": False,
    "use_earlystopping": True,
    "use_dense_cnn_lstm": True,
    "timestamp": timestamp,
    "time_steps": None,
    "sequence_length": 64,
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
    "lr_scheduler_gamma": 1.0,
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
    "added_dense_ft_hidden_size": 64
}

expdef_df = load_expdef_gestures(feateng_method=config["feature_engr"])
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
    config=config)

#full_path = os.path.join(cwd, 'ELEC573_Proj', 'models', 'generic_CNN_model.pth')
full_path = config['models_save_dir']
os.makedirs(os.path.dirname(full_path), exist_ok=True)  # Ensure the directory exists
print("Full Path:", full_path)
# TODO: pretty sure I have a save_model function...
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
    ft_loader = DataLoader(ft_dataset, batch_size=config["batch_size"], shuffle=True)

    indices = [i for i, datasplit_pid in enumerate(first_test_participant_data['participant_ids']) if datasplit_pid == participant_id]
    # ^ These indices should be the same as the above indices I think...
    intra_test_dataset = GestureDataset([first_test_participant_data['feature'][i] for i in indices], [first_test_participant_data['labels'][i] for i in indices])
    intra_test_loader = DataLoader(intra_test_dataset, batch_size=config["batch_size"], shuffle=True)

    finetuned_model, frozen_original_model, train_loss_log, test_loss_log = fine_tune_model(
        results['model'], ft_loader, config, config["timestamp"], test_loader=intra_test_loader, pid=participant_id
    )

if LOG_AND_VISUALIZE:
    # Example usage in main script
    visualize_model_performance(results)

    # Example usage in main script
    log_file = log_performance(results)
    print(f"Detailed performance log saved to: {log_file}")
