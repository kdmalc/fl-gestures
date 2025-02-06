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
NUM_CHANNELS = 80  # 16  # Hopefully changing this between 16 and 80 doesn't break things later
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
FINETUNE = False
LOG_AND_VISUALIZE = False
if NUM_CHANNELS == 80:
    APPLY_FEATENG = True
else:
    APPLY_FEATENG = False

MODEL_STR = "ELEC573Net"
config = {
    "learning_rate": 0.0001,
    "batch_size": 32,
    "optimizer": "adam",
    "weight_decay": 1e-4,
    "num_channels": NUM_CHANNELS,  #(int): Number of input channels.
    "sequence_length": 64,  # Not used for this one AFAIK
    "time_steps": None,  # Not used for this one AFAIK
    "num_classes": 10, #(int): Number of output classes.
    "num_epochs": 50,  # To use earlystopping or not... I guess I can check both...
    "use_earlystopping": False,  # Always use this to save time, in ft and earlier training
    "conv_layers": [(32, 5, 2), (64, 5, 2), (128, 5, 2)], 
    "fc_layers": [128],  #(list of int): List of integers specifying the sizes of fully connected layers.
    # I think I can just pass these in and it'll be fine
    "use_batchnorm": True,
    "padding": 1,  # I don't even think this appears in the new version...
    "pooling_layers": [None],  # In ELEC573Net pooling is on by default, see below
    "maxpool": 1,  # Just adding/passing this through 
    "cnn_dropout": 0.0,
    "fc_dropout": 0.3, 
    "lr_scheduler_gamma": 1.0, 
    # DEFAULT PARAMS FROM fine_tune_model() IN OLD CODE: num_epochs=20, lr=0.00001, use_weight_decay=True, weight_decay=0.05)
    "ft_learning_rate": 0.00001,
    "ft_weight_decay": 0.05,  # This is really high...
    "num_ft_epochs": 20,
    # UNSURE ABOUT HOW THIS WORKS
    "ft_batch_size": 32,  # How was this using 32 as its batch_size if there's only 30 total finetuning gestures...
    # METADATA
    "results_save_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\results\\{timestamp}",  # \\hyperparam_tuning
    "models_save_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\models\\{timestamp}",  # \\hyperparam_tuning
    "perf_log_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\results\\performance_logs", 
    "timestamp": timestamp,
    "verbose": False,
    "log_each_pid_results": False, 
    "save_ft_models": False  # Not even applicable here
}

##################################################
"""
path1 = 'C:\\Users\\kdmen\\Box\\Meta_Gesture_2024\\saved_datasets\\filtered_datasets\\$BStand_EMG_df.pkl'
with open(path1, 'rb') as file:
    raw_expdef_data_df = pickle.load(file)  # (204800, 19)

expdef_df = raw_expdef_data_df.groupby(['Participant', 'Gesture_ID', 'Gesture_Num']).apply(create_feature_vectors)
expdef_df = expdef_df.reset_index(drop=True)

all_participants = expdef_df['Participant'].unique()
# Shuffle the participants
np.random.shuffle(all_participants)
# Split into two groups
#train_participants = all_participants[:24]  # First 24 participants
test_participants = all_participants[24:]  # Remaining 8 participants

#convert Gesture_ID to numerical with new Gesture_Encoded column
label_encoder = LabelEncoder()
expdef_df['Gesture_Encoded'] = label_encoder.fit_transform(expdef_df['Gesture_ID'])
label_encoder2 = LabelEncoder()
expdef_df['Cluster_ID'] = label_encoder2.fit_transform(expdef_df['Participant'])
"""

expdef_df = load_expdef_gestures(apply_hc_feateng=APPLY_FEATENG)
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
#os.makedirs(os.path.dirname(full_path), exist_ok=True)  # Ensure the directory exists
os.makedirs(full_path, exist_ok=True)  # Ensure the directory exists
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
