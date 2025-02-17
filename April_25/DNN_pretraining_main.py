# THIS TRAINS A SINGLE GENERIC CNN AND TESTS ON WITHHELD USERS
## NO CLUSTERING!!!

import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from moments_engr import *
from DNN_FT_funcs import *
from hyperparam_tuned_configs import *
np.random.seed(42) 
import os
cwd = os.getcwd()
print("Current Working Directory: ", cwd)


MY_CONFIG = DynamicMomonaNet_config
MODEL_STR = 'DynamicMomonaNet'  # [CNN, CNNLSTM, DynamicMomonaNet, ELEC573Net]
finetune = False
do_normal_logging = True

print("Creating directories")
# Results
os.makedirs(MY_CONFIG["results_save_dir"])
print(f'Directory {MY_CONFIG["results_save_dir"]} created successfully!')
# Models
os.makedirs(MY_CONFIG["models_save_dir"])
print(f'Directory {MY_CONFIG["models_save_dir"]} created successfully!')

##################################################

expdef_df = load_expdef_gestures(feateng_method=MY_CONFIG["feature_engr"])

all_participants = expdef_df['Participant'].unique()
# Shuffle the participants for train/test user split --> UNIQUE
np.random.shuffle(all_participants)
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

#save_dir = MY_CONFIG["models_save_dir"]
#print("Save Path:", save_dir)
#torch.save(results["model"].state_dict(), full_path)
save_model(results["model"], MODEL_STR, MY_CONFIG["models_save_dir"], "pretrained", verbose=True, timestamp=MY_CONFIG["timestamp"])

if finetune:
    # Fine-tune on first test participant's data
    first_test_participant_data = data_splits['novel_trainFT']
    # Select just the first given participant ID from this dataset, probably for both train_data and labels

    participant_id = 'P132'
    # Filter based on participant_id
    indices = [i for i, pid in enumerate(first_test_participant_data['participant_ids']) if pid == participant_id]
    # Extract the corresponding features and labels
    features_p132 = [first_test_participant_data['features'][i] for i in indices]
    labels_p132 = [first_test_participant_data['labels'][i] for i in indices]

    fine_tuned_model = fine_tune_model(
        results['model'], 
        {'features':features_p132, 'labels':labels_p132}
    )

    # NEED TO EVALUATE FINETUNED MODEL, at least on said user's data...

if do_normal_logging:
    #visualize_model_performance(results)  # This isn't defined anymore... what happened...

    log_file = log_performance(results, base_filename=MODEL_STR, config=MY_CONFIG)
    print(f"Detailed performance log saved to: {log_file}")
