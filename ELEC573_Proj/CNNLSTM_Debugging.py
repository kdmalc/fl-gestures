# THIS TRAINS A SINGLE GENERIC CNN AND TESTS ON WITHHELD USERS
## NO CLUSTERING!!!

import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from moments_engr import *
from DNN_FT_funcs import *
np.random.seed(42) 
import os
cwd = os.getcwd()
print("Current Working Directory: ", cwd)


MY_CONFIG = MomonaNet_config
MODEL_STR = 'MomonaNet'  # [CNN, RNN, HybridCNNLSTM, CRNN, EMGHandNet, MomonaNet]
if MODEL_STR in ["CNN", "DynamicCNN"]:  # Is CNN just DynamicCNN now? Pretty sure yes
    do_feature_engr = True
else:
    do_feature_engr = False
finetune = False
do_normal_logging = True

##################################################

def load_expdef_gestures(apply_hc_feateng=True, filepath_pkl='C:\\Users\\kdmen\\Box\\Meta_Gesture_2024\\saved_datasets\\filtered_datasets\\$BStand_EMG_df.pkl'):
    with open(filepath_pkl, 'rb') as file:
        raw_expdef_data_df = pickle.load(file)  # (204800, 19)

    if apply_hc_feateng:
        expdef_df = raw_expdef_data_df.groupby(['Participant', 'Gesture_ID', 'Gesture_Num']).apply(create_feature_vectors)
        expdef_df = expdef_df.reset_index(drop=True)
    else:
        # Group by metadata columns and combine data into a matrix
        condensed_df = (
            raw_expdef_data_df.groupby(['Participant', 'Gesture_ID', 'Gesture_Num'], as_index=False)
            .apply(lambda group: pd.Series({
                'feature': group[raw_expdef_data_df.columns[3:]].to_numpy()
            }))
            .reset_index(drop=True)
        )
        # Combine metadata columns with the new data column
        expdef_df = pd.concat([raw_expdef_data_df[['Participant', 'Gesture_ID', 'Gesture_Num']].drop_duplicates().reset_index(drop=True), condensed_df['feature']], axis=1)
    
    #convert Gesture_ID to numerical with new Gesture_Encoded column
    label_encoder = LabelEncoder()
    expdef_df['Gesture_Encoded'] = label_encoder.fit_transform(expdef_df['Gesture_ID'])
    label_encoder2 = LabelEncoder()
    expdef_df['Cluster_ID'] = label_encoder2.fit_transform(expdef_df['Participant'])

    return expdef_df

expdef_df = load_expdef_gestures(apply_hc_feateng=do_feature_engr)

all_participants = expdef_df['Participant'].unique()
# Shuffle the participants
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
    num_epochs=50, config=MY_CONFIG)

full_path = os.path.join(cwd, 'ELEC573_Proj', 'models', 'generic_CNN_model.pth')
print("Full Path:", full_path)
torch.save(results["model"].state_dict(), full_path)

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
    visualize_model_performance(results)

    log_file = log_performance(results, base_filename=MODEL_STR, config=MY_CONFIG)
    print(f"Detailed performance log saved to: {log_file}")
