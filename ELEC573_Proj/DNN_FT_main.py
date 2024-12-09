import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

#import matplotlib.pyplot as plt
#import seaborn as sns

from moments_engr import *
from DNN_FT_funcs import *

np.random.seed(42) 

path1 = 'C:\\Users\\kdmen\\Box\\Meta_Gesture_2024\\saved_datasets\\filtered_datasets\\$BStand_EMG_df.pkl'

with open(path1, 'rb') as file:
    raw_userdef_data_df = pickle.load(file)  # (204800, 19)

userdef_df = raw_userdef_data_df.groupby(['Participant', 'Gesture_ID', 'Gesture_Num']).apply(create_feature_vectors)
#output is df with particpant, gesture_ID, gesture_num and feature (holds 80 len vector)
userdef_df = userdef_df.reset_index(drop=True)

all_participants = userdef_df['Participant'].unique()
# Shuffle the participants
np.random.shuffle(all_participants)
# Split into two groups
#train_participants = all_participants[:24]  # First 24 participants
test_participants = all_participants[24:]  # Remaining 8 participants

#convert Gesture_ID to numerical with new Gesture_Encoded column
label_encoder = LabelEncoder()
userdef_df['Gesture_Encoded'] = label_encoder.fit_transform(userdef_df['Gesture_ID'])
label_encoder2 = LabelEncoder()
userdef_df['Cluster_ID'] = label_encoder2.fit_transform(userdef_df['Participant'])

# Train base model
results = main_training_pipeline(
    userdef_df, 
    feature_column='feature', 
    target_column='Gesture_Encoded',
    all_participants=all_participants, 
    test_participants=test_participants,
    model_type='CNN',  # Or 'RNN'
    training_trials_per_gesture=8,
    num_epochs=50
)

# Fine-tune on first test participant's data
first_test_participant_data = prepare_data(
    userdef_df, 
    feature_column='feature', 
    target_column='Gesture_Encoded',
    participants=[test_participants[0]], 
    test_participants=[],
    trials_per_gesture=3
)['train']

fine_tuned_model = fine_tune_model(
    results['model'], 
    first_test_participant_data
)

# Example usage in main script
visualize_model_performance(results)

# Example usage in main script
log_file = log_performance(results)
print(f"Detailed performance log saved to: {log_file}")
