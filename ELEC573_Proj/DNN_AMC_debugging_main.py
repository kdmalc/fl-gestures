import pandas as pd
#import numpy as np
#np.random.seed(42) 
import random
from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import KFold
#from sklearn.cross_decomposition import CCA
#from sklearn.decomposition import PCA
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
#import matplotlib.pyplot as plt
#import seaborn as sns

from moments_engr import *
from agglo_model_clust import *
from DNN_FT_funcs import *
from DNN_AMC_funcs import *


MODEL_STR = "DynamicMomonaNet"
expdef_df = load_expdef_gestures(apply_hc_feateng=False)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

DynamicMomonaNet_config = {
    "num_channels": [NUM_CHANNELS],  #(int): Number of input channels.
    "sequence_length": [64],  #(int): Length of the input sequence.
    # ^ 32 wasn't working, it doesn't support time_steps to do multiple sequence batches for each slice
    "time_steps": [None],  # Required so that reused code doesn't break. Not used in DynamicMomonaNet
    "conv_layers": [[(32, 5, 1), (64, 3, 1), (128, 2, 1)]], 
    "pooling_layers": [[True, False, False, False]],  # Max pooling only after the first conv layer
    "use_dense_cnn_lstm": [True],  # Use dense layer between CNN and LSTM
    "lstm_hidden_size": [16],  #(int): Hidden size for LSTM layers.
    "lstm_num_layers": [1],  #(int): Number of LSTM layers.
    "fc_layers": [[128, 64]],  #(list of int): List of integers specifying the sizes of fully connected layers.
    "num_classes": [10], #(int): Number of output classes.
    # HYPERPARAMS
    "batch_size": [16],  #SHARED_BS #(int): Batch size.
    "lstm_dropout": [0.8],  #(float): Dropout probability for LSTM layers.
    "learning_rate": [0.001],
    "num_epochs": [500],
    "optimizer": ["adam"],
    "weight_decay": [1e-4],
    "dropout_rate": [0.3],
    "ft_learning_rate": [0.01],
    "num_ft_epochs": [500],
    "ft_weight_decay": [1e-4], 
    "ft_batch_size": [1], 
    "use_earlystopping": [True],  # Always use this to save time, in ft and earlier training
    # METADATA
    "results_save_dir": [f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\results\\hyperparam_tuning\\{timestamp}"],
    "models_save_dir": [f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\models\\hyperparam_tuning\\{timestamp}"], 
    "perf_log_dir": [f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\results\\performance_logs"], 
    "timestamp": [timestamp],
    "verbose": [True],
    "log_each_pid_results": [False], 
    "save_ft_models": [False]  # Not even applicable here
}

data_splits = make_data_split(expdef_df, num_gesture_training_trials=8, num_gesture_ft_trials=3)

# If I'm not using Pytorch, then I don't need dataloaders, need to revamp...
features_df = pd.DataFrame(data_splits['train']['feature'])
# Create a new column 'features' that contains all 80 columns as lists
features_df['feature'] = features_df.apply(lambda row: row.tolist(), axis=1)
# Keep only the new combined column
features_df = features_df[['feature']]
# Combine with labels and participant_ids into a single DataFrame
train_df = pd.concat([features_df, pd.Series(data_splits['train']['labels'], name='Gesture_Encoded'), pd.Series(data_splits['train']['participant_ids'], name='participant_ids')], axis=1)
label_encoder = LabelEncoder()
train_df['Cluster_ID'] = label_encoder.fit_transform(train_df['participant_ids'])

features_df = pd.DataFrame(data_splits['intra_subject_test']['feature'])
# Create a new column 'features' that contains all 80 columns as lists
features_df['feature'] = features_df.apply(lambda row: row.tolist(), axis=1)
# Keep only the new combined column
features_df = features_df[['feature']]
# Combine with labels and participant_ids into a single DataFrame
intra_test_df = pd.concat([features_df, pd.Series(data_splits['intra_subject_test']['labels'], name='Gesture_Encoded'), pd.Series(data_splits['intra_subject_test']['participant_ids'], name='participant_ids')], axis=1)
label_encoder = LabelEncoder()
intra_test_df['Cluster_ID'] = label_encoder.fit_transform(intra_test_df['participant_ids'])

features_df = pd.DataFrame(data_splits['cross_subject_test']['feature'])
# Create a new column 'features' that contains all 80 columns as lists
features_df['feature'] = features_df.apply(lambda row: row.tolist(), axis=1)
# Keep only the new combined column
features_df = features_df[['feature']]
# Combine with labels and participant_ids into a single DataFrame
cross_test_df = pd.concat([features_df, pd.Series(data_splits['cross_subject_test']['labels'], name='Gesture_Encoded'), pd.Series(data_splits['cross_subject_test']['participant_ids'], name='participant_ids')], axis=1)
label_encoder = LabelEncoder()
cross_test_df['Cluster_ID'] = label_encoder.fit_transform(cross_test_df['participant_ids'])

# Only clustering wrt intra_test results, not cross_test results, for now...
data_dfs_dict = {'train':train_df, 'test':intra_test_df}
merge_log, intra_cluster_performance, cross_cluster_performance = DNN_agglo_merge_procedure(data_dfs_dict, MODEL_STR, n_splits=2)

