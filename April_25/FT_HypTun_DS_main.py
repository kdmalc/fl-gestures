# NOTE: Hyperparam tuning is VERY slow bc"
## 1: must train a pretrained model from scratch every time
## 2: we do multiple datasplits for cross validation to ensure generalizability

from datetime import datetime
import os
cwd = os.getcwd()
print("Current Working Directory: ", cwd)

from global_seed import set_seed
set_seed()

from DNN_FT_funcs import *
from revamped_model_classes import *


NUM_CONFIGS = 20
NUM_TRAIN_TRIALS = 8
NUM_FT_TRIALS = 1
NUM_DATASPLITS = 1
NUM_CHANNELS = 1  # ... or 16...
MODEL_STR = "DynamicMomonaNet"
raise ValueError("Remember to set feateng_method to the correct FE")
expdef_df = load_expdef_gestures(feateng_method="None", filepath_pkl="D:\\Kai_MetaGestureClustering_24\\saved_datasets\\filtered_datasets\\$BStand_EMG_df.pkl")
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# Define the search space
## Is anything about these configs specific to DynamicMomonaNet? Or can I use the same one for each model?
DynamicMomonaNet_architecture_space = {
    "num_channels": [NUM_CHANNELS],  #(int): Number of input channels.
    "sequence_length": [64],  #(int): Length of the input sequence.
    # ^ 32 wasn't working, it doesn't support time_steps to do multiple sequence batches for each slice
    "time_steps": [None],  # Required so that reused code doesn't break. Not used in DynamicMomonaNet
    "conv_layers": [ #(list of tuples): List of tuples specifying convolutional layers. Each tuple should contain (out_channels, kernel_size, stride).
        ### ONE LAYER CNN ###
        # Basic starter
        #[(16, 5, 2)],
        # Larger receptive field
        #[(32, 10, 3)],
        ### TWO LAYER CNN ###
        # Standard progression
        [(32, 5, 1), (64, 3, 1)],
        # Downsampling focused
        [(16, 7, 2), (32, 5, 2)],
        # Channel-heavy
        [(64, 3, 1), (128, 3, 1)],
        ### THREE LAYER CNN ###
        # Moderate complexity
        [(32, 5, 1), (64, 3, 1), (128, 2, 1)],
        # Progressive downsampling
        #[(16, 7, 2), (32, 5, 2), (64, 3, 1)],
        # Larger Filters for Broader Features
        [(32, 7, 1), (64, 5, 1), (128, 3, 1)]
        ], 
    # Is this gonna work with just 1-2 layer networks? I think it should be fine...
    "pooling_layers": [[True, False, False, False], 
                       #[True, True, True, True], 
                       #[False, False, False, False], 
                       [True, True, False, False]  # This is basically all true since I'm only testing 2-3 conv layers
                       ],  # Max pooling only after the first conv layer
    "use_dense_cnn_lstm": [True, False],  # Use dense layer between CNN and LSTM
    "lstm_hidden_size": [4, 8, 12],  #(int): Hidden size for LSTM layers.
    "lstm_num_layers": [1],  #(int): Number of LSTM layers.
    "fc_layers": [[32], [32, 16], [128, 64]],  #(list of int): List of integers specifying the sizes of fully connected layers.
    "num_classes": [10] #(int): Number of output classes.
    # Not added yet, might not help? If my batch size (esp in finetuning) is too small...
    #"use_CNN_batchnorm": [True, False],   # This is not added yet
    #"use_CNNLSTMdense_batchnorm": [True, False],   # This is not added yet
    #"use_outputdense_batchnorm": [True, False]   # This is not added yet
}

DynamicMomonaNet_hyperparameter_space = {
    "batch_size": [4, 8, 16, 32],  #SHARED_BS #(int): Batch size.
    "lstm_dropout": [0.5, 0.8],  #(float): Dropout probability for LSTM layers.
    "learning_rate": [0.0001, 0.001, 0.01],
    "num_epochs": [100], #[30, 50, 70],
    "optimizer": ["adam", "sgd"],
    "weight_decay": [0.0, 1e-4],
    "cnn_dropout": [0.3, 0.5],
    "dense_cnnlstm_dropout": [0.3, 0.5], 
    "fc_dropout": [0.3, 0.5], 
    "ft_learning_rate": [0.0001, 0.001, 0.01],
    "num_ft_epochs": [100], #[10, 30, 50],
    "ft_weight_decay": [0.0, 1e-4], 
    "ft_batch_size": [1, 4, 10], 
    "use_earlystopping": [True],  # Always use this to save time, in ft and earlier training
    "finetune_strategy": ["full", "freeze_cnn", "freeze_cnn_lstm", "progressive_unfreeze"],
    #["full", "freeze_cnn", "freeze_cnn_lstm", "freeze_all_add_dense", "progressive_unfreeze"],
    #finetune_strategy: The fine-tuning method to use. Options:
    #    - "full": Train the entire model.
    #    - "freeze_cnn": Freeze CNN, train LSTM and dense layers.
    #    - "freeze_cnn_lstm": Freeze CNN + LSTM, train only dense layers.
    #    - "freeze_all_add_dense": Freeze entire model, add a new dense layer.
    #    - "progressive_unfreeze": Start with frozen CNN/LSTM, progressively unfreeze.
    "progressive_unfreezing_schedule": [1, 3, 5, 7], 
    "added_dense_ft_hidden_size": [64, 128], # Not working yet
    "lr_scheduler_gamma": [0.9, 0.95, 1.0]
}

metadata_config = {
    #"results_save_dir": [f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\results\\hyperparam_tuning\\{timestamp}"],
    #"models_save_dir": [f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\models\\hyperparam_tuning\\{timestamp}"], 
    #"perf_log_dir": [f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\results\\performance_logs"], 
    # BRC: 
    "results_save_dir": [f"C:\\Users\\YamagamiLab\\Desktop\\Dev\\fl-gestures\\ELEC573_Proj\\results\\hyperparam_tuning\\{timestamp}"],  # \\hyperparam_tuning
    "models_save_dir": [f"C:\\Users\\YamagamiLab\\Desktop\\Dev\\fl-gestures\\ELEC573_Proj\\models\\hyperparam_tuning\\{timestamp}"],  # \\hyperparam_tuning
    "perf_log_dir": [f"C:\\Users\\YamagamiLab\\Desktop\\Dev\\fl-gestures\\ELEC573_Proj\\results\\hyperparam_tuning\\performance_logs"], 
    "timestamp": [timestamp],
    "verbose": [False],
    "log_each_pid_results": [False], 
    "save_ft_models": [False]  # Don't need to be saved during hyperparam tuning
}

# Run the main function
results = hyperparam_tuning_for_ft(MODEL_STR, expdef_df, DynamicMomonaNet_hyperparameter_space, DynamicMomonaNet_architecture_space, metadata_config, 
                             num_configs_to_test=NUM_CONFIGS, num_datasplits_to_test=NUM_DATASPLITS, num_train_trials=NUM_TRAIN_TRIALS, num_ft_trials=NUM_FT_TRIALS)