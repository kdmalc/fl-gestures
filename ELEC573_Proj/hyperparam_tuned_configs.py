from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
NUM_CHANNELS = 16  # NOT USED WITH ELEC573Net rn


DynamicMomonaNet_config = {
    "feature_engr": None, 
    "weight_decay": 0.0,
    "verbose": False,
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


old_DynamicMomonaNet_config = {
    "num_channels": NUM_CHANNELS,  #(int): Number of input channels.
    "sequence_length": 64,  #(int): Length of the input sequence.
    # ^ 32 wasn't working, it doesn't support time_steps to do multiple sequence batches for each slice
    "time_steps": None,  # Required so that reused code doesn't break. Not used in DynamicMomonaNet
    "conv_layers": [(32, 5, 1), (64, 3, 1), (128, 2, 1)], 
    "pooling_layers": [True, False, False, False],  # Max pooling only after the first conv layer
    "use_dense_cnn_lstm": True,  # Use dense layer between CNN and LSTM
    "lstm_hidden_size": 16,  #(int): Hidden size for LSTM layers.
    "lstm_num_layers": 1,  #(int): Number of LSTM layers.
    "fc_layers": [128, 64],  #(list of int): List of integers specifying the sizes of fully connected layers.
    "num_classes": 10, #(int): Number of output classes.
    # HYPERPARAMS
    "batch_size": 16,  #SHARED_BS #(int): Batch size.
    "lstm_dropout": 0.8,  #(float): Dropout probability for LSTM layers.
    "cnn_dropout": 0.0,
    "dense_cnnlstm_dropout": 0.0, 
    "fc_dropout": 0.0, 
    "learning_rate": 0.001,
    "num_epochs": 500,
    "optimizer": "adam",
    "weight_decay": 1e-4,
    "ft_learning_rate": 0.01,
    "num_ft_epochs": 500,
    "ft_weight_decay": 1e-4, 
    "ft_batch_size": 1, 
    "use_earlystopping": False,  # Always use this to save time, in ft and earlier training
    # METADATA
    "results_save_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\results\\{timestamp}",  # \\hyperparam_tuning
    "models_save_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\models\\{timestamp}",  # \\hyperparam_tuning
    "timestamp": timestamp,
    "verbose": False,
    "log_each_pid_results": False, 
    "save_ft_models": False  # KEEP FALSE WHEN HYPERPARAM TUNING
}

# BEST PARAMS FOR GENERIC MODEL
ELEC573Net_config = {
    "feature_engr": "moments", 
    "learning_rate": 0.0001,
    "batch_size": 32,
    "optimizer": "adam",
    "weight_decay": 1e-4,
    "num_channels": 80,  #(int): Number of input channels.
    "sequence_length": 64,  # Not used for this one AFAIK
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