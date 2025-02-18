from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
NUM_CHANNELS = 16  # NOT USED WITH ELEC573Net rn
NUM_TRAIN_GESTURES = 8 
NUM_FT_GESTURES = 1  # Oneshot


# This one has not actually been hyperparamtuned...
DynamicCNNLSTM_config = {
    "model_str": "CNNLSTM",
    "feature_engr": None, 
    "time_steps": 1,
    "sequence_length": 64,
    "num_train_gesture_trials": NUM_TRAIN_GESTURES, 
    "num_ft_gesture_trials": NUM_FT_GESTURES,
    "num_pretrain_users": 24, 
    "num_testft_users": 8, 
    "weight_decay": 0.0,
    "fc_layers": [128, 64],
    "fc_dropout": 0.3,
    "dense_cnnlstm_dropout": 0.3,
    "conv_layers": [
        [64, 3, 1],
        [128, 3, 1]],
    "cnn_dropout": 0.3,
    "batch_size": 16,
    "added_dense_ft_hidden_size": 64, 
    "verbose": False,
    "use_dense_cnn_lstm": False,  # Not used with this network!
    "timestamp": timestamp,
    "pooling_layers": [True, True, True, True],
    "optimizer": "sgd",
    "num_epochs": 100,
    "num_classes": 10,
    "num_channels": NUM_CHANNELS,
    "user_split_json_filepath": "April_25\\fixed_user_splits\\24_8_user_splits.json",  # Mystery seed. Rerun with other user splits... ought to kfcv...
    "results_save_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\April_25\\results\\{timestamp}",  # \\hyperparam_tuning
    "models_save_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\April_25\\models\\{timestamp}",  # \\hyperparam_tuning
    "lstm_num_layers": 1,
    "lstm_hidden_size": 8,
    "lstm_dropout": 0.8,
    "log_each_pid_results": False,
    "learning_rate": 0.001,
    "finetune_strategy": "progressive_unfreeze",
    "progressive_unfreezing_schedule": 5,
    "ft_weight_decay": 0.0,
    "ft_learning_rate": 0.001,
    "ft_batch_size": 10,
    "num_ft_epochs": 100,
    "save_ft_models": False,
    "use_earlystopping": True,
    #"lr_scheduler_gamma": 1.0,
    "lr_scheduler_patience": 4, 
    "lr_scheduler_factor": 0.1, 
    "earlystopping_patience": 6,
    "earlystopping_min_delta": 0.01
}
DynamicCNN_config = {
    "model_str": "CNN",
    "feature_engr": None, 
    "time_steps": 1,
    "sequence_length": 64,
    "num_train_gesture_trials": NUM_TRAIN_GESTURES, 
    "num_ft_gesture_trials": NUM_FT_GESTURES,
    "num_pretrain_users": 24, 
    "num_testft_users": 8, 
    "weight_decay": 0.0,
    "fc_layers": [128, 64],
    "fc_dropout": 0.3,
    "conv_layers": [
        [64, 3, 1],
        [128, 3, 1]],
    "cnn_dropout": 0.3,
    "batch_size": 16,
    "added_dense_ft_hidden_size": 64, 
    "verbose": False,
    "timestamp": timestamp,
    "pooling_layers": [True, True, True, True],
    "optimizer": "sgd",
    "num_epochs": 100,
    "num_classes": 10,
    "num_channels": NUM_CHANNELS,
    "user_split_json_filepath": "April_25\\fixed_user_splits\\24_8_user_splits.json",  # Mystery seed. Rerun with other user splits... ought to kfcv...
    "results_save_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\April_25\\results\\{timestamp}",  # \\hyperparam_tuning
    "models_save_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\April_25\\models\\{timestamp}",  # \\hyperparam_tuning
    "log_each_pid_results": False,
    "learning_rate": 0.001,
    "finetune_strategy": "progressive_unfreeze",
    "progressive_unfreezing_schedule": 5,
    "ft_weight_decay": 0.0,
    "ft_learning_rate": 0.001,
    "ft_batch_size": 10,
    "num_ft_epochs": 100,
    "save_ft_models": False,
    "use_earlystopping": True,
    #"lr_scheduler_gamma": 1.0,
    "lr_scheduler_patience": 4, 
    "lr_scheduler_factor": 0.1, 
    "earlystopping_patience": 6,
    "earlystopping_min_delta": 0.01
}


DynamicMomonaNet_config = {
    "model_str": "DynamicMomonaNet",
    "num_train_gesture_trials": NUM_TRAIN_GESTURES, 
    "num_ft_gesture_trials": NUM_FT_GESTURES,
    "num_pretrain_users": 24, 
    "num_testft_users": 8, 
    "feature_engr": None, 
    "weight_decay": 0.0,
    "fc_layers": [128, 64],
    "fc_dropout": 0.3,
    "dense_cnnlstm_dropout": 0.3,
    "conv_layers": [
        [64, 3, 1],
        [128, 3, 1]],
    "cnn_dropout": 0.3,
    "batch_size": 16,
    "added_dense_ft_hidden_size": 64, 
    "verbose": False,
    "use_dense_cnn_lstm": True,
    "timestamp": timestamp,
    "time_steps": None,
    "sequence_length": 64,
    "pooling_layers": [True, True, True, True],
    "optimizer": "sgd",
    "num_epochs": 100,
    "num_classes": 10,
    "num_channels": NUM_CHANNELS,
    "user_split_json_filepath": "April_25\\fixed_user_splits\\24_8_user_splits.json",  # Mystery seed. Rerun with other user splits... ought to kfcv...
    "results_save_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\April_25\\results\\{timestamp}",  # \\hyperparam_tuning
    "models_save_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\April_25\\models\\{timestamp}",  # \\hyperparam_tuning
    "lstm_num_layers": 1,
    "lstm_hidden_size": 8,
    "lstm_dropout": 0.8,
    "log_each_pid_results": False,
    "learning_rate": 0.001,
    "finetune_strategy": "progressive_unfreeze",
    "progressive_unfreezing_schedule": 5,
    "ft_weight_decay": 0.0,
    "ft_learning_rate": 0.001,
    "ft_batch_size": 10,
    "num_ft_epochs": 100,
    "save_ft_models": False,
    "use_earlystopping": True,
    #"lr_scheduler_gamma": 1.0,
    "lr_scheduler_patience": 4, 
    "lr_scheduler_factor": 0.1, 
    "earlystopping_patience": 6,
    "earlystopping_min_delta": 0.01
}


# BEST PARAMS FOR GENERIC MODEL
ELEC573Net_config = {
    "model_str": "ELEC573Net",
    "num_train_gesture_trials": NUM_TRAIN_GESTURES, 
    "num_ft_gesture_trials": NUM_FT_GESTURES,
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
    "pooling_layers": None,  # In ELEC573Net pooling is on by default, see below. Should this be set to None or...
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
    "user_split_json_filepath": "April_25\\fixed_user_splits\\24_8_user_splits.json",  # Mystery seed. Rerun with other user splits... ought to kfcv...
    "results_save_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\April_25\\results\\{timestamp}",  # \\hyperparam_tuning
    "models_save_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\April_25\\models\\{timestamp}",  # \\hyperparam_tuning
    "timestamp": timestamp,
    "verbose": False,
    "log_each_pid_results": False, 
    "save_ft_models": False,  # Not even applicable here
    "use_earlystopping": True,
    #"lr_scheduler_gamma": 1.0,
    "lr_scheduler_patience": 4, 
    "lr_scheduler_factor": 0.1, 
    "earlystopping_patience": 6,
    "earlystopping_min_delta": 0.01
}