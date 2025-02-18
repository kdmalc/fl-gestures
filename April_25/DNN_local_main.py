from global_seed import set_seed
set_seed()

from moments_engr import *
from agglo_model_clust import *
from DNN_FT_funcs import *
from DNN_AMC_funcs import *
from hyperparam_tuned_configs import *
from full_study_funcs import *


MODEL_STR = "CNN"
MY_CONFIG = DynamicCNN_config
MY_CONFIG["feature_engr"] = "moments"  # None, "moments", "FS"
# Overwrite just in case...
MY_CONFIG["model_str"] = MODEL_STR
NUM_LOCAL_MODELS = 1
NUM_LOCAL_TRAIN_GESTURE_SAMPLES = 3

expdef_df = load_expdef_gestures(feateng_method=MY_CONFIG["feature_engr"])
all_participants = list(expdef_df['Participant'].unique())

# Choosing NUM_LOCAL_MODELS somewhat arbitrarily, don't need to look at all 32 results... save some computation ig
# Setting num_gesture_training_trials=3 so that this replicates finetuning data used for local
# num_gesture_ft_trials should not be used I don't think, not in main_training_pipeline anyways
data_splits = make_data_split(expdef_df, num_gesture_training_trials=NUM_LOCAL_TRAIN_GESTURE_SAMPLES, num_gesture_ft_trials=NUM_LOCAL_TRAIN_GESTURE_SAMPLES, num_train_users=NUM_LOCAL_MODELS)

# Need to make a dictionary of train/test loaders (guess I have to add cross as well for reusability)
unique_gestures = np.unique(data_splits['train']['labels'])
num_classes = len(unique_gestures)
input_dim = data_splits['train']['feature'].shape[1]

user_dict = prepare_data_for_local_models(data_splits, MODEL_STR, MY_CONFIG)

res_dict_lst = []
# Does len(list(set(data_splits['train']['participant_ids']))) == NUM_LOCAL_MODELS? NOPE!
#for p_idx, pid in enumerate(list(set(data_splits['train']['participant_ids']))):
for p_idx, pid in enumerate(list(set(data_splits['train']['participant_ids'][:NUM_LOCAL_MODELS]))):
    print(f"Training local model for {pid} ({p_idx+1}/{NUM_LOCAL_MODELS})")

    res_dict_lst.append(main_training_pipeline(data_splits=None, all_participants=all_participants, 
                                               test_participants=data_splits['cross_subject_test']['participant_ids'], 
                                               model_type=MODEL_STR, config=MY_CONFIG, 
                                               train_intra_cross_loaders=user_dict[pid]))

# res_dict_lst would either need to be saved, or extracted in a iPYNB so that we can plot train/test logs...

# Save the data to a file
## TODO: Ensure this is saving to the correct place
## TODO: This is the saving for the AMC, not for local.
# Need to save the results from local tho...
#print(f'{MY_CONFIG["results_save_dir"]}\\{MY_CONFIG["timestamp"]}')
#print()
#with open(f'{MY_CONFIG["results_save_dir"]}\\{MY_CONFIG["timestamp"]}_{MODEL_STR}_agglo_merge_res.pkl', 'wb') as f:
#    pickle.dump(merge_log, f)
#    pickle.dump(intra_cluster_performance, f)
#    pickle.dump(cross_cluster_performance, f)
#    pickle.dump(nested_clus_model_dict, f)
#print("Data has been saved successfully!")