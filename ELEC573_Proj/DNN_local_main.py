import pandas as pd
#import numpy as np
#import random
#np.random.seed(42) 
from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import KFold
#from sklearn.cross_decomposition import CCA
#from sklearn.decomposition import PCA

from moments_engr import *
from agglo_model_clust import *
from DNN_FT_funcs import *
from DNN_AMC_funcs import *
from hyperparam_tuned_configs import *


MODEL_STR = "DynamicMomonaNet"
MY_CONFIG = DynamicMomonaNet_config
NUM_LOCAL_MODELS = 15

expdef_df = load_expdef_gestures(apply_hc_feateng=False)
all_participants = list(expdef_df['Participant'].unique())

# Choosing NUM_LOCAL_MODELS somewhat arbitrarily, don't need to look at all 32 results... save some computation ig
# Setting num_gesture_training_trials=3 so that this replicates finetuning data used for local
# num_gesture_ft_trials should not be used I don't think, not in main_training_pipeline anyways
data_splits = make_data_split(expdef_df, num_gesture_training_trials=3, num_gesture_ft_trials=3, num_train_users=NUM_LOCAL_MODELS)

res_dict_lst = []
#for pid in train_df['participant_ids'].unique():
for p_idx, pid in enumerate(list(set(data_splits['train']['participant_ids']))):
    print(f"Training local model for {pid} ({p_idx}/{NUM_LOCAL_MODELS})")
    res_dict_lst.append(main_training_pipeline(data_splits, all_participants, data_splits['cross_subject_test']['participant_ids'], MODEL_STR, MY_CONFIG))

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