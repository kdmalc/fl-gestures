import pandas as pd
#import numpy as np
#np.random.seed(42) 
#import random
from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import KFold
#from sklearn.cross_decomposition import CCA
#from sklearn.decomposition import PCA

from moments_engr import *
from agglo_model_clust import *
from DNN_FT_funcs import *
from DNN_AMC_funcs import *
from hyperparam_tuned_configs import * 


MODEL_STR = "DynamicMomonaNet" #"DynamicMomonaNet"
FEATENG = None  # "moments" "FS" None
if FEATENG is not None and MODEL_STR=="DynamicMomonaNet":
    NUM_CHANNELS = 1
    SEQ_LEN = 80
    MY_CONFIG = DynamicMomonaNet_config
elif FEATENG is None and MODEL_STR=="DynamicMomonaNet":
    NUM_CHANNELS = 16
    SEQ_LEN = 64
    MY_CONFIG = DynamicMomonaNet_config
elif FEATENG=="moments" and MODEL_STR=="ELEC573Net":
    NUM_CHANNELS = 80
    SEQ_LEN = 1 
    MY_CONFIG = ELEC573Net_config
elif FEATENG=="FS" and MODEL_STR=="ELEC573Net":
    NUM_CHANNELS = 184
    SEQ_LEN = 1 
    MY_CONFIG = ELEC573Net_config
elif FEATENG is None and MODEL_STR=="ELEC573Net":
    NUM_CHANNELS = 16
    SEQ_LEN = 64  # I think this will break with ELEC573Net... not integrated AFAIK
    MY_CONFIG = ELEC573Net_config

MY_CONFIG["feature_engr"] = FEATENG
MY_CONFIG["num_channels"] = NUM_CHANNELS
MY_CONFIG["sequence_length"] = SEQ_LEN


expdef_df = load_expdef_gestures(feateng_method=MY_CONFIG["feature_engr"])
data_splits = make_data_split(expdef_df, MY_CONFIG)
# Fit LabelEncoder once on all participant IDs for consistency
all_participant_ids = data_splits['train']['participant_ids'] + data_splits['intra_subject_test']['participant_ids'] + data_splits['cross_subject_test']['participant_ids']
label_encoder = LabelEncoder()
label_encoder.fit(all_participant_ids)
# Process train and test sets
train_df = process_split(data_splits, 'train', label_encoder)
intra_test_df = process_split(data_splits, 'intra_subject_test', label_encoder)
cross_test_df = process_split(data_splits, 'cross_subject_test', label_encoder)
data_dfs_dict = {'train':train_df, 'test':intra_test_df}

# Only clustering wrt intra_test results, not cross_test results, for now...
data_dfs_dict = {'train':train_df, 'test':intra_test_df}
merge_log, intra_cluster_performance, cross_cluster_performance, nested_clus_model_dict = DNN_agglo_merge_procedure(data_dfs_dict, MODEL_STR, MY_CONFIG, n_splits=2)

# Save the data to a file
print(f'{MY_CONFIG["results_save_dir"]}')
# Create log directory if it doesn't exist
os.makedirs(MY_CONFIG["results_save_dir"], exist_ok=True)
# Include timestamp in file name? I think it is already included in results_save_dir?
with open(f'{MY_CONFIG["results_save_dir"]}\\{MY_CONFIG["timestamp"]}_{MODEL_STR}_agglo_merge_res.pkl', 'wb') as f:
    pickle.dump(merge_log, f)
    pickle.dump(intra_cluster_performance, f)
    pickle.dump(cross_cluster_performance, f)
    pickle.dump(nested_clus_model_dict, f)
print("Data has been saved successfully!")