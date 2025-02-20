import pandas as pd

from global_seed import set_seed
set_seed()

from sklearn.preprocessing import LabelEncoder

from moments_engr import *
from agglo_model_clust import *
from DNN_FT_funcs import *
from DNN_AMC_funcs import *
from hyperparam_tuned_configs import * 


MODEL_STR = "CNNModel3layer" #"DynamicMomonaNet" "ELEC573Net" "OriginalELEC573CNN"
FEATENG = "moments"  # "moments" "FS" None
MY_CONFIG = determine_config(MODEL_STR, feateng=FEATENG)
# TODO: Turn this on!
#MY_CONFIG["user_split_json_filepath"] = "April_25\\fixed_user_splits\\24_8_user_splits_RS17.json"

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