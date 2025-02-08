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


MODEL_STR = "DynamicMomonaNet" #"ELEC573Net"  #
MY_CONFIG = DynamicMomonaNet_config #ELEC573Net_config  #DynamicMomonaNet_config
expdef_df = load_expdef_gestures(apply_hc_feateng=False)

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
merge_log, intra_cluster_performance, cross_cluster_performance, nested_clus_model_dict = DNN_agglo_merge_procedure(data_dfs_dict, MODEL_STR, MY_CONFIG, n_splits=2)

# Save the data to a file
# This saving code hasn't been validated here yet
## TODO: Ensure this is saving to the correct place
print(f'{MY_CONFIG["results_save_dir"]}')
# Create log directory if it doesn't exist
os.makedirs(MY_CONFIG["results_save_dir"], exist_ok=True)
# Include timestamp in file name?
with open(f'{MY_CONFIG["results_save_dir"]}\\{MY_CONFIG["timestamp"]}_{MODEL_STR}_agglo_merge_res.pkl', 'wb') as f:
    pickle.dump(merge_log, f)
    pickle.dump(intra_cluster_performance, f)
    pickle.dump(cross_cluster_performance, f)
    pickle.dump(nested_clus_model_dict, f)
print("Data has been saved successfully!")