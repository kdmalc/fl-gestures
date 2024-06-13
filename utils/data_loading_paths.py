# Kai's laptop
kai_data_path = "C:\\Users\\kdmen\\Desktop\\Research\\Data\\$M\\saved_datasets\\"
kai_model_dir_path = 'C:\\Users\\kdmen\\Desktop\\Research\\Repos\\fl-gestures\\models\\Embedding\\Autoencoders\\'
kai_metadata_path = 'C:\\Users\\kdmen\\Desktop\\Research\\Data\\$M\\saved_datasets\\metadata_cols_df.pkl'

# BRC Desktop
brc_data_path = "D:\\Kai_MetaGestureClustering_24\\saved_datasets\\"
brc_model_dir_path = 'C:\\Users\\YamagamiLab\\Desktop\\Dev\\fl-gestures\\models\\Embedding\\Autoencoders\\'
brc_metadata_path = 'D:\\Kai_MetaGestureClustering_24\\saved_datasets\\metadata_cols_df.pkl'

# Relative path to $B standardize EMG and EMG+IMU (both) [no PCA!]
emg_dir = "filtered_datasets\\EMG_PPD\\"
both_dir = "filtered_datasets\\Both_PPD\\"
both_pca20_dir = "Both_PCA20\\"
both_pca40_dir = "Both_PCA40\\"
emg_pca3_dir = "EMG_PCA3\\"
emg_pca8_dir = "EMG_PCA8\\"
# Using the OLD data!
old_emg_dir = "EMG_PCA8\\"
old_both_dir = "Both_PCA40\\"

metadata_cols = ['Participant', 'Gesture_ID', 'Gesture_Num']

# THIS IS THE CODE TO USE / MATCH IN THE INDIVIDUAL NBs!
# emg $B standardized (16D)
#emg_training_users_df = pd.read_pickle(data_path+emg_dir+'training_users_df.pkl').drop(metadata_cols, axis=1)
#emg_test_users_df = pd.read_pickle(data_path+emg_dir+'test_users_df.pkl').drop(metadata_cols, axis=1)
# emg+imu $B standardized (88D)
#both_training_users_df = pd.read_pickle(data_path+both_dir+'training_users_df.pkl').drop(metadata_cols, axis=1)
#both_test_users_df = pd.read_pickle(data_path+both_dir+'test_users_df.pkl').drop(metadata_cols, axis=1)
# emg+imu $B standardized, post PCA 40 (40D)
#both_pca_training_users_df = pd.read_pickle(data_path+both_pca40_dir+'training_users_df.pkl').drop(metadata_cols, axis=1)
#both_pca_test_users_df = pd.read_pickle(data_path+both_pca40_dir+'test_users_df.pkl').drop(metadata_cols, axis=1)