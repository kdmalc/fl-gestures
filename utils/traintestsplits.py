import pandas as pd


def manual_train_test_split(df_t, metadata_cols_df, save_file_name, save_bool=True, save_path='D:\\Kai_MetaGestureClustering_24\\saved_datasets', user_holdout=True, gesture_holdout=False, held_out_user_pids=['P103','P109','P114','P124','P128','P005','P010'], held_out_test_gestures=['move', 'zoom-out', 'duplicate']):

    save_lst = []
    save_name_lst = []

    # Combine the data and metadata dfs again:
    # Ensure both DataFrames have the same index
    df_t.reset_index(drop=True, inplace=True)
    metadata_cols_df.reset_index(drop=True, inplace=True)
    # Concatenate the DataFrames
    metadata_PCA_df = pd.concat([metadata_cols_df, df_t], axis=1)
    save_lst.append(metadata_PCA_df)
    save_name_lst.append("full_dimreduc_df.pkl")

    if user_holdout:
        test_users_df = metadata_PCA_df[metadata_PCA_df['Participant'].isin(held_out_user_pids)]
        # Merge the DataFrames with an indicator
        merged_df = metadata_PCA_df.merge(test_users_df, how='left', indicator=True)
        # Filter out the rows that are in both DataFrames
        training_users_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns='_merge')
        save_lst.extend([test_users_df, training_users_df])
        #save_name_lst.extend([save_file_name+'_test_users_df.pkl', save_file_name+'_training_users_df.pkl'])
        save_name_lst.extend(['test_users_df.pkl', 'training_users_df.pkl'])
    if gesture_holdout:
        test_gestures_df = metadata_PCA_df[(metadata_PCA_df['Participant'].isin(held_out_user_pids)) & (data_df['Gesture_ID'].isin(held_out_gestures))]
        # Merge the DataFrames with an indicator
        merged_df = metadata_PCA_df.merge(test_gestures_df, how='left', indicator=True)
        # Filter out the rows that are in both DataFrames
        training_gestures_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns='_merge')
        save_lst.extend([test_gestures_df, training_gestures_df])
        #save_name_lst.extend([save_file_name+'_test_gestures_df.pkl', save_file_name+'_training_gestures_df.pkl'])
        save_name_lst.extend(['test_gestures_df.pkl', 'training_gestures_df.pkl'])
        
    if save_bool:
        for idx, ele in enumerate(save_lst):
            ele.to_pickle(save_path + '\\' + save_file_name + '\\' + save_name_lst[idx])

    return save_lst, save_name_lst