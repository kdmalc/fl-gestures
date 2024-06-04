import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def mean_subtraction_blockwise(block, end_IMU_range=72):
    raise ValueError("Use the other mean subtraction code, or come fix/validate this one")
    # DONT USE THIS ONE
    # Doesnt work because block size was (64,) for some reason... 
    # Probably need to remove metadata or call that out before calling this func...
    
    imu_block = block[:, :end_IMU_range]  # Extract IMU block
    emg_block = block[:, end_IMU_range:]  # Extract EMG block
    
    # Perform mean subtraction separately for IMU and EMG blocks
    imu_block_mean_subtracted = imu_block - imu_block.mean(axis=0)
    emg_block_mean_subtracted = emg_block - emg_block.mean(axis=0)
    
    # Concatenate the mean subtracted IMU and EMG blocks
    mean_subtracted_block = np.concatenate((imu_block_mean_subtracted, emg_block_mean_subtracted), axis=1)
    
    return mean_subtracted_block


def preprocess_df_by_gesture(data_df, preprocessing_approach, trial_length=64, metacol_start_idx=3):
    '''Assumes the first 3 columns are the metadata columns (Participant, Gesture_ID, Gesture_Num), which are removed for processing and then added back to the df when returning the processed df'''
    num_trials = data_df.shape[0] // trial_length
    preprocessed_data_lst = []
    metadata_cols = data_df.iloc[:, :metacol_start_idx]
    signal_cols = data_df.iloc[:, metacol_start_idx:]
    for trial_idx in range(num_trials):
        start_idx = trial_idx * trial_length
        end_idx = start_idx + trial_length
        trial_data = signal_cols.iloc[start_idx:end_idx]

        if preprocessing_approach.upper() == 'MEANSUBTRACTION':
            trial_data_preprocessed = trial_data - trial_data.mean()
        elif preprocessing_approach.upper() == 'MINMAXSCALER':
            scaler = MinMaxScaler()
            trial_data_preprocessed = scaler.fit_transform(trial_data)
        elif preprocessing_approach.upper() == 'STANDARDSCALER':
            scaler = StandardScaler()
            trial_data_preprocessed = scaler.fit_transform(trial_data)
        
        trial_data_preprocessed_df = pd.DataFrame(trial_data_preprocessed, columns=signal_cols.columns)
        preprocessed_data_lst.append(trial_data_preprocessed_df)
    preprocessed_df = pd.concat(preprocessed_data_lst, ignore_index=True)
    preprocessed_df = pd.concat([metadata_cols, preprocessed_df], axis=1)
    return preprocessed_df


def meansubtract_df_by_gesture(data_df, trial_length=64, metacol_start_idx=3):
    num_trials = data_df.shape[0] // trial_length
    mean_subtracted_data = []
    metadata_cols = data_df.iloc[:, :metacol_start_idx]
    signal_cols = data_df.iloc[:, metacol_start_idx:]
    for trial_idx in range(num_trials):
        start_idx = trial_idx * trial_length
        end_idx = start_idx + trial_length
        trial_data = signal_cols.iloc[start_idx:end_idx]
        trial_data_mean_subtracted = trial_data - trial_data.mean()
        trial_data_mean_subtracted_df = pd.DataFrame(trial_data_mean_subtracted, columns=signal_cols.columns)
        mean_subtracted_data.append(trial_data_mean_subtracted_df)
    mean_subtracted_df = pd.concat(mean_subtracted_data, ignore_index=True)
    mean_subtracted_df = pd.concat([metadata_cols, mean_subtracted_df], axis=1)
    return mean_subtracted_df


def interpolate_df(df, num_rows=64, columns_to_exclude=None):
    """
    Interpolates the dataframe to have a specified number of rows.
    Excludes specified columns from interpolation.
    """
    if columns_to_exclude is None:
        columns_to_exclude = []
    
    # Separate the columns to exclude
    excluded_columns_df = df[columns_to_exclude]
    subset_df = df.drop(columns=columns_to_exclude)
    
    # Interpolate the remaining columns
    x_old = np.linspace(0, 1, num=len(subset_df))
    x_new = np.linspace(0, 1, num=num_rows)
    interpolated_df = pd.DataFrame()

    for column in subset_df.columns:
        y_old = subset_df[column].values
        y_new = np.interp(x_new, x_old, y_old)
        interpolated_df[column] = y_new

    # Add the excluded columns back
    for col in columns_to_exclude:
        interpolated_df[col] = excluded_columns_df[col].iloc[0]

    # Reordering columns to match original DataFrame
    interpolated_df = interpolated_df[columns_to_exclude + list(subset_df.columns)]
    
    return interpolated_df


def interpolate_dataframe(df, num_rows=64):
    '''Old version from Ben's code (/Momona?), assumes no meta data'''
    # Create a new index array with num_rows evenly spaced values
    new_index = np.linspace(0, len(df) - 1, num_rows)
    interpolated_df = pd.DataFrame(columns=df.columns, index=new_index)
    # Interpolate each column of the DataFrame using the new index
    for column in df.columns:
        interpolated_df[column] = np.interp(new_index, np.arange(len(df)), df[column])

    interpolated_df.index = range(num_rows)
    
    return interpolated_df