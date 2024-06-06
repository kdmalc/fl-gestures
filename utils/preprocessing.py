import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def preprocess_df_by_gesture(data_df, preprocessing_approach, biosignal_switch_ix_lst=[72], trial_length=64):
    '''Assumes there is no metadata cols (eg only pass in the 2D dataframe of sensor data'''
    if len(biosignal_switch_ix_lst)>1:
        raise ValueError("Only one biosignal index switch is supported for now")
    
    num_trials = data_df.shape[0] // trial_length
    preprocessed_data_lst = []
    
    for trial_idx in range(num_trials):
        start_idx = trial_idx * trial_length
        end_idx = start_idx + trial_length
        trial_data = data_df.iloc[start_idx:end_idx]

        if trial_data.isna().any().any():
            print(f"Warning: NaNs detected in trial {trial_idx + 1} before preprocessing")

        if preprocessing_approach.upper() == 'MEANSUBTRACTION':
            # This already is applied on a per-column basis
            trial_data_preprocessed = trial_data - trial_data.mean()
        elif preprocessing_approach.upper() == 'MINMAXSCALER':
            scaler = MinMaxScaler()
            if len(biosignal_switch_ix_lst)==0:
                trial_data_preprocessed = scaler.fit_transform(trial_data)
            else:
                for biosignal_idx in biosignal_switch_ix_lst:
                    biosignal_block1 = trial_data.iloc[:, :biosignal_idx]
                    biosignal_block2 = trial_data.iloc[:, biosignal_idx:]
                    biosignal_block1_preprocessed = pd.DataFrame(scaler.fit_transform(biosignal_block1))
                    biosignal_block2_preprocessed = pd.DataFrame(scaler.fit_transform(biosignal_block2))
                    trial_data_preprocessed = pd.concat([biosignal_block1_preprocessed, biosignal_block2_preprocessed], axis=1)
        elif preprocessing_approach.upper() == 'STANDARDSCALER':
            scaler = StandardScaler()
            if len(biosignal_switch_ix_lst)==0:
                trial_data_preprocessed = scaler.fit_transform(trial_data)
            else:
                for biosignal_idx in biosignal_switch_ix_lst:
                    biosignal_block1 = trial_data.iloc[:, :biosignal_idx]
                    biosignal_block2 = trial_data.iloc[:, biosignal_idx:]
                    biosignal_block1_preprocessed = pd.DataFrame(scaler.fit_transform(biosignal_block1))
                    biosignal_block2_preprocessed = pd.DataFrame(scaler.fit_transform(biosignal_block2))
                    trial_data_preprocessed = pd.concat([biosignal_block1_preprocessed, biosignal_block2_preprocessed], axis=1)
            trial_data_preprocessed = scaler.fit_transform(trial_data)
        elif preprocessing_approach.upper() == '$B':
            # This already is applied on a per-column basis
            trial_data_centered = trial_data - trial_data.mean()
            if len(biosignal_switch_ix_lst)==0:
                trial_data_preprocessed = scaler.fit_transform(trial_data)
            else:
                for biosignal_idx in biosignal_switch_ix_lst:
                    biosignal_block1 = trial_data.iloc[:, :biosignal_idx]
                    biosignal_block2 = trial_data.iloc[:, biosignal_idx:]

                    gesture_std1 = np.std(biosignal_block1.values.flatten())
                    if gesture_std1 != 0:
                        biosignal_block1_preprocessed = pd.DataFrame(biosignal_block1 / gesture_std1)
                    else:
                        print("Gesture STD is equal to 0!")
                        biosignal_block1_preprocessed = pd.DataFrame(biosignal_block1)
        
                    gesture_std2 = np.std(biosignal_block2.values.flatten())
                    if gesture_std2 != 0:
                        biosignal_block2_preprocessed = pd.DataFrame(biosignal_block2 / gesture_std2)
                    else:
                        print("Gesture STD is equal to 0!")
                        biosignal_block2_preprocessed = pd.DataFrame(biosignal_block2)

                    trial_data_preprocessed = pd.concat([biosignal_block1_preprocessed, biosignal_block2_preprocessed], axis=1)

        trial_data_preprocessed_df = pd.DataFrame(trial_data_preprocessed, columns=data_df.columns)
        if trial_data_preprocessed_df.isna().any().any():
            print(f"Warning: NaNs detected in trial {trial_idx + 1} after preprocessing")
        preprocessed_data_lst.append(trial_data_preprocessed_df)
    
    preprocessed_df = pd.concat(preprocessed_data_lst, ignore_index=True)
    return preprocessed_df


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