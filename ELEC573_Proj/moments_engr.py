import numpy as np
import pandas as pd



def zero_order(df_freq):
    zero_order_moments_log = []
    zero_order_moments_raw = []

    for sensor in df_freq.columns:
        # Extract the frequency domain data for the sensor (column of the dataframe)
        time_data = df_freq[sensor].values
        
        # Step 1: Square the signal (power) at each frequency
        signal_squared = np.abs(time_data) ** 2
        
        # Step 2: Integrate (sum) the squared values to get the total power
        total_power = np.sum(signal_squared)
        
        # Step 3: Take the logarithm of the total power
        log_total_power = np.log(total_power)
        
        # Store the result in the dictionary
        zero_order_moments_log.append(log_total_power)
        zero_order_moments_raw.append(total_power)
        
    return zero_order_moments_log, zero_order_moments_raw


def first_order(df_freq, zero_order_raw):
    # Initialize lists to store the results for each sensor
    first_order_moments_log = []
    first_order_moments_raw = []
    
    for i,sensor in enumerate(df_freq.columns):
        # Extract the frequency domain data for the sensor (column of the dataframe)
        time_data = df_freq[sensor].values
        
        # Step 1: Take first derivative
        first_deriv = np.gradient(time_data)
        
        # Step 2: Square the signal (power) at each frequency of the first deriv
        signal_squared = np.abs(first_deriv) ** 2
        
        # Step 3: Integrate (sum) the squared values to get the total power
        total_power = np.sum(signal_squared)
        
        # Step 4: Take the logarithm of the total power
        log_total_power = np.log(total_power / (zero_order_raw[i]**2))
        
        # Store the results in the lists
        first_order_moments_log.append(log_total_power)
        first_order_moments_raw.append(total_power)

    # Convert lists to numpy arrays for consistency
    first_order_moments_log = np.array(first_order_moments_log)
    first_order_moments_raw = np.array(first_order_moments_raw)
    
    return first_order_moments_log, first_order_moments_raw


def second_order(df_freq, zero_order_raw):
    second_order_log = []
    second_order_raw = []

    for i, sensor in enumerate(df_freq.columns):
        # Extract the time domain data for the sensor (column of the dataframe)
        time_data = df_freq[sensor].values
        
        # Step 1: Take first derivative
        first_deriv = np.gradient(time_data)
        
        #Step 2: take second derivative
        second_deriv = np.gradient(first_deriv)
        
        # Step 3: Square the signal (power) at each frequency of the first deriv
        signal_squared = np.abs(second_deriv) ** 2
        
        # Step 4: Integrate (sum) the squared values to get the total power
        total_power = np.sum(signal_squared)
        
        # Step 4: Take the logarithm of the total power
        log_total_power = np.log(total_power/(zero_order_raw[i]**4))
        
        # Store the result in the dictionary
        second_order_log.append(log_total_power)
        second_order_raw.append(total_power)
    
    return second_order_log, second_order_raw


#working version of third_order that has precautions so no neg values inside log
def third_order(second_order_raw, first_order_raw, zero_order_raw):
    second_order_raw = np.array(second_order_raw)
    first_order_raw = np.array(first_order_raw)
    zero_order_raw = np.array(zero_order_raw)
    
    # Step 1: Compute the square roots (ensure no negative values before sqrt)
    sqrt_first_diff = np.sqrt(np.maximum(zero_order_raw - first_order_raw, 1e-10))  # Handle small or negative values
    sqrt_second_diff = np.sqrt(np.maximum(zero_order_raw - second_order_raw, 1e-10))  # Handle small or negative values
    
    # Step 2: Perform the dot product (ensure no division by zero by adding a small constant)
    dot_product = np.dot(sqrt_first_diff, sqrt_second_diff)
    dot_product = max(dot_product, 1e-10)  # Avoid division by zero or extremely small numbers
    
    # Step 3: Compute the sparseness ratio (ensure the result is positive)
    sparseness = zero_order_raw / dot_product
    
    # Step 4: Logarithm of the sparseness (ensure no negative or zero values inside the log)
    # We use np.maximum to ensure all values are at least 1e-10 to avoid taking log of non-positive values.
    safe_sparseness = np.maximum(sparseness, 1e-10)
    
    #print("Safe sparseness:", safe_sparseness)
    
    # Step 5: Apply the logarithm for third-order moments
    third_order_moments_log = np.log(safe_sparseness)
    
    return third_order_moments_log


def fourth_order(df_freq, zero_order_raw, first_order_raw, second_order_raw):
    # Initialize a list to store the fourth-order moments log for each sensor
    fourth_order_moments_logs = []
    
    for i,sensor in enumerate(df_freq.columns):
        # Extract the time domain data for the sensor (column of the dataframe)
        time_data = df_freq[sensor].values
        
        # Step 1: Take first derivative
        first_deriv = np.gradient(time_data)
        
        # Step 2: Absolute value of the signal (power) at each frequency of the first derivative
        signal_abs = np.abs(first_deriv)
        
        # Step 3: Integrate (sum) the absolute values to get the total power
        total_power = np.sum(signal_abs)
        
        # Step 4: Compute the fourth-order moments log for this sensor
        moment_log = np.log(np.sqrt((first_order_raw[i]**2) / 
                           (zero_order_raw[i] * second_order_raw[i])) / total_power)
        
        # Store the result in the list
        fourth_order_moments_logs.append(moment_log)
    
    # Convert list to numpy array for consistency
    fourth_order_moments_logs = np.array(fourth_order_moments_logs)
    
    return fourth_order_moments_logs


def create_feature_vectors(group):
    result_vector = []
    #can only run features on EMG columns
    ## This is a bit hardcoded in...
    emg_columns = [col for col in group.columns if col.startswith('EMG')]
    
    for emg_col in emg_columns:
        data = group[[emg_col]] #run on EMG columns. convert to df with single column to run with all the features
        
        #zero-order
        zero_order_log, zero_order_raw = zero_order(data)
        result_vector.append(zero_order_log)
        
        #first-order
        first_order_log, first_order_raw = first_order(data, zero_order_raw)
        result_vector.append(first_order_log)
        
        #second-order
        second_order_log, second_order_raw = second_order(data, zero_order_raw)
        result_vector.append(second_order_log)
        
        #third-order
        third_order_log = third_order(second_order_raw, first_order_raw, zero_order_raw)
        result_vector.append(third_order_log)
        
        #fourth-order
        fourth_order_log = fourth_order(data, zero_order_raw, first_order_raw, second_order_raw)
        result_vector.append(fourth_order_log)
        
    return pd.DataFrame({
        'Participant': [group['Participant'].iloc[0]],
        'Gesture_ID': [group['Gesture_ID'].iloc[0]],
        'Gesture_Num': [group['Gesture_Num'].iloc[0]],
        'feature': [np.array(result_vector)]
    })
    

# EXAMPLE
#userdef = df1.groupby(['Participant', 'Gesture_ID', 'Gesture_Num']).apply(create_feature_vectors)
#userdef = userdef.reset_index(drop=True)