# THIS FILE CONTAINS ALL FEATURE ENGINEERING

import numpy as np
import pandas as pd
from itertools import combinations


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
        # Step 3: Take the logarithm of the total 
        ## Note if total power is <= 0 then this will return NANs...
        log_total_power = np.log(total_power)
        # Store the result in the dictionary
        zero_order_moments_log.append(log_total_power)
        zero_order_moments_raw.append(total_power)
    # HAVE TO SAVE THE RAW BC IT IS USED DIRECTLY IN OTHERS FEATURES (could exponentiate ig...)
    return zero_order_moments_log, zero_order_moments_raw


def second_order(df_freq, zero_order_raw):
    # Initialize lists to store the results for each sensor
    second_order_moments_log = []
    second_order_moments_raw = []
    for i,sensor in enumerate(df_freq.columns):
        # Extract the frequency domain data for the sensor (column of the dataframe)
        time_data = df_freq[sensor].values
        # Step 1: Take first derivative
        first_deriv = np.gradient(time_data)
        # Step 2: Square the signal (power) at each frequency of the first deriv
        signal_squared = np.abs(first_deriv) ** 2
        # Step 3: Integrate (sum) the squared values to get the total power
        ## Current approach assumes that fs is uniform. Could use:
        ## total_power = np.trapz(signal_squared, dx=freq_step)  # if dx is known
        total_power = np.sum(signal_squared)
        # Step 4: Take the logarithm of the total power
        log_total_power = np.log(total_power / (zero_order_raw[i]**2))
        # Store the results in the lists
        second_order_moments_log.append(log_total_power)  # log(Normalized m2)
        second_order_moments_raw.append(total_power)  # m2
    # Convert lists to numpy arrays for consistency
    return np.array(second_order_moments_log), np.array(second_order_moments_raw)


def fourth_order(df_freq, zero_order_raw):
    fourth_order_log = []
    fourth_order_raw = []
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
        fourth_order_log.append(log_total_power)  # log(Normalized m4)
        fourth_order_raw.append(total_power)  # m4
    return np.array(fourth_order_log), np.array(fourth_order_raw)


def single_sparsity(fourth_order_raw, second_order_raw, zero_order_raw):
    # Sanity Check: all zero_order should be bigger than its corresponding second and fourth
    ## This allegedly checks pairwise, not setwise
    assert np.all(zero_order_raw >= second_order_raw), "Error: zero_order_raw must be >= second_order_raw"
    assert np.all(zero_order_raw >= fourth_order_raw), "Error: zero_order_raw must be >= fourth_order_raw"
    # Step 1: Compute the square roots (ensure no negative values before sqrt)
    sqrt_second_diff = np.sqrt(zero_order_raw - second_order_raw) 
    sqrt_fourth_diff = np.sqrt(zero_order_raw - fourth_order_raw) 
    # Step 2: Perform the dot product
    ## Is this a dot product or just multiplication...
    ## Should this be computed individually for each channel or all at once...
    dot_product = np.dot(sqrt_second_diff, sqrt_fourth_diff)
    dot_product = max(dot_product, 1e-10)
    # Step 3: Compute the sparseness ratio
    sparseness = zero_order_raw / dot_product
    # Step 4: Logarithm of the sparseness (ensure no negative or zero values inside the log)
    # We use np.maximum to ensure all values are at least 1e-10 to avoid taking log of non-positive values.
    ## I shouldn't need this...
    #safe_sparseness = np.maximum(sparseness, 1e-10)
    #print("Safe sparseness:", safe_sparseness)
    # Step 5: Apply the logarithm for third-order moments
    sparsity_log = np.log(sparseness)
    return np.array(sparsity_log)


def sparsity(zero_order_raw, second_order_raw, fourth_order_raw):
    # Sanity Check: Ensure zero_order is greater than or equal to second and fourth for each sensor
    assert np.all(zero_order_raw >= second_order_raw), "Error: zero_order_raw must be >= second_order_raw"
    assert np.all(zero_order_raw >= fourth_order_raw), "Error: zero_order_raw must be >= fourth_order_raw"
    # Step 1: Compute the square roots (element-wise)
    sqrt_second_diff = np.sqrt(zero_order_raw - second_order_raw) 
    sqrt_fourth_diff = np.sqrt(zero_order_raw - fourth_order_raw) 
    # Step 2: Element-wise multiplication (not dot product)
    elementwise_product = sqrt_second_diff * sqrt_fourth_diff  # No summing over all channels!
    # Step 3: Avoid division by zero
    elementwise_product = np.maximum(elementwise_product, 1e-10)  # Ensure denominator is nonzero
    # Step 4: Compute the sparseness ratio per sensor
    sparseness = zero_order_raw / elementwise_product
    # Step 5: Compute logarithm of sparseness
    sparsity_log = np.log(sparseness)
    # Return per-sensor sparsity as a numpy array
    return np.array(sparsity_log)  # This will have the same length as the number of channels


def old_irregularity_factor(df_freq, zero_order_raw, second_order_raw, fourth_order_raw):
    # Initialize a list to store the fourth-order moments log for each sensor
    irregularity_factor_log = []
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
        moment_log = np.log(np.sqrt((second_order_raw[i]**2) / 
                           (zero_order_raw[i] * fourth_order_raw[i])) / total_power)
        # Store the result in the list
        irregularity_factor_log.append(moment_log)
    # Convert list to numpy array for consistency
    return np.array(irregularity_factor_log)


def irregularity_factor(df_freq, zero_order_raw, second_order_raw, fourth_order_raw):
    # Initialize a list to store the irregularity factor log for each sensor
    irregularity_factor_log = []
    # Small constant for numerical safety
    epsilon = 1e-10
    for i, sensor in enumerate(df_freq.columns):
        # Extract the time domain data for the sensor (column of the dataframe)
        time_data = df_freq[sensor].values
        # Step 1: Compute first derivative
        first_deriv = np.gradient(time_data)
        # Step 2: Compute the absolute value of the first derivative
        signal_abs = np.abs(first_deriv)
        # Step 3: Integrate (sum) the absolute values to get WL
        WL = np.sum(signal_abs)  # WL is the integrated absolute derivative
        # Step 4: Compute IF = sqrt(m2^2 / (m0 * m4))
        m0 = zero_order_raw[i]
        m2 = second_order_raw[i]
        m4 = fourth_order_raw[i]
        # Ensure numerical safety in division
        denominator = np.maximum(m0 * m4, epsilon)  # Avoid divide-by-zero
        IF = np.sqrt((m2 ** 2) / denominator)
        # Step 5: Compute log(IF / WL), ensuring WL is nonzero
        WL = np.maximum(WL, epsilon)  # Avoid log(0)
        moment_log = np.log(IF / WL)
        # Store the result
        irregularity_factor_log.append(moment_log)
    # Convert list to numpy array for consistency
    return np.array(irregularity_factor_log)


def create_khushaba_spectralmomentsFE_vectors(group):
    result_vector = []
    #can only run features on EMG columns
    ## This is a bit hardcoded in...
    emg_columns = [col for col in group.columns if col.startswith('EMG')]
    for emg_col in emg_columns:
        data = group[[emg_col]] #run on EMG columns. convert to df with single column to run with all the features
        # zero-order
        zero_order_log, zero_order_raw = zero_order(data)
        result_vector.append(zero_order_log)
        # second-order
        second_order_log, second_order_raw = second_order(data, zero_order_raw)
        result_vector.append(second_order_log)
        # fourth-order
        fourth_order_log, fourth_order_raw = second_order(data, zero_order_raw)
        result_vector.append(fourth_order_log)
        # sparsity
        sparsity_log = sparsity(zero_order_raw, second_order_raw, fourth_order_raw)
        result_vector.append(sparsity_log)
        # irregularity factor
        irregularity_factor_log = irregularity_factor(data, zero_order_raw, second_order_raw, fourth_order_raw)
        result_vector.append(irregularity_factor_log)
    return pd.DataFrame({
        'Participant': [group['Participant'].iloc[0]],
        'Gesture_ID': [group['Gesture_ID'].iloc[0]],
        'Gesture_Num': [group['Gesture_Num'].iloc[0]],
        'feature': [np.array(result_vector)]
    })
    

# EXAMPLE
#userdef = df1.groupby(['Participant', 'Gesture_ID', 'Gesture_Num']).apply(create_khushaba_spectralmomentsFE_vectors)
#userdef = userdef.reset_index(drop=True)

###########################################################################

# python code to extract EMG features

"""
Written by Momona Yamagami, PhD 
Last updated: 2/15/2024

EMG features originally defined in: 
Abbaspour S, Lindén M, Gholamhosseini H, Naber A, Ortiz-Catalan M. Evaluation of surface EMG-based recognition algorithms for decoding hand movements. 
    Medical & biological engineering & computing. 2020 Jan;58:83-100.
    MAV, STD, DAMV, IAV, Var, WL, Cor, HMob, and HCom
IMU: only mean value extraction (MV)
    https://link.springer.com/article/10.1186/s12984-017-0284-4
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8932579

"""


"""
0) mean value (MV)
    MV = 1/N (\sum^N_{i=1}x_i)
    N: length of signal
    x_n: EMG signal in a segment
"""
def MV(s,fs=None):
    N = len(s)
    return 1/N*sum(s)


"""
1) mean absolute value (MAV)
    MAV = 1/N (\sum^N_{i=1}|x_i|)
    N: length of signal
    x_n: EMG signal in a segment
"""
def MAV(s,fs=None):
    N = len(s)
    return 1/N*sum(abs(s))


"""
2) standard deviation (STD)
    STD = \sqrt{1/(N-1) \sum^N_{i=1}(x_i-xbar)^2}
"""
def STD(s,fs=None):
    N = len(s)
    sbar = np.mean(s)
    return np.sqrt(1/(N-1)*sum((s-sbar)**2))


"""
3) variance of EMG (Var)
    Var = 1/(N-1)\sum^N_{i=1} x_i^2
"""
def Var(s,fs=None):
    N = len(s)
    return 1/(N-1)*sum(s**2)


"""
4) waveform length
    WL = sum (|x_i-x_{i-1})
"""
def WL(s,fs=None):
    return (sum(abs(s[1:]-s[:-1]))) / 1.0 # make sure convert to float64


"""
10) correlation coefficient (Cor)
    Cor(x,y)
    x, y: each pair of EMG channels in a time window
"""
def Cor(x,y,fs=None):
    xbar = np.mean(x)
    ybar = np.mean(y)
    num = abs(sum((x-xbar)*(y-ybar)))
    den = np.sqrt(sum((x-xbar)**2)*sum((y-ybar)**2))
    return num/den


""" 
11) Difference absolute mean value (DAMV)
    DAMV = 1/N sum_{i=1}^{N-1} |x_{i+1}-x_i|
"""
def DAMV(s,fs=None):
    N = len(s)
    return 1/N * sum(abs(s[1:]-s[:-1]))


"""
16) integrated absolute value (IAV)
    IAV = \sum^N_{i=1} |x_i|
"""
def IAV(s,fs=None):
    return (sum(abs(s))) / 1.0 # make sure convert to float64


"""
17) Hjorth mobility parameter (HMob)
    derivative correct?
"""
def HMob(s,fs=None):
    dt = 1/fs # 1/2000
    ds = np.gradient(s,dt) # compute derivative 
    return np.sqrt(Var(ds)/Var(s))


"""
18) Hjorth complexity parameter
    compares the similarity of the shape of a signal with
    pure sine wave 
    HCom = mobility(dx(t)/dt) / mobility(x(t))
"""
def HCom(s,fs=None):
    dt = 1/fs
    ds = np.gradient(s,dt)
    return HMob(ds,fs) / HMob(s,fs)


def create_abbaspour_FS_vectors(group):
    fs_vector = []
    
    # Extract EMG columns
    emg_columns = [col for col in group.columns if col.startswith('EMG')]
    # Compute features for individual channels
    emg_data = {col: np.array(group[col]) for col in emg_columns}

    # TODO: Rectify EMG data (so it is all positive and mean is not zero)?
    
    for emg_col, data in emg_data.items():
        fs_vector.append(Var(data))  # variance
        fs_vector.append(WL(data))  # wave length
        fs_vector.append(HMob(data, fs=2048))  # Hjorth 1
        fs_vector.append(HCom(data, fs=2048))  # Hjorth 2

    # Compute correlation coefficient between each pair of EMG channels
    for (col1, col2) in combinations(emg_columns, 2):
        fs_vector.append(Cor(emg_data[col1], emg_data[col2]))
        
    return pd.DataFrame({
        'Participant': [group['Participant'].iloc[0]],
        'Gesture_ID': [group['Gesture_ID'].iloc[0]],
        'Gesture_Num': [group['Gesture_Num'].iloc[0]],
        'feature': [np.array(fs_vector)]
    })
    