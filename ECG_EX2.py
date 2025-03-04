def ECG_extract_qrs2(ecg_signal ,  enable_plot =True  ):
    dpi=500
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks, butter, filtfilt, resample
    import neurokit2 as nk
    import pandas as pd
    import warnings

    # Suppress DeprecationWarning
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Close all previous plots
    # plt.close('all')

    # Parameters
    fs = 2000  # Original sampling frequency
    target_length = 5  # Target length of ECG signal in seconds
    target_samples = target_length * fs  # Target number of samples

    # Load the ECG signal
    try:
        a = ecg_signal[0]
    except:
        
        data = np.load('ecg.npz')
        ecg_signal = data['signal']

    # Smoothing the ECG signal
    e = []
    M = 5
    for i in range(len(ecg_signal) - M):
        mm = np.mean(ecg_signal[i:i + M])
        if not mm == ecg_signal[i]:
            e.append(ecg_signal[i])
    ecg_signal = e

    # Check if the ECG signal is shorter than the target length
    if len(ecg_signal) < target_samples:
        # Resample the ECG signal to the target length
        ecg_signal = resample(ecg_signal, 5 * target_samples)
        # print(f"ECG signal was resampled to {target_length} seconds.")
    
    # Time vector
    t = np.arange(len(ecg_signal)) / fs

    # Detect R-peaks
    _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=fs)
    
    

    # Delineate ECG waves
    # _, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=fs, method="peak")
    
    # Delineate and visualize all peaks of ECG complexes
    _, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=fs, method="peak", show=enable_plot, show_type='peaks')
    if enable_plot:
        plt.savefig('temp.png', bbox_inches='tight', pad_inches=0.5, dpi=dpi)
        plt.close()
    
    # Filter out NaN values from peak arrays
    def filter_nan(indices):
        if isinstance(indices, (list, np.ndarray)):  # Check if input is a list or array
            indices = np.array(indices)  # Convert to NumPy array
            if indices.size == 0:  # Check if the array is empty
                return np.array([], dtype=int)
            return indices[~np.isnan(indices)].astype(int)  # Filter NaN and convert to int
        return np.array([], dtype=int)  # Return empty array for invalid input

    # Filter peaks
    r_peaks = filter_nan(rpeaks['ECG_R_Peaks'])
    t_peaks = filter_nan(waves_peak.get('ECG_T_Peaks', []))
    p_peaks = filter_nan(waves_peak.get('ECG_P_Peaks', []))
    q_peaks = filter_nan(waves_peak.get('ECG_Q_Peaks', []))
    s_peaks = filter_nan(waves_peak.get('ECG_S_Peaks', []))

    # Calculate durations (in milliseconds)
    def calculate_duration(start, end):
        if len(start) > 0 and len(end) > 0:
            return (end[0] - start[0]) * (1000 / fs)  # Convert to milliseconds
        return np.nan

    P_dur = calculate_duration(p_peaks, q_peaks)  # P duration
    QRS_dur = calculate_duration(q_peaks, s_peaks)  # QRS duration
    T_dur = calculate_duration(t_peaks, t_peaks[1:]) if len(t_peaks) > 1 else np.nan  # T duration
    ST_dur = calculate_duration(s_peaks, t_peaks)  # ST duration
    PR_int = calculate_duration(p_peaks, r_peaks)  # PR interval
    QT_int = calculate_duration(q_peaks, t_peaks)  # QT interval
    QTC_int = QT_int / np.sqrt((60 * fs) / len(r_peaks)) if not np.isnan(QT_int) else np.nan  # Corrected QT interval
    Q_dur = calculate_duration(q_peaks, r_peaks)  # Q duration
    R_dur = calculate_duration(r_peaks, s_peaks)  # R duration
    S_dur = calculate_duration(s_peaks, t_peaks)  # S duration

    # Calculate amplitudes (in microvolts)
    def calculate_amplitude(peaks):
        if len(peaks) > 0:
            return ecg_signal[peaks[0]] * 1000  # Convert to microvolts
        return np.nan

    P_amp = calculate_amplitude(p_peaks)  # P amplitude
    Q_amp = calculate_amplitude(q_peaks)  # Q amplitude
    R_amp = calculate_amplitude(r_peaks)  # R amplitude
    S_amp = calculate_amplitude(s_peaks)  # S amplitude
    T_amp = calculate_amplitude(t_peaks)  # T amplitude
    ST_amp = calculate_amplitude(s_peaks)  # ST amplitude (using S peak)

    # Create a dictionary to store the results
    results = {
        'P_dur (ms)': P_dur,
        'QRS_dur (ms)': QRS_dur,
        'T_dur (ms)': T_dur,
        'ST_dur (ms)': ST_dur,
        'PR_int (ms)': PR_int,
        'QT_int (ms)': QT_int,
        'QTC_int (ms)': QTC_int,
        'Q_dur (ms)': Q_dur,
        'R_dur (ms)': R_dur,
        'S_dur (ms)': S_dur,
        'P_amp (uV)': P_amp,
        'Q_amp (uV)': Q_amp,
        'R_amp (uV)': R_amp,
        'S_amp (uV)': S_amp,
        'T_amp (uV)': T_amp,
        'ST_amp (uV)': ST_amp
    }
    
    
    Result2 ={
        't':t, 
        'ecg_signal':ecg_signal, 
        'r_peaks':r_peaks, 
        't_peaks':t_peaks, 
        'p_peaks':p_peaks, 
        'q_peaks':q_peaks, 
        's_peaks':s_peaks, 
        'r_peaks':r_peaks, 
        'rpeaks':rpeaks,      
              
              }
    

    return results,Result2


# if True:
#     results,Result2 = ECG_extract_qrs2([] , False)