'''
Created on Nov 11, 2022

@author: voodoocode
'''

import numpy as np

import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

import scipy.signal
import scipy.stats

def run_abs_mi(low_freq_data, high_freq_data, phase_window_half_size = 10, phase_step_width = 20):
    """
    Calculates the modulation index between a low frequency signal and a high frequency signal.
    
    :param low_freq_data: Single array of low frequency data.
    :param high_freq_data: Single array of high frequency data.
    :param phase_window_half_size: Width of the phase window used for calculation of frequency/phase histogram. Amplitude gets added to every phase bin within the window size. Larger windows result in more smooth/increased PAC estimates.
    :param phase_step_width: Step width/shift of the phase window used for calculation of frequency/phase histogram.
    
    :return: Amount of phase amplitude coupling measured using the modulation index.
    """
    phase_signal = np.angle(scipy.signal.hilbert(low_freq_data), deg = True)
    amplitude_signal = np.zeros(np.arange(-180, 181, phase_step_width).shape)
    
    for (phaseIdx, loc_phase) in enumerate(np.arange(-180, 181, phase_step_width)):
        phase_indices = np.argwhere(np.abs(phase_signal - loc_phase) < phase_window_half_size)

        if (len(phase_indices) == 0):
            amplitude_signal[phaseIdx] = np.nan
        else:
            amplitude_signal[phaseIdx] = np.mean(np.abs(high_freq_data[phase_indices])) # No need for a hilbert transform
        
    amplitude_signal = np.concatenate((amplitude_signal, amplitude_signal))
    amplitude_signal /= np.nansum(amplitude_signal)
    
    len_signal = len(amplitude_signal)
    uniform_signal = np.random.uniform(np.nanmin(amplitude_signal), np.nanmax(amplitude_signal), len_signal)
    
    score = (scipy.stats.entropy(amplitude_signal, uniform_signal, len_signal))/np.log(len_signal)

    return score

def run_hilbert_mi(low_freq_data, high_freq_data, phase_window_half_size = 10, phase_step_width = 20):
    """
    Calculates the modulation index between a low frequency signal and a high frequency signal.
    
    :param low_freq_data: Single array of low frequency data.
    :param high_freq_data: Single array of high frequency data.
    :param phase_window_half_size: Width of the phase window used for calculation of frequency/phase histogram. Amplitude gets added to every phase bin within the window size. Larger windows result in more smooth/increased PAC estimates.
    :param phase_step_width: Step width/shift of the phase window used for calculation of frequency/phase histogram.
    
    :return: Amount of phase amplitude coupling measured using the modulation index.
    """
    phase_signal = np.angle(scipy.signal.hilbert(low_freq_data), deg = True)
    amplitude_signal = np.zeros(np.arange(-180, 181, phase_step_width).shape)
    
    for (phaseIdx, loc_phase) in enumerate(np.arange(-180, 181, phase_step_width)):
        phase_indices = np.argwhere(np.abs(phase_signal - loc_phase) < phase_window_half_size)

        if (len(phase_indices) == 0):
            amplitude_signal[phaseIdx] = np.nan
        else:
            amplitude_signal[phaseIdx] = np.mean(np.abs(scipy.signal.hilbert(high_freq_data[phase_indices]))) # No need for a hilbert transform
        
    amplitude_signal = np.concatenate((amplitude_signal, amplitude_signal))
    amplitude_signal /= np.nansum(amplitude_signal)
    
    len_signal = len(amplitude_signal)
    uniform_signal = np.random.uniform(np.nanmin(amplitude_signal), np.nanmax(amplitude_signal), len_signal)
    
    score = (scipy.stats.entropy(amplitude_signal, uniform_signal, len_signal))/np.log(len_signal)

    return score

def run_entropy_mi(low_freq_data, high_freq_data, phase_window_half_size = 10, phase_step_width = 20):
    """
    Calculates the modulation index between a low frequency signal and a high frequency signal.
    
    :param low_freq_data: Single array of low frequency data.
    :param high_freq_data: Single array of high frequency data.
    :param phase_window_half_size: Width of the phase window used for calculation of frequency/phase histogram. Amplitude gets added to every phase bin within the window size. Larger windows result in more smooth/increased PAC estimates.
    :param phase_step_width: Step width/shift of the phase window used for calculation of frequency/phase histogram.
    
    :return: Amount of phase amplitude coupling measured using the modulation index.
    """
    phase_signal = np.angle(scipy.signal.hilbert(low_freq_data), deg = True)
    amplitude_signal = np.zeros(np.arange(-180, 181, phase_step_width).shape)
    
    for (phaseIdx, loc_phase) in enumerate(np.arange(-180, 181, phase_step_width)):
        phase_indices = np.argwhere(np.abs(phase_signal - loc_phase) < phase_window_half_size)

        if (len(phase_indices) == 0):
            amplitude_signal[phaseIdx] = np.nan
        else:
            amplitude_signal[phaseIdx] = np.mean(np.abs(scipy.signal.hilbert(high_freq_data[phase_indices]))) # No need for a hilbert transform
        
    amplitude_signal = np.concatenate((amplitude_signal, amplitude_signal))
    amplitude_signal /= np.nansum(amplitude_signal)
    
    entropy_value = scipy.stats.entropy(amplitude_signal)
    max_entropy = np.log(len(amplitude_signal))
    
    return (max_entropy - entropy_value)/max_entropy

def run_half_entropy_hilbert_mi(low_freq_data, high_freq_data, phase_window_half_size = 10, phase_step_width = 20):
    """
    Calculates the modulation index between a low frequency signal and a high frequency signal.
    
    :param low_freq_data: Single array of low frequency data.
    :param high_freq_data: Single array of high frequency data.
    :param phase_window_half_size: Width of the phase window used for calculation of frequency/phase histogram. Amplitude gets added to every phase bin within the window size. Larger windows result in more smooth/increased PAC estimates.
    :param phase_step_width: Step width/shift of the phase window used for calculation of frequency/phase histogram.
    
    :return: Amount of phase amplitude coupling measured using the modulation index.
    """
    phase_signal = np.angle(scipy.signal.hilbert(low_freq_data), deg = True)
    amplitude_signal = np.zeros(np.arange(-180, 181, phase_step_width).shape)
    
    for (phaseIdx, loc_phase) in enumerate(np.arange(-180, 181, phase_step_width)):
        phase_indices = np.argwhere(np.abs(phase_signal - loc_phase) < phase_window_half_size)

        if (len(phase_indices) == 0):
            amplitude_signal[phaseIdx] = np.nan
        else:
            amplitude_signal[phaseIdx] = np.mean(np.abs(scipy.signal.hilbert(high_freq_data[phase_indices]))) # No need for a hilbert transform
        
    amplitude_signal /= np.nansum(amplitude_signal)
    
    entropy_value = scipy.stats.entropy(amplitude_signal)
    max_entropy = np.log(len(amplitude_signal))
    
    return (max_entropy - entropy_value)/max_entropy

def generate_high_frequency_signal(n, frequency_sampling, frequency_within_bursts, random_noise_strength, 
                                   offset, burst_count, burst_length, 
                                   sinusoidal = True):
    signal = np.random.normal(0, 1, n) * random_noise_strength

    if (sinusoidal == False):
        for burst_start in np.arange(offset, n, n/burst_count):
            burst_end = burst_start + (burst_length/2)
            signal[int(burst_start):int(burst_end)] =  np.sin(2 * np.pi * frequency_within_bursts * np.arange(0, (int(burst_end) - int(burst_start))) / frequency_sampling)
    else:
        signal += (np.sin(2 * np.pi * 200 * np.arange(len(signal)) / 1000)) * (np.sin(2 * np.pi * 10 * np.arange(len(signal)) / 1000) + 1)/2

    return signal

def get_data(frequency_sampling = 1000, frequencies_between_bursts = [2, 5, 10, 15, 20, 25, 30], 
             tgt_frequency_between_bursts = 10, frequency_within_bursts = 200, 
             random_noise_strength = 0.00, data_sz = 1000, rep_cnt = 10):
    scores = list();
    for _ in range(rep_cnt):
        scores.append(list())
        data_range = np.arange(0,  data_sz)
            
        #Generate sample data
        burst_length = frequency_sampling / tgt_frequency_between_bursts
        burst_count = len(data_range) / frequency_sampling * tgt_frequency_between_bursts
        
        
        high_freq_signal = generate_high_frequency_signal(len(data_range), frequency_sampling, frequency_within_bursts,
                                                           random_noise_strength, 0, burst_count, burst_length, True)
        low_freq_signals = [np.sin(2 * np.pi * frequency_between_bursts * data_range / frequency_sampling) for frequency_between_bursts in frequencies_between_bursts]
        
        loc_scores = list();
        for low_freq_signal in low_freq_signals:

            np.random.seed(0)
            abs_mi_score = run_abs_mi(low_freq_signal, high_freq_signal) * 100
            np.random.seed(0)
            hilbert_mi_score = run_hilbert_mi(low_freq_signal, high_freq_signal) * 100
            np.random.seed(0)
            entropy_mi_score = run_entropy_mi(low_freq_signal, high_freq_signal) * 100
            np.random.seed(0)
            half_entropy_hilbert_mi_score = run_half_entropy_hilbert_mi(low_freq_signal, high_freq_signal) * 100
            
            loc_scores = [abs_mi_score, hilbert_mi_score, entropy_mi_score, half_entropy_hilbert_mi_score]
            
            scores[-1].append(loc_scores);
    
    return scores

def main():
    frequency_sampling = 1000
    frequencies_between_bursts = [2, 5, 10, 15, 20, 25, 30]
    tgt_frequency_between_bursts = 10
    frequency_within_bursts = 200
    random_noise_strength = 0.05
    data_sz = 1000
    rep_cnt = 10

    scores = get_data(frequency_sampling, frequencies_between_bursts, tgt_frequency_between_bursts,
                      frequency_within_bursts, random_noise_strength, data_sz, rep_cnt)
    scores = np.asarray(scores)
    scores = np.mean(scores, axis = 0)
    
    plt.plot(scores[:, 0], color = "red", linewidth = 6)
    plt.plot(scores[:, 1], color = "purple", linestyle='--', linewidth = 5)
    plt.plot(scores[:, 2], color = "green", linestyle=':', linewidth = 4)
    plt.plot(scores[:, 3], color = "orange", linestyle='-.', linewidth = 3)
    plt.show(block = True)

main()


