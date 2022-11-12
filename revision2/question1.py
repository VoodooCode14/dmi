'''
Created on Nov 11, 2022

@author: voodoocode
'''

import numpy as np

import finnpy.cfc.pac as pac

import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

import scipy.signal
import scipy.stats
import lmfit

def _sine(x, phase, amp):
    """
    Internal method. Used in run_dmi to estimate the direct modulation index. The amount of PAC is quantified via a sine fit. This sine is defined by the following paramters:
    
    :param x: Samples
    :param phase: Phase shift of the sine.
    :param amp: Amplitude of the sine.
    
    :return: Returns the fitted sine at the locations indicated by x.
    """
    freq = 1
    fs = 1
    return amp * (np.sin(2 * np.pi * freq * (x - ((phase + 180)/360)) / fs))

def run_mod_dmi(low_freq_data, high_freq_data,
         phase_window_half_size = 10, phase_step_width = 1,
         max_model_fit_iterations = 200):
    """
    Calculates the direct modulation index between a low frequency signal and a high frequency signal. Instead of the original modulation index based on entropy, this modulation index estimate is based on a sinusoidal fit. 
    
    :param low_freq_data: Single array of low frequency data.
    :param high_freq_data: Single array of high frequency data. Must have the same length as low_freq_data.
    :param phase_window_half_size: Width of the phase window used for calculation of frequency/phase histogram. Amplitude gets added to every phase bin within the window size. Larger windows result in more smooth, but also potentially increased PAC estimates.
    :param phase_step_width: Step width of the phase window used for calculation of frequency/phase histogram.
    :param max_model_fit_iterations: Maximum number of iterations applied during sine fitting.
    
    :return: Amount of phase amplitude coupling measured using the modulation index.
    """
    
    if (len(low_freq_data) != len(high_freq_data)):
        raise AssertionError("Both signals must have the same length")
    
    phase_signal = np.angle(scipy.signal.hilbert(low_freq_data), deg = True)
    amplitude_signal = np.zeros(np.arange(-180, 181, phase_step_width).shape)
    
    for (phaseIdx, loc_phase) in enumerate(np.arange(-180, 181, phase_step_width)):
        phase_indices = np.argwhere(np.abs(phase_signal - loc_phase) < phase_window_half_size)

        if (len(phase_indices) == 0):
            amplitude_signal[phaseIdx] = np.nan
        else:
            amplitude_signal[phaseIdx] = np.mean(np.abs(scipy.signal.hilbert(high_freq_data[phase_indices]))) # No need for a hilbert transform
    
    amplitude_signal -= np.nanpercentile(amplitude_signal,25)
    amplitude_signal /= np.nanpercentile(amplitude_signal,75)

    amplitude_signal = amplitude_signal * 2 - 1
    amplitude_signal*= 0.70710676
    
    params = lmfit.Parameters()
    params.add("phase", value = 0, min = -180, max = 180, vary = True)
    params.add("amp", value = 1, min = 0.95, max = 1.05, vary = True)
    model = lmfit.Model(_sine, nan_policy = "omit")
    result = model.fit(amplitude_signal, x = np.arange(0, 1, 1/len(amplitude_signal)),
                       params = params, max_nfev = max_model_fit_iterations)

    if (np.isnan(amplitude_signal).any() == True):
        amplitude_signal = np.where(np.isnan(amplitude_signal) == False)[0]

    error = np.sum(np.square(result.best_fit - amplitude_signal))/len(amplitude_signal)
    
    error = 1 if (error > 1) else error #Capping the error

    score = 1 - error
    score = 0 if (score < 0) else score

    return (score, result.best_fit, amplitude_signal)

def run_mod_mi(low_freq_data, high_freq_data, phase_window_half_size = 10, phase_step_width = 20):
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
            dmi_score = pac.run_dmi(low_freq_signal, high_freq_signal, phase_window_half_size = 4, phase_step_width = 2)[0]
            np.random.seed(0)
            mod_dmi_score = run_mod_dmi(low_freq_signal, high_freq_signal, phase_window_half_size = 4, phase_step_width = 2)[0]
            np.random.seed(0)
            mi_score = pac.run_mi(low_freq_signal, high_freq_signal) * 100
            np.random.seed(0)
            mod_mi_score = run_mod_mi(low_freq_signal, high_freq_signal) * 100
            
            loc_scores = [dmi_score, mod_dmi_score, mi_score, mod_mi_score]
            
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
    
    plt.plot(scores[:, 0], color = "green")
    plt.plot(scores[:, 1], color = "blue", linestyle='dashed')
    plt.plot(scores[:, 2], color = "red")
    plt.plot(scores[:, 3], color = "purple", linestyle='dashed')
    plt.show(block = True)

main()


