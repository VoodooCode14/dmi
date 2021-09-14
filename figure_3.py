'''
Created on Aug 7, 2020

@author: voodoocode
'''

import numpy as np

import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

import finn.cfc.pac as pac

np.random.seed(0)

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

def draw_figure(scores, best_fits, amplitude_signals, frequencies_between_bursts, random_noise_strengths):
    (_, axes) = plt.subplots(2, 2)
    
    axes = axes.reshape(-1)
    
    for ax_idx in range(scores.shape[2]):
        for noise_idx in range(scores.shape[0]):
            axes[ax_idx].plot(scores[noise_idx, :, ax_idx], label = ("noise %i" % int(random_noise_strengths[noise_idx] * 100)))
        axes[ax_idx].legend()
        axes[ax_idx].set_xticks(range(len(scores[noise_idx, :, ax_idx])))
        axes[ax_idx].set_xticklabels(frequencies_between_bursts)
        axes[ax_idx].set_title(pac_methods[ax_idx])
    axes[0].set_ylim((0, 1))

def get_data(frequency_sampling = 1000, frequencies_between_bursts = [2, 5, 10, 15, 20, 25, 30], 
             tgt_frequency_between_bursts = 10, frequency_within_bursts = 200, 
             random_noise_strengths = [0.01], data_sz = 1000, rep_cnt = 10):
    scores = list(); best_fits = list(); amplitude_signals = list()
    for _ in range(rep_cnt):
        data_range = np.arange(0,  data_sz)
            
        #Generate sample data
        burst_length = frequency_sampling / tgt_frequency_between_bursts
        burst_count = len(data_range) / frequency_sampling * tgt_frequency_between_bursts
        
        loc_scores = list(); loc_best_fits = list(); loc_amplitude_signals = list()
        for random_noise_strength in random_noise_strengths:
            loc_scores.append(list()); loc_best_fits.append(list()); loc_amplitude_signals.append(list())
            high_freq_signal = generate_high_frequency_signal(len(data_range), frequency_sampling, frequency_within_bursts,
                                                               random_noise_strength, 0, burst_count, burst_length, True)
            low_freq_signals = [np.sin(2 * np.pi * frequency_between_bursts * data_range / frequency_sampling) for frequency_between_bursts in frequencies_between_bursts]
            
            for low_freq_signal in low_freq_signals:
                tmp = pac.run_dmi(low_freq_signal, high_freq_signal, phase_window_half_size = 4, phase_step_width = 2)
                d_mi_score = tmp[0]
                plv_score = pac.run_plv(low_freq_signal, high_freq_signal)
                mvl_score = pac.run_mvl(low_freq_signal, high_freq_signal)
                mi_score = pac.run_mi(low_freq_signal, high_freq_signal) * 100
                loc_scores[-1].append([d_mi_score, plv_score, mvl_score, mi_score]); loc_best_fits[-1].append(tmp[1]); loc_amplitude_signals[-1].append(tmp[2])
        scores.append(loc_scores); best_fits.append(loc_best_fits); amplitude_signals.append(loc_amplitude_signals)
        
    return (scores, best_fits, amplitude_signals)

import scipy.stats

pac_methods = ["direct modulation index", "phase lag value", "mean vector length", "modulation index"]

def main(data_sz = 30000):

    frequency_sampling = 1000
    frequencies_between_bursts = [2, 5, 10, 15, 20, 25, 30]
    tgt_frequency_between_bursts = 10
    frequency_within_bursts = 200
    random_noise_strengths = [0.01, 0.25, 0.5, 1.0, 1.5]
    rep_cnt = 10

    (scores, best_fits, amplitude_signals) = get_data(frequency_sampling, frequencies_between_bursts, tgt_frequency_between_bursts,
                                                      frequency_within_bursts, random_noise_strengths, data_sz, rep_cnt)
    scores = np.asarray(scores); best_fits = np.asarray(best_fits); amplitude_signals = np.asarray(amplitude_signals)
    print(np.average(scores, axis = 0))    
    
    res = list()
    for noise_idx in range(len(random_noise_strengths)):
        res.append(list())
        for freq_idx in range(len(frequencies_between_bursts)):
            res[-1].append(list())
            for metric_idx in range(4):
                if ((scores[:, noise_idx, freq_idx, metric_idx] == scores[0, noise_idx, freq_idx, metric_idx]).all()):
                    res[-1][-1].append(np.nan)
                else:
                    res[-1][-1].append(scipy.stats.ttest_1samp(scores[:, noise_idx, freq_idx, metric_idx], 0)[1])
    res = np.asarray(res)
    print(np.nan_to_num(res))
    
    scores = np.average(scores, axis = 0); best_fits = np.average(best_fits, axis = 0); amplitude_signals = np.average(amplitude_signals, axis = 0)
    #visualization
    draw_figure(scores, best_fits, amplitude_signals, frequencies_between_bursts, random_noise_strengths)

main(30000)
main(300000)
plt.show(block = True)



