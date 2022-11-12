'''
Created on Aug 7, 2020

@author: voodoocode
'''

import numpy as np

import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt

import finnpy.cfc.pac as pac

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

def draw_figure(scores, best_fits, amplitude_signals, frequencies_between_bursts, tgt_frequency_between_bursts):
    (_, axes) = plt.subplots(1, len(frequencies_between_bursts))
    for (ax_idx, _) in enumerate(amplitude_signals):
        axes[ax_idx].plot(np.arange(0, 1, 1/len(amplitude_signals[ax_idx])), amplitude_signals[ax_idx], label = "original data")
        axes[ax_idx].plot(np.arange(0, 1, 1/len(amplitude_signals[ax_idx])), best_fits[ax_idx], label = "fitted curve")
        axes[ax_idx].set_title("DMI| PLV| MVL| MI  \n%.3f|%.3f|%.3f|%.3f" % (scores[ax_idx][0], scores[ax_idx][1], 
                                                                                            scores[ax_idx][2], scores[ax_idx][3]))

def get_data(frequency_sampling = 1000, frequencies_between_bursts = [2, 5, 10, 15, 20, 25, 30], 
             tgt_frequency_between_bursts = 10, frequency_within_bursts = 200, 
             random_noise_strength = 0.00, data_sz = 1000, rep_cnt = 10):
    scores = list(); best_fits = list(); amplitude_signals = list()
    for _ in range(rep_cnt):
        data_range = np.arange(0,  data_sz)
            
        #Generate sample data
        burst_length = frequency_sampling / tgt_frequency_between_bursts
        burst_count = len(data_range) / frequency_sampling * tgt_frequency_between_bursts
        
        
        high_freq_signal = generate_high_frequency_signal(len(data_range), frequency_sampling, frequency_within_bursts,
                                                           random_noise_strength, 0, burst_count, burst_length, True)
        low_freq_signals = [np.sin(2 * np.pi * frequency_between_bursts * data_range / frequency_sampling) for frequency_between_bursts in frequencies_between_bursts]
        
        loc_scores = list(); loc_best_fits = list(); loc_amplitude_signals = list()
        for low_freq_signal in low_freq_signals:
            tmp = pac.run_dmi(low_freq_signal, high_freq_signal, phase_window_half_size = 4, phase_step_width = 2)
            d_mi_score = tmp[0]
            plv_score = pac.run_plv(low_freq_signal, high_freq_signal)
            mvl_score = pac.run_mvl(low_freq_signal, high_freq_signal)
            mi_score = pac.run_mi(low_freq_signal, high_freq_signal) * 100
            loc_scores.append([d_mi_score, plv_score, mvl_score, mi_score]); loc_best_fits.append(tmp[1]); loc_amplitude_signals.append(tmp[2])
        scores.append(loc_scores); best_fits.append(loc_best_fits); amplitude_signals.append(loc_amplitude_signals)
        
    return (scores, best_fits, amplitude_signals)

import scipy.stats

def main(visualize = True):

    frequency_sampling = 1000
    frequencies_between_bursts = [2, 5, 10, 15, 20, 25, 30]
    tgt_frequency_between_bursts = 10
    frequency_within_bursts = 200
    random_noise_strength = 0.05
    data_sz = 1000
    rep_cnt = 10

    (scores, best_fits, amplitude_signals) = get_data(frequency_sampling, frequencies_between_bursts, tgt_frequency_between_bursts,
                                                      frequency_within_bursts, random_noise_strength, data_sz, rep_cnt)
    scores = np.asarray(scores); best_fits = np.asarray(best_fits); amplitude_signals = np.asarray(amplitude_signals)
    
    res = list()
    for freq_idx in range(len(frequencies_between_bursts)):
        res.append(list())
        for metric_idx in range(4):
            if ((scores[:, freq_idx, metric_idx] == scores[0, freq_idx, metric_idx]).all()):
                res[-1].append(np.nan)
            else:
                res[-1].append(scipy.stats.ttest_1samp(scores[:, freq_idx, metric_idx], 0)[1])
    res = np.asarray(res)
    print(np.nan_to_num(res, 0))
    
    scores = np.average(scores, axis = 0); best_fits = np.average(best_fits, axis = 0); amplitude_signals = np.average(amplitude_signals, axis = 0)
    print(scores)
    #visualization
    draw_figure(scores, best_fits, amplitude_signals, frequencies_between_bursts, tgt_frequency_between_bursts)

    if (visualize == True):
        plt.show(block = True)

main()


