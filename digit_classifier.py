"""
This file will implement a digit classifier using rule-based dsp methods.
As all digit waveforms are given, we could take that under consideration, of our RULE-BASED system.

We recommend you answer this after filling all functions in general_utilities.
"""
import torchaudio as ta
import soundfile as sf
import torch
import typing as tp
from pathlib import Path
import librosa
import matplotlib.pyplot as plt
import scipy
import numpy as np
from general_utilities import *

# --------------------------------------------------------------------------------------------------
#     Part A        Part A        Part A        Part A        Part A        Part A        Part A    
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# In this part we will get familiarized with the basic utilities defined in general_utilities
# --------------------------------------------------------------------------------------------------


def self_check_fft_stft():
    """
    Q:
    1. create 1KHz and 3Khz sine waves, each of 3 seconds length with a sample rate of 16KHz.
    2. In a single plot (3 subplots), plot (i) FFT(sine(1Khz)) (ii) FFT(sine(3Khz)), 
       (iii) FFT(sine(1Khz) + sine(3Khz)), make sure X axis shows frequencies. 
       Use general_utilities.plot_fft
    3. concatate [sine(1Khz), sine(3Khz), sine(1Khz) + sine(3Khz)] along the temporal axis, and plot
       the corresponding MAGNITUDE STFT using n_fft=1024. Make sure Y ticks are frequencies and X
       ticks are seconds.

    Include all plots in your PDF
    """
    onekh_wave = create_single_sin_wave(1000).unsqueeze(0)
    threekh_wave = create_single_sin_wave(3000).unsqueeze(0)
    waves = torch.stack([onekh_wave,threekh_wave, onekh_wave + threekh_wave])
    plot_fft(waves)
    concatenated_tensor = torch.cat((onekh_wave, threekh_wave, threekh_wave+onekh_wave))
    plot_spectrogram(concatenated_tensor.view(1, -1))

def audio_check_fft_stft():
    """
    Q:
    1. load all phone_*.wav files in increasing order (0 to 11)
    2. In a single plot (2 subplots), plot (i) FFT(phone_1.wav) (ii) FFT(phone_2.wav). 
       Use general_utilities.plot_fft
    3. concatate all phone_*.wav files in increasing order (0 to 11) along the temporal axis, and plot
       the corresponding MAGNITUDE STFT using n_fft=1024. Make sure Y ticks are frequencies and X
       ticks are seconds.

    Include all plots in your PDF
    """
    phones = []
    for x in range(12):
        wave, sr = load_wav("audio_files/phone_digits_8k/phone_" + str(x) + ".wav")
        phones.append(wave)
    plot_fft(torch.stack([phones[0], phones[1]]))
    concatenated_waves = torch.cat(phones).view(1,-1)
    plot_spectrogram(concatenated_waves)


# --------------------------------------------------------------------------------------------------
#     Part B        Part B        Part B        Part B        Part B        Part B        Part B    
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Digit Classifier
# --------------------------------------------------------------------------------------------------

def extract_digit_freq_helper() -> tp.Dict[int, tp.Tuple[int, int]]:
    freqs = {}

    for digit in range(12):
        wave, sr = load_wav("audio_files/phone_digits_8k/phone_" + str(digit) + ".wav")
        fft = do_fft(wave)
        magnitude = torch.abs(fft).numpy()
        frequencies = np.fft.fftfreq(magnitude.shape[-1], 1/sr)
        dominant_freq = frequencies[np.argsort(magnitude)[-2:]]
        freqs[digit] = dominant_freq
    return freqs

def classify_single_digit(wav: torch.Tensor) -> int:
    """
    Q:
    Write a RULE-BASED (if - else..) function to classify a given single digit waveform.
    Use ONLY functions from general_utilities file.

    Hint: try plotting the fft of all digits.
    
    wav: torch tensor of the shape (1, T).

    return: int, digit number
    """

    frequencies = extract_digit_freq_helper()
    fft = do_fft(wav)
    magnitude = torch.abs(fft).numpy()
    freq = np.fft.fftfreq(magnitude.shape[-1], 1/8000)
    dominant_freq = freq[np.argsort(magnitude)[-2:]]
    min_diff = float('inf')
    digit = -1
    for key,val in frequencies.items():
        diff = min(abs(val[0] - dominant_freq[0]) + abs(val[1] - dominant_freq[1]),
                   abs(val[0] - dominant_freq[1]) + abs(val[1] - dominant_freq[0]))
        if diff < min_diff:
            min_diff = diff
            digit = key
    if min_diff > 100:
        return -1
    return digit



def classify_digit_stream(wav: torch.Tensor) -> tp.List[int]:
    """
    Q:
    Write a RULE-BASED (if - else..) function to classify a waveform containing several digit stream.
    The input waveform will include at least a single digit in it.
    The input waveform will have digits waveforms concatenated on the temporal axis, with random zero
    padding in-between digits.
    You can assume that there will be at least 100ms of zero padding between digits
    The function should return a list of all integers pressed (in order).
    
    Use STFT from general_utilities file to answer this question.

    wav: torch tensor of the shape (1, T).

    return: List[int], all integers pressed (in order).
    """
    silence_threshold = 1e-2
    n_fft = 1024
    hop_length = n_fft // 4
    stft_result = do_stft(wav, n_fft)

    current_stft = torch.view_as_complex(stft_result[0])
    magnitude = torch.abs(current_stft)
    magnitude = magnitude[:magnitude.shape[0] // 2]
    num_frames = magnitude.shape[-1]

    digit_segment = []
    cur_segment = []
    for i in range(num_frames):
        if torch.max(magnitude[:, i]) > silence_threshold:
            cur_segment.append(i)
        elif len(cur_segment) > 0:
            digit_segment.append(cur_segment)
            cur_segment = []
    if len(cur_segment) > 0:
        digit_segment.append(cur_segment)

    digits = []
    for segment in digit_segment:
        start_idx = segment[0] * hop_length
        end_idx = (segment[-1]+1) * hop_length
        cur_wav = wav[:, start_idx:end_idx]
        digit = classify_single_digit(cur_wav)
        if digit != -1:
            digits.append(digit)
    return digits
