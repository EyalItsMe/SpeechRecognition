"""
In this file we will experiment with naively interpolating a signal on the time domain and on the frequency domain.

We reccomend you answer this file last.
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
from torch.nn.functional import interpolate


def naive_time_stretch_temporal(wav: torch.Tensor, factor: float):
    """
    Q:
      write a function that uses a simple linear interpolation across the temporal dimension
      stretching/squeezing a given waveform by a given factor.
      Use imported 'interpolate'.

    1. load audio_files/Basta_16k.wav
    2. use this function to stretch it by 1.2 and by 0.8.
    3. save files using ta.save(fpath, stretch_wav, 16000) and listen to the files. What happened?
       Explain what differences you notice and why that happened in your PDF file

    Do NOT include saved audio in your submission.
    """
    wav = wav.unsqueeze(1)
    return interpolate(wav, size=int(wav.size(-1) * factor), mode="linear", align_corners=False).squeeze(1)


def naive_time_stretch_stft(wav: torch.Tensor, factor: float):
    """
    Q:
      write a function that converts a given waveform to stft, then uses a simple linear interpolation 
      across the temporal dimension stretching/squeezing by a given factor and converts the stretched signal 
      back using istft.
      Use general_utilities for STFT / iSTFT and imported 'interpolate'.

    1. load audio_files/Basta_16k.wav
    2. use this function to stretch it by 1.2 and by 0.8.
    3. save files using ta.save(fpath, stretch_wav, 16000) and listen to the files. What happened?
       Explain what differences you notice and why that happened in your PDF file

    Do NOT include saved audio in your submission.
    """
    wav_one = wav[0].unsqueeze(0)
    stft_one = do_stft(wav_one,1024)
    plot_spectrogram(wav_one)
    wav_two = wav[1].unsqueeze(0)
    stft_two = do_stft(wav_two,1024)
    interpolate_one = interpolate(stft_one, size=int(stft_one.size(-1) * factor), mode="linear", align_corners=False)
    interpolate_two = interpolate(stft_two, size=int(stft_two.size(-1) * factor), mode="linear", align_corners=False)



if __name__ == "__main__":
    wave, sr = load_wav("audio_files/Basta_16k.wav")
    stretched_0_8 = naive_time_stretch_temporal(wave, 0.8)
    stretched_1_2 = naive_time_stretch_temporal(wave, 1.2)
    ta.save("stretched_Basta_0_8.wav", stretched_0_8, 16000)
    ta.save("stretched_Basta_1_2.wav", stretched_1_2, 16000)
    stretched_0_8 = naive_time_stretch_stft(wave, 0.8)
    stretched_1_2 = naive_time_stretch_stft(wave, 1.2)
    ta.save("stretched_Basta_0_8.wav", stretched_0_8, 16000)
    ta.save("stretched_Basta_1_2.wav", stretched_1_2, 16000)

