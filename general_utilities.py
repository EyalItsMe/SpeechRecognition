
"""
This file will define the general utility functions you will need for you implementation throughout this ex.
We suggest you start with implementing and testing the functions in this file.

NOTE: each function has expected typing for it's input and output arguments. 
You can assume that no other input types will be given and that shapes etc. will be as described.
Please verify that you return correct shapes and types, failing to do so could impact the grade of the whole ex.

NOTE 2: We STRONGLY encourage you to write down these function by hand and not to use Copilot/ChatGPT/etc.
Implementaiton should be fairly simple and will contribute much to your understanding of the course material.

NOTE 3: You may use external packages for fft/stft, you are requested to implement the functions below to 
standardize shapes and types.
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


def create_single_sin_wave(frequency_in_hz, total_time_in_secs=3, sample_rate=16000):
    timesteps = np.arange(0, total_time_in_secs * sample_rate) / sample_rate
    sig = np.sin(2 * np.pi * frequency_in_hz * timesteps)
    return torch.Tensor(sig).float()


def load_wav(abs_path: tp.Union[str, Path]) -> tp.Tuple[torch.Tensor, int]:
    """
    This function loads an audio file (mp3, wav).
    If you are running on a computer with gpu, make sure the returned objects are mapped on cpu.

    abs_path: path to the audio file (str or Path)
    returns: (waveform, sample_rate)
        waveform: torch.Tensor (float) of shape [1, num_channels]
        sample_rate: int, the corresponding sample rate
    """
    wav, sr = ta.load(abs_path)
    return wav, sr


def do_stft(wav: torch.Tensor, n_fft: int=1024) -> torch.Tensor:
    """
    This function performs STFT using win_length=n_fft and hop_length=n_fft//4.
    Should return the complex spectrogram.

    hint: see torch.stft.

    wav: torch tensor of the shape (1, T) or (B, 1, T) for the batched case.
    n_fft: int, denoting the number of used fft bins.

    returns: torch.tensor of the shape (1, n_fft, *, 2) or (B, 1, n_fft, *, 2), where last dim stands for real/imag entries.
    """
    wav = torch.squeeze(wav)
    stft = torch.stft(wav, n_fft=n_fft, onesided=False, return_complex=False)
    if len(wav.shape) == 1:
        stft = stft.unsqueeze(0)
    else:
        stft = stft.unsqueeze(1)
    return stft


def do_istft(spec: torch.Tensor, n_fft: int=1024) -> torch.Tensor:
    """
    This function performs iSTFT using win_length=n_fft and hop_length=n_fft//4.
    Should return the complex spectrogram.

    hint: see torch.istft.

    spec: torch.tensor of the shape (1, n_fft, *, 2) or (B, 1, n_fft, *, 2), where last dim stands for real/imag entries.
    n_fft: int, denoting the number of used fft bins.

    returns: torch tensor of the shape (1, T) or (B, 1, T) for the batched case.

    NOTE: you may need to use torch.view_as_complex.
    """
    if len(spec.shape) == 4:
        spec = spec.squeeze(0)
    else:
        spec = spec.squeeze(1)
    spec = torch.view_as_complex(spec)
    istft = torch.istft(spec, n_fft=n_fft, onesided=False, win_length=n_fft, hop_length=n_fft // 4,
                        return_complex=False)
    if len(istft.shape) == 1:
        istft = istft.unsqueeze(0)
    else:
        istft = istft.unsqueeze(1)

    return istft



def do_fft(wav: torch.Tensor) -> torch.Tensor:
    """
    This function performs fast fourier trasform (FFT) .

    hint: see scipy.fft.fft / torch.fft.rfft, you can convert the input tensor to numpy just make sure to cast it back to torch.

    wav: torch tensor of the shape (1, T).

    returns: corresponding FFT transformation considering ONLY POSITIVE frequencies, returned tensor should be of complex dtype.
    """
    wav_tensor = wav.squeeze(0)
    wav_tensor = torch.fft.rfft(wav_tensor)
    return wav_tensor


def plot_spectrogram(wav: torch.Tensor, n_fft: int=1024, sr=16000) -> None:
    """
    This function plots the magnitude spectrogram corresponding to a given waveform.
    The Y axis should include frequencies in Hz and the x axis should include time in seconds.

    wav: torch tensor of the shape (1, T) or (B, 1, T) for the batched case.

    NOTE: for the batched case multiple plots should be generated (sequentially by order in batch)
    """
    batch_size = wav.shape[0]

    if wav.dim() == 3:
        wav = wav.squeeze(1)

    for i in range(batch_size):
        stft_result = do_stft(wav[i], n_fft=n_fft).squeeze(0)
        stft_result = torch.view_as_complex(stft_result)
        magnitude_spectrogram = torch.abs(stft_result).numpy()
        magnitude_spectrogram = magnitude_spectrogram[:magnitude_spectrogram.shape[0] // 2]
        num_frames = magnitude_spectrogram.shape[-1]
        time_axis = np.arange(num_frames) * (n_fft // 4) / sr
        freq_axis = np.arange(magnitude_spectrogram.shape[0]) * sr / n_fft

        plt.figure(figsize=(10, 6))
        plt.imshow(magnitude_spectrogram, aspect='auto', origin='lower',
                   extent=[time_axis.min(), time_axis.max(), freq_axis.min(), freq_axis.max()])
        plt.colorbar(label='Magnitude')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Magnitude Spectrogram')
        plt.show()


def plot_fft(wav: torch.Tensor, sr=16000) -> None:
        """
        This function plots the FFT transform to a given waveform.
        The X axis should include frequencies in Hz.

        NOTE: As abs(FFT) reflects around zero, please plot only the POSITIVE frequencies.

        wav: torch tensor of the shape (1, T) or (B, 1, T) for the batched case.
        """
        batch_size = wav.shape[0]
        if wav.dim() == 3:
            wav = wav.squeeze(1)

        fig, axes = plt.subplots(batch_size, 1, figsize=(10, 6 * batch_size))
        if batch_size == 1:
            axes = [axes]
        for i in range(batch_size):
            fft_result = do_fft(wav[i])
            magnitude = torch.abs(fft_result).numpy()
            freq_bins = np.fft.rfftfreq(len(wav[i]), d=1.0/sr)
            axes[i].plot(freq_bins, magnitude)
            axes[i].set_xlabel('Frequency (Hz)')
            axes[i].set_ylabel('Magnitude')
            axes[i].set_title(f'FFT Magnitude Spectrum for sample {i + 1}')
            axes[i].grid(True)

        plt.show()

