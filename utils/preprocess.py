import torch
from torchaudio import transforms as T

from typing import Dict, Tuple

class Preprocessing:
    """
    Preprocessing contains utilities for transforming audio and text.
    
    """
    
    def __init__(self, 
                 out_channels: int = 2 , 
                 out_sampling_rate: int = 16000, 
                 n_mels: int = 80, 
                 n_fft : int = None, 
                 hop_length : int = None,
                 tokenizer = None):

        self.out_channels = out_channels
        self.out_sampling_rate = out_sampling_rate
        self.tokenizer = tokenizer
        
        if n_fft == None:
            n_fft = out_sampling_rate // 25 ## 25 ms n_fft window ##Conformer Paper
        
        if hop_length == None:
            hop_length = out_sampling_rate // 100 ## 10 ms hop length ##Conformer Paper
        
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
    
        self.mel_spec_transform = T.MelSpectrogram(sample_rate = self.out_sampling_rate,
                                                   n_fft = self.n_fft,
                                                   n_mels = self.n_mels,
                                                   hop_length = self.hop_length
                                                   )
        
    def standardize_channels(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Standardizes number of channels in a waveform. 
        
        Args:
            waveform: torch.Tensor (channel, timesteps)
        
        Returns:
            waveform: torch.Tensor (channel, timesteps)
        
        """
        
        num_channels = waveform.shape[0]
        
        if num_channels == self.out_channels:
            return waveform
        
        elif num_channels == 1:            
            return torch.cat((waveform, waveform))
        
        elif num_channels > 2:
            raise TypeError(f"Audio with more than 2 channels are not supported. Wanted maximum of 2 channels, found {num_channels} channels.")
        
        elif self.out_channels == 1:
            return torch.sum(waveform) / num_channels
        
        
    def standardize_sampling_rate(self, waveform: torch.Tensor, sampling_rate: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standardize Sampling Rate
        
        Args: waveform, sampling_rate
            waveform: torch.Tensor
            sampling_rate: torch.Tensor
            
        Returns: waveform, sampling_rate
            waveform: torch.Tensor
            sampling_rate: torch.Tensor
        """
        
        ##If there are more than 1 channel
        
        if waveform.shape[0] > 1:
        
            resampled = []

            for i in range(waveform.shape[0]):
                resampler = T.Resample(sampling_rate, self.out_sampling_rate)
                resampled_channel = resampler(waveform[i, :])
                resampled.append(resampled_channel)
        
            resampled = torch.stack(resampled)
            
            return resampled, self.out_sampling_rate
        
        ##If there is only 1 channel
        
        else:
            return T.Resample(sampling_rate, self.out_sampling_rate)(waveform[0, :]), self.out_sampling_rate
        

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Applies Feature Extraction Transforms to a waveform.
        
        Args: waveform
        
            waveform: The input for the feature extraction.
            
        Returns: melspec
        
            melspec: The Mel Spectrogram constructed from the input waveform of shape (channel, n_mels, time).
            
        """
        
        x = self.mel_spec_transform(waveform)
        
        return x
        
        
    def preprocess_waveform(self, waveform: torch.Tensor, sampling_rate: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        
        Preprocess Waveforms. Standardizes channels and sampling rate.
        
        Args: waveform, sampling_rate
        
            waveform: torch.Tensor
            
            sampling_rate: torch.Tensor
            
        Returns:

            waveform: torch.Tensor
            
            sampling_rate: torch.Tensor
        """
        
        
        waveform = self.standardize_channels(waveform)
        
        waveform, out_sampling_rate = self.standardize_sampling_rate(waveform, sampling_rate)
        
        return waveform, out_sampling_rate
    
    def tokenize_sentence(self, sentence: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize and convert a sentence to IDs.
        
        Args: sentence
            sentence: string to be tokenized
        
        Returns: 
            tokens: list of tokens numericalized
        """

        encoded = self.tokenizer.encode(sentence)
        
        return encoded
