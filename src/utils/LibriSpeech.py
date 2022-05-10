import torch
import os
import pandas
import numpy as np

import glob
import re
import functools
from pathlib import Path

from os.path import join as pathjoin
from os.path import abspath

import torchaudio

from torch.utils.data import Dataset

from typing import Union, Dict

from .preprocess import Preprocessing

def get_transcripts_dataframe(dataset_path: str) -> pandas.core.frame.DataFrame:
    
    """Generate a DataFrame for the LibriSpeech Dataset.
    
    Input:
        dataset_path: The path for the directory where the folders of the audio transcriptions are. 
            Example: dataset_path/103/1240/103-1240-000.flac, the dataset folder structure should be identical to this.

    Raises:
        ValueError: If the provided dataset path does not exist.

    Returns:
        pandas.core.frame.DataFrame: The DataFrame consisting of the filename, transcription and full file path.
    """
    
    
    ## Convert string path to PosixPath
    dataset_path = Path(dataset_path)
    
    ## Check if path exists
    if dataset_path.exists() == False:
        raise ValueError(f"Please provide a valid path. Input Dataset Path: {dataset_path}")
    
    ## Get all transcript files from the dataset path.
    transcript_files = glob.glob(f'{dataset_path}/**/*trans.txt', recursive=True)
    
    ## Compose a callable function
    read_func = functools.partial(pandas.read_csv, sep = r"(\d+-\d+-\d+) (\D*)", header= None, engine = "python", usecols=[1, 2], names = ["filename", "sentence"])

    ## Read all files then concat them together
    df = pandas.concat(map(read_func, transcript_files))
    
    ## Get the full filepath for each of the files
    df['filepath'] = df['filename'].apply(lambda x:abspath(pathjoin(dataset_path, pathjoin("/".join(x.split('-')[:2]), x) + '.flac')))
    
    ## Get the exact filename for the file
    df['filename'] = df['filename']+ '.flac'
    
    return df



class LibriSpeech(Dataset):
    
    """
    Utility Class for loading the LibriSpeech Dataset.
    """
    
    def __init__(self, dataset_path: str, out_channels: int = 2, out_sampling_rate: int = 16000, tokenizer = None):
        
 
        """
        Iterate over a subset LibriSpeech dataset.
        
        Args:
        ----------
        dataset_path: str: 
            The path for the directory where the folders of the audio transcriptions are. 
            Example: dataset_path/103/1240/103-1240-000.flac, the dataset folder structure should be identical to this.
        
        out_channels: int 
            Number of output channels for audio. 
            Mono = 1, Stereo = 2.
        
        out_sampling_rate: int
            sampling_rate used for standardizing.
        
        tokenizer: transformers.PreTrainedTokenizerFast
            tokenizer: tokenizer from huggingface used for tokenizing.
        
        Returns:
        ----------
        LibriSpeech Dataset object.
        
        """
        
        super(LibriSpeech).__init__()
                
        self.dataset_path = Path(dataset_path)
        
        ##Check that out_sampling_rate is an integer. Will probably need to specify how to convert Khz to Hz.
        assert isinstance(out_sampling_rate, int)
        self.out_sampling_rate = out_sampling_rate
        
        ##Check how many output channels are needed.
        if out_channels in [1, 2]:
            self.out_channels = out_channels
        else:
            raise ValueError("Only Mono (out_channels = 1) and Stereo (out_channels = 2) Supported.")
        
        ## Create DataFrame
        
        self.dataframe = get_transcripts_dataframe(self.dataset_path)
                
        ## Check if tokenizer is passed
        
        if tokenizer == None:
            raise ValueError("tokenizer cannot be None.")
        else:
            self.tokenizer = tokenizer
            
        ## Initialize Preprocessing
        
        self.preprocessing = Preprocessing(out_channels= self.out_channels, out_sampling_rate = self.out_sampling_rate, tokenizer = self.tokenizer)
        
        
    def __len__(self) -> int:
        return len(self.dataframe)
        
    
    def __getitem__(self, idx: Union[int, torch.Tensor, np.ndarray]) -> Dict[str, torch.Tensor]:
        
        if isinstance(idx, torch.Tensor):
            idx = list(idx)
            
        item = self.dataframe.iloc[idx]
        
        sentence = item['sentence']
        
        audio_file_path = item['filepath']
        
        waveform, source_sampling_rate = torchaudio.load(audio_file_path)
        waveform, out_sampling_rate = self.preprocessing.preprocess_waveform(waveform, source_sampling_rate)
        
        melspec = self.preprocessing.extract_features(waveform)
                
        item = {'waveform': waveform, 'sentence': sentence, 'melspec': melspec}
        
        return item
    
    
    def __repr__(self):
        return \
    f"""
    LibriSpeech Dataset
    -------------------
    
    Loading from {os.path.abspath(self.dataset_path)} directory.
        
    Number of Examples: {self.__len__()}
    
    Args:
        Sampling Rate: {self.out_sampling_rate}
        Output Channels: {self.out_channels}
    """
    