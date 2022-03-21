import torch
import os

import torchaudio

from torch.utils.data import Dataset

class CommonVoice(Dataset):
    
    def __init__(self, dataset_path: str, split_type: str = 'train', out_channels: int = 2, out_sampling_rate: int = 32000, tokenizer = None):
        
        import pandas as pd
 
        """
        Iterate over a split of CommonVoice dataset.
        
        Parameters
        ----------
        dataset_path: str
            The path where train.tsv, test.tsv and dev.tsv are located.
        
        split_type: str [train, test, dev]
            Loads one of train.tsv, test.tsv or dev.tsv
        
        out_channels: int 
            Number of output channels for audio. 
            Mono = 1, Stereo = 2.
        
        out_sampling_rate: int
            sampling_rate used for standardizing.
        
        tokenizer: transformers.PreTrainedTokenizerFast
            tokenizer: tokenizer from huggingface used for tokenizing.
        """
        
        super(CommonVoice).__init__()
                
        self.split_type = split_type
        self.dataset_path = dataset_path
        
        ##Check that out_sampling_rate is an integer. Will probably need to specify how to convert Khz to Hz.
        assert isinstance(out_sampling_rate, int)
        self.out_sampling_rate = out_sampling_rate
        
        ##Check how many output channels are needed.
        if out_channels in [1, 2]:
            self.out_channels = out_channels
        else:
            raise ValueError("Only Mono (out_channels = 1) and Stereo (out_channels = 2) Supported.")
        
        ##Check that dataset exists in the path specified and add clips path.
        if os.path.exists(dataset_path):
            self.clips_path = os.path.join(dataset_path, 'clips')
        else:
            raise ValueError(f"{dataset_path} doesn't exist, please provide a valid path to the dataset.")
        
        ##Check that split_type is one of train, test, dev.
        if split_type in ['train', 'test' , 'dev']:
            fullpath = os.path.join(dataset_path, split_type + '.tsv')
        else:
            raise ValueError("split_type must be one of train, test or dev")
        
        ## Load the dataframe
        self.dataframe = pd.read_csv(fullpath, sep = '\t')
        
        ## Check if tokenizer is passed
        
        if tokenizer == None:
            raise ValueError("tokenizer cannot be None.")
        else:
            self.tokenizer = tokenizer
        
        
    def __len__(self) -> int:
        return len(self.dataframe)
        
    def standardize_channels(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Standardizes number of channels in a waveform. 
        
        Args:
            waveform: torch.Tensor
        
        Returns:
            waveform: torch.Tensor
        
        """
        
        num_channels = waveform.shape[0]
        
        if num_channels == self.out_channels:
            return waveform
        
        elif num_channels == 1:            
            return torch.cat((waveform, waveform))
        
        elif num_channels > 2:
            raise TypeError(f"Audio with more than 2 channels are not supported. Wanted maximum of 2 channels, found {num_channels} channels.")
        
        
    def standardize_sampling_rate(self, waveform: torch.Tensor, sampling_rate: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Standardize Sampling Rate
        
        Args:
            waveform: torch.Tensor
            sampling_rate: torch.Tensor
            
        Returns:
            waveform: torch.Tensor
            sampling_rate: torch.Tensor
        """
        
        ##If there are more than 1 channels
        if waveform.shape[0] > 1:
        
            resampled = []

            for i in range(waveform.shape[0]):
                resampler = torchaudio.transforms.Resample(sampling_rate, self.out_sampling_rate)
                resampled_channel = resampler(waveform[i, :])
                resampled.append(resampled_channel)
        
            resampled = torch.stack(resampled)
            return resampled, self.out_sampling_rate
        
        ##If there is only 1 channel
        else:
            return Resample(sampling_rate, self.out_sampling_rate)(waveform[0, :]), self.out_sampling_rate
        
        
    def preprocess_waveform(self, waveform: torch.Tensor, sampling_rate: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Preprocess Waveforms. Standardizes channels and sampling rate.
        
        Args:
            waveform: torch.Tensor
            sampling_rate: torch.Tensor
            
        Returns:
            waveform: torch.Tensor
            sampling_rate: torch.Tensor
        """
        
        
        waveform = self.standardize_channels(waveform)
        
        waveform, out_sampling_rate = self.standardize_sampling_rate(waveform, sampling_rate)
        
        return waveform, out_sampling_rate
    
    def tokenize_sentence(self, sentence: str):
        """
        Tokenizes a given sentence.
        
        Args:
            sentence: string to be tokenized
        
        Returns:
            tokens: list of tokens numericalized
        """

        encoded = self.tokenizer.encode(sentence)
        
        return encoded
    
    def __getitem__(self, idx):
        
        if isinstance(idx, torch.Tensor):
            idx = list(idx)
            
        item = self.dataframe.iloc[idx]
        
        sentence = item['sentence']
        
        age = item['age']
        gender = item['gender']
        accent = item['accents']
        
        audio_file_path = os.path.join(self.clips_path, item['path'])
        
        waveform, source_sampling_rate = torchaudio.load(audio_file_path)
        waveform, out_sampling_rate = self.preprocess_waveform(waveform, source_sampling_rate)
                
        # item = {'waveform': waveform, 'sentence': sentence, 'age': age, 'gender' : gender, 'accent': accent}
        
        item = {'waveform': waveform, 'sentence': sentence}
        
        return item
    
    
    def __repr__(self):
        return \
    f"""
    CommonVoice Dataset
    -------------------
    
    Loading {self.split_type}.tsv from {os.path.abspath(self.dataset_path)} directory.
        
    Number of Examples: {self.__len__()}
    
    Args:
        Sampling Rate: {self.out_sampling_rate}
        Output Channels: {self.out_channels}
    """
    