import torch
import torch.nn.functional as F
import numpy as np
import random

from typing import List, Dict

class Collator:
    """
    
    Utility Class for Collation of Batch. Intended to be used as part of the PyTorch DataLoader.
    
    """
    
    def __init__(self, tokenizer = None, special_tokens = True):
        """
        Initializes the Collator.
        
        Args: tokenizer, special_tokens
        
            tokenizer: Should be a HuggingFace PreTrained Tokenizer.
            special_tokens: Add special tokens like <bos> and <eos> or not.
        
        """
        if tokenizer == None:
            raise ValueError("Please provide a valid tokenizer.")
        
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
    
    @staticmethod
    def seed_worker(worker_id):
        """
        Worker Init Function to ensure reproducibility.
        """
        
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def pad(self, tensor: torch.Tensor, target_length: int) -> torch.Tensor:
        
        """
        
        Pads tensor according to the given length.
        
        Args:
            tensor: torch.Tensor
                tensor to be padded
        
            out_length: int
                target length of tensor
        
        Returns:
            tensor: torch.Tensor
                padded tensor
        """
        
        if not isinstance(target_length, int):
            raise ValueError(f"target_length must be an integer. Wanted {int}, have {type(target_length)}")
        
        length = target_length - tensor.shape[-1]

        if length <= 0:
            return tensor
        
        else:
            return F.pad(tensor, (0, length), "constant", 0)
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        
        """
        
        Transforms lists of inputs in the batch to Tensors.
            
            1. Applies Tokenization to Sentences.
            
            2. Pads Waveforms and Mel Specs to the maximum lengths in the batch.
        
        Args: batch
        
            Batch containing the list of dicts from the CommonVoice Dataset.
            
        Returns: batch
        
            Dict of Tensors.
            
            dict = {
                "waveforms": *padded waveforms*, // (batch, channel, time, amplitude) 
                "sentences": *padded and tokenized sentences*, // (batch, tokens) 
                "mel_specs": *padded mel spectrograms: *} // (batch, channel, n_mels, timeframes)
                }
        """
    
        waveforms = [sample['waveform'] for sample in batch]
        sentences = [sample['sentence'] for sample in batch]
        melspecs = [sample['melspec'] for sample in batch]
        
        waveform_lengths = torch.Tensor([waveform.shape[-1] for waveform in waveforms])
        
        melspecs_lengths = torch.Tensor([melspec.shape[-1] for melspec in melspecs])
        
        max_len_waveform = int(waveform_lengths.max())
        
        max_len_melspecs = int(melspecs_lengths.max())
        
        tokenized_sentences = self.tokenizer(sentences, 
                                             padding=True, 
                                             return_tensors = 'pt', 
                                             return_attention_mask=False, 
                                             return_length = True, 
                                             return_special_tokens_mask=True,
                                             add_special_tokens = self.special_tokens) ##Sentence Lengths required by CTC Loss
        
        ## Tokenizer is just returning the maximum length of the batch rather than their true lengths
        token_lengths = tokenized_sentences['length'] - tokenized_sentences['special_tokens_mask'].sum(dim = 1)
        
        padded_waveforms = torch.stack([self.pad(waveform, max_len_waveform) for waveform in waveforms])
        
        padded_mel_specs = torch.stack([self.pad(melspec, max_len_melspecs) for melspec in melspecs])
        
        del waveforms, sentences, melspecs ## Clean up 
        
        return {"waveforms": padded_waveforms, "waveforms_lengths": waveform_lengths, 
                "sentences": tokenized_sentences['input_ids'], "sentence_lengths": token_lengths, 
                "melspecs": padded_mel_specs, "melspecs_lengths": melspecs_lengths}