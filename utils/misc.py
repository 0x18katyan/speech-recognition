from torchinfo import summary

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader

from datetime import datetime

import os

def get_summary(model: nn.Module, dataloader: DataLoader = None, data = None) -> str:
    
    """
    A very very dirty way to get model summary. TODO: Clean up a bit.
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    batch = next(iter(dataloader))
    
    sentences = batch['sentences'].to(device)
    sentence_lengths = batch['sentence_lengths'].to(device, dtype= torch.int32)

    melspecs = batch['melspecs'].to(device)
    melspecs_lengths = batch['melspecs_lengths'].to(device, dtype= torch.int32)

    melspecs = torch.transpose(melspecs, -1, -2) ## Changing to (batch, channel, time, n_mels) from (batch, channel, n_mels, time)
    
    return summary(model, 
                   input_data = [melspecs, melspecs_lengths],
                   device=device, 
                   dtypes = [torch.float32, torch.float32], 
                   col_width=16,
                   col_names=["output_size", "num_params"],
                   mode = 'eval', 
                   verbose = 0)

def get_writer(base_log_dir: str, comment = None) -> SummaryWriter:
    base_log_dir = 'logs/'

    start_time = datetime.now()
    start_time_fmt = start_time.strftime("%d-%m-%Y %H:%M:%S")

    run_log_dir = os.path.join(base_log_dir, start_time_fmt)

    writer = SummaryWriter(log_dir=run_log_dir, comment = comment)
    
    return writer