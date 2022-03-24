from torchinfo import summary

import torch

def get_summary(model, dataloader = None, data = None):
    
    """
    A very very dirty way to get model summary. TODO: Clean up a bit.
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for idx, batch in enumerate(dataloader):

        sentences = batch['sentences'].to(device)
        sentence_lengths = batch['sentence_lengths'].to(device, dtype= torch.int32)

        melspecs = batch['melspecs'].to(device)
        melspecs_lengths = batch['melspecs_lengths'].to(device, dtype= torch.int32)

        melspecs = torch.transpose(melspecs, -1, -2) ## Changing to (batch, channel, time, n_mels) from (batch, channel, n_mels, time)

        break
    
    return summary(encoder, 
                   input_data = [melspecs, melspecs_lengths],
                   device=device, 
                   dtypes = [torch.float32, torch.float32], 
                   col_width=16,
                   col_names=["output_size", "num_params"],
                   mode = 'eval', 
                   verbose = 0)