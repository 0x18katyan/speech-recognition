import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from torchaudio.models import Conformer

from typing import Tuple

class Encoder(nn.Module):
    
    def __init__(self, 
                 encoder_input_size: int = 80,
                 decoder_hidden_size: int = 256,
                 conformer_num_heads: int = 4, 
                 conformer_ffn_size: int = 80, 
                 conformer_num_layers: int = 4, 
                 conformer_conv_kernel_size: int = 31, 
                 encoder_rnn_hidden_size: int = 256,
                 encoder_rnn_num_layers: int = 1,
                 encoder_rnn_bidirectional: bool = True,
                 batch_first: bool = True,
                 dropout: float = 0.3):
            
        super(Encoder, self).__init__()
        
        directions = 2 if encoder_rnn_bidirectional else 1
        
        self.batch_first = batch_first
        
        self.conformer = Conformer(input_dim = encoder_input_size,
                                                     num_heads = conformer_num_heads,
                                                     ffn_dim = conformer_ffn_size,
                                                     num_layers = conformer_num_layers,
                                                     depthwise_conv_kernel_size = conformer_conv_kernel_size,
                                                     dropout = dropout)
        
        self.rnn = nn.GRU(input_size = encoder_input_size, 
                          hidden_size = encoder_rnn_hidden_size, 
                          num_layers = encoder_rnn_num_layers, 
                          batch_first = batch_first, 
                          dropout = dropout, 
                          bidirectional = encoder_rnn_bidirectional)
        
        self.proj_fc = nn.Linear(encoder_rnn_hidden_size * directions, decoder_hidden_size)
        self.tanh_layer = nn.Tanh()
        
                
    def forward(self, x: torch.Tensor, x_lens: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes the incoming spectrograms or waveforms using Conformer and an RNN.

        Args:
            x (torch.Tensor): feature inputs, should be 3D.
            x_lens (torch.Tensor): feature input lengths, used for identifying lengths before padding. None if batch_size = 1.

        Returns: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
            
            outputs: consist of the hidden states at each timestep, is a packed_sequence
            output_lengths: the length of outputs before padding
            hidden: is the last hidden state of the encoder rnn
        """
        if x_lens == None:
            outputs, output_lens = self.conformer.forward(x)
        else:    
            outputs, output_lens = self.conformer.forward(x, x_lens)
        
        outputs = pack_padded_sequence(outputs, 
                                       lengths = output_lens.to(device = 'cpu', dtype=torch.int64), 
                                       batch_first = self.batch_first, 
                                       enforce_sorted = False)
        
        outputs, hidden = self.rnn(outputs)
                
        outputs, output_lens = pad_packed_sequence(outputs,
                                                   batch_first = self.batch_first)
        
        concat_last_layer = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim = 1) ## Since, it is bidirectional, -2 and -1 are forward and backward pass of the last layer
        hidden = self.tanh_layer(self.proj_fc(concat_last_layer)) # project into decoder hidden_size
        hidden = hidden.unsqueeze(0)
        
        return outputs, output_lens, hidden
