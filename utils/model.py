import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import PackedSequence

import torchaudio

from typing import Tuple

class Encoder(nn.Module):
    
    def __init__(self, 
                 encoder_input_dim: int = 80, 
                 num_heads: int = 4, 
                 ffn_dim: int = 80, 
                 num_layers: int = 4, 
                 depthwise_conv_kernel_size: int = 31, 
                 dropout: float = 0.3,
                 **args):
        
        super(Encoder, self).__init__()
        
        self.conformer = torchaudio.models.Conformer(input_dim = encoder_input_dim,
                                                     num_heads = num_heads,
                                                     ffn_dim = ffn_dim,
                                                     num_layers = num_layers,
                                                     depthwise_conv_kernel_size = depthwise_conv_kernel_size,
                                                     dropout = dropout)
                
    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        x, x_lens = self.conformer.forward(x, x_lens)
        
        return x, x_lens
    
class RNNDecoder(nn.Module):
    
    def __init__(self, 
                 decoder_input_dim: int = 80, 
                 decoder_hidden_size: int = 256, 
                 num_layers: int = 1, 
                 bidirectional: bool = False, 
                 output_dim: int = None, 
                 **args):
        
        super(RNNDecoder, self).__init__()
        
        if output_dim == None:
            raise ValueError("Please specify the output size of the vocab.")
            
        directions = 2 if bidirectional == True else 1
        
        self.model = nn.GRU(input_size = decoder_input_dim, hidden_size = decoder_hidden_size, num_layers = num_layers, batch_first = False)
        self.ffn = nn.Sequential(nn.Linear(in_features = decoder_hidden_size * directions, out_features = 1024), 
                                 nn.GLU(), 
                                 nn.Dropout(0.5), 
                                 nn.Linear(in_features = 512, out_features = output_dim))
                                
    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]: 
        """
        Hidden state is needed, either in the form of encoder_hidden_state or decoder_hidden_state
        """
        
        if hidden_state == None:
            outputs, hidden_state = self.model(x)
        
        else:
            outputs, hidden_state = self.model(x, hidden_state)
        
        if isinstance(x, PackedSequence):
            outputs, _ = pad_packed_sequence(outputs)
        
        outputs = self.ffn(outputs)
        
        return outputs, hidden_state

class Model(nn.Module):
    
    def __init__(self, encoder_input_dim: int = 80,
                encoder_num_heads: int = 4, 
                encoder_ffn_dim: int = 144, 
                encoder_num_layers: int = 16, 
                encoder_depthwise_conv_kernel_size: int = 31, 
                decoder_hidden_size:int = 80,
                decoder_num_layers: int = 2,
                bidirectional_decoder: bool = False,
                vocab_size: int = None,
                padding_idx: int = None,
                sos_token_id: int = None):
        
        super(Model, self).__init__()
        
        self.encoder = Encoder(input_dim = encoder_input_dim,
                              num_heads = encoder_num_heads,
                              ffn_dim = encoder_ffn_dim,
                              depthwise_conv_kernel_size = encoder_depthwise_conv_kernel_size)
        
        self.decoder = RNNDecoder(input_dim = encoder_input_dim,
                                  hidden_size = decoder_hidden_size,
                                  num_layers = decoder_num_layers,
                                  bidirectional = bidirectional_decoder,
                                  output_dim = vocab_size)
        
        self.sos_token_id = sos_token_id
        
    def forward(self, x: torch.Tensor, x_lens: torch.Tensor):
        
        decoded = []
        
        bsz, msl, hdz = x.shape ##batch_size, max sequence length, hidden dimension size

        encoder_outputs = self.encoder(x, x_lens)
                
        decoder_inputs = encoder_outputs
        
        ## Start with the <sos> token
        x = torch.LongTensor([self.sos_token_id]).repeat(bsz).reshape(bsz, 1).to(device)

        for t in range(msl):
            
            if t == 0:
                decoder_output, decoder_hidden_state = self.decoder(x = decoder_inputs)            
            else:
                decoder_output, decoder_hidden_state = self.decoder(x = decoder_inputs, hidden_state = decoder_hidden_state)
            
            word = F.log_softmax(decoder_output, dim = -1) ## have to do log_softmax for CTC Loss
            
            topv, topi = decoder_output.topk(1)
            
            x = topv.squeeze().detach()
            
            decoded.append(topv)
            
        return encoder_outputs, torch.stack(decoded)