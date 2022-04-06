from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from src.model.attention import *

from src.utils.language_model import get_model

from src.model.encoder import Encoder
from src.model.decoder import RNNDecoder    

class Model(nn.Module):
    
    def __init__(self, 
                encoder_input_size: int = 80,
                conformer_num_heads: int = 4,
                conformer_ffn_size: int = 144,
                conformer_num_layers: int = 16,
                conformer_conv_kernel_size: int = 31,
                encoder_rnn_hidden_size: int = 256,
                encoder_rnn_num_layers: int = 1,
                encoder_rnn_bidirectional: int = False,
                decoder_embedding_size: int = 300,
                decoder_hidden_size: int = 256,
                decoder_num_layers: int = 1,
                decoder_attn_size: int = 84,
                dropout: float = 0.3,
                padding_idx: int = 6,
                sos_token_id: int = 1,
                eos_token_id: int = 2,
                vocab_size: int = 1000,
                batch_first: bool = True,
                device: str =  "cpu",
                *args,
                **kwargs):
        
        super(Model, self).__init__(*args, **kwargs)
                
        self.encoder_rnn_directions = 1 if encoder_rnn_bidirectional == False else 2
        
        self.encoder = Encoder(encoder_input_size = encoder_input_size,
                               decoder_hidden_size = decoder_hidden_size,
                               conformer_num_heads=conformer_num_heads,
                               conformer_ffn_size = conformer_ffn_size,
                               conformer_num_layers=conformer_num_layers,
                               conformer_conv_kernel_size=conformer_conv_kernel_size,
                               encoder_rnn_num_layers=encoder_rnn_num_layers,
                               encoder_rnn_hidden_size = encoder_rnn_hidden_size,
                               batch_first = batch_first,
                               dropout = dropout,
                               )
        
        self.decoder = RNNDecoder(encoder_rnn_hidden_size = encoder_rnn_hidden_size,
                                  encoder_rnn_bidirectional = encoder_rnn_bidirectional,
                                  decoder_embedding_size = decoder_embedding_size,
                                  decoder_hidden_size = decoder_hidden_size,
                                  decoder_num_layers = decoder_num_layers,
                                  decoder_attn_size = decoder_attn_size,
                                  vocab_size = vocab_size,
                                  batch_first = batch_first,
                                  dropout = dropout,
                                  device = device,
                                  padding_idx = padding_idx
                                  )
        self.padding_idx = padding_idx
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.batch_first = batch_first
        self.vocab_size = vocab_size
        self.device = device
        
        self.teacher_forcing = 0.75
        
        # self.LM = get_model("data/model/custom-lm/checkpoint-340000/")

        
    def forward(self, x: torch.Tensor, x_lens: torch.Tensor, y: torch.Tensor, y_lens: torch.Tensor) -> torch.Tensor:
        """Forward pass used during Training.

        Args:
            x (torch.Tensor): the source waveform or spectrogram inputs
            x_lens (torch.Tensor): the lengths of the source inputs before padding
            y (torch.Tensor): the target sentences to be predicted from the inputs
            y_lens (torch.Tensor): the lengths of the target sentences 

        Returns:
            torch.Tensor: the predicted tensor of shape = y.shape
        """
                
        encoder_outputs, encoder_output_lengths, encoder_hidden_state = self.encoder(x, x_lens)
        
        ## First decoder hidden state is the encoder hidden state
        decoder_hidden_state = encoder_hidden_state
        
        ##encoder_outputs: [seq_len, batch_size, hidden_dim]
        if self.batch_first:
            bsz, msl, hdz = x.shape ##batch_size, max sequence length, hidden dimension size
        else:
            msl, bsz, hdz = x.shape

        ## Replace all the eos tokens in the decoder inputs to pad since we do not want to input eos tokens into the model
        y[y== self.eos_token_id] = self.padding_idx 
        
        ## Select the sos token as the first input
        decoder_inputs = y[:, 0].unsqueeze(-1) ## the sos token

        max_tgt_len = y.shape[-1] - 1 ## Minus the sos tokens

        ## predicted_tensor shape [max_seq_len, batch_size, vocab_size]
        predicted_tensor = torch.zeros(max_tgt_len, y.shape[0], self.vocab_size, device = self.device)        
        

        for i in range(0, max_tgt_len): 
            decoder_outputs, decoder_hidden_state = self.decoder(decoder_inputs, encoder_outputs, decoder_hidden_state)
            decoder_hidden_state = decoder_hidden_state ## decoder hidden_state is [1, batch_size, hidden_size]

            topv, topi = torch.topk(decoder_outputs, k = 1, dim = -1) ## get the topv values and topindices, don't need right now might need for beam search
            
            if torch.rand(1) <= self.teacher_forcing and self.training:
                decoder_inputs = y[:, i + 1] ##because at i, the target is i+1
                decoder_inputs = decoder_inputs.unsqueeze(-1) ## Convert from [batch] to [batch, 1]
                
            else:
                
                decoder_inputs = topi.squeeze(1)
            
            predicted_tensor[i] = decoder_outputs.squeeze(1)
        
        ##predicted_tensor is changed from [seq, batch, vocab_size] to [batch, seq , vocab_size]
        predicted_tensor = predicted_tensor.permute(1,0,2)
        return predicted_tensor