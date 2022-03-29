from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence

import torchaudio

from typing import Tuple

from utils.attention import BadhanauAttention

class Encoder(nn.Module):
    
    def __init__(self, 
                 encoder_input_size: int = 80,
                 conformer_num_heads: int = 4, 
                 conformer_ffn_size: int = 80, 
                 conformer_num_layers: int = 4, 
                 conformer_conv_kernel_size: int = 31, 
                 encoder_rnn_hidden_size: int = 256,
                 encoder_rnn_num_layers: int = 1,
                 encoder_rnn_bidirectional: bool = False,
                 batch_first: bool = True,
                 dropout: float = 0.3):
            
        super(Encoder, self).__init__()
        
        self.batch_first = batch_first
        
        self.conformer = torchaudio.models.Conformer(input_dim = encoder_input_size,
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
        
        packed_encoder_outputs = pack_padded_sequence(outputs, 
                                                      lengths = output_lens.to(device = 'cpu', dtype=torch.int64), 
                                                      batch_first = self.batch_first, 
                                                      enforce_sorted = False)
        
        outputs, hidden = self.rnn(x)
        
        return outputs, output_lens, hidden
    
class RNNDecoder(nn.Module):
    
    def __init__(self,
                 encoder_rnn_hidden_size: int = 256,
                 decoder_embedding_size: int = 300,
                 decoder_hidden_size: int = 256,
                 decoder_attn_size: int = 84,
                 decoder_num_layers: int = 1,
                 decoder_bidirectional: bool = False, 
                 vocab_size: int = None,
                 batch_first: bool = True,
                 dropout: float = 0.3,
                 device: str = "cpu",
                 padding_idx: int = 0):
        
        super(RNNDecoder, self).__init__()
        
        self.decoder_hidden_dim = decoder_hidden_size
        self.device = device
        
        if vocab_size == None:
            raise ValueError("Please specify the output size of the vocab.")
        
        self.batch_first = batch_first
        
        directions = 2 if decoder_bidirectional == True else 1
        
        self.emb = nn.Embedding(num_embeddings = vocab_size, embedding_dim = decoder_embedding_size, padding_idx = padding_idx)
        
        self.rnn = nn.GRU(input_size = decoder_hidden_size + decoder_embedding_size,
                          
                          hidden_size = decoder_hidden_size, 
                          num_layers = decoder_num_layers, 
                          batch_first = batch_first, 
                          dropout = dropout)
        
        self.attention_layer = BadhanauAttention(encoder_hidden_dim = encoder_rnn_hidden_size, 
                                                 decoder_hidden_dim = decoder_hidden_size, 
                                                 attention_dim = decoder_attn_size)
        
        # self.attention_layer = Attention(enc_hid_dim=encoder_input_dim, dec_hid_dim=decoder_hidden_size)
        
        self.predictor = nn.Linear(in_features = decoder_hidden_size * directions, out_features = vocab_size)
        self.tanh_layer = nn.Tanh()

                                
    def forward(self, x: torch.LongTensor, encoder_outputs: torch.Tensor, hidden_state: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]: 
        """ Conducts a forward pass through the rnn decoder network.
        
        Args:
            x (torch.LongTensor): indices for the input, this is converted into embeddings.
            encoder_outputs (torch.Tensor): outputs from the encoder.
            hidden_state (torch.Tensor, optional): Hidden state is needed, either in the form of encoder_hidden_state or decoder_hidden_state. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: returns the outputs and the current hidden state of the rnn.
        """
        
        ## Add Batch dimension for unbatched inputs

        x_embedded = self.emb(x)
        x_embedded = F.gelu(x_embedded)
                
        output, context_vector = self.attention_layer(encoder_outputs, hidden_state.permute(1, 0, 2))
                        
        combined_input = torch.cat([output, x_embedded], dim = -1)
        
        outputs, hidden_state = self.rnn(combined_input, hidden_state)
        
        if isinstance(x_embedded, PackedSequence):
            outputs, _ = pad_packed_sequence(outputs)
        
        if self.batch_first == True:
            outputs = outputs.transpose(0, 1)
            hidden_state = hidden_state.transpose(0, 1)
        
        outputs = F.log_softmax(self.predictor(outputs).squeeze(0), dim = -1)
        
        return outputs, hidden_state

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
                decoder_bidirectional: int = False,
                decoder_attn_size: int = 84,
                dropout: float = 0.3,
                padding_idx: int = 4,
                sos_token_id: int = 0,
                vocab_size: int = 1000,
                batch_first: bool = True,
                device: str =  "cpu",
                *args,
                **kwargs):
        
        super(Model, self).__init__(*args, **kwargs)
                
        self.encoder_rnn_directions = 1 if encoder_rnn_bidirectional == False else 2
        self.decoder_rnn_directions = 1 if decoder_bidirectional == False else 2
        
        self.encoder = Encoder(encoder_input_size = encoder_input_size,
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
                                  decoder_embedding_size = decoder_embedding_size,
                                  decoder_hidden_size = decoder_hidden_size,
                                  decoder_num_layers = decoder_num_layers,
                                  decoder_bidirectional = decoder_bidirectional,
                                  decoder_attn_size = decoder_attn_size,
                                  vocab_size = vocab_size,
                                  batch_first = batch_first,
                                  dropout = dropout,
                                  device = device,
                                  padding_idx = padding_idx
                                  )
                
        self.sos_token_id = sos_token_id
        self.batch_first = batch_first
        self.vocab_size = vocab_size
        self.device = device
        
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
        
        decoder_inputs = encoder_hidden_state
        
        max_tgt_len = y.shape[-1] - 1 ## Minus the sos tokens

        ## predicted_tensor shape [max_seq_len, batch_size, vocab_size]
        predicted_tensor = torch.zeros(max_tgt_len, y.shape[0], self.vocab_size, device = self.device)
                
        decoder_inputs = y[:, 0].unsqueeze(1) ## for batch first, need to test
        
        for i in range(0, max_tgt_len): 

            decoder_outputs, decoder_hidden_state = self.decoder(decoder_inputs, encoder_outputs, decoder_hidden_state)
            decoder_hidden_state = decoder_hidden_state.permute(1, 0, 2)

            topv, topi = torch.topk(decoder_outputs, k = 1, dim = -1) ## get the topv values and topindices, don't need right now might need for beam search
            
            decoder_inputs = topi.squeeze(0) ## With no teacher forcing, might need to add code for teacher forcing
            
            predicted_tensor[i] = decoder_outputs
        
        ##predicted_tensor is changed from [seq, batch, vocab_size] to [batch, seq , vocab_size]
        predicted_tensor = predicted_tensor.permute(1,0,2)
        return predicted_tensor