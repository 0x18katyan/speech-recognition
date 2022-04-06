from sklearn import utils
import torch
import torch.nn as nn

from typing import Tuple
from src.model.attention import *

class RNNDecoder(nn.Module):
    
    def __init__(self,
                 encoder_rnn_hidden_size: int = 256,
                 encoder_rnn_bidirectional: bool = True, 
                 decoder_embedding_size: int = 300,
                 decoder_hidden_size: int = 256,
                 decoder_attn_size: int = 84,
                 decoder_num_layers: int = 1,
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
        
        directions = 2 if encoder_rnn_bidirectional else 1
        
        self.emb = nn.Embedding(num_embeddings = vocab_size, embedding_dim = decoder_embedding_size, padding_idx = padding_idx)
        
        self.rnn = nn.GRU( input_size = decoder_embedding_size, 
                          hidden_size = decoder_hidden_size, 
                          num_layers = decoder_num_layers, 
                          batch_first = batch_first, 
                          dropout = dropout)
        
        # self.attention_layer = BadhanauAttention(encoder_hidden_size = encoder_rnn_hidden_size, 
        #                                          decoder_hidden_size = decoder_hidden_size, 
        #                                          attention_dim = decoder_attn_size)
        
        # self.attention_layer = Attention(enc_hid_dim=encoder_input_dim, dec_hid_dim=decoder_hidden_size)
        self.attention_layer = DotProductAttention(enc_hid_dim=encoder_rnn_hidden_size, 
                                                   dec_hid_dim=decoder_hidden_size)
        
        self.projector = nn.Linear(in_features = (encoder_rnn_hidden_size * directions) + decoder_hidden_size, out_features = decoder_hidden_size)
              
        self.predictor = nn.Linear(in_features = decoder_hidden_size, out_features = vocab_size)
        self.tanh_layer = nn.Tanh()
        self.softmax_layer = nn.LogSoftmax(dim = -1)
        
        self.dropout = nn.Dropout(dropout)
        

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
        x_embedded = F.relu(self.dropout(x_embedded)) ##dropout then activate
        
        ##Run through the RNN
        try:
            outputs, hidden_state = self.rnn(x_embedded, hidden_state)
        except:
            print(f"x.shape: {x.shape}, x_embedded.shape: {x_embedded.shape}, hidden_state.shape: {hidden_state.shape}")
            outputs, hidden_state = self.rnn(x_embedded, hidden_state)
        
        attention_weights = self.attention_layer(encoder_outputs, hidden_state)
        
        ##Apply the alignment scores to the encoder_hidden_states and calculate the weighted average
        ##context_vector shape == [batch_size, 1 , encoder_hidden_size]
        context_vector = torch.bmm(attention_weights.permute(0, 2, 1), encoder_outputs) ## context vector is how luong defines this
                
        ## changing hidden_state from [seq, batch, hidden] to [batch, seq, hidden]
        concat = torch.concat([hidden_state.permute(1, 0, 2), context_vector], dim = -1)
        
        concat_proj = self.tanh_layer(self.projector(concat))
        
        outputs = self.predictor(concat_proj)
        outputs = self.softmax_layer(outputs)
        
        return outputs, hidden_state
