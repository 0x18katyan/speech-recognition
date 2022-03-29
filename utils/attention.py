import torch
import torch.nn as nn
import torch.nn.functional as F

class BadhanauAttention(nn.Module):
    """
    Badhanau Attention Model
    
    """
    
    def __init__(self, encoder_hidden_dim: int, decoder_hidden_dim: int, attention_dim: int):
        """

        Args:
            encoder_hidden_dim (int): the hidden dimension size of the encoder model
            decoder_hidden_dim (int): the hidden dimension size of the decoder model
            attention_dim (int): the hidden dimesion size for the attention model
        """
        
        super(BadhanauAttention, self).__init__()

        self.U_a = nn.Linear(in_features = encoder_hidden_dim, out_features = attention_dim, bias = False)

        self.W_a = nn.Linear(in_features = decoder_hidden_dim, out_features = attention_dim, bias = False)
                
        self.v_t = nn.Linear(in_features = attention_dim, out_features = 1, bias = False)
        
        self.ffn = nn.Linear(in_features = encoder_hidden_dim + decoder_hidden_dim, out_features = decoder_hidden_dim, bias = False)

        self.tanh_layer = nn.Tanh()
        
    def forward(self, encoder_hidden_states: torch.Tensor, decoder_hidden_state: torch.Tensor) -> torch.Tensor:
        """Returns the alignment scores for the encoder_hidden_states and the decoder_hidden_state.

        Args:
            encoder_hidden_states (torch.Tensor): the hidden states from the encoder
            decoder_hidden_state (torch.Tensor): the hidden state from the decoder

        Returns:
            torch.Tensor: the alignment score of the encoder_hidden_states.
        """
        
        ##score is the e_ij
        
        # print(f"encoder_hidden_states.shape: {encoder_hidden_states.shape}")
        
        # print(f"decoder_hidden_state.shape: {decoder_hidden_state.shape}")
        
        U_a_out = self.U_a(encoder_hidden_states)        
        W_a_out = self.W_a(decoder_hidden_state)
                
        e_ij = self.tanh_layer(W_a_out + U_a_out)
                
        e_ij = self.v_t(e_ij)

        ##Get the alignment scores for encoder_outputs and t-1 decoder_hidden_state
        alignment_scores = F.softmax(e_ij, dim = 1)
        
        ##Apply the alignment scores to the encoder_hidden_states and calculate the weighted average
        ##context_vector shape == [batch_size, 1 , encoder_hidden_states]
        
        context_vector = torch.bmm(alignment_scores.permute(0, 2, 1), encoder_hidden_states) ## context vector is how luong defines this
        
        # print(f"context_vector.shape: {context_vector.shape}, decoder_hidden_state.shape: {decoder_hidden_state.shape}")
        
        output = self.tanh_layer(self.ffn(torch.concat([decoder_hidden_state, context_vector], dim = -1)))

        return output, context_vector