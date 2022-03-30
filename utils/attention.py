from json import decoder
import torch
import torch.nn as nn
import torch.nn.functional as F

class BadhanauAttention(nn.Module):
    """
    Badhanau Attention Model
    
    """
    
    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int, attention_dim: int):
        """

        Args:
            encoder_hidden_dim (int): the hidden dimension size of the encoder model
            decoder_hidden_dim (int): the hidden dimension size of the decoder model
            attention_dim (int): the hidden dimesion size for the attention model
        """
        
        super(BadhanauAttention, self).__init__()

        self.U_a = nn.Linear(in_features = encoder_hidden_size, out_features = attention_dim, bias = False)

        self.W_a = nn.Linear(in_features = decoder_hidden_size, out_features = attention_dim, bias = False)
                
        self.v_t = nn.Linear(in_features = attention_dim, out_features = 1, bias = False)
        
        self.ffn = nn.Linear(in_features = encoder_hidden_size + decoder_hidden_size, out_features = decoder_hidden_size, bias = False)

        self.tanh_layer = nn.Tanh()
        
    def forward(self, encoder_hidden_states: torch.Tensor, decoder_hidden_state: torch.Tensor) -> torch.Tensor:
        """Returns the alignment scores for the encoder_hidden_states and the decoder_hidden_state.

        Args:
            encoder_hidden_states (torch.Tensor): the hidden states from the encoder
            decoder_hidden_state (torch.Tensor): the hidden state from the decoder

        Returns:
            torch.Tensor: the alignment score of the encoder_hidden_states.
        """
        
        print(f"""encoder_hidden_states.shape: {encoder_hidden_states.shape},
              decoder_hidden_state.shape: {decoder_hidden_state.shape}""")
                
        
        U_a_out = self.U_a(encoder_hidden_states)        
        W_a_out = self.W_a(decoder_hidden_state)
        
        ## e_ij is also called energy?    
        e_ij = self.tanh_layer(W_a_out + U_a_out)

        e_ij = self.v_t(e_ij)

        ##Get the alignment scores for encoder_outputs and t-1 decoder_hidden_state
        alignment_scores = F.softmax(e_ij, dim = 1)
        
        return alignment_scores
    
##source: https://github.com/bentrevett/pytorch-seq2seq, used for validation
    
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, encoder_outputs, hidden):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention= [batch size, src len]
        
        return F.softmax(attention, dim=1)
    
    
class DotProductAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
    def forward(self, encoder_outputs, hidden):
        
        scores = torch.bmm(encoder_outputs, hidden.permute(0, 2, 1))
        
        return F.softmax(scores, dim=1)