import torch
import torch.nn as nn
import torch.nn.functional as F

class BadhanauAttention(nn.Module):
        
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, attention_dim):
        super(BadhanauAttention, self).__init__()
    
        self.W_a = nn.Linear(in_features = decoder_hidden_dim, out_features = attention_dim)
        self.U_a = nn.Linear(in_features = encoder_hidden_dim, out_features = attention_dim)
                
        self.v_t = nn.Linear(in_features = attention_dim, out_features = 1)
    
    def forward(self, encoder_hidden_states, decoder_hidden_state):
        
        ##score is the e_ij
        
        W_a_out = self.W_a(decoder_hidden_state)
        U_a_out = self.U_a(encoder_hidden_states)
        
        e_ij = F.tanh(W_a_out + U_a_out)
        e_ij = self.v_t(e_ij).transpose(1,2)
    
        return e_ij
