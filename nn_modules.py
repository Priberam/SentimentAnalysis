import torch
cuda_available = torch.cuda.is_available()
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            random = torch.randn(din.size()).cuda() if cuda_available else torch.randn(din.size())
            return din + torch.autograd.Variable(random * self.stddev)
        return din

# model inspired from http://aclweb.org/anthology/S17-2126
# Adapted from the original at https://github.com/bentrevett/pytorch-sentiment-analysis
# as well as https://github.com/cbaziotis/datastories-semeval2017-task4
class RNN(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embedding_dim, 
                 hidden_dim, 
                 output_dim, 
                 n_layers, 
                 bidirectional, 
                 attention,
                 dropout_final,
                 dropout_attention,
                 dropout_words,
                 dropout_rnn,
                 dropout_rnn_U,
                 noise,
                 final_layer):
        super().__init__()        

        self.attention=attention
        self.dropout_final_v     = dropout_final
        self.dropout_attention_v = dropout_attention
        self.dropout_words_v     = dropout_words
        self.dropout_rnn_v       = dropout_rnn
        self.dropout_rnn_U_v     = dropout_rnn_U
        self.noise               = noise
        self.final_layer         = final_layer

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if noise>0:
            self.noisylayer = GaussianNoise(noise)       
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=self.dropout_rnn_U_v) 
        
        if attention:
            self.attention_tanh = nn.Tanh() 
            self.attention_weights = nn.Linear(hidden_dim*2, 1)
            self.attention_softmax =  nn.Softmax(dim=0)

        self.fc = nn.Linear(hidden_dim*2, output_dim)

        self.dropout_words       = nn.Dropout(self.dropout_words_v)
        self.dropout_rnn         = nn.Dropout(self.dropout_rnn_v)
        self.dropout_attention   = nn.Dropout(self.dropout_attention_v)
        self.dropout_final       = nn.Dropout(self.dropout_final_v)
        
    def forward(self, x):       
        _x = x.cuda() if cuda_available else x
        _embedded = self.embedding(_x)       

        if self.noise>0:
            _embedded_with_noise = self.noisylayer(_embedded)
        
        if self.dropout_words_v > 0:
            embedded = self.dropout_words(_embedded_with_noise if self.noise>0 else _embedded)
        else:
            embedded = _embedded_with_noise if self.noise>0 else _embedded
        
        output, (hidden, cell) = self.rnn(embedded)    
        if self.dropout_rnn_v > 0:
            if self.attention == "simple":
                output = self.dropout_rnn(output)   
            else:
                hidden = self.dropout_rnn(hidden)


        if self.attention == "simple":
            transpweights_dot_output = self.attention_weights(output) #shape    (transpweights_dot_output) = (sentence_length, batch, 1)
            tanhed_m = self.attention_tanh(transpweights_dot_output) # shape(m) =   (sentence_length, batch, dim)
            alpha = self.attention_softmax (tanhed_m)   # shape(alpha) =    (sentence_length, batch, 1)
            r = torch.bmm(output.permute(1,2,0), alpha.permute(1,0,2)) #shape(r) =  (batch, dim, 1)
            r = r.squeeze(2)
            if self.dropout_attention_v > 0:
                r = self.dropout_attention(r)
        else:            
            rnn_hiddens = (hidden[-2,:,:], hidden[-1,:,:])
            concatenation = torch.cat(rnn_hiddens, dim=1)
            r = concatenation       
        
        if self.final_layer:
            if self.dropout_final_v > 0:
                r = self.dropout_final(r)

        return F.log_softmax(self.fc(r.squeeze(0)), dim=1)        

