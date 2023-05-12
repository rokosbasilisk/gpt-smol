"""
a smol gpt model

"""
import numpy as np 
import string 
import torch.nn as nn
import torch as th

alphabet = list(string.printable)
vector_dim = len(alphabet)
weight_dim = vector_dim # since no linear-un-embedding layers!
max_length = 100 # since the above reason
token_dict = {}

def adjust_length(charsequence):
    charsequence_ = ''.join(list(charsequence)[:max_length]) if (len(charsequence)>max_length) else charsequence+' '*(max_length-len(charsequence))
    return(charsequence_)

for letter_idx in range(len(alphabet)):
    _ = np.zeros(len(alphabet))
    _[letter_idx] = 1.0
    token_dict[alphabet[letter_idx]] = _

def encode(charsequence):
    _ = list(charsequence)
    vectorsequence = []
    for letter in _:
        vectorsequence.append(token_dict[letter])
    return(np.vstack(vectorsequence))

def decode(vectorsequence):
    charsequence = []
    for vector in vectorsequence:
        charsequence.append(list(token_dict.keys())[int(np.dot(np.ceil(vector),np.arange(len(vector))))])
    return(''.join(charsequence))


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_w = nn.Linear(vector_dim,weight_dim, bias=True)
        self.key_w = nn.Linear(vector_dim,weight_dim, bias=True)
        self.value_w = nn.Linear(vector_dim,weight_dim, bias=True)
        self.softmax = nn.Softmax()
    
    def forward(self,char_embeddings):
        char_embeddings = th.Tensor(char_embeddings) 
        query = self.query_w(char_embeddings)
        key = self.key_w(char_embeddings)
        v = self.value_w(char_embeddings)
        qk = query.T@key
        qkv = self.softmax(qk)@v.T
        return(qkv)

class Transformer:
    def __init__(self,n_layers=8):
        self.n_layers = n_layers
        self.layers = [Attention() for layer_idx in range(self.n_layers)]
    def forward(self,emb):
        idx =  0
        while(idx<self.n_layers):
            emb = self.layers[idx](emb)
            idx += 1 
        return(emb)

if __name__ == '__main__':
    trx = Transformer()
    for i in range(100):
        try:
            print(decode(np.array(trx.forward(encode(adjust_length("hello world"))).detach().numpy())))
        except:
            print("dimension error haha!")
