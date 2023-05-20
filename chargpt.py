'''
character level gpt 
1) code with minimum description length
2) think fast-and-slow

'''
import torch.nn as nn
import torch 
import numpy as np
import string 

vocab = list(string.printable)
n_embed = 256
n_vocab = len(vocab)

def encode(text):
    text_vector = list(text)
    _ = [] 
    for char in text_vector:
        enc = np.zeros(n_vocab)
        enc[vocab.index(char)] = 1.0
        _.append(enc)
    return(np.vstack(_))

def decode(encoding):
    text = ''
    for enc in encoding:
        text = text + vocab[np.argmax(enc)]
    return(text)

# 30-min (done)

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(n_embed,n_embed)
        self.k = nn.Linear(n_embed,n_embed)
        self.v = nn.Linear(n_embed,n_embed)
        self.sf = nn.Softmax()
    def forward(self,x):
        # get q,k,v and return softmax(qxkT)xv
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return(self.sf(q@k.T)@v)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(n_embed,n_embed)
        self.rl = nn.ReLU()
        self.l2 = nn.Linear(n_embed,n_embed)
    def forward(self,x):
        return(self.l2(self.rl(self.l1(x))))

# one-hour

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()
        self.attn = CausalSelfAttention()
    def forward(self,x):
        return(self.mlp(self.attn(x)))

class Model(nn.Module):
    def __init__(self,n_blocks):
        super().__init__()
        self.n_blocks = n_blocks 
        self.layers = [Block() for block in range(self.n_blocks)]
        self.embed  = nn.Linear(n_vocab,n_embed)
        self.de_embed = nn.Linear(n_embed,n_vocab)
    def forward(self,x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return(self.de_embed(x))

# one-hour

'''
show a single forward pass on a random text, encode and decode it
'''

class Trainer:
    def __init__(self):
        pass

'''
make it sequence-length,batch-wise
train the model using Trainer class 
and run predictions on it
'''
# one-hour

if __name__== '__main__':
    print("is encode/decode working ? "+str("hello world"==decode(encode("hello world"))))
    model = Model(12)
    x = torch.randn(2,100,n_vocab) #Batch-size,sequence-length,embedding-dimension
    print(model(x))
    #print(decode(model(torch.Tensor(encode("testing"))).detach().numpy()))
