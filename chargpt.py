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
n_seq_length = 128

def pad(text):
    if(len(text)<n_seq_length):
        text = text+(n_seq_length-len(text))*' '
    else:
        text = text[:n_seq_length]
    return(text)

def encode(text):

    text_vector = list(pad(text))
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
        return(self.sf((q@k.transpose(-2,-1)))@v)


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
        print(x.shape)
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return(self.de_embed(x))

class CharDataset(torch.utils.data.Dataset):
    """
    Emits batches of characters
    """


    def __init__(self, data,block_size):

        self.block_size = block_size
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size
        self.data = data


    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
    
class trainer:
    def __init__(self):
        pass

if __name__== '__main__':

    dataset = CharDataset(open("input.txt",'r').read(),n_seq_length)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
    model = Model(12)
    iter_num = 0
    data_iter = iter(train_loader)
    loss = torch.nn.NLLLoss()
    while True:

        # fetch the next batch (x, y) and re-init iterator if needed
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
            
        #batch = [t.to('cuda') for t in batch]
        batch = [t for t in batch]
        x, y = batch

        # forward the model
        logits = loss(model(x),y)

        # backprop and update the parameters
        model.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
        optimizer.step()

        trigger_callbacks('on_batch_end')
        iter_num += 1


