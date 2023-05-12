"""
a smol gpt model

"""
import numpy as np 
import string 
import torch.nn as nn


max_length = 64
alphabet = list(string.printable)
token_dict = {}

def adjust_length(charsequence):
    charsequence_ = ''.join(list(charsequence)[:64]) if (len(charsequence)>64) else charsequence+' '*(64-len(charsequence))
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
        pass
    def forward(self):
        pass

if __name__ == '__main__':
    print(decode(encode(adjust_length("hello world"))))
    #input_sequence = "hello world"
    #smol = Smol()
    #print(smol.forward(input_sequence))

