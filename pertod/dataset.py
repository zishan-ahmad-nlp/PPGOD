import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
# from uttils import add_special_tokens

class GPT21024Dataset(Dataset):

    def __init__(self, root_dir, model_checkpoint, mode='test',length=None):
        self.root_dir = root_dir
        # self.tokenizer = add_special_tokens()
        # self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)

        # with open(ids_file,'r') as f:
            # if mode=='train':
            #     self.idxs = np.array(json.load(f)['train_ids'])
            # elif mode=='valid':
            #     self.idxs = np.array(json.load(f)['valid_ids'])
            # elif mode=='test':
            #     self.idxs = np.array(json.load(f)['test_ids'])

            # self.idxs = self.idxs -min(self.idxs)
        lines = None
        with open(f'{root_dir}/{mode}.lex') as f:
            lines = f.readlines()
        self.data = lines        
        self.idxs = range(len(lines))
        self.len = len(lines)
        # self.test_dir = test_dir 
        # self.test_idxs = os.listdir(test_dir)
        # self.mode = mode
        # if len == None:
        #     self.len = len(self.idxs)
        # else:
        #     self.len = length

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        line = self.data[idx]
        # text = self.tokenizer.encode(self.tokenizer.pad_token)*1024
        context = line.split('<|response|>')[0]
        content = self.tokenizer.encode(context)
        # print(content)
        return content