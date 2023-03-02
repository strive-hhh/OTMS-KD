import sys
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import BertTokenizer

from torchtext.legacy import data as tdata
from torchtext.vocab import GloVe, Vectors

csv.field_size_limit(sys.maxsize)


class BertDataset(Dataset):

    def __init__(self, data_path, max_length_doc, DIR, tokenizer = BertTokenizer):  
        super(BertDataset, self).__init__()

        texts, labels = [], []
        with open(data_path) as csv_file:
            reader = csv.reader(csv_file, quotechar='"')               
            for idx, line in enumerate(reader):
                text = ""
                for tx in line[1:]:                
                    text += tx.lower()
                    text += " "
                label = int(line[0])             

                texts.append(text)
                labels.append(label)

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer.from_pretrained(DIR)           
        self.max_length_doc = max_length_doc
    
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_id = self.tokenizer.pad_token_id

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        text = self.texts[index]

        document_encode = self.tokenizer.encode(text) 

        if len(document_encode) > self.max_length_doc:
            document_encode = document_encode[:self.max_length_doc - 1] + [self.sep_token_id]
        else:                                   
            document_encode.extend([self.pad_id] * (self.max_length_doc - len(document_encode)))

        document_encode = np.stack(arrays=document_encode, axis=0)
        seq_mask = [1 if i!=self.pad_id else 0 for i in document_encode]

        return torch.tensor(document_encode), torch.tensor(seq_mask), label




def read_data(data_path, text_field, label_field):
    fields = [('text', text_field), ('label', label_field)]
    examples = []

    with open(data_path) as csv_file:
        reader = csv.reader(csv_file, quotechar='"')           
        for idx, line in enumerate(reader):
            text = ""
            for tx in line[1:]:
                text += tx.lower()
                text += " "
            label = int(line[0])                
            examples.append(tdata.Example.fromlist([text, label], fields))

    return examples, fields
    

def data_iter(train_path, dev_path, test_path, text_field, label_field, batch_size, name, embedding_dim=50):      
    train_examples, train_fields = read_data(train_path, text_field, label_field)           
    dev_examples, dev_fields = read_data(dev_path, text_field, label_field)
    test_examples, test_fields = read_data(test_path, text_field, label_field)              

    train_dataset = tdata.Dataset(train_examples, train_fields)
    dev_dataset = tdata.Dataset(dev_examples, dev_fields)
    test_dataset = tdata.Dataset(test_examples, test_fields)
         
    text_field.build_vocab(train_dataset, vectors=GloVe(name=name, dim=embedding_dim), max_size=50000)          
   
    vocabulary = text_field.vocab
    
    train_iter = tdata.Iterator(train_dataset, batch_size=batch_size, shuffle=False, sort=False, sort_within_batch=False, repeat=False)  
    val_iter = tdata.Iterator(dev_dataset, batch_size=batch_size)           
    test_iter = tdata.Iterator(test_dataset, batch_size=batch_size)
    return train_iter, val_iter, test_iter, vocabulary

