import torch
from torch import nn
import transformers
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_size, SENTENCE_LIMIT_SIZE, filter_num=100, kernel_lst=(3,4,5), dropout=0.5):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
                             nn.Sequential(nn.Conv2d(1, filter_num, (kernel, embedding_dim)),       
                                            nn.ReLU(),
                                            nn.MaxPool2d((SENTENCE_LIMIT_SIZE - kernel + 1, 1)))
                              for kernel in kernel_lst])
        self.fc = nn.Linear(filter_num * len(kernel_lst), output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)            # (batch, word_num, embedding_dim)
        x = x.unsqueeze(1)               # (batch, channel_num, word_num, embedding_dim)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)      
        out = out.view(x.size(0), -1)    
        out = self.dropout(out)
        logit = self.fc(out)            

        return logit


class EnsembleCNN(nn.Module):
    def __init__(self, vocab1, vocab2, vocab3, emb_size1, emb_size2, emb_size3, NUM_LABELS, SENTENCE_LIMIT_SIZE):
        super(EnsembleCNN, self).__init__()
        self.model1 = TextCNN(len(vocab1), emb_size1, NUM_LABELS, SENTENCE_LIMIT_SIZE)
        self.model2 = TextCNN(len(vocab2), emb_size2, NUM_LABELS, SENTENCE_LIMIT_SIZE)
        self.model3 = TextCNN(len(vocab3), emb_size3, NUM_LABELS, SENTENCE_LIMIT_SIZE)
        self.weight = nn.Parameter(torch.Tensor([1/3, 1/3, 1/3]))

    def forward(self, x1, x2, x3):

        out1 = self.model1(x1)
        out2 = self.model2(x2)
        out3 = self.model3(x3)

        w = F.softmax(self.weight, dim=-1)
        pred_final = w[0] * out1 + w[1] * out2 + w[2] * out3
        return out1, out2, out3, pred_final


