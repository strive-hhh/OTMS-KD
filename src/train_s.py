# Train the student model
import time
import os
from tqdm import tqdm
import logging
import random 
import numpy as np
import torch
from torch import nn, optim
from torchtext.legacy import data as tdata
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize 

import utils
import parser
from data_loader import data_iter
from models import EnsembleCNN
from transformers import BertConfig, BertForSequenceClassification


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train(model, train_iter1, train_iter2, train_iter3, teacher_outputs, optimizer, criterion_T):
    avg_acc, avg_loss = [], []
    model.train()      

    with tqdm(total=len(train_iter1)) as t:
        for i, batch in enumerate(zip(train_iter1, train_iter2, train_iter3)):
            text1, text2, text3, label = batch[0].text.cuda(), batch[1].text.cuda(), batch[2].text.cuda(), batch[2].label
        
            out0, out1, out2, pred_final = model(text1, text2, text3)
            
            loss = criterion_T(out0, teacher_outputs[i]) + criterion_T(out1, teacher_outputs[i]) + criterion_T(out2, teacher_outputs[i])
            loss += criterion_T(pred_final, teacher_outputs[i])              

            acc = utils.binary_acc(torch.argmax(pred_final.cpu(), dim=1), label)   
            avg_acc.append(acc)
            avg_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.update()

    avg_acc = np.array(avg_acc).mean()
    avg_loss = np.array(avg_loss).mean()
    train_metrics = {'train_acc': avg_acc,
                     'train_loss': avg_loss
                     }
    logging.info(train_metrics)
    return avg_acc, avg_loss



def evaluate(model, iter1, iter2, iter3):
    avg_acc = []
    model.eval()        

    y_pred, y_true = [], []

    with torch.no_grad():
        with tqdm(total=len(iter1)) as t:
            for batch in zip(iter1, iter2, iter3):
                text1, text2, text3, label = batch[0].text.cuda(), batch[1].text.cuda(), batch[2].text.cuda(), batch[2].label
                _, _, _, pred_final = model(text1, text2, text3)

                y_true.append(label) 
                y_pred.append(torch.argmax(pred_final.cpu(), dim=1)) 

                acc = utils.binary_acc(torch.argmax(pred_final.cpu(), dim=1), label)
                avg_acc.append(acc)
                t.update()

    avg_acc = np.array(avg_acc).mean()
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    test_metrics = {'test_acc': avg_acc}
    logging.info(test_metrics)
    return avg_acc, y_pred, y_true


def get_teacher_outputs(args):
    teacher_outputs = []
    idx, lines = 0, []
    with open(os.path.join(args.data_dir, args.testset_dir[:-3] + 'csv'), 'r') as fr:         
        for line in fr.readlines():               
            if idx == args.batch_size:
                teacher_outputs.append(torch.tensor(lines).cuda())
                idx = 0
                lines = []
            if idx < args.batch_size:
                lines.append(eval(line[:-1]))
                idx += 1
    # last batch
    if idx < args.batch_size:
        teacher_outputs.append(torch.tensor(lines).cuda())
    return teacher_outputs



if __name__ == '__main__':
    parser = parser.default_parser()
    args = parser.parse_args()
    print('args = ', args)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    checkpoint_path = os.path.join(args.checkpoint_dir, 'bert_3cnn')       
    os.makedirs(checkpoint_path, exist_ok=True)
    utils.set_logger(os.path.join(checkpoint_path, 'train.log'))

    logging.info('create dataset')
    
    text_fields = tdata.Field(tokenize=lambda x: word_tokenize(x), lower=True, fix_length=args.max_length, batch_first=True)  
    label_fields = tdata.LabelField(dtype=torch.int, sequential=False, use_vocab=False)

    train_iter1, dev_iter1, test_iter1, vocab1 = data_iter(os.path.join(args.data_dir, args.testset_dir), os.path.join(args.data_dir, "dev.csv"), os.path.join(args.data_dir, "test.csv"), 
                                                text_fields, label_fields, batch_size=args.batch_size, name='6B', embedding_dim=args.emb_size1)
    train_iter2, dev_iter2, test_iter2, vocab2 = data_iter(os.path.join(args.data_dir, args.testset_dir), os.path.join(args.data_dir, "dev.csv"), os.path.join(args.data_dir, "test.csv"), 
                                                text_fields, label_fields, batch_size=args.batch_size, name='twitter.27B', embedding_dim=args.emb_size1)               
    train_iter3, dev_iter3, test_iter3, vocab3 = data_iter(os.path.join(args.data_dir, args.testset_dir), os.path.join(args.data_dir, "dev.csv"), os.path.join(args.data_dir, "test.csv"), 
                                                text_fields, label_fields, batch_size=args.batch_size, name='42B', embedding_dim=args.emb_size2) 

    PAD_IDX = vocab1.stoi[text_fields.pad_token]         
    UNK_IDX = vocab1.stoi[text_fields.unk_token]         

    pretrained_emb1 = vocab1.vectors 
    pretrained_emb2 = vocab2.vectors 
    pretrained_emb3 = vocab3.vectors 


    logging.info('create model')
    model = EnsembleCNN(vocab1, vocab2, vocab3, args.emb_size1, args.emb_size1, args.emb_size2, args.num_labels, args.max_length)
    model.model1.embedding.weight.data.copy_(pretrained_emb1)
    model.model1.embedding.weight.data[UNK_IDX] = torch.zeros(args.emb_size1)
    model.model1.embedding.weight.data[PAD_IDX] = torch.zeros(args.emb_size1)

    model.model2.embedding.weight.data.copy_(pretrained_emb2)
    model.model2.embedding.weight.data[UNK_IDX] = torch.zeros(args.emb_size1)
    model.model2.embedding.weight.data[PAD_IDX] = torch.zeros(args.emb_size1)

    model.model3.embedding.weight.data.copy_(pretrained_emb3)
    model.model3.embedding.weight.data[UNK_IDX] = torch.zeros(args.emb_size2)
    model.model3.embedding.weight.data[PAD_IDX] = torch.zeros(args.emb_size2)

    model = model.cuda()

    num_params = (sum(p.numel() for p in model.parameters())/1000000.0)
    logging.info('Total params: %.2fM' % num_params)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay = args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    criterion_T = utils.KL_Loss(args.temperature)

    logging.info('teacher model deals unlabeled data')
    teacher_outputs = get_teacher_outputs(args)


    logging.info('start training')
    max_val_acc = 0
    for epoch in range(args.num_epochs):
        logging.info("Epoch {}/{}".format(epoch + 1, args.num_epochs))
        train_acc, train_loss = train(model, train_iter1, train_iter2, train_iter3, teacher_outputs, optimizer, criterion_T)
        val_acc, val_pred, val_true = evaluate(model, dev_iter1, dev_iter2, dev_iter3)
        
        if epoch > 0 and val_acc > max_val_acc:
            max_val_acc = val_acc
            utils.save_model(model, os.path.join(checkpoint_path, 'best.pt'))


    utils.load_model(model, os.path.join(checkpoint_path, 'best.pt'))
    test_acc, test_pred, test_true = evaluate(model, test_iter1, test_iter2, test_iter3)
    print('classification report:\n%s' % classification_report(test_true, test_pred, digits=4))   


