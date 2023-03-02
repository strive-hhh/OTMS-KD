# Supervised Sentiment Classification
# Finetune the teacher model, Predict the unlabeled data;
import os
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import utils
from tqdm import tqdm
from data_loader import BertDataset
from transformers import get_linear_schedule_with_warmup
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, AdamW, BertModel

from sklearn.metrics import classification_report
import time
import logging
import parser

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(model, train_loader, optimizer, scheduler):  
    avg_loss, avg_acc = [], []
    model.train()

    for batch in tqdm(train_loader):
        b_input_ids, b_input_mask, b_labels = batch[0].long().cuda(), batch[1].long().cuda(), batch[2].long().cuda()
        output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        loss, logits = output[0], output[1] 
        avg_loss.append(loss.item())

        acc = utils.binary_acc(torch.argmax(logits, dim=1), b_labels.flatten())
        avg_acc.append(acc)

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)     
        optimizer.step()            
        scheduler.step()              

    avg_acc = np.array(avg_acc).mean()
    avg_loss = np.array(avg_loss).mean()
    train_metrics = {'train_acc': avg_acc,
                     'train_loss': avg_loss
                     }
    logging.info(train_metrics)
    return avg_loss, avg_acc



def evaluate(model, loader):
    avg_acc = []
    model.eval()         

    y_pred, y_true = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            b_input_ids, b_input_mask, b_labels = batch[0].long().cuda(), batch[1].long().cuda(), batch[2].long().cuda()
            
            output = model(b_input_ids, attention_mask=b_input_mask)
           
            y_true.append(b_labels.cpu().numpy()) 
            y_pred.append(torch.argmax(output[0], dim=1).cpu().numpy())         

            acc = utils.binary_acc(torch.argmax(output[0], dim=1), b_labels.flatten())
            avg_acc.append(acc)

    avg_acc = np.array(avg_acc).mean()
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    test_metrics = {'test_acc': avg_acc}
    logging.info(test_metrics)
    return avg_acc, y_pred, y_true



if __name__ == '__main__':
    parser = parser.default_parser()
    args = parser.parse_args()
    print('args = ', args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    checkpoint_path = os.path.join(args.checkpoint_dir, 'bert')       
    os.makedirs(checkpoint_path, exist_ok=True)
    utils.set_logger(os.path.join(checkpoint_path, 'train.log'))


    logging.info("load data")
    train_dataset = BertDataset(data_path = os.path.join(args.data_dir, "train.csv"), max_length_doc=args.max_length, DIR=args.model_dir)
    dev_dataset = BertDataset(data_path = os.path.join(args.data_dir, 'dev.csv'), max_length_doc=args.max_length, DIR=args.model_dir)
    test_dataset = BertDataset(data_path= os.path.join(args.data_dir, args.testset_dir), max_length_doc=args.max_length, DIR=args.model_dir)  

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)


    logging.info('create model')
    bert_config = BertConfig.from_pretrained(args.model_dir + 'config.json', num_labels=args.num_labels)
    model = BertForSequenceClassification.from_pretrained(args.model_dir, config=bert_config)  

    model_file = args.model_dir + 'pytorch_model.bin' 
    model.load_state_dict(torch.load(model_file), strict=False)           
    model = model.cuda()

    num_params = (sum(p.numel() for p in model.parameters())/1000000.0)
    logging.info('Total params: %.2fM' % num_params)


    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr = args.learning_rate)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)


    if args.do_train:       
        logging.info('start training')
        max_val_acc = 0
        for epoch in range(args.num_epochs):
            logging.info("Epoch {}/{}".format(epoch + 1, args.num_epochs))
            train_loss, train_acc = train(model, train_loader, optimizer, scheduler)
            val_acc, val_pred, val_true = evaluate(model, dev_loader)
            if val_acc > max_val_acc:
                max_val_acc = val_acc 
                utils.save_model(model, os.path.join(checkpoint_path, 'best.pt'))

        utils.load_model(model, os.path.join(checkpoint_path, 'best.pt'))         
        test_acc, y_pred, y_true = evaluate(model, test_loader)
        print('classification report:\n%s' % classification_report(y_true, y_pred, digits=4))   

    else:                   
        logging.info('predict the unlabeled data')
        utils.load_model(model, os.path.join(checkpoint_path, 'best.pt'))

        teacher_outputs = []
        fw = open(os.path.join(args.data_dir, args.testset_dir[:-3] + 'txt'), 'w')
        for batch in test_loader:
            with torch.no_grad():
                b_input_ids, b_input_mask = batch[0].long().cuda(), batch[1].long().cuda()
                output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                teacher_outputs.append(output[0])     
                for item in output[0]:
                    fw.write(str(item.tolist()) + '\n')

