import argparse
import logging
import os
import random
import torch

HOME_DATA_FOLDER = "../OTMS-KD"
TASK_NAME = 'yelp'

def default_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        default=os.path.join(HOME_DATA_FOLDER, 'datasets', TASK_NAME),
                        type=str,
                        help="The path of dataset.")
    parser.add_argument("--model_dir",
                        default = './bert-base-uncased/',                       
                        type=str,
                        help="The path of pretrained model file")

    parser.add_argument("--checkpoint_dir",
                        default=os.path.join(HOME_DATA_FOLDER, 'checkpoints', TASK_NAME),
                        type=str,
                        help="The output directory where the model checkpoints will be written.")



    # train_t.py
    # do_train=True  "test.csv"
    # (do_train=False)  "unlabel_10+train.csv"
    parser.add_argument("--do_train",
                        action="store_true",
                        default=False, 
                        help="Weather to finetune the taecher model")

    parser.add_argument("--testset_dir",
                         default="test.csv",          
                         type=str,
                         help="The path of test data.")
    


    # student model  128 1e-3 1e-5 20
    # teacher model  128 2e-5 1e-3 10
    parser.add_argument("--batch_size",
                        default=128,                                            
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=1e-3,
                        type=float,
                        help="The weight decay for Adam.")
    parser.add_argument("--num_epochs",
                        default=10,
                        type=int,
                        help="Total number of training epochs to perform.")


    # general parameters
    parser.add_argument('--seed',
                        default=123,
                        type=int,
                        help="Random seed for initialization")

    parser.add_argument("--max_length",
                        default=50,                                   
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--temperature",
                        default=1.0,
                        type=float,
                        help="The temperature for distillation.")

    parser.add_argument("--emb_size1",
                        default=50,
                        type=int,
                        help="One embedding size of three models.")
    parser.add_argument("--emb_size2",
                        default=300,
                        type=int,
                        help="Another embedding size of three models.")

    parser.add_argument("--num_labels",
                        default=2,                    
                        type=int,
                        help="The number of classification labels.")
    
    return parser

