# OTMS-KD
This is the implementation of the paper [One-Teacher and Multiple-Student Knowledge Distillation on Sentiment Classification]

## Requirements

To run our code, please install all the dependency packages by using the following command:
```
pip install -r requirements.txt
pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchtext==0.9.0
```
**Note**: Different versions of server graphics cards and dependency packages may lead to different results from the paper. However, the trend should still hold if you tune hyper-parameters no matter what versions you use.

## Prepare the data

In supervised sentiment classification task, we use the data of YELP.
```
python datasets/preprocess.py
```

## Finetune the teacher model
We choose BERT-base as the teacher model and 'bert-base-uncased' can also replaced by 'bert-large', 'distilbert-base' etc.  
```
python src/train_t.py --do_train --testset_dir="test.csv" --batch_size=128 --learning_rate=2e-5 --weight_decay=1e-3 --num_epochs=10
```

## Predict the unlabeled data

```
python src/train_t.py --testset_dir="unlabel_10+train.csv"
```

## Train the student model
To train the student model, we need to first download the embeddings of glove.6B.50d.txt, glove.twitter.27B.50d.txt, glove.42B.300d.txt (see https://nlp.stanford.edu/projects/glove/) and put them into the directory '.vector_cache'.

```
python src/train_s.py --testset_dir="unlabel_10+train.csv" --batch_size=128 --learning_rate=1e-3 --weight_decay=1e-5 --num_epochs=20
```
