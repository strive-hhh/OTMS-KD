# 将unlabel_10.csv和train.csv合并
import os

HOME_DATA_FOLDER = "../OTMS-KD/datasets/yelp/"

def mergeFiles(file1, file2):
    # 先检查unlabel_10+train.csv是否存在
    new_file = file1[:-4] + '+' + file2
    if os.path.isfile(HOME_DATA_FOLDER + new_file):
        return
    
    with open(HOME_DATA_FOLDER + file1) as fr1:
        cont1 = fr1.readlines()
    
    with open(HOME_DATA_FOLDER + file2) as fr2:
        cont2 = fr2.readlines()

    cont = cont1 + cont2
    with open(HOME_DATA_FOLDER + new_file, 'a+') as fw:
        for line in cont:
            fw.write(line)
    
mergeFiles('unlabel_10.csv', 'train.csv')


