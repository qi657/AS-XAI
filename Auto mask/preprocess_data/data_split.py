# -- coding: utf-8 --
# @Time : 2023/9/10 21:10
# @Author : Harper
# @Email : sunc696@gmail.com
# @File : data_split.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import collections
import math
import shutil
import pandas as pd


data_name = 'leaves'  # 'leaves' or '12_kind_cat'

if data_name == 'leaves':
    data_dir = 'D:/datasets/classify-leaves/'  # Please input your data path of classify-leaves

    def read_csv_labels(fname):
        with open(fname, 'r') as f:
            lines = f.readlines()[1:]
        tokens = [l.rstrip().split(',') for l in lines]
        return dict(((name, label) for name, label in tokens))

    labels = read_csv_labels(os.path.join(data_dir, 'train.csv')) 


    def copyfile(filename, target_dir):
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy(filename, target_dir)


    def reorg_train_valid(data_dir, labels, valid_ratio):
        n = collections.Counter(labels.values()).most_common()[-1][1]
        n_valid_per_label = max(1, math.floor(n * valid_ratio))
        label_count = {}
        for train_file in labels:
            label = labels[train_file] 
            fname = os.path.join(data_dir, train_file)
            copyfile(
                fname,
                os.path.join(data_dir, 'train_valid_test', 'train_valid', label)) 
            if label not in label_count or label_count[label] < n_valid_per_label: 
                copyfile(
                    fname,
                    os.path.join(data_dir, 'train_valid_test', 'valid', label))
                label_count[label] = label_count.get(label, 0) + 1
            else:
                copyfile(
                    fname,
                    os.path.join(data_dir, 'train_valid_test', 'train', label))
        return n_valid_per_label 


    def reorg_test(data_dir):
        test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        for test_file in test['image']:
            copyfile(
                os.path.join(data_dir, test_file),
                os.path.join(data_dir, 'train_valid_test', 'test', 'unknown'))


    def reorg_leave_data(data_dir, valid_ratio):
        labels = read_csv_labels(os.path.join(data_dir, 'train.csv')) 
        reorg_train_valid(data_dir, labels, valid_ratio)
        reorg_test(data_dir) 

    batch_size = 128
    valid_ratio = 0.1 
    if not os.path.exists(data_dir + "\\" + "train_valid_test"): 
        print("start!")
        reorg_leave_data(data_dir, valid_ratio)
    else:
        print("Already exists!")
    print('finish!')


if data_name == '12_kind_cat':
    def classify_data(txt_path,labels):
        fh = open(txt_path, 'r', encoding='utf-8')
        lines = fh.readlines()

        for line in lines:
            line = line.strip('\n') 
            line = line.rstrip()  
            words = line.split() 
            imgs_name = words[0] 
            srcfile = 'D:/datasets/12_kind_cat/'+imgs_name
            print(srcfile)
            imgs_label = int(words[1])
            #print(type(imgs_label))
            print(srcfile)
            if imgs_label ==0:
                shutil.copy(srcfile, 'D:/datasets/12_kind_cat/test/'+labels[0])
            elif imgs_label ==1:
                shutil.copy(srcfile, 'D:/datasets/12_kind_cat/test/'+labels[1])
            elif imgs_label ==2:
                shutil.copy(srcfile, 'D:/datasets/12_kind_cat/test/'+labels[2])
            elif imgs_label ==3:
                shutil.copy(srcfile, 'D:/datasets/12_kind_cat/test/'+labels[3])
            elif imgs_label ==4:
                shutil.copy(srcfile, 'D:/datasets/12_kind_cat/test/'+labels[4])
            elif imgs_label ==5:
                shutil.copy(srcfile, 'D:/datasets/12_kind_cat/test/'+labels[5])
            elif imgs_label ==6:
                shutil.copy(srcfile, 'D:/datasets/12_kind_cat/test/'+labels[6])
            elif imgs_label ==7:
                shutil.copy(srcfile, 'D:/datasets/12_kind_cat/test/'+labels[7])
            elif imgs_label ==8:
                shutil.copy(srcfile, 'D:/datasets/12_kind_cat/test/'+labels[8])
            elif imgs_label ==9:
                shutil.copy(srcfile, 'D:/datasets/12_kind_cat/test/'+labels[9])
            elif imgs_label ==10:
                shutil.copy(srcfile, 'D:/datasets/12_kind_cat/test/'+labels[10])
            elif imgs_label ==11:
                shutil.copy(srcfile, 'D:/datasets/12_kind_cat/test/'+labels[11])
        print("Copy files Successfully!")

    if __name__ == '__main__':
        label = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011"]
        for i in label:
            os.mkdir('D:/datasets/12_kind_cat/test/'+i)

        # classify_data('D:/datasets/12_kind_cat/train_list.txt',label)
        classify_data('D:/datasets/12_kind_cat/val_split_list.txt', label)

