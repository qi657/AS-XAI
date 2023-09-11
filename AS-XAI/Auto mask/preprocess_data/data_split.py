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
        """读取 `fname` 来给标签字典返回一个文件名。"""
        with open(fname, 'r') as f:
            lines = f.readlines()[1:]
        tokens = [l.rstrip().split(',') for l in lines]
        return dict(((name, label) for name, label in tokens))

    labels = read_csv_labels(os.path.join(data_dir, 'train.csv')) # 存放训练集标签的文件


    def copyfile(filename, target_dir):
        """将文件复制到目标目录。"""
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy(filename, target_dir)


    def reorg_train_valid(data_dir, labels, valid_ratio):
        # 下面的collections.Counter就是统计label这个字典中有几个类别（返回字典）；.most_common()则转换成元组；[-1][1]则返回最后一个元组的第二个值(因为这个类别数量最小)
        n = collections.Counter(labels.values()).most_common()[-1][1] # n就是数量最少类别对应的数量
        n_valid_per_label = max(1, math.floor(n * valid_ratio)) # 根据最小类别数量，得出验证集的数量
        label_count = {}
        for train_file in labels: # 返回训练集中的图片名字列表(我们看到，训练标签转换成的字典，key就是训练集的图片名字)
            label = labels[train_file] # 每张图片 对应的标签
            fname = os.path.join(data_dir, train_file) # 每个图片的完整路径
            copyfile(
                fname,
                os.path.join(data_dir, 'train_valid_test', 'train_valid', label)) # 将图片复制到指定的目录下，这个是为了交叉验证使用，这里和训练集没区别
            if label not in label_count or label_count[label] < n_valid_per_label: # 制作验证集。注：标签名作为key,value为每个标签的图片数量
                copyfile(
                    fname,
                    os.path.join(data_dir, 'train_valid_test', 'valid', label))
                label_count[label] = label_count.get(label, 0) + 1 # 统计每个标签的图片数量
            else: # 制作训练集
                copyfile(
                    fname,
                    os.path.join(data_dir, 'train_valid_test', 'train', label))
        return n_valid_per_label # 返回验证集的数量


    # 在预测期间整理测试集，以方便读取
    def reorg_test(data_dir):
        test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        for test_file in test['image']: # 获取测试集图片的名字，复制到指定文件夹下
            copyfile(
                os.path.join(data_dir, test_file),
                os.path.join(data_dir, 'train_valid_test', 'test', 'unknown'))


    # 调用前面定义的函数，进行整理数据集
    def reorg_leave_data(data_dir, valid_ratio):
        labels = read_csv_labels(os.path.join(data_dir, 'train.csv')) # 是个字典
        reorg_train_valid(data_dir, labels, valid_ratio) # 生成训练集和验证集
        reorg_test(data_dir) # 生成测试集

    batch_size = 128
    valid_ratio = 0.1 # 验证集的比例
    if not os.path.exists(data_dir + "\\" + "train_valid_test"): # 判断是否已经制作好了数据集
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
            line = line.strip('\n')  # 移除字符串首尾的换行符
            line = line.rstrip()  # 删除末尾空
            words = line.split()  # 以空格为分隔符 将字符串分成两部分
            imgs_name = words[0]  # imgs中包含有图像路径和标签
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
        # 创建文件夹
        label = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011"]
        for i in label:
            os.mkdir('D:/datasets/12_kind_cat/test/'+i)

        # classify_data('D:/datasets/12_kind_cat/train_list.txt',label)
        classify_data('D:/datasets/12_kind_cat/val_split_list.txt', label)

