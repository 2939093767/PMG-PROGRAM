import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from utils.transforms import cvtColor


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

# ---------------------------------------------- #
# 训练集处理                                      
# ---------------------------------------------- #
class train_Dataset(Dataset):
    def __init__(self, input_shape, root,image_path, num_classes, random, trans):
        self.trans = trans
        self.input_shape    = input_shape
        self.lines          = root        #txt路径
        self.image_root     = image_path  #图片路径
        self.num_classes    = num_classes
        self.random         = random
        #   路径和标签
        self.paths  = []
        self.labels = []
        self.load_dataset()
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        #   创建全为零的矩阵
        #images = np.zeros((3, 3, self.input_shape[0], self.input_shape[1]))
        images = []
        labels = np.zeros((3))
        #   先获得两张同一个人的人脸
        #   用来作为anchor和positive
        c               = random.randint(0, self.num_classes - 1)
        selected_path   = self.paths[self.labels[:] == c]
        while len(selected_path) < 2:
            c               = random.randint(0, self.num_classes - 1)
            selected_path   = self.paths[self.labels[:] == c]

        #   随机选择两张
        image_indexes = np.random.choice(range(0, len(selected_path)), 2)
        #   打开图片并放入矩阵
        image = cvtColor(Image.open(selected_path[image_indexes[0]]))
        images.append(image)
        labels[0] = c
        
        image = cvtColor(Image.open(selected_path[image_indexes[1]]))
        #image = self.trans(image)
        images.append(image)
        labels[1] = c
        #   取出另外一个人的人脸
        different_c         = list(range(self.num_classes))
        different_c.pop(c)
        different_c_index   = np.random.choice(range(0, self.num_classes - 1), 1)
        current_c           = different_c[different_c_index[0]]
        selected_path       = self.paths[self.labels == current_c]
        while len(selected_path)<1:
            different_c_index   = np.random.choice(range(0, self.num_classes - 1), 1)
            current_c           = different_c[different_c_index[0]]
            selected_path       = self.paths[self.labels == current_c]
        
        #   随机选择一张
        image_indexes       = np.random.choice(range(0, len(selected_path)), 1)
        image               = cvtColor(Image.open(selected_path[image_indexes[0]]))
        images.append(image)
        labels[2]           = current_c
        for i,img in enumerate(images):
            images[i] = torch.unsqueeze(self.trans(img),0)
        images = torch.concatenate(images)
        return images, labels

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a
    
    def load_dataset(self):
        with open(os.path.join(self.lines,"train.txt"),"r+") as f:
            paths = f.readlines()
        self.length = len(paths)
        for path in paths:
            path_split = path.split(" ")
            self.paths.append(os.path.join(self.image_root,path_split[1].replace("\n","")))
            self.labels.append(int(path_split[0]))
        self.paths  = np.array(self.paths, dtype=np.object_)
        self.labels = np.array(self.labels)



# ---------------------------------------------- #
# 测试集处理                                          
# ---------------------------------------------- #
class test_Dataset(Dataset):
    def __init__(self, input_shape, info_path,image_path, num_classes, random, trans):
        
        self.trans = trans
        self.input_shape    = input_shape
        self.lines          = info_path        #txt路径
        self.image_root     = image_path  #图片路径
        self.length         = len(info_path)
        self.num_classes    = num_classes
        self.random         = random
        #   路径和标签
        self.paths  = []
        self.labels = []

        self.load_dataset()
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        #   创建全为零的矩阵
        #images = np.zeros((2, 3, self.input_shape[0], self.input_shape[1]))
        images = []
        labels = np.zeros((2))
        #   随机抽选样本
        c               = random.randint(0, self.num_classes - 1)
        selected_path   = self.paths[self.labels[:] == c]
        while len(selected_path) < 2:
            c               = random.randint(0, self.num_classes - 1)
            selected_path   = self.paths[self.labels[:] == c]
        #随机负样本正样本 1:1 
        i = random.randint(0,1)
        if i == 1:
            #   随机选择两张
            image_indexes = np.random.choice(range(0, len(selected_path)), 2)
            #   打开图片并放入矩阵
            image = cvtColor(Image.open(selected_path[image_indexes[0]]))
            #   缩放图片
            images.append(image)
            labels[0] = c
            #   第二张
            image = cvtColor(Image.open(selected_path[image_indexes[1]]))
            images.append(image)
            labels[1] = c
            id_label = 1
        else:
            image_indexes = np.random.choice(range(0, len(selected_path)), 1)
            #   打开图片并放入矩阵
            image = cvtColor(Image.open(selected_path[image_indexes[0]]))
            #   缩放图片
            #image = self.trans(image)
            images.append(image)
            labels[0] = c
            #   取出另外一个人的人脸
            different_c         = list(range(self.num_classes))
            different_c.pop(c)
            different_c_index   = np.random.choice(range(0, self.num_classes - 1), 1)
            current_c           = different_c[different_c_index[0]]
            selected_path       = self.paths[self.labels == current_c]
            while len(selected_path)<1:
                different_c_index   = np.random.choice(range(0, self.num_classes - 1), 1)
                current_c           = different_c[different_c_index[0]]
                selected_path       = self.paths[self.labels == current_c]
            #   随机选择一张
            image_indexes       = np.random.choice(range(0, len(selected_path)), 1)
            image               = cvtColor(Image.open(selected_path[image_indexes[0]]))
            #image = self.trans(image)
            images.append(image)
            labels[1]           = current_c
            id_label = 0
        for i,img in enumerate(images):
            images[i] = torch.unsqueeze(self.trans(img),0)
        images = torch.concatenate(images)
        
        return images, labels,id_label

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a
    
    def load_dataset(self):
        with open(os.path.join(self.lines,"test.txt"),"r+") as f:
            paths = f.readlines()
        self.length         = len(paths)
        for path in paths:
            path_split = path.split(" ")
            self.paths.append(os.path.join(self.image_root,path_split[1].replace("\n","")))
            self.labels.append(int(path_split[0]))
        self.paths  = np.array(self.paths, dtype=np.object_)
        self.labels = np.array(self.labels)
        



# ---------------------------------------------- #
# train数据集处理，实际batchsize = 输入batchsize * 3       
# ---------------------------------------------- #
def dataset_train_collate(batch):
    images = []
    labels = []
    for img, label in batch:
        images.append(img.unsqueeze(0))
        labels.append(label)
    images = torch.concatenate(images)
    
    images1 = images[:, 0, :, :, :]
    images2 = images[:, 1, :, :, :]
    images3 = images[:, 2, :, :, :]
    images = torch.concatenate([images1, images2, images3], 0)
    
    labels1 = np.array(labels)[:, 0]
    labels2 = np.array(labels)[:, 1]
    labels3 = np.array(labels)[:, 2]
    labels = np.concatenate([labels1, labels2, labels3], 0)
    #images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    labels  = torch.from_numpy(np.array(labels)).long()
    
    return images, labels



# ---------------------------------------------- #
# test数据集处理，实际batchsize = 输入batchsize * 2        
# ---------------------------------------------- #
def dataset_test_collate(batch):
    images = []
    labels = []
    id_labels = []
    for img, label, id_label in batch:
        images.append(img.unsqueeze(0))
        labels.append(label)
        id_labels.append(id_label)
    images = torch.concatenate(images)

    images1 = images[:, 0, :, :, :]
    images2 = images[:, 1, :, :, :]
    images = torch.concatenate([images1, images2], 0)
    
    labels1 = np.array(labels)[:, 0]
    labels2 = np.array(labels)[:, 1]
    labels = np.concatenate([labels1, labels2], 0)
    
    #images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    labels  = torch.from_numpy(np.array(labels)).long()
    id_labels = torch.from_numpy(np.array(id_labels)).long()
    return images, labels, id_labels







