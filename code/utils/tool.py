import numpy as np
import os
import torch
import shutil
import logging
import time
from torch.nn import Module
from tqdm import tqdm


# ---------------------------------------------- #
# 训练特征切割
# 锚本：正样本：负样本 = 1：1：1                             
# ---------------------------------------------- #
def split_feature(feature):
    batch = feature.size(0)
    origin = feature[0:batch//3]
    true = feature[batch//3:batch//3*2]
    false = feature[batch//3*2:]
    return origin,true,false
# ---------------------------------------------- #
# 测试样本分割  
# 锚本：其他 = 1 : 1                                      
# ---------------------------------------------- #
def split_feature_test(feature):
    batch = feature.size(0)
    origin = feature[0:batch//2]
    other = feature[batch//2:]
    return origin,other

def test_split(feature,thread=10):
    batch = feature.size(0)//2
    orgin,other = split_feature_test(feature)
    output = []
    for i in range(batch):
        dis1 = orgin[i].cpu().detach().numpy()
        dis2 = other[i].cpu().detach().numpy()
        l1 = np.linalg.norm(dis1-dis2)
        if l1>thread:
            output.append(0)
        else:
            output.append(1)
    return output
    
    
    
    
    
# ---------------------------------------------- #
# 模型参数及大小计算                                       
# ---------------------------------------------- #
def calculate_model_size(model:Module):
    """
    计算并打印 PyTorch 模型的参数量及大致内存占用。
    
    参数:
    model (torch.nn.Module): 需要计算的 PyTorch 模型。
    """
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # numel() 返回张量中元素的数量
    # requires_grad 为 True 表示该参数在训练过程中需要求梯度
    # 通常情况下，一个float32类型的参数占用4字节内存
    size_mb = params * 4 / 1024**2
    print(f"模型参数量: {params}, 大约占用内存: {size_mb:.2f} MB")



# ---------------------------------------------- #
# 学习率调整                                      
# ---------------------------------------------- #
def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1
    return float(lr / 2 * cos_out)


# ---------------------------------------------- #
# 参数保存                                           
# ---------------------------------------------- #
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    
    weight_dir = f"./weight/{state['data_name']}/{state['epoch']}"
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    epoch = state['epoch']
    prec1 = state['prec1']
    file_path = os.path.join(weight_dir, 'epoch_{:04d}_top1_{:02d}_{}'.format(int(epoch), int(prec1), filename))  
    torch.save(state, file_path)
    
# ---------------------------------------------- #
# 最佳模型保存                                          
# ---------------------------------------------- #
def save_best_model(state):
    weight_dir = f"./weight/{state['data_name']}"
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    file_path = os.path.join(weight_dir,f"best_model_{state['prec']:1f}.pth")
    torch.save(state["model"],file_path)
        






# ---------------------------------------------- #
# 进度条记录                                         
# ---------------------------------------------- #
class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        super(TqdmHandler, self).__init__()

    def emit(self, msg):
        msg = self.format(msg)
        tqdm.write(msg)
        time.sleep(1)





# ---------------------------------------------- #
# 定时器                                            
# ---------------------------------------------- #
class Timer(object):

    def __init__(self):
        self.start = time.time()
        self.last = time.time()

    def tick(self, from_start=False):
        this_time = time.time()
        if from_start:
            duration = this_time - self.start
        else:
            duration = this_time - self.last
        self.last = this_time
        return duration
    


# ---------------------------------------------- #
# 数据单位                                           
# ---------------------------------------------- #
class AverageMeter(object):
    """Keep track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ---------------------------------------------- #
# 训练信息保存                                         
# ---------------------------------------------- #
def save_train_info(epoch, args, loss_concat:AverageMeter,loss_split:AverageMeter,train_loss:AverageMeter,prec:AverageMeter):
    """
    loss may contain several parts
    """
    root_dir = os.path.abspath('./')
    log_dir = os.path.join(root_dir, 'log') 
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_file = os.path.join(log_dir, 'log_train.txt')
    with open(log_file, 'a') as f:
        f.write(f'DFL-CNN <==> Train <==> Epoch: [{epoch}/{args.epochs}]\n'
                f'Loss_concat ({loss_concat.avg:.4f})\t'
                f'Loss_split ({loss_split.avg:.4f})\t'
                f'train_loss  ({train_loss.avg:.4f})\t'
                f'Prec ({prec.avg:.3f})\n')

# ---------------------------------------------- #
# pmg测试信息保存                                         
# ---------------------------------------------- #
def save_test_info(epoch,state:dict):
    root_dir = os.path.abspath('./')
    log_dir = os.path.join(root_dir, 'log') 
    # check log_dir 
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_file = os.path.join(log_dir, 'log_test.txt')
    with open(log_file, 'a') as f:
        f.write(f'<==> Test <==> Epoch: [{epoch}] acc_concat:{state["total_acc_concat"].avg} acc_com:{state["total_acc_com"].avg}\n')
        f.write(f'prec:{state["pre1"].avg:2f}/{state["pre2"].avg:2f} rec:{state["rec1"].avg:2f}{state["rec2"].avg:2f} f1s:{state["f1s1"].avg:2f}{state["f1s2"].avg:2f}\n')
        f.write(f'split_acc:{state["split_acc"].avg:2f} \n')