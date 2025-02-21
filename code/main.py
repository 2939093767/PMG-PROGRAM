import argparse
import torch.backends.cudnn as cudnn
import torch
import os
import time

from model.pmg import load_PMG_model
from torch import nn
from utils.tool import *
from utils.dataloader import *
from utils.transforms import *
from train import train_pmg
from test import test_pmg

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataroot', default=r"./data/tshudog",metavar='DIR',           
                    help='path to dataset')
parser.add_argument('--infopath', default=r"./metedata/tshudog",metavar='DIR',           
                    help='path to train/test info path')
parser.add_argument('--nclass', default=130, type=int,
                    help='num of classes')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--trainbatch', default=12, type=int,
                    metavar='N', help='train-batch size')
parser.add_argument('--testbatch', '--test_batch_size', default=12, type=int,
                    metavar='N', help='test-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.00005, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='momentum', help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--resume', default=r'', type=str, metavar='PATH',
                    help='path to latest checkpoint')
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU nums to use.')
parser.add_argument('--eval_epoch', default=2, type=int,
                    help='every eval_epoch we will evaluate')
parser.add_argument('--input_size', default=224, type=int,
                    help='224/448')
parser.add_argument('--split_thread', default=14.5, type=float,
                    help='split thread for similarity')



best_prec1 = 0
def main():
    print('<==> Part1 : prepare for parameters <==> Begin')
    global args,best_prec1
    args = parser.parse_args()
    print('<==> Part1 : prepare for parameters <==> Done')
    
    print('<==> Part2 : Load Network  <==> Begin')
    #模型初始化
    net = load_PMG_model(args.nclass,input_size=args.input_size,pretrain=True)
    if args.gpu is not None:
        net = nn.DataParallel(net, device_ids=range(args.gpu))
        net = net.cuda()
        cudnn.benchmark = True
    #打印大小
    calculate_model_size(net)
    #损失函数和优化器
    optimizer_cls = torch.optim.SGD([
        {'params': net.module.classifier_concat.parameters(), 'lr': args.lr},
        {'params': net.module.conv_block1.parameters(), 'lr':args.lr},
        {'params': net.module.classifier1.parameters(), 'lr': args.lr},
        {'params': net.module.conv_block2.parameters(), 'lr': args.lr},
        {'params': net.module.classifier2.parameters(), 'lr': args.lr},
        {'params': net.module.conv_block3.parameters(), 'lr': args.lr},
        {'params': net.module.classifier3.parameters(), 'lr': args.lr},
        {'params': net.module.features.parameters(), 'lr': args.lr}
    ],
        momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_split = torch.optim.SGD([
        {'params': net.module.slitp.parameters(), 'lr': args.lr}
    ],
        momentum=args.momentum, weight_decay=args.weight_decay)
    
    
    if args.resume:
        if os.path.isfile(args.resume):
            net = torch.load(args.resume)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer_cls.load_state_dict(checkpoint['optimizer_cls'])
            optimizer_split.load_state_dict(checkpoint['optimizer_split'])
            print('<==> Part2 : Load Network  <==> Continue from {} epoch {}'.format(args.resume, checkpoint['epoch']))
            print('<==> Part2 : Load Network  <==> Continue from {} epoch 0'.format(args.resume))
        else:
            print('<==> Part2 : Load Network  <==> Failed')
    print('<==> Part2 : Load Network  <==> Done')
    
    print('<==> Part3 : Load Dataset  <==> Begin')
    dataroot = os.path.abspath(args.dataroot)
    inforroot = os.path.abspath(args.infopath)
    train_transform = get_train_transform(args.input_size)
    train_dataset = train_Dataset(args.input_size,inforroot,dataroot,args.nclass,False,trans=train_transform)
    test_transform = get_test_transform(args.input_size)
    test_dataset = test_Dataset(args.input_size,inforroot,dataroot,args.nclass,False,trans=test_transform)
    trainloader = DataLoader(train_dataset, batch_size=args.trainbatch,collate_fn=dataset_train_collate)
    testloader = DataLoader(test_dataset, batch_size=args.testbatch,collate_fn=dataset_test_collate)
    print('<==> Part3 : Load Dataset  <==> Done')

    print('<==> Part4 : Train and Test  <==> Begin')
    for epoch in range(args.start_epoch, args.epochs):
        s_t = time.time()
        #调整学习率
        for nlr in range(len(optimizer_cls.param_groups)):
            optimizer_cls.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.epochs, args.lr)
        for nlr in range(len(optimizer_split.param_groups)):
            optimizer_split.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.epochs, args.lr)
        #训练一轮
        net,optimizer_cls,optimizer_split = train_pmg(args, trainloader, net, [optimizer_cls,optimizer_split], epoch)
        
        #每n轮验证一次
        if epoch % args.eval_epoch == 0:
            prec1 = test_pmg(args, testloader, net, epoch)
            #记录
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'data_name' : os.path.basename(args.infopath),
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer_cls' : optimizer_cls.state_dict(),
                'optimizer_split' : optimizer_split.state_dict(),
                'prec1'     : prec1,
            }) 
            
            if is_best:
                save_best_model({
                    'data_name' : os.path.basename(args.infopath),
                    'model':net,
                    'prec':best_prec1
                })
                
        e_t = time.time()
        print(f"一轮消耗时间：{e_t-s_t}")
        torch.cuda.empty_cache()
        
        
if __name__ == "__main__":
    main()