import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from PIL import Image
from utils.transforms import get_test_transform
from utils.draw_pic import *


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--model_path', default=r'model_weight/pmg_tshudog_448/best_model_77.871287.pth',metavar='DIR',           
                    help='path to dataset')
parser.add_argument('--input_size', default=448,metavar='DIR',type=int,          
                    help='224/448')
parser.add_argument('--save_image_cam_path', default='./Data_visualization/attention_hotmap',type=str,
                    help='path to save image_cam')
parser.add_argument('--save_result_image_path', default='./Data_visualization/result.jpg',type=str,
                    help='path to save result')
parser.add_argument('--thread', default=0.7,type=float,
                    help='thread for class')
args = parser.parse_args()


def predict(image1,image2,model_path):
    net = torch.load(model_path,torch.device('cuda'))
    image_tensor1 = get_test_transform(args.input_size)(image1).unsqueeze(0)
    image_tensor2 = get_test_transform(args.input_size)(image2).unsqueeze(0)
    _,_,_,cls1,dis1 = net(image_tensor1)
    _,_,_,cls2,dis2 = net(image_tensor2)
    cls1 = torch.nn.Softmax(-1)(cls1)
    cls2 = torch.nn.Softmax(-1)(cls2)
    ma1,cls_pred1 = torch.max(cls1.data,1)
    ma2,cls_pred2 = torch.max(cls2.data,1)
    if ma1 < args.thread:
        cls_pred1 = None
    elif ma2 < args.thread:
        cls_pred2 = None
    print(cls_pred1)
    print(cls_pred2)
    dis1 = dis1.cpu().detach().numpy()
    dis2 = dis2.cpu().detach().numpy()
    l1 = np.linalg.norm(dis1-dis2)
    draw_distance_and_cls(image1,image2,cls_pred1,cls_pred2,l1,args.save_result_image_path)
    if cls_pred1 != None:
        model_attention_hotmap(image1,image_tensor1,cls_pred1,net,args.input_size,args.save_image_cam_path)
    if cls_pred2 != None:
        model_attention_hotmap(image2,image_tensor2,cls_pred2,net,args.input_size,args.save_image_cam_path)
    
    return cls_pred1,cls_pred2,l1
    
    
    
if __name__ == "__main__":
    while not os.path.exists(args.model_path):
        print(f"'{args.model_path}' is error")
        model = input("model_path:")
    while True:
        image1 = Image.open(input("image1_path:"))
        image2 = Image.open(input("image2_path:"))
        predict(image1,image2,model_path=args.model_path)
        