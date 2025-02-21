import os
import torch
import argparse
from model.pmg import load_PMG_model

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--modelroot', default=r"",metavar='DIR',           
                    help='path to dataset')
parser.add_argument('--saveroot', default=r"",metavar='DIR',           
                    help='path to train/test info path')
parser.add_argument('--model_class', default="checkpoint",
                    help='checkpoint/model')
parser.add_argument('--nclass', default=37, type=int, metavar='N',
                    help='number of class')
parser.add_argument('--output_cls', default="onnx", type=int, metavar='N',
                    help='onnx/pth')
args = parser.parse_args()


# ---------------------------------------------- #
# 加载模型                                           
# ---------------------------------------------- #
def check_model():
    if args.model_class == "checkpoint":
        state = torch.load(args.modelroot)
        model = load_PMG_model(model.nclass)
        model.load_state_dict(state['state_dict'])
    elif args.model_class == "model":
        model = torch.load(args.modelroot)
    else:
        print("model_class error, please input checkpoint/model")
        return None
    return model


# ---------------------------------------------- #
# onnx模型输出                                       
# ---------------------------------------------- #
def onnx_file_export():
    model = check_model()
    if not model:
        return 0
    input = torch.randn((1,3,448,448))
    save_path = os.path.join(args.saveroot,"onnx_file.onnx")
    torch.onnx.export(model,input,save_path,
                      input_names=['input'], output_names=['output'])
    
    
# ---------------------------------------------- #
# pth模型输出                                     
# ---------------------------------------------- #
def pth_file_export():
    model = check_model()
    if not model:
        return 0
    save_path = os.path.join(args.saveroot,"pth_file.pth")
    torch.save(model,save_path)
    
    
if __name__ == "__main__":
    if args.output_cls == "onnx":
        onnx_file_export()
    elif args.output_cls == "pth":
        pth_file_export()