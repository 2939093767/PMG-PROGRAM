import torch
import os
import numpy as np
from utils.transforms import cvtColor,get_test_transform
from utils.tool import AverageMeter,save_test_info,test_split
from utils.target_fn import evaluation,evaluation_split
from PIL import Image
from tqdm import tqdm

def test_pmg(args, test_loader, model, epoch):
    model.eval()
    test_loss = AverageMeter()
    total_acc_concat = AverageMeter()
    total_acc_com = AverageMeter()
    pre1 = AverageMeter()
    pre2 = AverageMeter()
    rec1 = AverageMeter()
    rec2 = AverageMeter()
    f1s1 = AverageMeter()
    f1s2 = AverageMeter()
    split_acc = AverageMeter()
    
    with torch.no_grad():
        val_bar = tqdm(test_loader, ncols=100)
        for data in val_bar:
            inputs,targets,id_targets = data
            inputs, targets, id_targets = inputs.to("cuda"), targets.to("cuda"), id_targets.to("cuda")
            
            batchsize = inputs.size(0)
            #forward
            output_1, output_2, output_3, output_concat, output_split= model(inputs)
            #split count
            id_output = test_split(output_split,thread=args.split_thread)
            outputs_com = output_1 + output_2 + output_3 + output_concat
            
            loss = torch.nn.CrossEntropyLoss()(output_concat, targets)
            #acc
            acc_concat,precision1,recall1,f1_score1 = evaluation(targets,output_concat.data)
            acc_com,precision2,recall2,f1_score2 = evaluation(targets,outputs_com.data)
            id_output = torch.from_numpy(np.array(id_output)).long().to("cuda")
            split_com,_,_,_= evaluation_split(id_targets,id_output)
            
            #log
            pre1.update(precision1*100,batchsize)
            rec1.update(recall1*100,batchsize)
            f1s1.update(f1_score1*100,batchsize)
            
            pre2.update(precision2*100,batchsize)
            rec2.update(recall2*100,batchsize)
            f1s2.update(f1_score2*100,batchsize)
            
            total_acc_concat.update(acc_concat*100,batchsize)
            total_acc_com.update(acc_com*100,batchsize)
            test_loss.update(loss.item(),batchsize)
            
            split_acc.update(split_com*100,batchsize//2)
            
            #updata tqdm
            val_bar.set_description(f'Testing')
            val_bar.set_postfix(acc_concat=total_acc_concat.avg, loss=test_loss.avg)
            
        print(f"test_prec_com:{total_acc_com.avg}      test_prec_concat:{total_acc_concat.avg}")
        #test_similarity(get_test_transform(args.input_size),model)
        save_test_info(epoch,{
            'total_acc_concat':total_acc_concat,
            'total_acc_com':total_acc_com,
            'pre1':pre1,
            'pre2':pre2,
            'rec1':rec1,
            'rec2':rec2,
            'f1s1':f1s1,
            'f1s2':f1s2,
            'split_acc':split_acc,
        })
    return max(total_acc_com.avg,total_acc_concat.avg)


def test_similarity(transform,model):
    images = []
    root = "data/test_image"
    image_list = os.listdir(root)
    for i in range(0,len(image_list)) :
        path = os.path.join(root,image_list[i])
        image = cvtColor(Image.open(path))
        image = transform(image)
        images.append(image)
    _,_,_,_,out = model(torch.stack(images).cuda())
    for i in range(0,len(out)-1):
        out1 = out[i].cpu().detach().numpy()
        out2 = out[i+1].cpu().detach().numpy()
        l1 = np.linalg.norm(out1-out2)
        print(f"{i}-{i+1}:{l1}")