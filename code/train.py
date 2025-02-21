import torch
from torch import nn
from utils.transforms import jigsaw_generator
from utils.tool import split_feature,AverageMeter,save_train_info
from utils.target_fn import evaluation
from tqdm import tqdm

def train_pmg(args, train_loader, model, optimizer, epoch):
    
    loss_concat = AverageMeter()
    train_loss = AverageMeter()
    split_lo = AverageMeter()
    acc = AverageMeter()
    
    # switch to train mode
    optimizer1 = optimizer[0]
    optimizer2 = optimizer[1]
    
    CELoss = nn.CrossEntropyLoss()
    triploss = nn.TripletMarginWithDistanceLoss()
    training_bar = tqdm(train_loader, ncols=100)
    
    model.train()
    for data in training_bar:
        inputs,targets = data
        batchsize = inputs.shape[0]
        if inputs.shape[0] < batchsize:
            continue
        
        inputs = inputs.cuda()
        targets = targets.cuda()
        #data, targets = Variable(inputs), Variable(targets)

        # Step 1
        optimizer1.zero_grad()
        inputs1 = jigsaw_generator(inputs, 8, args.input_size)
        output_1, _, _, _, _ = model(inputs1)
        loss1 = CELoss(output_1, targets) * 1
        loss1.backward()
        optimizer1.step()

        # Step 2
        optimizer1.zero_grad()
        inputs2 = jigsaw_generator(inputs, 4, args.input_size)
        _, output_2, _, _,_ = model(inputs2)
        loss2 = CELoss(output_2, targets) * 1
        loss2.backward()
        optimizer1.step()

        # Step 3
        optimizer1.zero_grad()
        inputs3 = jigsaw_generator(inputs, 2, args.input_size)
        _, _, output_3, _,_ = model(inputs3)
        loss3 = CELoss(output_3, targets) * 1
        loss3.backward()
        optimizer1.step()

        # Step 4
        optimizer1.zero_grad()
        _, _, _, output_concat, output_split = model(inputs)
        concat_loss = CELoss(output_concat, targets) * 2
        concat_loss.backward()
        optimizer1.step()
        
        #Step 5
        optimizer2.zero_grad()
        _, _, _, _, output_split = model(inputs)
        origin,true,false = split_feature(output_split)
        split_loss = triploss(origin,true,false)
        split_loss.backward()
        optimizer2.step()
        
        #计算准确率

        accuracy,_,_,_ = evaluation(targets,output_concat)
        acc.update(accuracy*100,batchsize)

        loss_concat.update(concat_loss.item(),batchsize)
        train_loss.update(loss1.item()+loss2.item()+loss3.item()+concat_loss.item(),batchsize)
        split_lo.update(split_loss.item(),batchsize//3)
        #训练信息打印
        training_bar.set_description(f'Train Epoch [{epoch}/{args.epochs}]')
        training_bar.set_postfix(loss_concat=loss_concat.avg, acc=acc.avg)
    save_train_info(epoch,args,loss_concat,split_lo,train_loss,acc)
    
    return model,optimizer1,optimizer2