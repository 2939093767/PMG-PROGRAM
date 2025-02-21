import numpy as np
import random
from torchvision import transforms
from PIL import Image



# ---------------------------------------------- #
# 测试图片预处理                                        
# ---------------------------------------------- #
def get_test_transform(size):
    transform_test = transforms.Compose([
            transforms.Lambda(lambda img:scale_keep_ar_min_fixed(img, size)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225))
                                ])
    return transform_test




# ---------------------------------------------- #
# 训练图片预处理                                        
# ---------------------------------------------- #
def get_train_transform(size):
    transform_train = transforms.Compose([
            transforms.Lambda(lambda img:scale_keep_ar_min_fixed(img, size)),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225))
                                ])
    return transform_train




# ---------------------------------------------------------------------------- #
#   自适应图片大小处理                          
# ---------------------------------------------------------------------------- #
def scale_keep_ar_min_fixed(img, fixed_min):
    ow, oh = img.size
    if ow < oh:
        nw = fixed_min
        nh = nw * oh // ow
    else:
        nh = fixed_min 
        nw = nh * ow // oh
    return img.resize((nw, nh), Image.BICUBIC)



#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 



# ---------------------------------------------- #
# 粒度切割函数                                    
# ---------------------------------------------- #
def jigsaw_generator(images, n, input_size=224):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    block_size = input_size // n
    rounds = n ** 2
    random.shuffle(l)
    jigsaws = images.clone()
    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                y * block_size:(y + 1) * block_size].clone()
        jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

    return jigsaws