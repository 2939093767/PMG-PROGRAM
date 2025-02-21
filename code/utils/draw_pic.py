import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from utils.transforms import *
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image



def draw_distance_and_cls(image1,image2,cls1,cls2,dis,save_path=None):
    fig,axes = plt.subplots(1,2)
    axes[0].imshow(np.array(image1))
    axes[1].imshow(np.array(image2))
    axes[0].axis('off')
    axes[1].axis('off')
    if cls1 != None:
        plt.text(-300, -30, 'cls1:%.3f' % cls1, ha='center', va= 'bottom',fontsize=11)
    else :
        plt.text(-300, -30, 'cls1:None', ha='center', va= 'bottom',fontsize=11)
    plt.text(0, -30, 'Distance:%.3f' % dis, ha='center', va= 'bottom',fontsize=11)
    if cls2 != None:
        plt.text(300, -30, 'cls2:%.3f' % cls2, ha='center', va= 'bottom',fontsize=11)
    else:
        plt.text(300, -30, 'cls2:None', ha='center', va= 'bottom',fontsize=11)
    plt.show()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    
    

count = 0
def model_attention_hotmap(image_draw,image_tensor,cls,model,size,save_path=None):
    global count
    transform_draw = transforms.Compose([
                transforms.Lambda(lambda img:scale_keep_ar_min_fixed(img, size)),
                transforms.CenterCrop(size)])
    rgb_img= transform_draw(image_draw)
    rgb_img = np.float32(rgb_img)/255
    target_layers_1 = [model.module.conv_block1]
    target_layers_2 = [model.module.conv_block2]
    target_layers_3 = [model.module.conv_block3]
    # 选取合适的类激活图，但是ScoreCAM和AblationCAM需要batch_size
    for i in [target_layers_1,target_layers_2,target_layers_3]:
        count += 1
        cam = GradCAMPlusPlus(model=model,target_layers=i)
        target = [ClassifierOutputTarget(cls),ClassifierOutputTarget(cls),ClassifierOutputTarget(cls),ClassifierOutputTarget(cls)]
        # 上方preds需要设定，比如ImageNet有1000类，这里可以设为200
        grayscale_cam = cam(input_tensor=image_tensor, targets=target)
        grayscale_cam = grayscale_cam[0, :]
        cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        pil_image = Image.fromarray(cam_img)
        if save_path:
            print(os.path.join(save_path,f'output_cam_class{cls.item()}_{count}.jpg'))
            pil_image.save(os.path.join(save_path,f'output_cam_class{cls.item()}_{count}.jpg'))