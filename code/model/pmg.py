import torch.nn as nn
import torch
from torchvision.models.mobilenetv3 import mobilenet_v3_large,MobileNet_V3_Large_Weights

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
    
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class mobilenetv3_feature(nn.Module):
    def __init__(self,pretrained=True):
        super(mobilenetv3_feature, self).__init__()
        # k channels for one class, nclass is total classes, therefore k * nclass for conv6
        
        if pretrained:
            self.back = mobilenet_v3_large(MobileNet_V3_Large_Weights.IMAGENET1K_V2).features
        else:
            self.back = mobilenet_v3_large().features
            
        self.c1 = nn.Sequential(*list(self.back)[:7])
        self.c2 = nn.Sequential(*list(self.back)[7:13])
        self.c3 = nn.Sequential(*list(self.back)[13:17])
    def forward(self,x):
        f_3 = self.c1(x)
        f_4 = self.c2(f_3)
        f_5 = self.c3(f_4)
        return f_3,f_4,f_5
    
    
class PMG_mobile_448_triploss(nn.Module):
    def __init__(self, feature_size, classes_num, pretrained):
        super(PMG_mobile_448_triploss, self).__init__()
        self.features = mobilenetv3_feature(pretrained)
        self.max1 = nn.MaxPool2d(kernel_size=56, stride=56)
        self.max2 = nn.MaxPool2d(kernel_size=28, stride=28)
        self.max3 = nn.MaxPool2d(kernel_size=14, stride=14)
        self.num_ftrs = [40,112,960]
        self.elu = nn.ELU(inplace=True)

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(960 * 3),
            nn.Linear(960 * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
            # nn.BatchNorm1d(classes_),
            # nn.ELU(inplace=True),
            
        )
        
        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs[0], feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs[2], kernel_size=3, stride=1, padding=1, relu=True)
        )
        
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs[2]),
            nn.Linear(self.num_ftrs[2], feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs[1], feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs[2], kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs[2]),
            nn.Linear(self.num_ftrs[2], feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs[2], feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs[2], kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs[2]),
            nn.Linear(self.num_ftrs[2], feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
            
        )

        self.slitp = nn.Sequential(
            nn.BatchNorm1d(960 * 3),
            nn.Linear(960 * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True)
        )
    def forward(self, x):
        xf3, xf4, xf5 = self.features(x)
        
        xl1 = self.conv_block1(xf3)
        xl2 = self.conv_block2(xf4)
        xl3 = self.conv_block3(xf5)
        
        xl1 = self.max1(xl1)
        xl1 = xl1.view(xl1.size(0), -1)
        xc1 = self.classifier1(xl1)

        xl2 = self.max2(xl2)
        xl2 = xl2.view(xl2.size(0), -1)
        xc2 = self.classifier2(xl2)

        xl3 = self.max3(xl3)
        xl3 = xl3.view(xl3.size(0), -1)
        xc3 = self.classifier3(xl3)
        
        x_concat = torch.cat((xl1, xl2, xl3), -1)
        
        x_concat_cls = self.classifier_concat(x_concat)
        x_concat_like = self.slitp(x_concat)
        return xc1, xc2, xc3, x_concat_cls,x_concat_like
    
    
    
class PMG_mobile_224_triploss(nn.Module):
    def __init__(self, feature_size, classes_num, pretrained):
        super(PMG_mobile_224_triploss, self).__init__()
        self.features = mobilenetv3_feature(pretrained)
        self.max1 = nn.MaxPool2d(kernel_size=28, stride=28)
        self.max2 = nn.MaxPool2d(kernel_size=14, stride=14)
        self.max3 = nn.MaxPool2d(kernel_size=7, stride=7)
        self.num_ftrs = [40,112,960]
        self.elu = nn.ELU(inplace=True)

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(960 * 3),
            nn.Linear(960 * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )
        
        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs[0], feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs[2], kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs[2]),
            nn.Linear(self.num_ftrs[2], feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs[1], feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs[2], kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs[2]),
            nn.Linear(self.num_ftrs[2], feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs[2], feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs[2], kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs[2]),
            nn.Linear(self.num_ftrs[2], feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.slitp = nn.Sequential(
            nn.BatchNorm1d(960 * 3),
            nn.Linear(960 * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True)
        )
    def forward(self, x):
        xf3, xf4, xf5 = self.features(x)
        
        xl1 = self.conv_block1(xf3)
        xl2 = self.conv_block2(xf4)
        xl3 = self.conv_block3(xf5)
        
        xl1 = self.max1(xl1)
        xl1 = xl1.view(xl1.size(0), -1)
        xc1 = self.classifier1(xl1)

        xl2 = self.max2(xl2)
        xl2 = xl2.view(xl2.size(0), -1)
        xc2 = self.classifier2(xl2)

        xl3 = self.max3(xl3)
        xl3 = xl3.view(xl3.size(0), -1)
        xc3 = self.classifier3(xl3)
        
        x_concat = torch.cat((xl1, xl2, xl3), -1)
        
        x_concat_cls = self.classifier_concat(x_concat)
        x_concat_like = self.slitp(x_concat)
        
        return xc1, xc2, xc3, x_concat_cls,x_concat_like
    
    
def load_PMG_model(nclass,input_size,pretrain=True):
    if input_size == 224:
        return PMG_mobile_224_triploss(256,nclass,pretrain)
    elif input_size == 448:
        return PMG_mobile_448_triploss(256,nclass,pretrain)


if __name__ == "__main__":
    model = PMG_mobile_448_triploss(256,120)
    _,_,_,_,out = model(torch.randn((2,3,448,448)))
    print(out.shape)