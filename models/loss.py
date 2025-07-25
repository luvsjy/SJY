# import torch
# from torchvision import models
# import torch.nn as nn
#
# def l1_loss_mask(inputs, targets, mask):
#     loss = torch.sum(torch.abs(inputs - targets), dim=[1,2,3])
#     xs = targets.size()
#     ratio = torch.sum(mask, dim=[1,2,3]) * xs[1]  # mask: BWH1. if mask = BWH3, then remove '*3'
#     loss_mean = torch.mean(loss / (ratio + 1e-12))  # avoid mask = 0
#     return loss_mean
#
# class VGG16FeatureExtractor(nn.Module):
#     def __init__(self):
#         super().__init__()
#         vgg16 = models.vgg16(pretrained=True)
#         self.enc_1 = nn.Sequential(*vgg16.features[:5])
#         self.enc_2 = nn.Sequential(*vgg16.features[5:10])
#         self.enc_3 = nn.Sequential(*vgg16.features[10:17])
#
#         # fix the encoder
#         for i in range(3):
#             for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
#                 param.requires_grad = False
#
#     def forward(self, image):
#         results = [image]
#         for i in range(3):
#             func = getattr(self, 'enc_{:d}'.format(i + 1))
#             results.append(func(results[-1]))
#         return results[1:]
#
# def style_loss(A_feats, B_feats):
#     assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
#     loss_value = 0.0
#     for i in range(len(A_feats)):
#         A_feat = A_feats[i]
#         B_feat = B_feats[i]
#         _, c, w, h = A_feat.size()
#         A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
#         B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
#         A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
#         B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
#         loss_value += torch.mean(torch.abs(A_style - B_style)/(c * w * h))
#     return loss_value
#
# def TV_loss(x):
#     h_x = x.size(2)
#     w_x = x.size(3)
#     h_tv = torch.mean(torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]))
#     w_tv = torch.mean(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]))
#     return h_tv + w_tv
#
# def perceptual_loss(A_feats, B_feats):
#     assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
#     loss_value = 0.0
#     for i in range(len(A_feats)):
#         A_feat = A_feats[i]
#         B_feat = B_feats[i]
#         loss_value += torch.mean(torch.abs(A_feat - B_feat))
#     return loss_value
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# L1 Loss with Masking for Gray Images (Single Channel)
def l1_loss_mask(inputs, targets, mask):
    # 计算L1损失，只在H和W维度上求和，不需要考虑通道数
    loss = torch.sum(torch.abs(inputs - targets), dim=[1, 2, 3])
    xs = targets.size()
    ratio = torch.sum(mask, dim=[1, 2, 3]) * xs[1]  # 只考虑H和W的尺寸
    loss_mean = torch.mean(loss / (ratio + 1e-12))  # 避免mask为0
    return loss_mean

# VGG16 Feature Extractor - For Gray Images (Single Channel)
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)

        # 修改VGG16的输入通道数，从3改为1
        vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # 冻结特征提取器的权重
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        # 如果输入是灰度图（单通道），直接使用
        if image.size(1) == 1:  # 单通道灰度图
            results = [image]
            for i in range(3):
                func = getattr(self, 'enc_{:d}'.format(i + 1))
                results.append(func(results[-1]))
            return results[1:]
        else:
            # 如果是三通道（RGB图像），按常规处理
            results = [image]
            for i in range(3):
                func = getattr(self, 'enc_{:d}'.format(i + 1))
                results.append(func(results[-1]))
            return results[1:]
# Style Loss - For Gray Images
def style_loss(A_feats, B_feats):
    assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value = 0.0
    for i in range(len(A_feats)):
        A_feat = A_feats[i]
        B_feat = B_feats[i]
        _, c, w, h = A_feat.size()
        A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
        B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
        A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
        B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
        # 对于灰度图像，c=1，w和h保持不变
        loss_value += torch.mean(torch.abs(A_style - B_style) / (c * w * h))
    return loss_value

# Total Variation (TV) Loss for Gray Images
def TV_loss(x):
    h_x = x.size(2)
    w_x = x.size(3)
    h_tv = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :h_x-1, :]))
    w_tv = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x-1]))
    return h_tv + w_tv

# Perceptual Loss - For Gray Images
def perceptual_loss(A_feats, B_feats):
    assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value = 0.0
    for i in range(len(A_feats)):
        A_feat = A_feats[i]
        B_feat = B_feats[i]
        # 比较灰度图像的特征图
        loss_value += torch.mean(torch.abs(A_feat - B_feat))
    return loss_value
