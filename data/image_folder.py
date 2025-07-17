"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

# import torch.utils.data as data
#
# from PIL import Image
# import os
# import os.path
#
# IMG_EXTENSIONS = [
#     '.jpg', '.JPG', '.jpeg', '.JPEG',
#     '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
#     '.tif', '.TIF', '.tiff', '.TIFF','.npy',
# ]
#
#
# def is_image_file(filename):
#     return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
#
#
# def make_dataset(dir, max_dataset_size=float("inf")):
#     images = []
#     assert os.path.isdir(dir), '%s is not a valid directory' % dir
#
#     for root, _, fnames in sorted(os.walk(dir)):
#         for fname in fnames:
#             if is_image_file(fname):
#                 path = os.path.join(root, fname)
#                 images.append(path)
#     return images[:min(max_dataset_size, len(images))]
#
#
# def default_loader(path):
#     return Image.open(path).convert('RGB')
#
#
# class ImageFolder(data.Dataset):
#
#     def __init__(self, root, transform=None, return_paths=False,
#                  loader=default_loader):
#         imgs = make_dataset(root)
#         if len(imgs) == 0:
#             raise(RuntimeError("Found 0 images in: " + root + "\n"
#                                "Supported image extensions are: " +
#                                ",".join(IMG_EXTENSIONS)))
#
#         self.root = root
#         self.imgs = imgs
#         self.transform = transform
#         self.return_paths = return_paths
#         self.loader = loader
#
#     def __getitem__(self, index):
#         path = self.imgs[index]
#         img = self.loader(path)
#         if self.transform is not None:
#             img = self.transform(img)
#         if self.return_paths:
#             return img, path
#         else:
#             return img
#
#     def __len__(self):
#         return len(self.imgs)
import torch.utils.data as data
import numpy as np
import os
import os.path
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF', '.npy',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


# 修改 default_loader，支持加载 .npy 文件
def npy_loader(path):
    img = np.load(path)  # 加载 .npy 文件
    if img.ndim == 2:  # 如果是单通道灰度图像
        img = img[None, :, :]  # 转换为 (1, H, W)
    elif img.ndim == 3:  # 如果是彩色图像
        img = img.transpose(2, 0, 1)  # 转换为 (C, H, W)
    return img


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=npy_loader):  # 使用新的 loader
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)  # 使用 npy_loader
        if self.transform is not None:
            img = self.transform(img)  # 应用转换操作
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
