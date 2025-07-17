# import os.path
# from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
# from PIL import Image
# import random
# import torch
# import torch.nn.functional as F
#
# class UnalignedDataset(BaseDataset):
#     """
#     This dataset class can load unaligned/unpaired datasets.
#
#     It requires two directories to host training images from domain A '/path/to/data/trainA'
#     and from domain B '/path/to/data/trainB' respectively.
#     You can train the model with the dataset flag '--dataroot /path/to/data'.
#     Similarly, you need to prepare two directories:
#     '/path/to/data/testA' and '/path/to/data/testB' during test time.
#     """
#
#     def __init__(self, opt):
#         """Initialize this dataset class.
#
#         Parameters:
#             opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
#         """
#         BaseDataset.__init__(self, opt)
#         if opt.phase == 'train':
#             self.dir_A = opt.train_imgroot  # create a path '/path/to/data/trainA'
#             self.dir_B = opt.train_maskroot  # create a path '/path/to/data/trainB'
#         else:
#             self.dir_A = opt.test_imgroot  # create a path '/path/to/data/trainA'
#             self.dir_B = opt.test_maskroot  # create a path '/path/to/data/trainB'
#
#         self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
#         self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
#         self.A_size = len(self.A_paths)  # get the size of dataset A
#         self.B_size = len(self.B_paths)  # get the size of dataset B
#         self.transform_A = get_transform(self.opt, grayscale=(opt.input_nc == 1))
#         self.transform_B = get_transform(self.opt, grayscale=True)
#
#     def __getitem__(self, index):
#         """Return a data point and its metadata information.
#
#         Parameters:
#             index (int)      -- a random integer for data indexing
#
#         Returns a dictionary that contains A, B, A_paths and B_paths
#             A (tensor)       -- an image in the input domain
#             B (tensor)       -- its corresponding image in the target domain
#             A_paths (str)    -- image paths
#             B_paths (str)    -- image paths
#         """
#         A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
#         if self.opt.serial_batches:   # make sure index is within then range
#             index_B = index % self.B_size
#         else:   # randomize the index for domain B to avoid fixed pairs.
#             index_B = random.randint(0, self.B_size - 1)
#         B_path = self.B_paths[index_B]
#         A_img = Image.open(A_path).convert('RGB')
#         B_img = Image.open(B_path).convert('L')
#         # apply image transformation
#         A = self.transform_A(A_img)
#         B = self.transform_B(B_img)
#
#         return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
#
#     def __len__(self):
#         """Return the total number of images in the dataset.
#
#         As we have two datasets with potentially different number of images,
#         we take a maximum of
#         """
#         return min(self.A_size, self.B_size)
import os.path
import numpy as np
import random
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets, especially with .npy files.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        if opt.phase == 'train':
            self.dir_A = opt.train_imgroot  # Create a path '/path/to/data/trainA'
            self.dir_B = opt.train_maskroot  # Create a path '/path/to/data/trainB'
        else:
            self.dir_A = opt.test_imgroot  # Create a path '/path/to/data/testA'
            self.dir_B = opt.test_maskroot  # Create a path '/path/to/data/testB'

        # Get all paths for the images and masks (which are in .npy format)
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # Load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # Load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # Get the size of dataset A
        self.B_size = len(self.B_paths)  # Get the size of dataset B
        self.transform_A = get_transform(self.opt, grayscale=(opt.input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=True)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain (loaded from .npy file)
            B (tensor)       -- its corresponding image in the target domain (loaded from .npy file)
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within the range
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs
            index_B = random.randint(0, self.B_size - 1)

        B_path = self.B_paths[index_B]
        # Load .npy files
        A_img = np.load(A_path)  # Load A domain image from .npy file
        B_img = np.load(B_path)  # Load B domain image from .npy file

        # Convert numpy arrays to torch tensors
        A = torch.from_numpy(A_img).float()  # Convert to float tensor
        B = torch.from_numpy(B_img).float()  # Convert to float tensor

        # 调整维度
        if A.ndimension() == 2:  # If A is a grayscale image (H, W)
            A = A.unsqueeze(0)  # Add the channel dimension to make it (1, H, W)
        if B.ndimension() == 2:  # If B is a grayscale image (H, W)
            B = B.unsqueeze(0)  # Add the channel dimension to make it (1, H, W)
        # # # Apply necessary transformations
        # A_pil_img = Image.fromarray(A_img)
        # B_pil_img = Image.fromarray(B_img)
        # A = self.transform_A(A_pil_img)
        # B = self.transform_B(B_pil_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return min(self.A_size, self.B_size)
