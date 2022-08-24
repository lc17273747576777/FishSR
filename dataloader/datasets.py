from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms.functional as TF
import random
import cv2

class Datasets(Dataset):
    def __init__(self, image_size, scale, randomflag=False):
        self.image_size = image_size
        self.scale = scale
        self.randomflag = randomflag
        if not os.path.exists('datasets'):
            raise Exception(f"[!] dataset is not exited")

        self.image_file_name = sorted(os.listdir(os.path.join('F:/WJ_project/dataset/realSR/crop576/', 'HR')))

    def __getitem__(self, item):
        file_name = self.image_file_name[item]
        # high_resolution = Image.open(os.path.join('./datasets', 'training/HR', file_name)).convert('RGB')
        # low_resolution = Image.open(os.path.join('./datasets', 'training/LR', file_name)).convert('RGB')
        high_resolution = cv2.imread(os.path.join('F:/WJ_project/dataset/realSR/crop576/', 'HR', file_name))
        low_resolution = cv2.imread(os.path.join('F:/WJ_project/dataset/realSR/crop576/', 'LR_2D', file_name))
        h, w, c = high_resolution.shape
        # if self.scale != 1:
        #     high_resolution = cv2.resize(high_resolution, dsize=(0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LANCZOS4)
        if self.randomflag:
            # high_resolution = cv2.bilateralFilter(high_resolution, d=3, sigmaSpace=80, sigmaColor=80)
            xstart = random.randint(0, w - self.image_size)
            ystart = random.randint(0, h - self.image_size)
        else:
            xstart = 0
            ystart = 0
        high_resolution_crop = high_resolution[int(self.scale*ystart):int(self.scale*(ystart + self.image_size)), int(self.scale*(xstart)):int(self.scale*(xstart + self.image_size)), :]
        low_resolution_crop = low_resolution[ystart:ystart + self.image_size, xstart:xstart + self.image_size, :]
        # if random() > 0.5:
        #     high_resolution = TF.vflip(high_resolution)
        #     low_resolution = TF.vflip(low_resolution)
        #
        # if random() > 0.5:
        #     high_resolution = TF.hflip(high_resolution)
        #     low_resolution = TF.hflip(low_resolution)

        high_resolution_yuv = cv2.cvtColor(high_resolution_crop, cv2.COLOR_BGR2YUV)
        low_resolution_yuv = cv2.cvtColor(low_resolution_crop, cv2.COLOR_BGR2YUV)

        high_resolution_y = TF.to_tensor(Image.fromarray(high_resolution_yuv[:, :, 0]))
        low_resolution_y = TF.to_tensor(Image.fromarray(low_resolution_yuv[:, :, 0]))
        high_resolution_u = TF.to_tensor(Image.fromarray(high_resolution_yuv[:, :, 1]))
        low_resolution_u = TF.to_tensor(Image.fromarray(low_resolution_yuv[:, :, 1]))
        high_resolution_v = TF.to_tensor(Image.fromarray(high_resolution_yuv[:, :, 2]))
        low_resolution_v = TF.to_tensor(Image.fromarray(low_resolution_yuv[:, :, 2]))
        images = {'lr': low_resolution_y, 'lr_u': low_resolution_u,'lr_v': low_resolution_v, 'hr': high_resolution_y, 'hr_u': high_resolution_u, 'hr_v': high_resolution_v}

        return images

    def __len__(self):
        return len(self.image_file_name)
