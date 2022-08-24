from model.ESRGAN import ESRGAN , ESRGAN_tiny
import os
from glob import glob
import torch
from torchvision.utils import save_image
import torch.nn as nn


class Tester:
    def __init__(self, config, data_loader):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = config.checkpoint_dir
        self.data_loader = data_loader
        self.scale_factor = config.scale_factor
        self.sample_dir = config.sample_dir
        self.num_epoch = config.num_epoch
        self.image_size = config.image_size
        self.upsampler = nn.Upsample(scale_factor=self.scale_factor)
        self.epoch = config.epoch
        self.build_model()

    def test(self):
        self.generator.eval()
        total_step = len(self.data_loader)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        for step, image in enumerate(self.data_loader):
            low_resolution_y = image['lr'].to(self.device)
            high_resolution_y = image['hr'].to(self.device)
            fake_high_resolution_y = self.generator(low_resolution_y)
            low_resolution_y = self.upsampler(low_resolution_y)
            print(f"[Batch {step}/{total_step}]... ")
            low_resolution_u = image['lr_u'].to(self.device)
            low_resolution_v = image['lr_v'].to(self.device)
            high_resolution_u = image['hr_u'].to(self.device)
            high_resolution_v = image['hr_v'].to(self.device)
            fake_high_resolution_u = low_resolution_u
            fake_high_resolution_v = low_resolution_v
            result_y = torch.cat((low_resolution_y, fake_high_resolution_y, high_resolution_y), 2)
            result_u = torch.cat((low_resolution_u, fake_high_resolution_u, high_resolution_u), 2)
            result_v = torch.cat((low_resolution_v, fake_high_resolution_v, high_resolution_v), 2)
            result = torch.cat((result_y, result_u, result_v), 2)
            save_image(result, os.path.join(self.sample_dir, f"SR_{step}.png"))

    def build_model(self):
        self.generator = ESRGAN_tiny(1, 1, nf=32, gc=16, n_basic_block=6, scale_factor=self.scale_factor).to(self.device)
        self.load_model()

    def load_model(self):
        print(f"[*] Load model from {self.checkpoint_dir}")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.listdir(self.checkpoint_dir):
            raise Exception(f"[!] No checkpoint in {self.checkpoint_dir}")

        generator = glob(os.path.join(self.checkpoint_dir, f'generator_{self.epoch - 1}.pth'))

        self.generator.load_state_dict(torch.load(generator[0]))