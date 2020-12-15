import os.path as osp
import torch
import cv2
import random
import numpy as np

from torch.utils import data
from torchvision import transforms as T

class DataSet(data.Dataset):

    def __init__(self, config, img_transform):
        self.img_transform = img_transform
        self.hr_dir = config['TRAINING_CONFIG']['HR_IMG_DIR']
        self.lr_dir = config['TRAINING_CONFIG']['LR_IMG_DIR']
        self.up_scale = config['MODEL_CONFIG']['UP_SCALE']
        self.patch_size = config['TRAINING_CONFIG']['PATCH_SIZE']
        self.data_list = list(range(1, 801))
        self.h_flip = T.RandomHorizontalFlip(p=1.0)
        self.v_flip = T.RandomVerticalFlip(p=1.0)

        if 'mild' in self.lr_dir:
            self.post_fix_lr = 'x{}m.png'.format(self.up_scale)
        else:
            self.post_fix_lr = 'x{}.png'.format(self.up_scale)

    def __getitem__(self, index):
        select_num = str(self.data_list[index]).zfill(4)

        lr_image = cv2.imread(osp.join(self.lr_dir, '{}{}'.format(select_num, self.post_fix_lr)))
        hr_image = cv2.imread(osp.join(self.hr_dir, '{}.png'.format(select_num)))

        ih, iw = lr_image.shape[:2]

        tp = self.up_scale * self.patch_size
        ip = tp // self.up_scale

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        tx, ty = self.up_scale * ix, self.up_scale * iy

        lr_image = self.np2tensor(lr_image[iy:iy + ip, ix:ix + ip, :])
        hr_image = self.np2tensor(hr_image[ty:ty + tp, tx:tx + tp, :])

        if torch.rand(1) > 0.5:
            lr_image = self.h_flip(lr_image)
            hr_image = self.h_flip(hr_image)

        if torch.rand(1) > 0.5:
            lr_image = self.v_flip(lr_image)
            hr_image = self.v_flip(hr_image)

        if torch.rand(1) > 0.5:
            lr_image = self.h_flip(lr_image)
            hr_image = self.h_flip(hr_image)

        return self.img_transform(lr_image), self.img_transform(hr_image)

    def __len__(self):
        """Return the number of images."""
        return len(self.data_list)

    def np2tensor(self, img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(1 / 255.0)

        return tensor

def get_loader(config):

    img_transform = list()

    #img_transform.append(T.ToTensor())
    img_transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    img_transform = T.Compose(img_transform)

    dataset = DataSet(config, img_transform)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['TRAINING_CONFIG']['BATCH_SIZE'],
                                  shuffle=(config['TRAINING_CONFIG']['MODE'] == 'train'),
                                  num_workers=config['TRAINING_CONFIG']['NUM_WORKER'],
                                  drop_last=True)
    return data_loader
