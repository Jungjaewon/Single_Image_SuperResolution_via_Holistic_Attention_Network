import os.path as osp
import torch
from torch.utils import data
from torchvision import transforms as T
from PIL import Image

class DataSet(data.Dataset):

    def __init__(self, config, img_transform):
        self.img_transform = img_transform
        self.hr_dir = config['TRAINING_CONFIG']['HR_IMG_DIR']
        self.lr_dir = config['TRAINING_CONFIG']['LR_IMG_DIR']
        self.up_scale = config['MODEL_CONFIG']['UP_SCALE']
        self.data_list = list(range(1, 801))
        self.h_flip = T.RandomHorizontalFlip(p=1.0)
        self.v_flip = T.RandomVerticalFlip(p=1.0)

        if 'mild' in self.lr_dir:
            self.post_fix_lr = 'x{}m.png'.format(self.up_scale)
        else:
            self.post_fix_lr = 'x{}.png'.format(self.up_scale)

    def __getitem__(self, index):
        select_num = str(self.data_list[index]).zfill(4)

        lr_image = Image.open(osp.join(self.lr_dir, '{}{}'.format(select_num, self.post_fix_lr)))
        hr_image = Image.open(osp.join(self.hr_dir, '{}.png'.format(select_num)))

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

def get_loader(config):

    img_transform = list()

    img_transform.append(T.ToTensor())
    img_transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    img_transform = T.Compose(img_transform)

    dataset = DataSet(config, img_transform)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=1,
                                  shuffle=(config['TRAINING_CONFIG']['MODE'] == 'train'),
                                  num_workers=config['TRAINING_CONFIG']['NUM_WORKER'],
                                  drop_last=True)
    return data_loader
