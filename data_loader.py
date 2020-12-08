import os.path as osp
import glob
import random
from torch.utils import data
from torchvision import transforms as T
from PIL import Image

class DataSet(data.Dataset):

    def __init__(self, config, img_transform_LR, img_transform_HR):
        self.img_transform_LR = img_transform_LR
        self.img_transform_HR = img_transform_HR
        self.img_dir = config['TRAINING_CONFIG']['IMG_DIR']

        self.LR_data_list = glob.glob(osp.join(config['TRAINING_CONFIG']['LR_IMG_DIR'], '*.png'))
        self.HR_data_list = glob.glob(osp.join(config['TRAINING_CONFIG']['HR_IMG_DIR'], '*.png'))
        #random.seed(config['TRAINING_CONFIG']['CPU_SEED'])

    def __getitem__(self, index):
        hr_image = Image.open(random.choice(self.HR_data_list)).convert('RGB')
        lr_image = Image.open(random.choice(self.LR_data_list)).convert('RGB')
        return self.img_transform_LR(lr_image), self.img_transform_HR(hr_image)

    def __len__(self):
        """Return the number of images."""
        return max(len(self.LR_data_list), len(self.HR_data_list))


def get_loader(config):

    img_transform_LR = list()
    img_transform_HR = list()
    lr_img_size = config['MODEL_CONFIG']['LR_IMG_SIZE']
    hr_img_size = config['MODEL_CONFIG']['HR_IMG_SIZE']

    img_transform_LR .append(T.Resize((lr_img_size, lr_img_size)))
    img_transform_LR .append(T.ToTensor())
    img_transform_LR .append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    img_transform_LR = T.Compose(img_transform_LR)

    img_transform_HR.append(T.Resize((hr_img_size, hr_img_size)))
    img_transform_HR.append(T.ToTensor())
    img_transform_HR.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    img_transform_HR = T.Compose(img_transform_LR)

    dataset = DataSet(config, img_transform_LR, img_transform_HR)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['TRAINING_CONFIG']['BATCH_SIZE'],
                                  shuffle=(config['TRAINING_CONFIG']['MODE'] == 'train'),
                                  num_workers=config['TRAINING_CONFIG']['NUM_WORKER'],
                                  drop_last=True)
    return data_loader
