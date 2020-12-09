import os
import time
import datetime
import torch
import torch.nn as nn
import os.path as osp
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import save_image


class Solver(object):

    def __init__(self, config, data_loader):
        """Initialize configurations."""
        self.data_loader = data_loader

        self.up_scale = config['MODEL_CONFIG']['UP_SCALE']
        self.target_img = Image.open(config['MODEL_CONFIG']['TARGET_IMG']).convert('RGB')
        target_transform = list()
        target_transform.append(T.ToTensor())
        target_transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        target_transform = T.Compose(target_transform)
        self.target_tensor = target_transform(self.target_img)

        assert self.up_scale in [2, 4]

        self.epoch         = config['TRAINING_CONFIG']['EPOCH']
        self.batch_size    = config['TRAINING_CONFIG']['BATCH_SIZE']
        self.g_lr          = float(config['TRAINING_CONFIG']['G_LR'])
        self.d_lr          = float(config['TRAINING_CONFIG']['D_LR'])
        self.lambda_g_fake = config['TRAINING_CONFIG']['LAMBDA_G_FAKE']
        self.lambda_g_recon = config['TRAINING_CONFIG']['LAMBDA_G_RECON']
        self.lambda_d_fake = config['TRAINING_CONFIG']['LAMBDA_D_FAKE']
        self.lambda_d_real = config['TRAINING_CONFIG']['LAMBDA_D_REAL']
        self.lambda_d_gp     = config['TRAINING_CONFIG']['LAMBDA_GP']
        self.d_critic      = config['TRAINING_CONFIG']['D_CRITIC']
        self.g_critic      = config['TRAINING_CONFIG']['G_CRITIC']

        self.gan_loss = config['TRAINING_CONFIG']['GAN_LOSS']
        assert self.gan_loss in ['lsgan', 'wgan', 'vanilla']

        self.optim = config['TRAINING_CONFIG']['OPTIM']
        self.beta1 = config['TRAINING_CONFIG']['BETA1']
        self.beta2 = config['TRAINING_CONFIG']['BETA2']
        if self.gan_loss == 'lsgan':
            self.adversarial_loss = torch.nn.MSELoss()
        elif self.gan_loss =='vanilla':
            self.adversarial_loss = torch.nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()

        self.cpu_seed = config['TRAINING_CONFIG']['CPU_SEED']
        self.gpu_seed = config['TRAINING_CONFIG']['GPU_SEED']
        #torch.manual_seed(config['TRAINING_CONFIG']['CPU_SEED'])
        #torch.cuda.manual_seed_all(config['TRAINING_CONFIG']['GPU_SEED'])

        self.g_spec = config['TRAINING_CONFIG']['G_SPEC'] == 'True'
        self.d_spec = config['TRAINING_CONFIG']['D_SPEC'] == 'True'

        self.gpu = config['TRAINING_CONFIG']['GPU']
        self.use_tensorboard = config['TRAINING_CONFIG']['USE_TENSORBOARD']

        # Directory
        self.train_dir  = config['TRAINING_CONFIG']['TRAIN_DIR']
        self.log_dir    = os.path.join(self.train_dir, config['TRAINING_CONFIG']['LOG_DIR'])
        self.sample_dir = os.path.join(self.train_dir, config['TRAINING_CONFIG']['SAMPLE_DIR'])
        self.result_dir = os.path.join(self.train_dir, config['TRAINING_CONFIG']['RESULT_DIR'])
        self.model_dir  = os.path.join(self.train_dir, config['TRAINING_CONFIG']['MODEL_DIR'])

        # Steps
        self.log_step       = config['TRAINING_CONFIG']['LOG_STEP']
        self.sample_step    = config['TRAINING_CONFIG']['SAMPLE_STEP']
        self.save_step      = config['TRAINING_CONFIG']['SAVE_STEP']
        self.save_start     = config['TRAINING_CONFIG']['SAVE_START']
        self.lr_decay_step  = config['TRAINING_CONFIG']['LR_DECAY_STEP']
        self.decay_start  = config['TRAINING_CONFIG']['DECAY_START']

        self.build_model()

        if self.use_tensorboard == 'True':
            self.build_tensorboard()

    def build_model(self):

        if self.up_scale == 2:
            self.D_L = LR_Discriminator32(spec_norm=self.d_spec, LR=0.02)
            self.D_H = HR_Discriminator64(spec_norm=self.d_spec, LR=0.02)
            self.G_L = DSN64(spec_norm=False, LR=0.02, inner_channel=32)
            self.G_H = SRN(spec_norm=False, LR=0.02, inner_channel=32, up_scale=self.up_scale)

        elif self.up_scale == 4:
            self.D_L = LR_Discriminator32(spec_norm=self.d_spec, LR=0.02)
            self.D_H = HR_Discriminator128(spec_norm=self.d_spec, LR=0.02)
            self.G_L = DSN128(spec_norm=False, LR=0.02, inner_channel=64)
            self.G_H = SRN(spec_norm=False, LR=0.02, inner_channel=64, up_scale=self.up_scale)

        self.g_l_optimizer = torch.optim.Adam(self.G_L.parameters(), self.g_lr, (self.beta1, self.beta2))
        self.g_h_optimizer = torch.optim.Adam(self.G_H.parameters(), self.g_lr, (self.beta1, self.beta2))
        self.d_l_optimizer = torch.optim.Adam(self.D_L.parameters(), self.d_lr, (self.beta1, self.beta2))
        self.d_h_optimizer = torch.optim.Adam(self.D_H.parameters(), self.d_lr, (self.beta1, self.beta2))

        self.print_network(self.G_L, 'G_L')
        self.print_network(self.G_H, 'G_H')
        self.print_network(self.D_L, 'D_L')
        self.print_network(self.D_H, 'D_H')

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        #print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

        with open(os.path.join(self.train_dir,'model_arch.txt'), 'a') as fp:
            print(model, file=fp)
            print(name, file=fp)
            print("The number of parameters: {}".format(num_params),file=fp)

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_l_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.g_h_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_l_optimizer.param_groups:
            param_group['lr'] = d_lr
        for param_group in self.d_h_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_l_optimizer.zero_grad()
        self.g_h_optimizer.zero_grad()
        self.d_l_optimizer.zero_grad()
        self.d_h_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.gpu)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def get_state_dict(self, path):

        if path.startswith("module-"):
            state_dict = torch.load(path, map_location=lambda storage, loc: storage)
            # state_dict = torch.load(model_path, map_location=self.device)
            # https://github.com/computationalmedia/semstyle/issues/3
            # https://github.com/pytorch/pytorch/issues/10622
            # https://discuss.pytorch.org/t/saving-and-loading-torch-models-on-2-machines-with-different-number-of-gpu-devices/6666/2
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            return new_state_dict
        else:
            return torch.load(path, map_location=lambda storage, loc: storage)

    def restore_model(self, epoch):

        D_L_path = osp.join(self.model_dir, '*{}-D_L.ckpt'.format(epoch))
        D_H_path = osp.join(self.model_dir, '*{}-D_H.ckpt'.format(epoch))
        G_L_path = os.path.join(self.model_dir, '*{}-G_L.ckpt'.format(epoch))
        G_H_path = os.path.join(self.model_dir, '*{}-G_H.ckpt'.format(epoch))

        self.G_L.load_state_dict(self.get_state_dict(G_L_path))
        self.G_H.load_state_dict(self.get_state_dict(G_H_path))
        self.D_L.load_state_dict(self.get_state_dict(D_L_path))
        self.D_H.load_state_dict(self.get_state_dict(D_H_path))

        return epoch

    def calculate_d_loss(self, G, D, input_image, gt_image):
        fake_hr_image = G(input_image)
        real_score = D(gt_image)
        fake_score = D(fake_hr_image.detach())
        d_loss_real = -torch.mean(real_score)
        d_loss_fake = torch.mean(fake_score)
        alpha = torch.rand(gt_image.size(0), 1, 1, 1).to(self.gpu)
        x_hat = (alpha * gt_image.data + (1 - alpha) * fake_hr_image.data).requires_grad_(True)
        out_src = self.D_H(x_hat)
        d_loss_gp = self.gradient_penalty(out_src, x_hat)
        d_loss = self.lambda_d_real * d_loss_real + self.lambda_d_fake * d_loss_fake + self.lambda_d_gp * d_loss_gp
        return d_loss

    def train(self):

        # Set data loader.
        data_loader = self.data_loader
        iterations = len(self.data_loader)
        print('iterations : ', iterations)
        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        #fixed_lr_image, fixed_hr_image = next(data_iter)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        start_time = time.time()
        print('Start training...')
        for e in range(self.epoch):

            for i in range(iterations):
                try:
                    lr_image, hr_image = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    lr_image, hr_image = next(data_iter)

                lr_image = lr_image.to(self.gpu)
                hr_image = hr_image.to(self.gpu)

                loss_dict = dict()
                if (i + 1) % self.d_critic == 0:

                    d_h_loss = self.calculate_d_loss(self.G_H, self.D_H, input_image=lr_image, gt_image=hr_image)
                    d_l_loss = self.calculate_d_loss(self.G_L, self.D_L, input_image=hr_image, gt_image=lr_image)

                    # Backward and optimize.
                    self.reset_grad()
                    d_h_loss.backward()
                    d_l_loss.backward()
                    self.d_h_optimizer.step()
                    self.d_l_optimizer.step()

                    # Logging.
                    loss_dict['D/d_h_loss'] = d_h_loss.item()
                    loss_dict['D/d_l_loss'] = d_l_loss.item()

                if (i + 1) % self.g_critic == 0:
                    g_srn_cycle = torch.mean(self.D_L(self.G_L(self.G_H(lr_image)))) -torch.mean(self.D_L(lr_image)) + self.l1_loss(self.G_L(self.G_H(lr_image)), lr_image)
                    g_dsn_cycle = torch.mean(self.D_H(self.G_H(self.G_L(lr_image)))) -torch.mean(self.D_H(hr_image)) + self.l1_loss(self.G_H(self.G_L(hr_image)), hr_image)

                    self.reset_grad()
                    g_srn_cycle.backward()
                    g_dsn_cycle.backward()
                    self.g_h_optimizer.step()
                    self.g_l_optimizer.step()

                    # Logging.
                    loss_dict['G/g_dsn_cycle'] = g_dsn_cycle.item()
                    loss_dict['G/g_srn_cycle'] = g_srn_cycle.item()

                if (i + 1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Epoch [{}/{}], Elapsed [{}], Iteration [{}/{}]".format(e+1, self.epoch, et, i + 1, iterations)
                    for tag, value in loss_dict.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

            if (e + 1) % self.sample_step == 0:
                fake_hr = self.G_H(self.target_tensor)
                sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(e + 1))
                save_image(self.denorm(fake_hr.data.cpu()), sample_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(self.sample_dir))
            # Save model checkpoints.
            if (e + 1) % self.save_step == 0 and (e + 1) >= self.save_start:
                D_L_path = osp.join(self.model_dir, '*{}-D_L.ckpt'.format(e + 1))
                D_H_path = osp.join(self.model_dir, '*{}-D_H.ckpt'.format(e + 1))
                G_L_path = os.path.join(self.model_dir, '*{}-G_L.ckpt'.format(e + 1))
                G_H_path = os.path.join(self.model_dir, '*{}-G_H.ckpt'.format(e + 1))

                torch.save(self.G_L.state_dict(), G_L_path)
                torch.save(self.G_H.state_dict(), G_H_path)
                torch.save(self.D_L.state_dict(), D_L_path)
                torch.save(self.D_H.state_dict(), D_H_path)
                print('Saved model checkpoints into {}...'.format(self.model_dir))

            # decay learning rate
            if (e + 1) % self.lr_decay_step == 0 and (e + 1) >= self.decay_start:
                g_lr = g_lr * 0.1
                d_lr = d_lr * 0.1
                self.update_lr(g_lr, d_lr)

        print('Training is finished')

    def test(self):
        pass

