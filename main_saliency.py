
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import util.lr_sched as lr_sched
import matplotlib.pyplot as plt
import requests


import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import datasets, transforms
import timm


# assert timm.__version__ == "0.3.2"  # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models.vision_transformer import PatchEmbed, Block

import models_mae
import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import torch.nn as nn

from models_mae import MaskedAutoencoderViT

import models_vit

# from engine_finetune import train_one_epoch, evaluate

#xx
def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')

    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')



    #___________________#
    parser.add_argument('--result_dir', default='./result')
    parser.add_argument('--masking_ratio', default=0., type=float)

    return parser


class xd():
    def xd(self):
        print('xd')
class Saliency_model(MaskedAutoencoderViT):
    def __init__(self, encoder):
        super(Saliency_model, self).__init__(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4)
        self.encoder = encoder
        self.decoder_pred = nn.Linear(self.decoder_embed_dim, self.patch_size ** 2 * 1, bias=True)  # decoder to patch

    def unpatchify2(self, x):
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs

    def patchify2(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2 *1)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 1))
        return x


    def loss(self, img, target):
        target = self.patchify2(target)
        loss = (img - target) ** 2
        loss = loss.mean(dim=-1)
        loss = loss.mean()
        return loss

    def forward(self, img, target, mask_ratio):
        latent, mask, ids_restore = self.encoder(img, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.loss(pred, target)
        return loss, pred


def main(args):
    # misc.init_distributed_mode(args)
    folder_path = args.result_dir
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # cudnn.benchmark = True
    # dataset_train = build_dataset(is_train=True, args=args)
    # dataset_val = build_dataset(is_train=False, args=args)

    def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
        # build model
        model = getattr(models_mae, arch)()
        # load model
        checkpoint = torch.load(chkpt_dir, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)
        return model

    chkpt_dir = 'demo/mae_visualize_vit_base.pth'
    encoder = prepare_model(chkpt_dir, 'mae_vit_base_patch16')
    encoder =encoder.to(device )

    for param in encoder.parameters():
        param.requires_grad = False

    model = Saliency_model(encoder.forward_encoder)
    model = model.to(device)


    class CustomDataset(Dataset):
        def __init__(self, root_dir1, root_dir2, transform=None):
            self.root_dir1 = root_dir1
            self.root_dir2 = root_dir2
            self.transform = transform
            self.images1 = os.listdir(root_dir1)
            self.images2 = os.listdir(root_dir2)

        def __len__(self):
            return min(len(self.images1), len(self.images2))  # Ensure both sets have the same number of images

        def __getitem__(self, idx):
            img_name1 = os.path.join(self.root_dir1, self.images1[idx])
            img_name2 = os.path.join(self.root_dir2, os.path.splitext(self.images1[idx])[0] + '.png')

            image1 = Image.open(img_name1)
            image2 = Image.open(img_name2)

            if self.transform:
                image1 = self.transform(image1)
                image2 = self.transform(image2)
            return image1, image2

    transform = transforms.Compose([
        transforms.CenterCrop((400, 400)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset_train = CustomDataset(root_dir1='data/salicon/images/train', root_dir2='data/salicon/maps-2/train', transform=transform)
    dataset_val = CustomDataset(root_dir1='data/salicon/images/val', root_dir2='data/salicon/maps-2/val', transform=transform)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    batch_size = 32

    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    data_loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)



    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module


    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay, layer_decay=args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)


    for epoch in range(args.start_epoch, args.epochs):
        accum_iter = args.accum_iter

        # if args.distributed:
        #     data_loader_train.sampler.set_epoch(epoch)
        for i, (samples, targets) in enumerate(data_loader_train):
            samples = samples.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss, pred = model(samples, targets, args.masking_ratio)
            loss.sum().backward()
            optimizer.step()

            # if i % 25 ==0:
            #     with torch.no_grad():
            #         image1 = samples[0]  # Assuming you want to display the first image
            #         image2 = targets[0]  # Similarly for the second DataLoader
            #
            #         # Display the first image using Matplotlib
            #         plt.imshow(image1.permute(1, 2, 0))  # Permute dimensions for visualization (C, H, W) to (H, W, C)
            #         plt.title('Image from DataLoader 1')
            #         plt.show()
            #
            #         # Display the second image from the second DataLoader
            #         plt.imshow(image2.permute(1, 2, 0))  # Permute dimensions for visualization (C, H, W) to (H, W, C)
            #         plt.title('Image from DataLoader 2')
            #         plt.show()
            #
            #         predx = model.unpatchify2(pred)
            #         predx = predx[0]
            #         plt.imshow(predx.permute(1, 2, 0))  # Permute dimensions for visualization (C, H, W) to (H, W, C)
            #         plt.title('pred')
            #         plt.show()



            if i % accum_iter == 0:
                lr_sched.adjust_learning_rate(optimizer, i / len(data_loader_train) + epoch, args)

        if args.output_dir:
            torch.save(model.state_dict(), os.path.join(folder_path, 'model_state.pth'))
            torch.save(optimizer.state_dict(), os.path.join(folder_path, 'optimizer_state.pth'))


        with torch.no_grad():
            samples, targets = next(iter(data_loader_val))
            samples = samples.to(device)
            targets = targets.to(device)
            loss, pred = model(samples, targets, args.masking_ratio)
            print(f"MSE val: {loss}")

            image1 = samples[0]  # Assuming you want to display the first image
            image2 = targets[0]  # Similarly for the second DataLoader
            plt.imsave(os.path.join(folder_path, f'img{epoch}.jpg'), image1.permute(1, 2, 0).cpu().numpy())

            image_np = image2.squeeze().cpu().numpy()  # Assuming the tensor has shape (1, 224, 224)
            image_np_uint8 = (image_np * 255).astype('uint8')  # Convert to uint8 (0-255 range)
            image2_pil = Image.fromarray(image_np_uint8)
            image2_pil.save(os.path.join(folder_path, f'target{epoch}.png'))

            predx = model.unpatchify2(pred)
            predx=predx[0]
            image_np = predx.squeeze().cpu().numpy()  # Assuming the tensor has shape (1, 224, 224)
            image_np_uint8 = (image_np * 255).astype('uint8')  # Convert to uint8 (0-255 range)
            image2_pil = Image.fromarray(image_np_uint8)
            image2_pil.save(os.path.join(folder_path, f'result{epoch}.png'))


if __name__ == '__main__':
    print("test")
    print(os.getcwd())
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    main(args)
