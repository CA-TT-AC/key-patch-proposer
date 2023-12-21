import argparse
import numpy as np
import os

import torch

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check

import util.misc as misc

import models_mae
from custom_datasets import ImageNetDatasetPerClass
from KPP_utils import *

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--data_path', default='/data/discover-07/xuj/workspace/datasets/imagenette2-320/train/n03000684', type=str,
                    help='dataset path')
    parser.add_argument('--ckpt', default='/data/discover-07/xuj/workspace/my_mae/ckpt/mae_visualize_vit_base.pth',
                        help='path to ckpt')
    parser.add_argument('--output_dir', default='./log/MPsearch',
                    help='path where to save, empty for no saving')
    parser.add_argument('--mask_ratio', default=0.75)
    parser.add_argument('--visualize', default=False)
    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    return parser


def MPsearch():
    args = get_args_parser()
    args = args.parse_args()
    os.makedirs(os.path.join(args.output_dir, 'pics'), exist_ok=True)
    # simple augmentation
    transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            # transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    dataset = ImageNetDatasetPerClass(args.data_path, transform=transform)
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    # sampler = torch.utils.data.RandomSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    print(dataset)
    device = 'cuda'
    model = models_mae.__dict__[args.model]()
    model.to(device)
    state_dict = torch.load(args.ckpt)['model']
    # print(state_dict.keys())
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        for i, (data, image_name) in enumerate(data_loader):
            print('image:', i, '/', len(data_loader))
            image_name = image_name[0]
            loss_list = []
            patch_list = [98]
            data = data.to(device)
            x = model.patch_embed(data)
            x = x + model.pos_embed[:, 1:, :] 
            len_keep = int(x.shape[1] * (1 - args.mask_ratio))
            if args.mask_ratio == 0:
                len_keep = x.shape[1]-1
            n_patches = x.shape[1]
            for cur_len_keep in range(len_keep):
                if (cur_len_keep+1)%20 == 0:
                    print('step:', cur_len_keep+1, '/', len_keep)
                input, ids_restore, remaining_patches, mask = search(n_patches, patch_list, cur_len_keep, x)
                out = model.forward_encoder_without_patchembed(input)
                pred = model.forward_decoder(out, ids_restore)
                loss = model.forward_batch_loss(data, pred, mask)
                cur_loss = float(min(loss).cpu())
                # print('current min loss:', cur_loss)
                loss_list.append(cur_loss)
                idx = torch.argmin(loss)
                patch_idx = int(remaining_patches[idx])
                patch_list.append(patch_idx)
                mask = mask[idx, :].unsqueeze(0)
                pred = pred[idx, :].unsqueeze(0)
            if args.visualize:
                visulize_for_show(model, pred, data, mask, output_dir=os.path.join(args.output_dir, 'pics', str(i)+'.jpg'))
            # update_data(image_name, patch_list, loss_list, args.data_path)
                

            
            
    
if __name__ == '__main__':
    MPsearch()