# 假设 x.shape[1] 是所有可能的 patches 的数量
import json
import os
from matplotlib import pyplot as plt
import numpy as np
import torch

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched

def show_image(image, title=''):
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def visulize(model, pred, data, mask, output_dir):
    y = model.unpatchify(pred)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', data).cpu()

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")
    plt.savefig(output_dir)


def visulize_for_show(model, pred, data, mask, output_dir):
    y = model.unpatchify(pred)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', data).cpu()

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    plt.figure(figsize=(12, 3))

    plt.subplot(1, 3, 1)
    show_image(x[0])
    plt.axis('off')  

    plt.subplot(1, 3, 2)
    show_image(im_masked[0])
    plt.axis('off')  

    plt.subplot(1, 3, 3)
    show_image(im_paste[0])
    plt.axis('off')  

    plt.subplots_adjust(left=0, right=0.5, wspace=0, hspace=0)

    plt.savefig(output_dir, bbox_inches='tight', pad_inches=0)

def update_data(image_name, patch_list, loss_list, dir):
    file_path = os.path.join(dir, "patch_ids.json")
    # Attempt to read existing data from a file
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    # Update or add new data
    data[image_name] = {'patch': patch_list, 'loss': loss_list}

    # Write the updated data back to the file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def search(n_patches, patch_list, cur_len_keep, x,device='cuda'):
    all_patches = set(range(n_patches))
    # selected patches
    selected_patches = set(patch_list)
    # remove selected patches，get patches waiting for selection
    remaining_patches = list(all_patches - selected_patches)
    patch_list_tensor = torch.tensor(patch_list)
    remaining_patches_tensor = torch.tensor(remaining_patches)
    patch_list_expanded = patch_list_tensor.unsqueeze(0).expand(len(remaining_patches), -1)
    remaining_patches_expanded = remaining_patches_tensor.unsqueeze(1)
    combined_index = torch.cat((patch_list_expanded, remaining_patches_expanded), dim=1).to(device)
    n_groups = combined_index.shape[0]
    x_expanded = x.expand(n_groups, -1, -1)
    input = x_expanded[torch.arange(n_groups).unsqueeze(1), combined_index]
    complete_indices = torch.arange(n_patches).repeat(n_groups, 1)
    mask = torch.ones_like(complete_indices).to(device)
    mask = mask.scatter_(1, combined_index, False)
    
    remaining_indices = complete_indices[mask.bool()].reshape(n_groups, -1).to(device)
    # print(combined_index, remaining_indices)
    ids_keep = torch.cat([combined_index, remaining_indices], dim=1).to(device)
    ids_restore = torch.argsort(ids_keep, dim=1)
    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([n_groups, n_patches], device=x.device)
    mask[:, :cur_len_keep+2] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return input, ids_restore, remaining_patches, mask




def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets, patch_ids) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        patch_ids = patch_ids.to(device, non_blocking=True)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples, patch_ids)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        patch_ids = batch[2]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        patch_ids = patch_ids.to(device, non_blocking=True)
        # compute output
        with torch.cuda.amp.autocast():
            output = model(images, patch_ids)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}