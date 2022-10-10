import math
import sys
from typing import Iterable, Optional

import torch
from torch.distributions.normal import Normal
import torch.nn.functional as F
import torch.distributed as dist

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
from util.datasets import AddNoise
from util.consistency import consistency_loss
import util.lr_sched as lr_sched
import time
import datetime

import PIL
from torchvision import transforms, datasets


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

    # gaussian augmentation
    noised = AddNoise(args.sigma)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        samples = noised(samples)
        samples = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(samples)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
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

def train_one_epoch_con_reg(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    '''
    training one epoch with consistency regularization
    '''
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    # gaussian augmentation
    noised = AddNoise(args.sigma)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # consistency regularization
        # (b*s, c, h, w)
        samples = samples.repeat(args.num_noise_sample, 1, 1, 1)
        targets = targets.repeat(args.num_noise_sample)
        samples = noised(samples)
        samples = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(samples)
        

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss_nat = criterion(outputs, targets)
            loss_reg = consistency_loss(outputs.chunk(args.num_noise_sample), args.reg_lbd, args.reg_eta)
            loss = loss_nat + loss_reg

        loss_value = loss.item()
        loss_nat = loss_nat.item()
        loss_reg = loss_reg.item()

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
        metric_logger.update(loss_nat=loss_nat)
        metric_logger.update(loss_reg=loss_reg)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_nat_reduce = misc.all_reduce_mean(loss_nat)
        loss_reg_reduce = misc.all_reduce_mean(loss_reg)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss_nat', loss_nat_reduce, epoch_1000x)
            log_writer.add_scalar('loss_reg', loss_reg_reduce, epoch_1000x)
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
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        images = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(images)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
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

@torch.no_grad()
def evaluate_radius_0(data_loader, model, device, sigma=0.25, num_sample=100, stride=50):
    '''
    randomized smoothing on radius 0, actually a model ensembling.
    num_sample: the times of sampling, should be a multiple of stride
    sigma: the std of noise
    '''
    print('Test certified accuracy on radius 0.0 by sampling {} times.'.format(num_sample))
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    # gaussian augmentation
    noised = AddNoise(sigma)

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # sample
        batch_size, c, h, w = images.size()
        # (b, s, c, h, w)
        images = images.unsqueeze(1).repeat(1, stride, 1, 1, 1)
        # (b*s, c, h, w)
        images = images.reshape(-1, c, h, w)
        # samlping step by step to avoid CUDA out of memory
        predict = []
        for _ in range(num_sample // stride):
            noisy = noised(images)
            noisy = transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)(noisy)
            # compute output
            with torch.cuda.amp.autocast():
                # (b*s, num_classes)
                output = model(noisy)
            # (b, s)
            predict.append(output.argmax(-1).reshape(batch_size, -1))
        predict = torch.cat(predict, dim=-1)
        predict, _ = predict.mode()
        acc1 = (predict == target).sum().float() * 100 / batch_size
        metric_logger.meters['acc1_r0'].update(acc1.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Certified acc@1 {top1.global_avg:.3f}'.format(top1=metric_logger.acc1_r0))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def certify_evaluate_dist(data_loader, model, device, threshold=[0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5], num=10000):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    iter = 0
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        batch_size = images.shape[0]
        assert batch_size == 1

        with torch.cuda.amp.autocast():
            output, radius = model.certify(images, 100, num, 0.001, 1000, target.item())
            
        for thres in threshold:
            correct = float(radius >= thres and output == target)
            metric_logger.meters['Acc@r={radius:.2f}'.format(radius=thres)].update(correct, n=batch_size)
        
        iter += 1
        if iter % 50 == 0:
            dist.barrier()
            print('synchronize at iter {}'.format(iter))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}