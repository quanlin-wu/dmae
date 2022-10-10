import torch
import numpy as np
import argparse
import os
from pathlib import Path
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import util.misc as misc
from util.datasets import build_dataset, build_dataset_with_interval
from util.smooth import Smooth

import models_vit

from engine_finetune import certify_evaluate_dist


def get_args_parser():
    parser = argparse.ArgumentParser('Test of certified accuracy', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # * Finetuning params
    parser.add_argument('--global_pool', action='store_true')
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    parser.add_argument('--linprobe', action='store_true',
                        help='activated when testing linear-probing')
    parser.set_defaults(global_pool=True)

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default=None,
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--eval', action='store_true', default=True,
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
    
    # certified parameters
    parser.add_argument('--sigma', default=None, type=float,
                        help='standard deviation for randomized smoothing')
    parser.add_argument('--sample_interval', default=50, type=int,
                        help="the interval of sampling during test")

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_val = build_dataset_with_interval(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    print('len(sampler_val)', len(sampler_val))
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )
    if args.linprobe:
        # for linear prob only
        # hack: revise model's head with BN
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=None, loss_scaler=None)
    
    # switch to evaluation mode
    model.eval()
    threshold=[0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    if args.sigma:
        smoothed_classifier = Smooth(model, num_classes, args.sigma)
        test_stats = certify_evaluate_dist(data_loader_val, smoothed_classifier, device, threshold)
        print('* Load model from {}'.format(args.resume))
        print('* Interval of sampling: {}(number of datapoints: {})'.format(args.sample_interval, len(dataset_val)))
        print('* Randomized smoothing with sigma {}'.format(args.sigma))
        print('* Certiﬁed test accuracy:')
        for thres in threshold:
            print('* Acc@r={radius:.2f} {acc:.3f}'.format(
                radius=thres, 
                acc=test_stats['Acc@r={radius:.2f}'.format(radius=thres)]))
    else: # test on sigma = (0.25, 0.5, 1.0)
        for sigma in [0.25, 0.5, 1.0]:
            smoothed_classifier = Smooth(model, num_classes, sigma)
            test_stats = certify_evaluate_dist(data_loader_val, smoothed_classifier, device, threshold)
            print('* Load model from {}'.format(args.resume))
            print('* Interval of sampling: {}(number of datapoints: {})'.format(args.sample_interval, len(dataset_val)))
            print('* Randomized smoothing with sigma {}'.format(sigma))
            print('* Certiﬁed test accuracy:')
            for thres in threshold:
                print('* Acc@r={radius:.2f} {acc:.3f}'.format(
                    radius=thres, 
                    acc=test_stats['Acc@r={radius:.2f}'.format(radius=thres)]))
    exit(0)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # if args.output_dir:
    #    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    num_classes = 1000
    main(args)
