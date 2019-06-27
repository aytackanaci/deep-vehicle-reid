from __future__ import print_function
from __future__ import division

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
import time
import datetime
import os.path as osp
import math
import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torch.optim import lr_scheduler

from tensorboardX import SummaryWriter

from default_parser import (
    init_parser, imagedata_kwargs, videodata_kwargs,
    optimizer_kwargs, lr_scheduler_kwargs, engine_run_kwargs
)
# from args import argument_parser, image_dataset_kwargs, optimizer_kwargs, lr_scheduler_kwargs
from torchreid.data.datamanager import ImageDataManager
from torchreid import models
from torchreid.utils import (
    Logger, set_random_seed, check_isfile, resume_from_checkpoint, load_checkpoint,
    load_pretrained_weights, compute_model_complexity, collect_env_info
)
from torchreid.losses import CrossEntropyLoss, DeepSupervision
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.loggers import RankLogger
from torchreid.utils.torchtools import count_num_param, open_all_layers, open_specified_layers, \
    load_pretrained_weights, save_checkpoint
from torchreid.utils.reidtools import visualize_ranked_results
from torchreid.metrics import evaluate_rank
from torchreid.metrics import accuracy
# from torchreid.optimizers import init_optimizer
from torchreid.optim.optimizer import build_optimizer
# from torchreid.lr_schedulers import init_lr_scheduler
from torchreid.optim.lr_scheduler import build_lr_scheduler

def exp_name(cfg):
    name = [
        'e_' + cfg.prefix,
        'S_' + '-'.join(cfg.sources),
        'T_' + '-'.join(cfg.targets),
        cfg.arch,
        'E',
        '' if cfg.resume == '' else 'r',
        '' if cfg.fixbase_epoch is 0 else 'warmup' + str(cfg.fixbase_epoch),
        str(cfg.stepsize),
        'm' + str(cfg.max_epoch),
        'P',
        'b' + str(cfg.batch_size),
        cfg.optim,
        'lr' + str(cfg.lr),
        'wd' + str(cfg.weight_decay),
        ]

    return '_'.join(name)

# global variables
parser = init_parser()
args = parser.parse_args()
# args.start_eval = args.max_epoch - 20 -1
args.save_dir = exp_name(args)


def main():
    global args

    set_random_seed(args.seed)
    if not args.use_avai_gpus: os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False
    log_name = 'test.log' if args.evaluate else 'train.log'
    sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print('==========\nArgs:{}\n=========='.format(args))

    if use_gpu:
        print('Currently using GPU {}'.format(args.gpu_devices))
        cudnn.benchmark = True
    else:
        warnings.warn('Currently using CPU, however, GPU is highly recommended')

    print('Initializing image data manager')
    print(imagedata_kwargs(args))
    dm = ImageDataManager(**imagedata_kwargs(args))
    trainloader, testloader_dict = dm.return_dataloaders()

    print('Initializing model: {}'.format(args.arch))
    model = models.build_model(name=args.arch, num_classes=dm.num_train_pids, loss='softmax', pretrained=not args.no_pretrained, use_gpu=use_gpu)
    print('Model size: {:.3f} M'.format(count_num_param(model)))

    if args.load_weights and check_isfile(args.load_weights):
        load_pretrained_weights(model, args.load_weights)

    model = nn.DataParallel(model).cuda() if use_gpu else model

    criterion = CrossEntropyLoss(num_classes=dm.num_train_pids, use_gpu=use_gpu, label_smooth=args.label_smooth)
    optimizer = build_optimizer(model, **optimizer_kwargs(args))
    scheduler = build_lr_scheduler(optimizer, **lr_scheduler_kwargs(args))
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=1e-04)

    def cyclical_lr(stepsize, min_lr=1e-3, max_lr=0.1):

        # Scaler: we can adapt this if we do not want the triangular CLR
        scaler = lambda x: 1.

        # Lambda function to calculate the LR
        lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

        # Additional function to see where on the cycle we are
        def relative(it, stepsize):
            cycle = math.floor(1 + it / (2 * stepsize))
            x = abs(it / stepsize - 2 * cycle + 1)
            return max(0, (1 - x)) * scaler(cycle)

        return lr_lambda

    factor = 6
    end_lr = args.lr
    step_size = 4*len(trainloader)
    clr = cyclical_lr(step_size, min_lr=end_lr/factor, max_lr=end_lr)
    clr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])

    if args.resume and check_isfile(args.resume):
        args.start_epoch = resume_from_checkpoint(args.resume, model, optimizer=optimizer)

    if args.evaluate:
        print('Evaluate only')

        for name in args.targets:
            print('Evaluating {} ...'.format(name))
            queryloader = testloader_dict[name]['query']
            galleryloader = testloader_dict[name]['gallery']
            distmat = test(model, queryloader, galleryloader, use_gpu, return_distmat=True)

            if args.visrank:
                visualize_ranked_results(
                    distmat, dm.return_testdataset_by_name(name),
                    save_dir=osp.join(args.save_dir, 'ranked_results', name),
                    topk=args.visrank_topk
                )
        return

    time_start = time.time()
    ranklogger = RankLogger(args.sources, args.targets)
    print('=> Start training')


    # Tensorboard
    writer = SummaryWriter(osp.join('runs', args.save_dir))


    find_clr = False
    if find_clr:
        print("Findindg upper bound for CLR")
        lr_find_epochs = 10
	start_lr = 1e-4
	end_lr = 1

        optimizer = torch.optim.SGD(model.parameters(), start_lr)


	lr_lambda = lambda x: math.exp(x * math.log(end_lr / start_lr) / (lr_find_epochs * len( trainloader)))
	clr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        lr_find_loss = []
        lr_find_lr = []
	for epoch in range(lr_find_epochs):
            lr_find_loss_e, lr_find_lr_e = train(epoch, model, clr_scheduler, criterion, optimizer, trainloader, use_gpu, writer)
            lr_find_loss.extend(lr_find_loss_e)
            lr_find_lr.extend(lr_find_lr_e)

        plt.ylabel("loss")
        plt.xlabel("loss")
        plt.xscale("log")
        plt.plot(lr_find_lr, lr_find_loss)
        plt.savefig('fig1.png', bbox_inches='tight' )

        plt.ylabel("lr")
        plt.xlabel("step")
        plt.plot(range(len(lr_find_lr)), lr_find_lr)
        plt.savefig('fig2.png', bbox_inches='tight' )


        sys.exit(1)

    if args.fixbase_epoch > 0:
        print('Train {} for {} epochs while keeping other layers frozen'.format(args.open_layers, args.fixbase_epoch))
        initial_optim_state = optimizer.state_dict()

        for epoch in range(args.fixbase_epoch):
            train(epoch, model, criterion, optimizer, trainloader, use_gpu, writer, fixbase=True)
            writer.add_scalar('train/loss', loss, epoch+1)

        print('Done. All layers are open to train for {} epochs'.format(args.max_epoch))
        optimizer.load_state_dict(initial_optim_state)

    for epoch in range(args.start_epoch, args.max_epoch):
        loss = train(epoch, model, clr_scheduler, criterion, optimizer, trainloader, use_gpu, writer)
        writer.add_scalar('train/loss', loss, epoch+1)


        if (epoch + 1) > args.start_eval and args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
            print('=> Test')

            for name in args.targets:
                print('Evaluating {} ...'.format(name))
                queryloader = testloader_dict[name]['query']
                galleryloader = testloader_dict[name]['gallery']
                rank1, mAP = test(model, queryloader, galleryloader, use_gpu)
                writer.add_scalar(name + '_test/top1', rank1, epoch+1)
                writer.add_scalar(name + '_test/mAP', mAP, epoch+1)
                ranklogger.write(name, epoch + 1, rank1)

            save_checkpoint({
                'state_dict': model.state_dict(),
                'rank1': rank1,
                'epoch': epoch + 1,
                'arch': args.arch,
                'optimizer': optimizer.state_dict(),
            }, args.save_dir)

    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Elapsed {}'.format(elapsed))
    ranklogger.show_summary()


def train(epoch, model, scheduler, criterion, optimizer, trainloader, use_gpu, writer, fixbase=False):
    losses = AverageMeter()
    accs = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_iterations = len(trainloader)

    find = False
    if find:
        lr_find_loss = []
        lr_find_lr = []
        n_iter = 0
        smoothing = 0.05


    model.train()

    if fixbase:
        open_specified_layers(model, args.open_layers)
    else:
        open_all_layers(model)

    end = time.time()
    for batch_idx, (imgs, pids, _, _) in enumerate(trainloader):
        data_time.update(time.time() - end)

        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        outputs = model(imgs)
        if isinstance(outputs, (tuple, list)):
            loss = DeepSupervision(criterion, outputs, pids)
        else:
            loss = criterion(outputs, pids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()
        writer.add_scalar('iter/loss', loss, epoch*epoch_iterations+batch_idx)

        if find:
            lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
            lr_find_lr.append(lr_step)

            # smooth the loss
            if n_iter==0:
                lr_find_loss.append(loss)
            else:
                loss = smoothing  * loss + (1 - smoothing) * lr_find_loss[-1]
                lr_find_loss.append(loss)

            n_iter += 1

        batch_time.update(time.time() - end)

        losses.update(loss.item(), pids.size(0))

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.2f} ({acc.avg:.2f})\t'.format(
                   epoch + 1, batch_idx + 1, len(trainloader),
                   batch_time=batch_time,
                   data_time=data_time,
                   loss=losses,
                   acc=accs
            ))

        end = time.time()

    if find:
        return lr_find_loss, lr_find_lr
    return losses.avg

def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20], return_distmat=False):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(queryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print('Extracted features for query set, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        end = time.time()
        for batch_idx, (imgs, pids, camids, _) in enumerate(galleryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print('Extracted features for gallery set, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

    print('=> BatchTime(s)/BatchSize(img): {:.3f}/{}'.format(batch_time.avg, args.batch_size))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print('Computing CMC and mAP')
    cmc, mAP = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)

    print('Results ----------')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r-1]))
    print('------------------')

    if return_distmat:
        return distmat
    return cmc[0], mAP


if __name__ == '__main__':
    main()
