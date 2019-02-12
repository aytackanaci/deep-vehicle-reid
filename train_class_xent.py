from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

from args import argument_parser, image_dataset_kwargs, optimizer_kwargs
from torchreid.data_manager import ImageDataManager
from torchreid.class_data_manager import ClassDataManager
from torchreid import models
from torchreid.losses import CrossEntropyLoss, DeepSupervision
from torchreid.utils.iotools import save_checkpoint, check_isfile
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.loggers import Logger, RankLogger
from torchreid.utils.torchtools import count_num_param, open_all_layers, open_specified_layers
from torchreid.utils.reidtools import visualize_ranked_results
from torchreid.utils.generaltools import set_random_seed
from torchreid.eval_metrics import evaluate, accuracy
from torchreid.optimizers import init_optimizer

def exp_name(cfg):
    name = [
        'e_' + cfg.prefix,
        'S_' + '-'.join(cfg.source_names),
        'T_' + '-'.join(cfg.target_names),
        cfg.arch,
        'E',
        '' if cfg.resume == '' else 'r',
        '' if cfg.fixbase_epoch is 0 else 'warmup' + str(cfg.fixbase_epoch),
        str(cfg.stepsize),
        'm' + str(cfg.max_epoch),
        'P',
        'b' + str(cfg.train_batch_size),
        cfg.optim,
        'lr' + str(cfg.lr),
        'wd' + str(cfg.weight_decay),
        ]

    return '_'.join(name)

# read config
parser = argument_parser()
args = parser.parse_args()
args.start_eval = args.max_epoch - 20 -1
args.save_dir = exp_name(args)


def main():
    global args

    set_random_seed(args.seed)
    if not args.use_avai_gpus: os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False
    log_name = 'log_test.txt' if args.evaluate else 'log_train.txt'
    sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
    else:
        print("Currently using CPU, however, GPU is highly recommended")

    print("Initializing image data manager")
    dm = ClassDataManager(use_gpu, **image_dataset_kwargs(args))
    trainloader, testloader_dict = dm.return_dataloaders()

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dm.num_train_pids, input_size=args.width, loss={'xent'}, use_gpu=use_gpu)
    print("Model size: {:.3f} M".format(count_num_param(model)))
    # print(model)
    # sys.exit(0)

    criterion = CrossEntropyLoss(num_classes=dm.num_train_pids, use_gpu=use_gpu, label_smooth=args.label_smooth)
    optimizer = init_optimizer(model.parameters(), **optimizer_kwargs(args))
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True, threshold=1e-04)

    if args.load_weights and check_isfile(args.load_weights): # load pretrained weights but ignore layers that don't match in size
        checkpoint = torch.load(args.load_weights)
        pretrain_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
        print("Loaded pretrained weights from '{}'".format(args.load_weights))

    if args.resume and check_isfile(args.resume):
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch'] + 1
        print("Loaded checkpoint from '{}'".format(args.resume))
        print("- start_epoch: {}\n- rank1: {}".format(args.start_epoch, checkpoint['rank1']))

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")

        for name in args.target_names:
            print("Evaluating {} ...".format(name))
            testloader = testloader_dict[name]['test']
            test_set = dm.return_testdataset_by_name(name)
            rank1 = test(model, test_set, name, testloader, use_gpu, visualize=args.visualize_ranks)

        return

    start_time = time.time()
    top1logger = RankLogger(args.source_names, args.target_names)
    train_time = 0
    print("=> Start training")


    if args.fixbase_epoch > 0:
        print("Train {} for {} epochs while keeping other layers frozen".format(args.open_layers, args.fixbase_epoch))
        initial_optim_state = optimizer.state_dict()

        for epoch in range(args.fixbase_epoch):
            start_train_time = time.time()
            loss, prec1 = train(epoch, model, criterion, optimizer, trainloader, use_gpu, fixbase=True)
            print('Epoch: [{:02d}] [Average Loss:] {:.4f}\t [Average Prec.:] {:.2%}'.format(epoch+1, loss, prec1))
            train_time += round(time.time() - start_train_time)

        print("Done. All layers are open to train for {} epochs".format(args.max_epoch))
        optimizer.load_state_dict(initial_optim_state)

    for epoch in range(args.start_epoch, args.max_epoch):
        start_train_time = time.time()
        loss, prec1 = train(epoch, model, criterion, optimizer, trainloader, use_gpu)
        print('Epoch: [{:02d}] [Average Loss:] {:.4f}\t [Average Prec.:] {:.2%}'.format(epoch+1, loss, prec1))
        train_time += round(time.time() - start_train_time)

        scheduler.step()

        if (epoch + 1) > args.start_eval and args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
            print("=> Test")

            for name in args.target_names:
                print("Evaluating {} ...".format(name))
                testloader = testloader_dict[name]['test']

                test_set = dm.return_testdataset_by_name(name)

                if epoch+1 == args.max_epoch:
                    rank1 = test(model, test_set, name, testloader, use_gpu, visualize=True)
                else:
                    rank1 = test(model, test_set, name, testloader, use_gpu)

                top1logger.write(name, epoch + 1, rank1)

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, False, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))


    # save last checkpoint
    save_checkpoint({
        'state_dict': state_dict,
        'rank1': rank1,
        'epoch': epoch,
    }, False, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    top1logger.show_summary()


def train(epoch, model, criterion, optimizer, trainloader, use_gpu, fixbase=False):
    losses = AverageMeter()
    precisions = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    if fixbase or args.always_fixbase:
        open_specified_layers(model, args.open_layers)
    else:
        open_all_layers(model)

    end = time.time()
    for batch_idx, (imgs, pids) in enumerate(trainloader):
        data_time.update(time.time() - end)

        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        outputs = model(imgs)
        if isinstance(outputs, (tuple, list)):
            loss = DeepSupervision(criterion, outputs, pids)
        else:
            loss = criterion(outputs, pids)

        prec, = accuracy(outputs.data, pids.data)
        prec1 = prec[0]  # get top 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        losses.update(loss.item(), pids.size(0))
        precisions.update(prec1, pids.size(0))

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0:02d}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {prec.val:.2%} ({prec.avg:.2%})\t'.format(
                   epoch + 1, batch_idx + 1, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses, prec=precisions))

        end = time.time()

    return losses.avg, precisions.avg


def test(model, test_set, name, testloader, use_gpu, ranks=[1, 5], visualize=False):
    batch_time = AverageMeter()

    model.eval()

    num_correct = 0
    num_total = len(testloader)
    with torch.no_grad():
        for batch_idx, (imgs, pids) in enumerate(testloader):
            if use_gpu:
                imgs, pids = imgs.cuda(), pids.cuda()

            end = time.time()
            outputs = model(imgs)
            batch_time.update(time.time() - end)

            prec, = accuracy(outputs.data, pids.data, return_count=True)
            prec1 = prec[0]  # get top 1

            num_correct += prec1

    top1 = num_correct / num_total
    print("=> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch_size))

    print("Results ----------")
    print("top1: {:.2%}".format(top1))
    # print("CMC curve")
    # for r in ranks:
    #     print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")

    return top1


if __name__ == '__main__':
    main()
