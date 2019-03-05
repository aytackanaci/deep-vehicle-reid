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
from torch.nn import functional as F

from args import argument_parser, image_dataset_kwargs, optimizer_kwargs
from torchreid.data_manager import ImageDataManager
from torchreid import models
from torchreid.losses import CrossEntropyLoss, DeepSupervision, KLDivLoss
from torchreid.utils.iotools import save_checkpoint, check_isfile
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.loggers import Logger, RankLogger
from torchreid.utils.torchtools import count_num_param, open_all_layers, open_specified_layers
from torchreid.utils.reidtools import visualize_ranked_results
from torchreid.utils.generaltools import set_random_seed
from torchreid.eval_metrics import evaluate, accuracy
from torchreid.optimizers import init_optimizer

def exp_name(cfg, train_o, train_l, dropout):
    name = [
        'e_' + cfg.prefix,
        'S_' + '-'.join(cfg.source_names),
        'T_' + '-'.join(cfg.target_names),
        cfg.arch,
        'E',
        '' if cfg.resume == '' else 'r',
        '' if cfg.fixbase_epoch is not 0 else 'warmup' + str(cfg.fixbase_epoch),
        str(cfg.stepsize),
        'm' + str(cfg.max_epoch),
        'P',
        'b' + str(cfg.train_batch_size),
        cfg.optim,
        'lr' + str(cfg.lr),
        'wd' + str(cfg.weight_decay),
        'do' + str(dropout),
        'orient' + str(train_o),
        'landmarks' + str(train_l),
        'lmRegress' if cfg.regress_landmarks else 'lmClass'
        ]

    return '_'.join(name)


train_orient=True
train_landmarks=True

def main():
    global args, train_orient, train_landmarks

    # read config
    parser = argument_parser()
    args = parser.parse_args()
    args.fixbase_epoch = 0
    args.arch = 'mpfl'

    dropout = 0.001
    
    if args.use_landmarks_only and args.use_orient_only:
        print('Error: Only one of --use_orient_only or --use_landmarks_only can be selected.')
        sys.exit(1)

    if args.use_orient_only:
        print('Training only ID and orient branches')
        train_landmarks=False
    elif args.use_landmarks_only:
        print('Training only ID and landmark branches')
        train_orient=False
        
    args.save_dir = exp_name(args, train_orient, train_landmarks, dropout)

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

    print("Initializing Landmarks data manager")
    dm = ImageDataManager(use_gpu, **image_dataset_kwargs(args))
    trainloader_lm, trainloader, testloader_dict = dm.return_dataloaders(landmarks=True)
    # sys.exit(0)

    if not trainloader_lm:
        print('Warning: landmarks train loader not given, only id labels will be used for training')

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dm.num_train_pids, num_orients=dm.num_train_orients, num_landmarks=dm.num_train_landmarks, input_size=args.width, loss={'xent'}, use_gpu=use_gpu, train_orient=train_orient, train_landmarks=train_landmarks, regress_landmarks=args.regress_landmarks, dropout=dropout)
    print("Model size: {:.3f} M".format(count_num_param(model)))
    print(model)

    criterion = {}
    criterion['id'] = CrossEntropyLoss(num_classes=dm.num_train_pids, use_gpu=use_gpu, label_smooth=args.label_smooth)
    criterion['id_soft'] = KLDivLoss(num_classes=dm.num_train_pids, use_gpu=use_gpu, label_smooth=False)
    criterion['orient'] = CrossEntropyLoss(num_classes=dm.num_train_orients, use_gpu=use_gpu, label_smooth=args.label_smooth)
    criterion['landmarks'] = CrossEntropyLoss(num_classes=dm.num_train_landmarks, use_gpu=use_gpu, label_smooth=args.label_smooth)
    optimizer = init_optimizer(model.parameters(), **optimizer_kwargs(args))
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True, threshold=1e-04)

    if args.load_weights:
        if check_isfile(args.load_weights): # load pretrained weights but ignore layers that don't match in size
            checkpoint = torch.load(args.load_weights)
            pretrain_dict = checkpoint['state_dict']
            model_dict = model.state_dict()
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
            model_dict.update(pretrain_dict)
            model.load_state_dict(model_dict)
            print("Loaded pretrained weights from '{}'".format(args.load_weights))
        else:
            print("Error! Cannot load pretrained weights from '{}'".format(args.load_weights))
            
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
            queryloader = testloader_dict[name]['query']
            galleryloader = testloader_dict[name]['gallery']
            test_set = dm.return_testdataset_by_name(name)
            rank1, mAP = test(model, test_set, name, queryloader, galleryloader, use_gpu, visualize=args.visualize_ranks)

        return

    start_time = time.time()
    ranklogger = RankLogger(args.source_names, args.target_names)
    maplogger = RankLogger(args.source_names, args.target_names)
    train_time = 0
    print("=> Start training")


    if args.fixbase_epoch > 0:
        print("Train {} for {} epochs while keeping other layers frozen".format(args.open_layers, args.fixbase_epoch))
        initial_optim_state = optimizer.state_dict()

        for epoch in range(args.fixbase_epoch):
            start_train_time = time.time()
            loss, prec1 = train(epoch, model, criterion, optimizer, trainloader_lm, use_gpu, fixbase=True)
            if trainloader:
                loss, prec1 = train(epoch, model, criterion, optimizer, trainloader, use_gpu, fixbase=True)
            print('Epoch: [{:02d}] [Average Loss:] {:.4f}\t [Average Prec.:] {:.2%}'.format(epoch+1, loss, prec1))
            train_time += round(time.time() - start_train_time)

        print("Done. All layers are open to train for {} epochs".format(args.max_epoch))
        optimizer.load_state_dict(initial_optim_state)

    feedback_consensus = True
    for epoch in range(args.start_epoch, args.max_epoch):
        start_train_time = time.time()
        print('Training on landmark data')
        loss, prec1 = train(epoch, model, criterion, optimizer, trainloader_lm, use_gpu, feedback_consensus=feedback_consensus)
        if trainloader:
            print('Training on non-landmark data')
            loss, prec1 = train(epoch, model, criterion, optimizer, trainloader, use_gpu, feedback_consensus=feedback_consensus)
        print('Epoch: [{:02d}] [Average Loss:] {:.4f}\t [Average Prec.:] {:.2%}'.format(epoch+1, loss, prec1))
        train_time += round(time.time() - start_train_time)

        scheduler.step()

        if (epoch + 1) > args.start_eval and args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
            print("=> Test")

            for name in args.target_names:
                print("Evaluating {} ...".format(name))
                queryloader = testloader_dict[name]['query']
                galleryloader = testloader_dict[name]['gallery']

                test_set = dm.return_testdataset_by_name(name)

                if epoch+1 == args.max_epoch:
                    rank1, mAP = test(model, test_set, name, queryloader, galleryloader, use_gpu, visualize=True)
                else:
                    rank1, mAP = test(model, test_set, name, queryloader, galleryloader, use_gpu, visualize=args.visualize_ranks)

                ranklogger.write(name, epoch + 1, rank1)
                maplogger.write(name, epoch + 1, mAP)

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
    ranklogger.show_summary()
    maplogger.show_summary()


def train(epoch, model, criterion, optimizer, trainloader, use_gpu, fixbase=False, feedback_consensus=True):
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
    for batch_idx, data in enumerate(trainloader):
        data_time.update(time.time() - end)

        update_lmo = False

        # Only have id prediction
        img, pids = data[0:2]
        if use_gpu:
            img, pids = img.cuda(), pids.cuda(),
        
        if len(data) > 4:
            # We have landmark and orientation labels
            porient, plandmarks = data[3:5]
            plandmarks = plandmarks.float()
            if use_gpu:
                porient, plandmarks =  porient.cuda(), plandmarks.cuda()
            update_lmo = True

        y_id, y_orient, y_landmarks, y_orient_id, y_landmarks_id, y_consensus = model(img)

        loss_id = criterion['id'](y_id, pids)
        loss_orient_id = criterion['id'](y_orient_id, pids)
        loss_landmarks_id = criterion['id'](y_landmarks_id, pids)

        if update_lmo:
            loss_orient = criterion['orient'](y_orient, porient)
            loss_landmarks = criterion['landmarks'](y_landmarks, plandmarks, one_hot=True)

        if feedback_consensus:
            loss_consensus_id = criterion['id_soft'](y_id, y_consensus)
            loss_consensus_orient = criterion['id_soft'](y_orient_id, y_consensus)
            loss_consensus_landmarks = criterion['id_soft'](y_landmarks_id, y_consensus)

        loss_consensus_labels = criterion['id'](y_consensus, pids)

        prec, = accuracy(y_consensus.data, pids.data)
        prec1 = prec[0]  # get top 1
        
        optimizer.zero_grad()

        # Individual branch losses
        total_loss = loss_id
        if feedback_consensus:
            total_loss += loss_consensus_id

        if train_orient:
            total_loss_orient = loss_orient_id
            if update_lmo:
                total_loss_orient += loss_orient
            if feedback_consensus:
                total_loss_orient += loss_consensus_orient

            total_loss += total_loss_orient
            
        if train_landmarks:
            total_loss_landmarks = loss_landmarks_id
            if update_lmo:
                total_loss_landmarks += loss_landmarks
            if feedback_consensus:
                total_loss_landmarks += loss_consensus_landmarks

            total_loss += total_loss_landmarks
            
        # Consensus loss according to labels
        total_loss += loss_consensus_labels

        # Now propagate back all losses
        total_loss.backward()
        
        optimizer.step()

        batch_time.update(time.time() - end)

        losses.update(total_loss.item(), pids.size(0))
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


def test(model, test_set, name, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20], visualize=False):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (img, pids, camids, _) in enumerate(queryloader):
            if use_gpu: img = img.cuda()

            end = time.time()
            features = model(img)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        end = time.time()
        for batch_idx, (img, pids, camids, _) in enumerate(galleryloader):
            if use_gpu: img = img.cuda()

            end = time.time()
            features = model(img)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("=> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch_size))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP, all_AP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)

    if visualize:
        visualize_ranked_results(
            distmat, all_AP, test_set, name,
            save_path=args.save_dir,
            topk=100
        )

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")

    return cmc[0], mAP


if __name__ == '__main__':
    main()
