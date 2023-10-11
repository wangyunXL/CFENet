import os
import datetime
import random
import time
import cv2
import numpy as np
import logging
import argparse
import math
import json
from visdom import Visdom
import os.path as osp
from shutil import copyfile
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler



from tensorboardX import SummaryWriter

from model.our_PSPNet import OneModel

from utils import our_dataset
from utils import transform, transform_tri, config
from utils.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, get_model_para_number, setup_seed, get_logger, get_save_path, \
                                    is_same_model, fix_bn, sum_list, check_makedirs, voc_cmap

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
# os.environ["CUDA_VISIBLE_DEVICES"] = '8'

CMAP = voc_cmap()
def decode_target(mask):
    return CMAP[mask]


# def get_parser():
#     parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
#     parser.add_argument('--arch', type=str, default='PSPNet') #
#     parser.add_argument('--viz', action='store_true', default=False)
#     parser.add_argument('--config', type=str, default='/root/autodl-tmp/my_work/BAM/config/coco/base/coco_split0_resnet50_base.yaml', help='config file') # coco/coco_split0_resnet50.yaml
#     parser.add_argument('--local_rank', type=int, default=-1, help='number of cpu threads to use during batch generation')
#     parser.add_argument('--opts', help='see config/coco/coco_split0_resnet50_base.yaml for all options', default=None, nargs=argparse.REMAINDER)
#     args = parser.parse_args()
#     assert args.config is not None
#     cfg = config.load_cfg_from_cfg_file(args.config)
#     cfg = config.merge_cfg_from_args(cfg, args)
#     if args.opts is not None:
#         cfg = config.merge_cfg_from_list(cfg, args.opts)
#     return cfg


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Few-Shot Semantic Segmentation')
    parser.add_argument('--arch', type=str, default='New_base_Model') #
    parser.add_argument('--viz', action='store_true', default=True)
    parser.add_argument('--config', type=str, default='/root/autodl-tmp/my_work/BAM/config/pascal/base/pascal_split3_resnet50_base.yaml', help='config file') # coco/coco_split0_resnet50.yaml
    parser.add_argument('--local_rank', type=int, default=-1, help='number of cpu threads to use during batch generation')
    parser.add_argument('--opts', help='see config/pascal/pascal_split2_resnet50_base.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg = config.merge_cfg_from_args(cfg, args)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_save_path(args):
    backbone_str = 'resnet'+str(args.layers)
    args.snapshot_path = os.path.join("/root/autodl-tmp/my_work/BAM", '{}/{}/split{}/{}/snapshot'
    .format(args.arch, args.data_set, args.split, backbone_str))
    args.result_path = os.path.join("/root/autodl-tmp/my_work/BAM", '{}/{}/split{}/{}/result'
    .format(args.arch, args.data_set, args.split, backbone_str))
    print("snapshot_path: ", args.snapshot_path)
    print("result_path: ", args.result_path)

def get_model(args):
    """
    Number of Parameters: 52M
    Number of Learnable Parameters: 52M
    """

    model = OneModel(args)
    optimizer = model.get_optim(model, args, LR=args.base_lr)

    if hasattr(model,'freeze_modules'):
        model.freeze_modules(model)

    model = model.cuda()

    # Resume
    get_save_path(args)
    check_makedirs(args.snapshot_path)
    check_makedirs(args.result_path)

    if args.resume:
        resume_path = osp.join(args.snapshot_path, args.resume)
        if os.path.isfile(resume_path):
            logger.info("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            new_param = checkpoint['state_dict']
            try: 
                model.load_state_dict(new_param)
            except RuntimeError:                   # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                model.load_state_dict(new_param)
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
        else:

            logger.info("=> no checkpoint found at '{}'".format(resume_path))

    # Get model para.
    total_number, learnable_number = get_model_para_number(model)

    print('Number of Parameters: %dM' % (total_number*1e-6))
    print('Number of Learnable Parameters: %dM' % (learnable_number*1e-6))
    # print('Number of Parameters: %d' % (total_number))
    # print('Number of Learnable Parameters: %d' % (learnable_number))

    time.sleep(5)
    return model, optimizer



def main():
    global args, logger, writer
    args = get_parser()
    logger = get_logger()
    # print(args)

    if args.manual_seed is not None:
        setup_seed(args.manual_seed, args.seed_deterministic)             # 固定随机种子设置

    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0

# ----------------------------获取模型和优化器-------------------------------
    logger.info("=> creating model ...")
    model, optimizer = get_model(args)
    # logger.info(model)
    if args.viz:
        writer = SummaryWriter(args.result_path)

# ----------------------  DATASET  ----------------------
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    # Train
    train_transform = transform.Compose([
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.padding_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.padding_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    # print("args.train_h:{}, args.train_w".format(args.train_h, args.train_w))
    if args.data_set == 'pascal' or args.data_set == 'coco':
        train_data = our_dataset.BaseData(split=args.split, mode='train', data_root=args.data_root, data_list=args.train_list, \
                                    data_set=args.data_set, use_split_coco=args.use_split_coco, \
                                    transform=train_transform, batch_size=args.batch_size)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, \
                                                pin_memory=False, sampler=None, drop_last=True, \
                                                shuffle=True)
    # Val
    if args.evaluate:
        if args.resized_val:
            val_transform = transform.Compose([
                transform.Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])
        else:
            val_transform = transform.Compose([
                transform.test_Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])
        if args.data_set == 'pascal' or args.data_set == 'coco':    
            val_data = our_dataset.BaseData(split=args.split, mode='val', data_root=args.data_root, data_list=args.val_list, \
                                        data_set=args.data_set, use_split_coco=args.use_split_coco, \
                                        transform=val_transform, batch_size=args.batch_size_val)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=False, sampler=None)
        if args.ori_resize:
            assert args.batch_size_val == 1

# ----------------------  TRAINVAL  ----------------------
    global best_miou, best_epoch, keep_epoch, val_num
    best_miou = 0
    best_epoch = 0
    keep_epoch = 0
    val_num = 0

    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if keep_epoch == args.stop_interval:
            break
        if args.fix_random_seed_val:                       # 每个epoch都重新设置随机种子
            setup_seed(args.manual_seed + epoch, args.seed_deterministic)

        epoch_log = epoch + 1
        keep_epoch += 1


        check_makedirs(args.log_dir)
        train_log = "split3_train_base_log.txt"
        train_logpath = os.path.join(args.log_dir, train_log)
        val_log = "split3_val_base_epoch_log.txt"
        val_logpath = os.path.join(args.log_dir, val_log)


        # ----------------------  TRAIN  ----------------------
        train(train_loader, model, optimizer, epoch, train_logpath)
        print("args.save_freq: {}, args.snapshot_path:{}".format(args.save_freq, args.snapshot_path))

        # save model for <resuming>          epoch除以保存模型频率，判断是否需要进行模型保存
        if (epoch % args.save_freq == 0) and (epoch > 0):
            filename = args.snapshot_path + '/epoch_{}.pth'.format(epoch)
            logger.info('Saving checkpoint to: ' + filename)
            if osp.exists(filename):
                os.remove(filename)            
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)

        # -----------------------  VAL  -----------------------
        if args.evaluate and epoch%1==0:
            with torch.no_grad():
                mIoU = validate(val_loader, model, val_logpath, epoch)
                val_num += 1
                if args.viz:  # 可视化mIoU_val
                    writer.add_scalar('mIoU_val', mIoU, epoch_log)


        # save model for <testing>                           当获取到新的最佳mIOU值时，保存模型
            if (mIoU > best_miou):
                best_miou, best_epoch = mIoU, epoch
                keep_epoch = 0
                filename = args.snapshot_path + '/train_epoch_' + str(epoch) + '_{:.4f}'.format(best_miou) + '.pth'
                logger.info('Saving checkpoint to: ' + filename)
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
                copyfile(filename, args.snapshot_path + '/best.pth')   # shutil.copyfile(src, dst,follow_symlinks)  https://www.yisu.com/zixun/278645.html
                # args.save_freq: 10,          args.snapshot_path:exp/pascal/BAM/split0/resnet50/snapshot

    total_time = time.time() - start_time
    t_m, t_s = divmod(total_time, 60)
    t_h, t_m = divmod(t_m, 60)
    total_time = '{:02d}h {:02d}m {:02d}s'.format(int(t_h), int(t_m), int(t_s))

    print('\nEpoch: {}/{} \t Total running time: {}'.format(epoch_log, args.epochs, total_time))
    print('The number of models validated: {}'.format(val_num))
    print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  Final Best Result   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(args.arch + '\t Group:{} \t Best_mIoU:{:.4f} \t Best_step:{}'.format(args.split, best_miou, best_epoch))
    print('>'*80)
    print ('%s' % datetime.datetime.now())


def train(train_loader, model, optimizer, epoch, train_logpath):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    model.train()

    end = time.time()
    val_time = 0.
    max_iter = args.epochs * len(train_loader)           # 最大迭代次数

    print('Warmup: {}'.format(args.warmup))

    for i, (input, target, cls_label, name_list) in enumerate(train_loader):

        data_time.update(time.time() - end - val_time)              # 每次数据读取所用的时间
        current_iter = epoch * len(train_loader) + i + 1            # 目前的迭代次数
        
        poly_learning_rate(optimizer, args.base_lr, current_iter, max_iter, power=args.power, index_split=args.index_split, warmup=args.warmup, warmup_step=len(train_loader)//2)

        input = input.cuda(non_blocking=True)                       # input.shape:  torch.Size([3, 3, 417, 417])
        target = target.cuda(non_blocking=True)
        cls_label = cls_label.cuda(non_blocking=True)               # cls_label.shape:  torch.Size([16, 20, 1, 1])
        
        output, main_loss, cls_loss = model(x=input, y=target, cls_label=cls_label)  # output.shape:  torch.Size([16, 473, 473])
        loss = main_loss + cls_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()                                   # 反向传播完成

        n = input.size(0)                                  # batch_size

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)    # 开始计算交并比
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
        
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)  # allAcc     计算准确率, 此处的sum()是将tensor中的所有元素求和，即0和1对应的像素个数求总和
        
        loss_meter.update(loss.item(), n)        # 此处更新，取n=batch_size

        batch_time.update(time.time() - end - val_time)
        end = time.time()

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0:
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '                      
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                        batch_time=batch_time,
                                                        data_time=data_time,
                                                        remain_time=remain_time,
                                                        loss_meter=loss_meter,
                                                        accuracy=accuracy))
            if args.viz:
                writer.add_scalar('loss_train', loss_meter.val, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)


    logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch, args.epochs, mIoU, mAcc, allAcc))

    train_log_txt_formatter = "{time_str}\t [Epoch] {epoch:03d}\t [mIoU]{mIoU}\t [mAcc] {mAcc}\t             \
                                        [allAcc] {allAcc} [Main_loss] {main_loss}\t [cls_loss]{cls_loss}\t [loss]{loss}\n"
    log_write = train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"),
                                    epoch = epoch,
                                    mIoU = mIoU,  
                                    mAcc = mAcc,   
                                    allAcc = allAcc,
                                    main_loss = main_loss_meter.avg,
                                    cls_loss = cls_loss_meter.avg,
                                    loss = loss_meter.avg
                                    )
    
    with open(train_logpath, "a") as f:
        f.write(log_write)


def validate(val_loader, model, val_logpath, epoch):

    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    class_intersection_meter = [0]*(args.classes-1)       # args.classes=61/16
    class_union_meter = [0]*(args.classes-1)

    if args.manual_seed is not None and args.fix_random_seed_val:
        setup_seed(args.manual_seed, args.seed_deterministic)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model.eval()
    end = time.time()
    val_start = end

    iter_num = 0

    for i, logits in enumerate(val_loader):
        iter_num += 1
        data_time.update(time.time() - end)

        if args.batch_size_val == 1:
            input, target, ori_label, cls_label, name_list = logits
            ori_label = ori_label.cuda(non_blocking=True)
        else:
            input, target, cls_label, name_list = logits
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        cls_label = cls_label.cuda(non_blocking=True)


        start_time = time.time()
        output, cls_loss = model(x=input, y=target, cls_label=cls_label)
        model_time.update(time.time() - start_time)

        if args.ori_resize:
            longerside = max(ori_label.size(1), ori_label.size(2))
            backmask = torch.ones(ori_label.size(0), longerside, longerside, device='cuda')*255
            backmask[0, :ori_label.size(1), :ori_label.size(2)] = ori_label
            target = backmask.clone().long()

        output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)

        loss = criterion(output, target)
        loss = loss + cls_loss

        output = output.max(1)[1]              # output.shape:  torch.Size([6, 473, 473])


        # for i in range(output.shape[0]):
        #     name = name_list[0][i]
        #     pred = output[i].cpu().numpy()
        #     label = target[i].cpu().numpy()
        #     colorized_pred = decode_target(pred).astype('uint8')
        #     colorized_label = decode_target(label).astype('uint8')
        #     # colorized_preds = Image.fromarray(colorized_preds)
        #     if loss > 0.5:
        #         pred_save_dir = os.path.join(args.pred_root, "epoch_{}/pred_miss".format(epoch))
        #         target_save_dir = os.path.join(args.pred_root, "epoch_{}/pred_miss".format(epoch))
        #         check_makedirs(pred_save_dir)
        #         check_makedirs(target_save_dir)
        #         pred_save_path = os.path.join(pred_save_dir, "{}_pred.png".format(name))
        #         target_save_path = os.path.join(target_save_dir, "{}_label.png".format(name))
        #         cv2.imwrite(pred_save_path, colorized_pred)
        #         cv2.imwrite(target_save_path, colorized_label)
        #     else:
        #         pred_save_dir = os.path.join(args.pred_root, "epoch_{}".format(epoch))
        #         target_save_dir = os.path.join(args.pred_root, "epoch_{}".format(epoch))
        #         check_makedirs(pred_save_dir)
        #         check_makedirs(target_save_dir)
        #         pred_save_path = os.path.join(pred_save_dir, "{}_pred.png".format(name))
        #         target_save_path = os.path.join(target_save_dir, "{}_label.png".format(name))
        #         cv2.imwrite(pred_save_path, colorized_pred)
        #         cv2.imwrite(target_save_path, colorized_label)


        intersection, union, new_target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, new_target = intersection.cpu().numpy(), union.cpu().numpy(), new_target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)
        for idx in range(1,len(intersection)):                    # 类别数量61或者16
            class_intersection_meter[idx-1] += intersection[idx]  # 将对应类别的交集像素个数存入列表对应类别索引处
            class_union_meter[idx-1] += union[idx]                # # 将对应类别的并集像素个数存入列表对应类别索引处

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), input.size(0))
        cls_loss_meter.update(cls_loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if ((iter_num % 100 == 0) or (iter_num == len(val_loader))):          # 每100个迭代，或者1个epoch，记录一次
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Cls_Loss {cls_loss_meter.val:.4f} ({cls_loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(iter_num, len(val_loader),
                                                        data_time=data_time,
                                                        batch_time=batch_time,
                                                        loss_meter=loss_meter,
                                                        cls_loss_meter=cls_loss_meter,
                                                        accuracy=accuracy))
    val_time = time.time()-val_start

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    
    class_iou_class = []
    class_miou = 0
    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i]/(class_union_meter[i]+ 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou
    class_miou = class_miou*1.0 / len(class_intersection_meter)

    
    base_dict = {}

    logger.info('meanIoU---Val result: mIoU {:.4f}.'.format(class_miou))
    for i in range(len(class_intersection_meter)):
        logger.info('Class_{} Result: iou_b {:.4f}.'.format(i+1, class_iou_class[i]))
        base_dict["Class{}_iou_b".format(i)] = class_iou_class[i]
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    print('total time: {:.4f}, avg inference time: {:.4f}, count: {}'.format(val_time, model_time.avg, iter_num))

    log_stats = {
        'epoch': epoch + 1,
        'mIoU': mIoU,
        'mIoU_f_class_miou' : class_miou,   
        **{f'val_{k}_iou_f': v for k, v in base_dict.items()},
    }
    
    with open(val_logpath, "a") as f:
        f.write(json.dumps(log_stats) + '\n\n')

    return class_miou

if __name__ == '__main__':
    main()
