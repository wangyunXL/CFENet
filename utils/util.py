import os
import numpy as np
from PIL import Image
import random
import logging
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
from matplotlib import font_manager
from matplotlib import rcParams
import seaborn as sns
import pandas as pd
import math
from seaborn.distributions import distplot
from tqdm import tqdm
from scipy import ndimage
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.init as initer


value_scale = 255
mean = [0.485, 0.456, 0.406]
mean = [item * value_scale for item in mean]
std = [0.229, 0.224, 0.225]
std = [item * value_scale for item in std]

def unNormalize(feat):
    ori_img = np.zeros_like(feat)
    ori_img[0, :, :] = feat[0, :, :] * std[0] + mean[0]
    ori_img[1, :, :] = feat[1, :, :] * std[1] + mean[1]
    ori_img[2, :, :] = feat[2, :, :] * std[2] + mean[2]
    ori_img = ori_img.transpose(1,2,0).astype("uint8")
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)  
    
    return ori_img

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def step_learning_rate(optimizer, base_lr, epoch, step_epoch, multiplier=0.1):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = base_lr * (multiplier ** (epoch // step_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def poly_learning_rate(optimizer, base_lr, curr_iter, max_iter, power=0.9, index_split=-1, scale_lr=10., warmup=False, warmup_step=500):
    """poly learning rate policy"""
    if warmup and curr_iter < warmup_step:           #warmup_step = len(train_loader) // 2
        lr = base_lr * (0.1 + 0.9 * (curr_iter/warmup_step))
    else:
        lr = base_lr * (1 - float(curr_iter) / max_iter) ** power

    # if curr_iter % 50 == 0:   
    #     print('Base LR: {:.4f}, Curr LR: {:.4f}, Warmup: {}.'.format(base_lr, lr, (warmup and curr_iter < warmup_step)))     

    for index, param_group in enumerate(optimizer.param_groups):
        if index <= index_split:
            param_group['lr'] = lr
        else:
            param_group['lr'] = lr * scale_lr   # 10x LR


def generate_sample(img, mask, mode = None):
    h,w,c = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  
    mask_3d_color = np.zeros((h,w,c), dtype="uint8")
    
    color_red = np.zeros((h,w), dtype="uint8")
    color_green = np.zeros((h,w), dtype="uint8")
    color_blue = np.zeros((h,w), dtype="uint8")
    target_pix = np.where(mask == 1)
    if mode == "red":
        color_red[target_pix] = 255
        color_green[target_pix] = 75
        color_blue[target_pix] = 75
    elif mode == "green":
        color_red[target_pix] = 75
        color_green[target_pix] = 255
        color_blue[target_pix] = 75
    elif mode == "blue":
        color_red[target_pix] = 75
        color_green[target_pix] = 75
        color_blue[target_pix] = 255
    
    mask_3d_color[:,:,0] = color_red 
    mask_3d_color[:,:,1] = color_green 
    mask_3d_color[:,:,2] = color_blue
    result = cv2.addWeighted(img, 1, mask_3d_color, 0.8, 0)

    return result



def Process_label(label):
    class_id = np.unique(label).tolist()
    mask = np.zeros_like(label, dtype=np.uint8)
    fg_pix = np.where(label==1)
    ignore_pix = np.where(label==255)
    if class_id == [0,1]:
        mask = label
    elif class_id == [0,255]:
        mask[ignore_pix] = 1
    elif class_id == [0,1,255]:
        mask[fg_pix] = 1
    
    return mask

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)               
    target = target.view(-1)            
    output[target == ignore_index] = ignore_index 
    intersection = output[output == target]       
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1) 
    area_output = torch.histc(output, bins=K, min=0, max=K-1)      
    area_target = torch.histc(target, bins=K, min=0, max=K-1)      
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path,i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)

def init_weights(model, conv='kaiming', batchnorm='normal', linear='kaiming', lstm='kaiming'):
    """
    :param model: Pytorch Model which is nn.Module
    :param conv:  'kaiming' or 'xavier'
    :param batchnorm: 'normal' or 'constant'
    :param linear: 'kaiming' or 'xavier'
    :param lstm: 'kaiming' or 'xavier'
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if conv == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif conv == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of conv error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):#, BatchNorm1d, BatchNorm2d, BatchNorm3d)):
            if batchnorm == 'normal':
                initer.normal_(m.weight, 1.0, 0.02)
            elif batchnorm == 'constant':
                initer.constant_(m.weight, 1.0)
            else:
                raise ValueError("init type of batchnorm error.\n")
            initer.constant_(m.bias, 0.0)

        elif isinstance(m, nn.Linear):
            if linear == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif linear == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of linear error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if lstm == 'kaiming':
                        initer.kaiming_normal_(param)
                    elif lstm == 'xavier':
                        initer.xavier_normal_(param)
                    else:
                        raise ValueError("init type of lstm error.\n")
                elif 'bias' in name:
                    initer.constant_(param, 0)

def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color

def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)     
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3    


        cmap[i] = np.array([r, g, b])            

    cmap = cmap/255 if normalized else cmap
    return cmap


# ------------------------------------------------------
def get_model_para_number(model):
    total_number = 0
    learnable_number = 0 
    for para in model.parameters():
        total_number += torch.numel(para)     
        if para.requires_grad == True:        
            learnable_number+= torch.numel(para)
    return total_number, learnable_number

def setup_seed(seed=2021, deterministic=False):
    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def get_save_path(args):
    backbone_str = 'resnet'+str(args.layers)
    args.snapshot_path = os.path.join("../CFENet", 'train_model/{}/{}/{}/split{}/{}/snapshot'
    .format(args.arch, args.data_set, args.shot, args.split, backbone_str))
    args.result_path = os.path.join("../CFENet", 'train_model/{}/{}/{}/split{}/{}/result'
    .format(args.arch, args.data_set, args.shot, args.split, backbone_str))
    print("snapshot_path: ", args.snapshot_path)
    print("result_path: ", args.result_path)

def get_train_val_set(args):
    if args.data_set == 'pascal':
        class_list = list(range(1, 21)) 
        if args.split == 3: 
            sub_list = list(range(1, 16)) 
            sub_val_list = list(range(16, 21))
        elif args.split == 2:
            sub_list = list(range(1, 11)) + list(range(16, 21)) 
            sub_val_list = list(range(11, 16)) 
        elif args.split == 1:
            sub_list = list(range(1, 6)) + list(range(11, 21)) 
            sub_val_list = list(range(6, 11))
        elif args.split == 0:
            sub_list = list(range(6, 21))
            sub_val_list = list(range(1, 6))

    elif args.data_set == 'coco':
        if args.use_split_coco:
            print('INFO: using SPLIT COCO (FWB)')
            class_list = list(range(1, 81))
            if args.split == 3:
                sub_val_list = list(range(4, 81, 4))
                sub_list = list(set(class_list) - set(sub_val_list))                    
            elif args.split == 2:
                sub_val_list = list(range(3, 80, 4))
                sub_list = list(set(class_list) - set(sub_val_list))    
            elif args.split == 1:
                sub_val_list = list(range(2, 79, 4))
                sub_list = list(set(class_list) - set(sub_val_list))    
            elif args.split == 0:
                sub_val_list = list(range(1, 78, 4))
                sub_list = list(set(class_list) - set(sub_val_list))    
        else:
            print('INFO: using COCO (PANet)')
            class_list = list(range(1, 81))
            if args.split == 3:
                sub_list = list(range(1, 61))
                sub_val_list = list(range(61, 81))
            elif args.split == 2:
                sub_list = list(range(1, 41)) + list(range(61, 81))
                sub_val_list = list(range(41, 61))
            elif args.split == 1:
                sub_list = list(range(1, 21)) + list(range(41, 81))
                sub_val_list = list(range(21, 41))
            elif args.split == 0:
                sub_list = list(range(21, 81)) 
                sub_val_list = list(range(1, 21))
                
    return sub_list, sub_val_list

def is_same_model(model1, model2):
    flag = 0
    count = 0
    for k, v in model1.state_dict().items():
        model1_val = v
        model2_val = model2.state_dict()[k]
        if (model1_val==model2_val).all():
            pass
        else:
            flag+=1
            print('value of key <{}> mismatch'.format(k))
        count+=1

    return True if flag==0 else False

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def sum_list(list):
    sum = 0
    for item in list:
        sum += item
    return sum

def show_cam_on_image(img, mask, save_root, cls_list, cat_idx, query_name, training):     
    for i in range(img.shape[0]):

        cls = cls_list[cat_idx[i]]
        if training == True: 
            save_path = os.path.join(save_root, "train", "class_{}_{}.png".format(cls, query_name[i]))
        else:
            save_path = os.path.join(save_root, "val", "class_{}_{}.png".format(cls, query_name[i]))
        check_makedirs(save_path)                

        image = img[i]
        mask = mask[i]        # img.shape:torch.Size([3, 473, 473]), mask.shape:(473, 473)
        factor = np.unique(mask).tolist()
        print(mask.shape)
        print(factor)

        img_flatten = image.flatten(1)           
        img_min = torch.min(img_flatten, dim=1)[0].unsqueeze(-1).unsqueeze(-1)   
        img_max = torch.max(img_flatten, dim=1)[0].unsqueeze(-1).unsqueeze(-1)
        image = (image - img_min) / (img_max - img_min)
        image = image.permute(1,2,0)              
        image = image.detach().cpu().numpy()

        image = np.float32(image) * 255.
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + image
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)
        cv2.imwrite(save_path, cam)

def show_cam(img, mask, save_root, cls, query_name, training):    
    mask = mask.cpu().numpy()

    for i in range(mask.shape[0]):          
        label = mask[i]           
        image = img.detach().cpu()

        if training == True: 
            save_path = os.path.join(save_root, "train/{}".format(query_name))
        else:
            save_path = os.path.join(save_root, "val/{}".format(query_name))
        check_makedirs(save_path)               
        save_path = os.path.join(save_path, "class_{}_{}.png".format(query_name, cls, query_name))

        # print("save_path: ", save_path)

        img_flatten = image.flatten(1)             
        img_min = torch.min(img_flatten, dim=1)[0].unsqueeze(-1).unsqueeze(-1)   
        img_max = torch.max(img_flatten, dim=1)[0].unsqueeze(-1).unsqueeze(-1)
        image = (image - img_min) / (img_max - img_min)
        image = image.permute(1,2,0).numpy()              

        heatmap = cv2.applyColorMap(np.uint8(255 * label), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + image
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)
        cv2.imwrite(save_path, cam)


# def crf_inference_label(img, labels, t=10, n_labels=21, gt_prob=0.7):

#     h, w = img.shape[:2]

#     d = dcrf.DenseCRF2D(w, h, n_labels)

#     unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)

#     d.setUnaryEnergy(unary)
#     d.addPairwiseGaussian(sxy=3, compat=3)
#     d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=10)

#     q = d.inference(t)

#     return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)
