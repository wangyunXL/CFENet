import torch
import os
from torch import nn
from torch._C import device
import torch.nn.functional as F
from torch.nn import BatchNorm2d as BatchNorm        

import numpy as np
import random
import time
import cv2

import model.resnet as models

from model.ASPP import ASPP
from model.PSPNet import OneModel as PSPNet
from utils.util import get_train_val_set, check_makedirs
from model.channel_attn import channel_attn

from model.feature import extract_feat_res, extract_feat_vgg
from functools import reduce
from operator import add


np.seterr(divide='ignore', invalid='ignore')
id_to_class = {"0":"BG", "1":"plane", "2":"bicycle", "3":"bird", "4":"boat", "5":"bottle",
				"6":"bus", "7":"car", "8":"cat", "9":"chair", "10":"cow", "11":"dining_table",
				"12":"dog", "13":"horse", "14":"motorbike", "15":"person", "16":"potted_plant",
				"17":"sheep", "18":"sofa", "19":"train", "20":"monitor"}

def get_corr_chanfeat(query_feat, supp_feat_list, mask_list):
    softmax = nn.Softmax(dim=1)
    corr_q_feat_list = []
    # corr_attn_list = []
    for i in range(len(supp_feat_list)):
        supp_feat = supp_feat_list[i]
        supp_mask = mask_list[i]

        n,c,h,w = query_feat.shape
        q_feat = query_feat.reshape(n,c,h*w)
        q_feat_T = q_feat.permute(0,2,1)

        s_feat = supp_feat*supp_mask
        s_feat = s_feat.reshape(n,c,h*w)
        s_feat_T = s_feat.permute(0,2,1)

        aff = torch.matmul(s_feat_T, q_feat)
        # aff = softmax(aff)
        aff = softmax(aff.clamp(min=0))
        corr_feat = torch.matmul(s_feat, aff)
        corr_feat = corr_feat / (corr_feat.norm(dim=1, p=2, keepdim=True)+1e-5)
        corr_feat = corr_feat.reshape(n,c,h,w)
        corr_q_feat = query_feat * corr_feat
        # chan_corr_feat = torch.matmul(chan_corr, q_feat).reshape(n,-1,h,w)
        # corr_feat_list.append(corr_feat)
        corr_q_feat_list.append(corr_q_feat)
        # corr_attn_list.append(chan_corr_feat)

    # corr_feat = torch.cat(corr_feat_list, dim=1)         
    corr_q_feat = torch.cat(corr_q_feat_list, dim=1)        
    # corr_q_feat = (corr_q_feat * weight_soft).sum(1, True)
    # corr_attn_feat = torch.cat(corr_attn_list, dim=1)    

    # return corr_feat, corr_q_feat, corr_attn_feat
    return corr_q_feat

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_pro = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_pro

def get_gram_matrix(fea):       
    b, c, h, w = fea.shape        
    fea = fea.reshape(b, c, h*w)    
    fea_T = fea.permute(0, 2, 1)    
    fea_norm = fea.norm(2, 2, True)      
    fea_T_norm = fea_T.norm(2, 1, True)          
    gram = torch.bmm(fea, fea_T)/(torch.bmm(fea_norm, fea_T_norm) + 1e-7)   
    return gram                 


class MHSA(nn.Module):
    def __init__(self, n_dims=None, height=60, width=60, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, height, 1 ]), requires_grad=True) 
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, width]), requires_grad=True)  

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, height, width = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)    
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)          
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)        


        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)     

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)  
        content_position = torch.matmul(content_position, q)      

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))    
        out = out.view(n_batch, -1, height, width)

        return out

class Correlation:

    @classmethod
    def multilayer_correlation(cls, query_feats, support_feats, stack_ids):
        eps = 1e-5
        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            queryShape = query_feat.shape#b,c,h,w
            corrI=[]
            realSupI=[]
            for j in range(len(support_feat)):#b
                queryIJ=query_feat[j].flatten(start_dim=1)#c,hw
                queryIJNorm=queryIJ/(queryIJ.norm(dim=0, p=2, keepdim=True) + eps)
                supIJ=support_feat[j]#c,hw
                supIJNorm=supIJ/(supIJ.norm(dim=0, p=2, keepdim=True) + eps)
                corr=(queryIJNorm.permute(1,0)).matmul(supIJNorm)
                corr = corr.clamp(min=0)
                corr=corr.mean(dim=1,keepdim=True)
                corr=(corr.permute(1,0)).unsqueeze(0)#1,1,hw
                corrI.append(corr)#b,1,hw

            corrI=torch.cat(corrI,dim=0)#b,1,h,w
            corrI=corrI.reshape((corrI.shape[0],corrI.shape[1],queryShape[-2],queryShape[-1]))#b,1,h,w
            corrs.append(corrI)#n,b,1,h,w

        corr_l4 = torch.cat(corrs[-stack_ids[0]:],dim=1).contiguous()#b,n,h,w    
        corr_l3 = torch.cat(corrs[-stack_ids[1]:-stack_ids[0]],dim=1).contiguous()   
        corr_l2 = torch.cat(corrs[-stack_ids[2]:-stack_ids[1]],dim=1).contiguous()  


        return [corr_l4, corr_l3, corr_l2] #,[sup_l4,sup_l3,sup_l2]

class Attention(nn.Module):
    """
    Guided Attention Module (GAM).

    Args:
        in_channels: interval channel depth for both input and output
            feature map.
        drop_rate: dropout rate.
    """

    def __init__(self, in_channels, drop_rate=0.5):
        super().__init__()
        self.DEPTH = in_channels
        self.DROP_RATE = drop_rate
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=self.DEPTH, out_channels=256,
                      kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=1),
            nn.Dropout(p=drop_rate),
            nn.Sigmoid())

    @staticmethod
    def mask(embedding, mask):
        h, w = embedding.size()[-2:]
        mask = F.interpolate(mask, size=(h, w), mode='nearest')
        mask=mask
        return mask * embedding

    def forward(self, *x):
        Fs, Ys = x
        att = F.adaptive_avg_pool2d(self.mask(Fs, Ys), output_size=(1, 1))
        g = self.gate(att)
        Fs = g * Fs
        return Fs


def Enhan_corr(corr):
    img_list = []
    for i in range(corr.shape[0]):
        img_corr = corr[i]
        h,w = img_corr.squeeze(0).shape
        img = img_corr.squeeze(0).detach().cpu().numpy()
        img = (img * 255).astype("uint8")
        img_flatten = img.flatten()
        hist, bins = np.histogram(img_flatten, 256, [0,256])
        cdf = hist.cumsum()
        np.seterr(divide="ignore", invalid="ignore")
        img_dst = ((cdf - cdf.min()) * 255) / (cdf.max() - cdf.min())
        en_img = img_dst[img_flatten].reshape(h,w)
        en_img = en_img / np.max(en_img)
        en_img = en_img.astype(np.float32)

        en_img = torch.from_numpy(en_img).cuda()
        img_list.append(en_img.unsqueeze(0))
    enhan_img = torch.cat(img_list, dim=0)
    return enhan_img



class OneModel(nn.Module):
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()

        self.cls_type = cls_type  # 'Base' or 'Novel'
        self.layers = args.layers
        self.zoom_factor = args.zoom_factor
        self.shot = args.shot
        self.vgg = args.vgg
        self.dataset = args.data_set
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.sub_list, self.sub_val_list = get_train_val_set(args)

        self.print_freq = args.print_freq/2

        self.pretrained = True
        self.classes = 2
        if self.dataset == 'pascal':
            self.base_classes = 15
            self.gap = 5
        elif self.dataset == 'coco':
            self.base_classes = 60
            self.gap = 20
        
        assert self.layers in [50, 101, 152]
    
        PSPNet_ = PSPNet(args)
        backbone_str = 'vgg' if args.vgg else 'resnet' + str(args.layers)
        weight_path = r'../CFENet/base_train_model/PSPNet/{}/split{}/{}/snapshot/best.pth'.format(args.data_set, args.split, backbone_str)
        new_param = torch.load(weight_path, map_location=torch.device('cpu'))['state_dict']
        try:
            PSPNet_.load_state_dict(new_param)
        except RuntimeError:                   # 1GPU loads mGPU model
            for key in list(new_param.keys()):
                new_param[key[7:]] = new_param.pop(key)
            PSPNet_.load_state_dict(new_param)


        if backbone_str == 'vgg':
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
            self.nsimlairy = [1,3,3]
        elif backbone_str == 'resnet50':
            self.feat_ids = list(range(3, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
            self.nsimlairy = [3,6,4]
        elif backbone_str == 'resnet101':
            self.feat_ids = list(range(3, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
            self.nsimlairy = [3,23,4]
        else:
            raise Exception('Unavailable backbone: %s' % backbone_str)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]


        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = PSPNet_.layer0, PSPNet_.layer1, PSPNet_.layer2, PSPNet_.layer3, PSPNet_.layer4
        # Base Learner
        self.learner_base = nn.Sequential(PSPNet_.ppm, PSPNet_.cls)
        

        # Meta Learner
        reduce_dim = 256
        self.low_fea_id = args.low_fea[-1]     
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512              
        self.down_query = nn.Sequential(     
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        self.down_supp = nn.Sequential(    
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        self.attn = channel_attn(in_dim=self.shot*reduce_dim, out_dim=reduce_dim)
        self.init_merge1 = nn.Sequential(     
            nn.Conv2d(reduce_dim*2+1, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.init_merge2 = nn.Sequential(      
            nn.Conv2d(reduce_dim*3+64, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.supp_attention = Attention(in_channels=reduce_dim)
        self.hyper_final = nn.Sequential(
            nn.Conv2d(sum(nbottlenecks[-3:]), 64, kernel_size=1,padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            )
        self.ASPP_meta = ASPP(reduce_dim)
        self.ASPP1 = ASPP(reduce_dim)
        self.mhsa = MHSA(n_dims=reduce_dim, height=args.down_h, width=args.down_w, heads=4)
        self.res1_meta = nn.Sequential(      
            nn.Conv2d(reduce_dim*11, reduce_dim*2, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.res2_meta = nn.Sequential(     
            nn.Conv2d(reduce_dim*2, reduce_dim*2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim*2, reduce_dim*2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True))
        self.cls_meta = nn.Sequential(         
            nn.Conv2d(reduce_dim*2, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1))    

        # Gram and Meta
        self.gram_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.gram_merge.weight = nn.Parameter(torch.tensor([[1.0],[0.0]]).reshape_as(self.gram_merge.weight))

        # Learner Ensemble
        self.cls_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.cls_merge.weight = nn.Parameter(torch.tensor([[1.0],[0.0]]).reshape_as(self.cls_merge.weight))
        
        # K-Shot Reweighting
        if args.shot > 1:
            self.kshot_trans_dim = args.kshot_trans_dim    
            if self.kshot_trans_dim == 0:    
                self.kshot_rw = nn.Conv2d(self.shot, self.shot, kernel_size=1, bias=False)   
                self.kshot_rw.weight = nn.Parameter(torch.ones_like(self.kshot_rw.weight) / args.shot)
            else:
                self.kshot_rw = nn.Sequential(
                    nn.Conv2d(self.shot, self.kshot_trans_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.kshot_trans_dim, self.shot, kernel_size=1))


    def get_optim(self, model, args, LR):
        if args.shot > 1:
            optimizer = torch.optim.SGD(
                [     
                {'params': model.down_query.parameters()},
                {'params': model.down_supp.parameters()},
                {'params': model.supp_attention.parameters()},
                {'params': model.attn.parameters()},
                {'params': model.hyper_final.parameters()},
                {'params': model.init_merge1.parameters()},
                {'params': model.init_merge2.parameters()},
                {'params': model.mhsa.parameters()},
                {'params': model.ASPP_meta.parameters()},
                {'params': model.ASPP1.parameters()},
                {'params': model.res1_meta.parameters()},
                {'params': model.res2_meta.parameters()},       
                {'params': model.cls_meta.parameters()},
                {'params': model.gram_merge.parameters()},
                {'params': model.cls_merge.parameters()},
                {'params': model.kshot_rw.parameters()},       
                ], lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(
                [     
                {'params': model.down_query.parameters()},
                {'params': model.down_supp.parameters()},
                {'params': model.supp_attention.parameters()},
                {'params': model.attn.parameters()},
                {'params': model.hyper_final.parameters()},
                {'params': model.init_merge1.parameters()},
                {'params': model.init_merge2.parameters()},
                {'params': model.mhsa.parameters()},
                {'params': model.ASPP_meta.parameters()},
                {'params': model.ASPP1.parameters()},
                {'params': model.res1_meta.parameters()},
                {'params': model.res2_meta.parameters()},        
                {'params': model.cls_meta.parameters()},
                {'params': model.gram_merge.parameters()},
                {'params': model.cls_merge.parameters()},
                ], lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
        return optimizer

    def freeze_modules(self, model):
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
        for param in model.learner_base.parameters():
            param.requires_grad = False
    


    def get_corr_map(self, query_feat, supp_feat_list, weight_soft):
        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(supp_feat_list):
            
            q = query_feat                         
            s = tmp_supp_feat
            bsize, ch_sz, sp_sz, _ = q.size()[:]

            tmp_query = q
            tmp_query = tmp_query.reshape(bsize, ch_sz, -1)       
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)     

            tmp_supp = s               
            tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1)         
            tmp_supp = tmp_supp.permute(0, 2, 1)
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

            similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)  
            similarity = similarity.max(1)[0].reshape(bsize, sp_sz*sp_sz)          
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)  
            corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)       
            corr_query_mask_list.append(corr_query)                    
        corr_query_mask = torch.cat(corr_query_mask_list, 1)            
        corr_query_mask = (weight_soft * corr_query_mask).sum(1,True)

        return corr_query_mask

    def mask_feature(self, features, support_mask):#bchw
        bs=features[0].shape[0]
        initSize=((features[0].shape[-1])*2,)*2
        support_mask = (support_mask).float()
        support_mask = F.interpolate(support_mask, initSize, mode='bilinear', align_corners=True)
        for idx, feature in enumerate(features):
            feat=[]
            if support_mask.shape[-1]!=feature.shape[-1]:
                support_mask = F.interpolate(support_mask, feature.size()[2:], mode='bilinear', align_corners=True)
            for i in range(bs):
                featI=feature[i].flatten(start_dim=1)#c,hw
                maskI=support_mask[i].flatten(start_dim=1)#hw
                featI = featI * maskI
                maskI=maskI.squeeze()
                meanVal=maskI[maskI>0].mean()
                realSupI=featI[:,maskI>=meanVal]
                if maskI.sum()==0:
                    realSupI=torch.zeros(featI.shape[0],1).cuda()
                feat.append(realSupI)#[b,]ch,w
            features[idx] = feat#nfeatures ,bs,ch,w
        return features


    # que_img, sup_img, sup_mask, que_mask(meta), que_mask(base), cat_idx(meta)
    def forward(self, x, s_x, s_y, y_m, y_b, cat_idx=None, name_dict=None, epoch=None):   
        x_size = x.size()          # torch.Size([4, 3, 641, 641])
        bs = x_size[0]
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)     # h,w:  641 641
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        # Query Feature        
        with torch.no_grad():
            query_feats, query_backbone_layers = self.extract_feats(x, [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4], self.feat_ids,
                                                                   self.bottleneck_ids, self.lids)
        if self.vgg:
            query_feat = F.interpolate(query_backbone_layers[2], size=(query_backbone_layers[3].size(2),query_backbone_layers[3].size(3)),\
                 mode='bilinear', align_corners=True)
            query_feat = torch.cat([query_backbone_layers[3], query_feat], 1)
        else:
            query_feat = torch.cat([query_backbone_layers[3], query_backbone_layers[2]], 1)
        query_feat = self.down_query(query_feat)            

        # Base and Meta
        base_out = self.learner_base(query_backbone_layers[4])
        base_out_soft = base_out.softmax(1)
        if self.training and self.cls_type == 'Base':          
            c_id_array = torch.arange(self.base_classes+1, device='cuda')   
            base_map_list = []
            class_map_list = []
            for b_id in range(bs):                                      
                c_id = cat_idx[b_id] + 1                                    
                c_mask = (c_id_array!=0)&(c_id_array!=c_id)
                non_mask = (c_id_array==0)&(c_id_array==c_id)
                base_map_list.append(base_out_soft[b_id, c_mask,:,:].unsqueeze(0).sum(1,True))
                class_map_list.append(base_out_soft[b_id, non_mask,:,:].unsqueeze(0).sum(1,True))
            base_map = torch.cat(base_map_list, 0)    
        else:
            base_map = base_out_soft[:,1:,:,:].sum(1,True)                  

        supp_pro_list = []
        final_supp_list = []
        mask_list = []
        supp_feat_list = []
        corrs = []
        supp_feat2_list = []
        supp_feat3_list = []
        mask_down_list = []
        supp_attn = 0
        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                support_feats, support_backbone_layers = self.extract_feats(s_x[:,i,:,:,:], [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4],
                                                                        self.feat_ids, self.bottleneck_ids, self.lids)
                final_supp_list.append(support_backbone_layers[4])
                
            # supp_out = self.learner_base(support_backbone_layers[4])
            if self.vgg:
                supp_feat = F.interpolate(support_backbone_layers[2], size=(support_backbone_layers[3].size(2),support_backbone_layers[3].size(3)),
                                            mode='bilinear', align_corners=True)
                supp_feat = torch.cat([support_backbone_layers[3], supp_feat], 1)
            else:
                supp_feat = torch.cat([support_backbone_layers[3], support_backbone_layers[2]], 1)
            
            mask_down = F.interpolate(mask, size=(support_backbone_layers[3].size(2), support_backbone_layers[3].size(3)), mode='bilinear', align_corners=True)
            supp_feat = self.down_supp(supp_feat)
            supp_attn += self.supp_attention(supp_feat, mask_down)
            supp_pro = Weighted_GAP(supp_feat, mask_down)
            supp_pro_list.append(supp_pro)
            supp_feat2_list.append(support_backbone_layers[2])
            supp_feat3_list.append(support_backbone_layers[3])
            supp_feat_list.append(supp_feat)
            mask_down_list.append(mask_down)

            support_feats_1 = self.mask_feature(support_feats, mask)
            corr = Correlation.multilayer_correlation(query_feats, support_feats_1, self.stack_ids)
            corrs.append(corr)      # corr_type: list, corrs_type: list, len(corr): 3
            # supp_out_list.append(supp_out)

        supp_attn = supp_attn / self.shot
        corrs_shot = [corrs[0][i] for i in range(len(self.nsimlairy))]
        for ly in range(len(self.nsimlairy)):
            for s in range(1, self.shot):
                corrs_shot[ly] +=(corrs[s][ly])

        hyper_4 = corrs_shot[0] / self.shot
        hyper_3 = corrs_shot[1] / self.shot
        if self.vgg: 
            hyper_2 = F.interpolate(corr[2], size=(corr[1].size(2),corr[1].size(3)), mode='bilinear', align_corners=True)
        else:
            hyper_2 = corrs_shot[2] / self.shot
        hyper_final = torch.cat([hyper_2, hyper_3, hyper_4],1)
        hyper_final = self.hyper_final(hyper_final)

        # K-Shot Reweighting
        que_gram = get_gram_matrix(query_backbone_layers[2])  
        norm_max = torch.ones_like(que_gram).norm(dim=(1, 2))   
        est_val_list = []
        for supp_item in supp_feat2_list:
            supp_gram = get_gram_matrix(supp_item)
            gram_diff = que_gram - supp_gram
            est_val_list.append((gram_diff.norm(dim=(1, 2)) / norm_max).reshape(bs, 1, 1, 1))  # norm2    [(bs),(bs)...]
        est_val_total = torch.cat(est_val_list, 1)  # [bs, shot, 1, 1]
        if self.shot > 1:
            val1, idx1 = est_val_total.sort(1)   
            val2, idx2 = idx1.sort(1)           
            weight = self.kshot_rw(val1)         
            idx3 = idx1.gather(1, idx2)
            weight = weight.gather(1, idx3)      
            weight_soft = torch.softmax(weight, 1)            
        else:
            weight_soft = torch.ones_like(est_val_total)
        est_val = (weight_soft * est_val_total).sum(1, True)  # [bs, 1, 1, 1]


        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
            q = query_backbone_layers[4]
            s = tmp_supp_feat_4
            bsize, ch_sz, sp_sz, _ = q.size()[:]

            tmp_query = q
            tmp_query = tmp_query.reshape(bsize, ch_sz, -1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_supp = s
            tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1)
            tmp_supp = tmp_supp.permute(0, 2, 1)
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

            similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
            similarity = similarity.max(1)[0].reshape(bsize, sp_sz * sp_sz)
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                        similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=(query_backbone_layers[3].size()[2], query_backbone_layers[3].size()[3]),
                                       mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)
        corr_query_mask = torch.cat(corr_query_mask_list, 1)            
        corr_query_mask = (weight_soft * corr_query_mask).sum(1, True)

        # Support Prototype
        supp_pro = torch.cat(supp_pro_list, 2)  # [bs, 256, shot, 1]
        supp_pro = (weight_soft.permute(0, 2, 1, 3) * supp_pro).sum(2, True)
        concat_feat = supp_pro.expand_as(query_feat)

        corr_q_feat = get_corr_chanfeat(query_feat, supp_feat_list, mask_down_list)
        corr_q_feat = self.attn(corr_q_feat)

        merge_feat1 = torch.cat([query_feat, corr_query_mask, concat_feat], dim=1)
        merge_feat2 = torch.cat([query_feat, corr_q_feat, hyper_final, supp_attn], dim=1)

        specify_feat1 = self.init_merge1(merge_feat1)
        specify_feat2 = self.init_merge2(merge_feat2)   
           
        query_meta1 = self.ASPP_meta(specify_feat1)               
        query_mhsa = self.mhsa(specify_feat2)  
        query_meta2 = self.ASPP1(specify_feat2)
        query_meta = torch.cat([query_meta1, query_meta2, query_mhsa], dim=1)
        query_meta = self.res1_meta(query_meta)                  
        query_meta = self.res2_meta(query_meta) + query_meta    

        meta_out = self.cls_meta(query_meta) 
        meta_out_soft = meta_out.softmax(1)               
        meta_map_bg = meta_out_soft[:,0:1,:,:]           
        meta_map_fg = meta_out_soft[:,1:,:,:]            


        est_map = est_val.expand_as(meta_map_fg)
        meta_map_bg = self.gram_merge(torch.cat([meta_map_bg,est_map], dim=1))       
        meta_map_fg = self.gram_merge(torch.cat([meta_map_fg,est_map], dim=1))      

        merge_map = torch.cat([meta_map_bg, base_map], 1)       
        merge_bg = self.cls_merge(merge_map)                     
        final_out = torch.cat([merge_bg, meta_map_fg], dim=1)    

        # Output Part
        if self.zoom_factor != 1:
            meta_out = F.interpolate(meta_out, size=(h, w), mode='bilinear', align_corners=True)
            base_out = F.interpolate(base_out, size=(h, w), mode='bilinear', align_corners=True)
            final_out = F.interpolate(final_out, size=(h, w), mode='bilinear', align_corners=True)       

        # loss
        if self.training:
            main_loss = self.criterion(final_out, y_m.long())        # Query_pred & Query_GT         
            aux_loss1 = self.criterion(meta_out, y_m.long())         # Supp_seg_pred & Supp_GT
            aux_loss2 = self.criterion(base_out, y_b.long())         # Query_seg_pred & Query_GT    
            return final_out.max(1)[1], main_loss, aux_loss1, aux_loss2
        else:
            return final_out, meta_out, base_out 
