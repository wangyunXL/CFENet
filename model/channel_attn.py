import torch
import torch.nn as nn
import torch.utils.data as Data


# Convolutional Block Attention Module
import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class channel_attn(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(channel_attn, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, bias=False), nn.ReLU())
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, bias=False), nn.ReLU())
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        down_x = self.conv1(x)
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        out = down_x * self.sigmoid(avgout + maxout)


        return out

def get_corr_chanfeat(query_feat, supp_feat_list, mask_list):
    softmax = nn.Softmax(dim=1)
    corr_q_feat_list = []
    for i in range(len(supp_feat_list)):
        supp_feat = supp_feat_list[i]
        supp_mask = mask_list[i]

        n,c,h,w = query_feat.shape
        q_feat = query_feat.reshape(n,c,h*w)

        s_feat = supp_feat*supp_mask
        s_feat = s_feat.reshape(n,c,h*w)
        s_feat_T = s_feat.permute(0,2,1)

        aff = torch.matmul(s_feat_T, q_feat)
        aff = softmax(aff.clamp(min=0))
        corr_feat = torch.matmul(s_feat, aff)
        corr_feat = corr_feat / (corr_feat.norm(dim=1, p=2, keepdim=True)+1e-5)
        corr_feat = corr_feat.reshape(n,c,h,w)
        corr_q_feat = query_feat * corr_feat
        corr_q_feat_list.append(corr_q_feat)

    corr_q_feat = torch.cat(corr_q_feat_list, dim=1)      
    return corr_q_feat
