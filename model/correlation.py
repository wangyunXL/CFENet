r""" Provides functions that builds/manipulates correlation tensors """
import torch
import torch.nn as nn

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

    # corr_feat = torch.cat(corr_feat_list, dim=1)             # 空间位置加权
    corr_q_feat = torch.cat(corr_q_feat_list, dim=1)         # 将调整后的特征相乘
    # corr_q_feat = (corr_q_feat * weight_soft).sum(1, True)
    # corr_attn_feat = torch.cat(corr_attn_list, dim=1)        # 通道加权

    # return corr_feat, corr_q_feat, corr_attn_feat
    return corr_q_feat, corr_q_feat_list


def get_corr_supp(query_feat, supp_feat_list):
    corr_feat_list = []
    corr_q_feat_list = []
    corr_attn_list = []
    for i in range(len(supp_feat_list)):
        supp_feat = supp_feat_list[i]

        n,c,h,w = query_feat.shape
        q_feat = query_feat.reshape(n,c,h*w)
        q_feat_T = q_feat.permute(0,2,1)

        s_feat = supp_feat
        s_feat = s_feat.reshape(n,c,h*w)
        s_feat_T = s_feat.permute(0,2,1)

        aff = torch.matmul(q_feat_T, s_feat)
        corr_feat = torch.matmul(q_feat, aff)
        chan_corr = torch.matmul(q_feat, s_feat_T)
        corr_feat = corr_feat / (corr_feat.norm(dim=1, p=2, keepdim=True)+1e-5)
        corr_feat = corr_feat.reshape(n,c,h,w)
        corr_q_feat = supp_feat * corr_feat
        chan_corr_feat = torch.matmul(chan_corr, supp_feat)
        corr_feat_list.append(corr_feat)
        corr_q_feat_list.append(corr_q_feat)
        corr_attn_list.append(chan_corr_feat)
    corr_feat = torch.cat(corr_feat_list, dim=1)             # 空间位置加权
    corr_q_feat = torch.cat(corr_q_feat_list, dim=1)         # 将调整后的特征相乘
    corr_attn_feat = torch.cat(corr_attn_list, dim=1)        # 通道加权

    return corr_feat, corr_q_feat, corr_attn_feat

class Correlation:

    @classmethod
    def pixel_relation(cls, query_feat, support_feats):
        eps = 1e-5

        corrs = []
        for i in range(len(support_feats)):
            support_feat = support_feats[i]
            bsz, ch, hb, wb = support_feat.size()
            support_feat = support_feat.view(bsz, ch, -1)
            # support_feat_norm = torch.norm(support_feat, dim=1, p=2, keepdim=True)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)

            bsz, ch, ha, wa = query_feat.size()
            query_feat = query_feat.view(bsz, ch, -1)
            # query_feat_norm = torch.norm(query_feat, dim=1, p=2, keepdim=True)
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)


            corr = torch.bmm(query_feat.transpose(1, 2), support_feat)
            corr = corr.clamp(min=0)    # 取非负
            corr = corr.mean(dim=2,keepdim=True).squeeze(2)
            corr = corr.view(bsz, hb, wb)
            corr = corr / (torch.norm(corr, dim=2, keepdim=True)+1e-5)
            corrs.append(corr.unsqueeze(1))

        return corrs


# def get_corr_map(self, query_feat, supp_feat_list):
#     corr_query_mask_list = []
#     cosine_eps = 1e-7
#     for i, tmp_supp_feat in enumerate(supp_feat_list):
#         resize_size = tmp_supp_feat.size(2)
    
#         q = query_feat                                    # torch.Size([4, 2048, 60, 60])
#         s = tmp_supp_feat
#         bsize, ch_sz, sp_sz, _ = q.size()[:]

#         tmp_query = q
#         tmp_query = tmp_query.reshape(bsize, ch_sz, -1)          # torch.Size([4, 2048, 3600])
#         tmp_query_norm = torch.norm(tmp_query, 2, 1, True)       # torch.Size([4, 1, 3600])

#         tmp_supp = s               
#         tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1)            # torch.Size([4, 2048, 3600])
#         tmp_supp = tmp_supp.permute(0, 2, 1)
#         tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

#         similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   # 计算余弦相似度值: torch.Size([4, 3600, 3600])
#         similarity = similarity.max(1)[0].reshape(bsize, sp_sz*sp_sz)           # 取最大的余弦相似度值，并reshape为：(b,h*w)
#         similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)  # min-max归一化（归一化到(0,1)之间）
#         corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)         # 生成先验掩码图
#         corr_query_mask_list.append(corr_query)                      #  pascal:    corr_query:（bsize, 1, 60, 60）

    


#     corr_query_mask = torch.cat(corr_query_mask_list, 1)             #  pascal:    corr_query:（bsize, shot, 60, 60）

#     return corr_query_mask



# def PCM(self, corr1, corr2):      # (Pixel correlation Module)
#     n,c,h,w = corr1.shape
#     corr1_flatten = corr1.reshape(n,c,h*w)
#     corr2_flatten = corr2.reshape(n,c,h*w)
#     relevance = torch.matmul(corr2_flatten.permute(0,2,1), corr2_flatten)
#     relevance = relevance / (torch.norm(relevance, dim=1, keepdim=True)+1e-5)
#     # relevance = relevance / (torch.norm(relevance, dim=2, keepdim=True)+1e-5)
#     new_corr1 = torch.matmul(corr1_flatten, relevance).reshape(n,-1,h,w)

#     return new_corr1

# def feat_PCM(self, masked_feat, query_feat):
#     n,c,h,w = query_feat.shape
#     masked_feat_flatten = masked_feat.reshape(n,c,h*w)
#     query_feat_flatten = query_feat.reshape(n,c,h*w)              # query_feat_flatten:torch.Size([24, 256, 3600])
#     masked_feat_flatten = masked_feat_flatten / (torch.norm(masked_feat_flatten, dim=2, keepdim=True)+1e-5)
#     masked_flatten = masked_feat_flatten.permute(0,2,1)

#     # aff = F.relu(torch.matmul(query_flatten, masked_feat_flatten))
#     aff = F.relu(torch.matmul(masked_feat_flatten, masked_flatten))
#     aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-5)      # aff.shape:torch.Size([24, 256, 256])
#     # print("query_feat_flatten:{}, aff.shape:{}".format(query_feat_flatten.shape, aff.shape))
#     new_feat = torch.matmul(aff, query_feat_flatten)

#     new_feat = (new_feat / (torch.norm(new_feat, p=2, dim=2, keepdim=True)+1e-5))
#     new_feat = new_feat.view(n,-1,h,w)

#     return new_feat

# def get_supp_corr(self, supp_feat_list, query_feat):
#     corr_supp_mask_list = []
#     cosine_eps = 1e-7
#     for i, tmp_supp_feat in enumerate(supp_feat_list):
        
#         q = query_feat                                    # torch.Size([4, 2048, 60, 60])
#         s = tmp_supp_feat
#         bsize, ch_sz, sp_sz, _ = q.size()[:]

#         tmp_supp = s               
#         tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1)            # torch.Size([4, 2048, 3600])
#         tmp_supp_norm = torch.norm(tmp_supp, 2, 1, True) 

#         tmp_query = q
#         tmp_query = tmp_query.reshape(bsize, ch_sz, -1)          # torch.Size([4, 2048, 3600])
#         tmp_query = tmp_query.permute(0, 2, 1)
#         tmp_query_norm = torch.norm(tmp_query, 2, 2, True) 
        

#         similarity = torch.bmm(tmp_query, tmp_supp)/(torch.bmm(tmp_query_norm, tmp_supp_norm) + cosine_eps)   # 计算余弦相似度值: torch.Size([4, 3600, 3600])
#         similarity = similarity.max(1)[0].reshape(bsize, sp_sz*sp_sz)           # 取最大的余弦相似度值，并reshape为：(b,h*w)
#         similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)  # min-max归一化（归一化到(0,1)之间）
#         corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)         # 生成先验掩码图
#         corr_supp_mask_list.append(corr_query)                      #  pascal:    corr_query:（bsize, 1, 60, 60）

#     return corr_supp_mask_list

# def get_q_masked_feat(self, use_mask_list, mask_bg, feat3):
#     feat3_list = []
#     for i in range(feat3.shape[0]):   
#         if use_mask_list[i] == True:
#             m_feat3 = feat3[i] * mask_bg[i].unsqueeze(0)         
#         else:
#             m_feat3 = feat3[i]
#         feat3_list.append(m_feat3.unsqueeze(0))

#     masked_feat3 = torch.cat(feat3_list, dim=0)
#     masked_side3 = self.side3(masked_feat3)

#     return masked_side3

# def handle_mask(self, corr_feat):
#     en_corr = Enhan_corr(corr_feat)
#     corr_mask = Trans_mask(en_corr)
#     return corr_mask


# def get_masked_qfeat(self, query_feat, supp_attn):
#     n,c,h,w = query_feat.shape
#     q_feat = query_feat.reshape(n,-1,h*w)
#     s_attn = supp_attn.reshape(n,-1,h*w)
#     aff = torch.matmul(s_attn.permute(0,2,1), q_feat)
#     aff = aff.softmax(dim=1)
#     masked_feat = torch.matmul(q_feat, aff)
#     masked_feat = masked_feat / (torch.norm(masked_feat, dim=2, p=2, keepdim=True)+1e-5)
#     masked_feat = masked_feat.reshape(n,-1,h,w)
#     return masked_feat