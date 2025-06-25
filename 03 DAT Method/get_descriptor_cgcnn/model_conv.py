from __future__ import print_function, division

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len):
        super(ConvLayer, self).__init__()

        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len,
                                 2 * self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len * 2)).view(N, M, self.atom_fea_len * 2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class ConvNetDat(nn.Module):
    def __init__(self, orig_atom_fea_len, nbr_fea_len):
        # 先调用父类的 __init__ 方法
        super(ConvNetDat, self).__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, 64)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=64,nbr_fea_len=nbr_fea_len) for _ in range(3)])

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        # print("卷积操作之前forward收到的参数维度如下：")
        # print("atom_fea:", atom_fea.shape)
        # print("nbr_fea:", nbr_fea.shape)
        # print("nbr_fea_idx:", nbr_fea_idx.shape)
        # print("crystal_atom_idx:", crystal_atom_idx.shape)

        # 对输入的原子特征进行嵌入操作,将原始的原子特征转换为特定维度的特征表示
        atom_fea = self.embedding(atom_fea)
        # print("嵌入后的原子特征维度 atom_fea: ", atom_fea.shape)

        # 多个卷积层的迭代
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
            # print("before pooling atom_fea: ", atom_fea.shape)

        # 池化操作
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        # print("池化后 after pooling atom_fea: ", crys_fea.shape)

        return crys_fea

    def pooling(self, atom_fea, crystal_atom_idx):
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) == \
               atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)
