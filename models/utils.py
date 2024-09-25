import h5py
import math
import torch
import torch.nn as nn


def get_omics_sizes_from_dataset(hdf5_file):
    category_counts = {}
    with h5py.File(hdf5_file, 'r') as f:
        first_case_id = list(f.keys())[0]
        omics_group = f[first_case_id]['omics']
        for category in omics_group.keys():
            category_counts.setdefault(category, 0)
            category_counts[category] = len(omics_group[category])
    sorted_counts = [category_counts[category] for category in sorted(category_counts.keys())]
    return sorted_counts


def get_rnaseq_size_from_dataset(hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        first_case_id = list(f.keys())[0]
        omics_group = f[first_case_id]['genomics']
        return len(omics_group['rnaseq'])


def get_cnv_size_from_dataset(hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        first_case_id = list(f.keys())[0]
        omics_group = f[first_case_id]['genomics']
        return len(omics_group['cnv'])


def l1_reg(model):
    reg = None
    for W in model.parameters():
        if reg is None:
            reg = torch.abs(W).sum()
        else:
            reg = reg + torch.abs(W).sum()
    return reg


def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()
