import torch
from torch.utils.data import Dataset


class MultimodalDataset(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        self.indexes = list(self.hdf5_file.keys())

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        index = self.indexes[index]
        case_group = self.hdf5_file[index]

        clinical_group = case_group['clinical']
        survival_months = torch.tensor(clinical_group['survival_months'][()])
        survival_class = torch.tensor(clinical_group['survival_class'][()])
        censorship = torch.tensor(clinical_group['censorship_status'][()])

        omics_group = case_group['omics']
        omics_data = [torch.tensor(omics_group[dataset][:]) for dataset in omics_group]

        wsi_group = case_group['wsi']
        patches_embeddings = torch.tensor(wsi_group['patches'][()])

        return survival_months, survival_class, censorship, omics_data, patches_embeddings


class GenomicDataset(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        self.indexes = list(self.hdf5_file.keys())

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        index = self.indexes[index]
        case_group = self.hdf5_file[index]

        clinical_group = case_group['clinical']
        survival_months = torch.tensor(clinical_group['survival_months'][()])
        survival_class = torch.tensor(clinical_group['survival_class'][()])
        censorship = torch.tensor(clinical_group['censorship_status'][()])

        omics_group = case_group['genomics']
        rnaseq = torch.tensor(omics_group['rnaseq'][()])
        cnv = torch.tensor(omics_group['cnv'][()])

        return survival_months, survival_class, censorship, rnaseq, cnv
