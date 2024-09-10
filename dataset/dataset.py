import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from scipy import stats


class MultimodalDatasetV2(Dataset):
    def __init__(self, file, config, use_signatures=False, top_rnaseq=None, remove_incomplete_samples=True, inference=False, normalize=True):
        self.data = pd.read_csv(file)
        survival_class, _ = pd.qcut(self.data['survival_months'], q=4, retbins=True, labels=False)
        self.data['survival_class'] = survival_class
        if inference:
            self.patches_dir = config['inference']['dataset']['patches_dir']
        else:
            self.patches_dir = config['dataset']['patches_dir']
        if self.patches_dir is None:
            self.patches_dir = ''

        if remove_incomplete_samples:
            slide_index = 0
            complete_data_only = []
            for slide in self.data['slide_id']:
                slide_name = slide.replace('.svs', '.pt')
                if os.path.exists(os.path.join(self.patches_dir, slide_name)):
                    complete_data_only.append(self.data.iloc[slide_index])
                slide_index += 1
            self.data = pd.DataFrame(complete_data_only)
            self.data.reset_index(drop=True, inplace=True)
            print(f'Remaining samples after removing incomplete: {len(self.data)}')

        # SLIDES
        self.slides = {}
        for slide in self.data['slide_id']:
            slide_name = slide.replace('.svs', '.pt')
            if os.path.exists(os.path.join(self.patches_dir, slide_name)):
                patches_embeddings = torch.load(os.path.join(self.patches_dir, slide_name))
                self.slides[slide_name] = patches_embeddings

        # RNA
        self.rnaseq = self.data.iloc[:, self.data.columns.str.endswith('_rnaseq')].astype(float)
        if top_rnaseq is not None:
            rnaseq = self.data[self.data.columns[self.data.columns.str.contains('_rnaseq')]]
            mad = stats.median_abs_deviation(rnaseq, axis=0)
            sort_idx = np.argsort(mad)[-top_rnaseq:]
            self.rnaseq = rnaseq[rnaseq.columns[sort_idx]]
        self.rnaseq_size = len(self.rnaseq.columns)
        if normalize:
            self.rnaseq = 2 * (self.rnaseq - self.rnaseq.min()) / (self.rnaseq.max() - self.rnaseq.min()) - 1
        print(f'RNA data size: {self.rnaseq_size}')
        # CNV
        self.cnv = self.data.iloc[:, self.data.columns.str.endswith('_cnv')].astype(float)
        self.cnv_size = len(self.cnv.columns)
        print(f'CNV data size: {self.cnv_size}')
        # MUT
        self.mut = self.data.iloc[:, self.data.columns.str.endswith('_mut')].astype(float)
        self.mut_size = len(self.mut.columns)
        print(f'MUT data size: {self.mut_size}')

        # Signatures
        self.use_signatures = use_signatures
        if self.use_signatures:
            self.signature_sizes = []
            self.signature_data = {}
            if inference:
                signatures_file = config['inference']['dataset']['signatures']
            else:
                signatures_file = config['dataset']['signatures']
            signatures_df = pd.read_csv(signatures_file)
            self.signatures = signatures_df.columns
            for signature_name in self.signatures:
                columns = {}
                for gene in signatures_df[signature_name].dropna():
                    gene += '_rnaseq'
                    if gene in self.data.columns:
                        columns[gene] = self.data[gene]
                self.signature_data[signature_name] = pd.DataFrame(columns)
                self.signature_sizes.append(len(self.signature_data[signature_name].columns))
            print(f'Signatures size: {self.signature_sizes}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        survival_months = self.data['survival_months'][index]
        survival_class = self.data['survival_class'][index]
        censorship = self.data['censorship'][index]

        slide_name = self.data['slide_id'][index].replace('.svs', '.pt')
        # patches_embeddings = torch.load(os.path.join(self.patches_dir, slide_name))
        patches_embeddings = self.slides[slide_name]

        if not self.use_signatures:
            rnaseq = self.rnaseq.iloc[index].values
            cnv = self.cnv.iloc[index].values
            mut = self.mut.iloc[index].values
            omics_data = {
                'rnaseq': torch.tensor(rnaseq, dtype=torch.float32),
                'cnv': torch.tensor(cnv, dtype=torch.float32),
                'mut': torch.tensor(mut, dtype=torch.float32)
            }
        else:
            omics_data = []
            for signature in self.signatures:
                signature_data = self.signature_data[signature].iloc[index].astype(float).values
                omics_data.append(torch.tensor(signature_data, dtype=torch.float32))

        return survival_months, survival_class, censorship, omics_data, patches_embeddings


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
