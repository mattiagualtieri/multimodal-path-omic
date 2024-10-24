import torch
import os
import pandas as pd
import numpy as np
import h5py

from torch.utils.data import Dataset
from scipy import stats


class MultimodalGeneExprPredDataset(Dataset):
    def __init__(self, file: str, config, gene: str, use_signatures=False, top_rnaseq=None, remove_incomplete_samples=True, inference=False, standardize=False, normalize=False):
        self.data = pd.read_csv(file)

        if config['dataset']['decider_only']:
            print('Using DECIDER data only')
            self.data = self.data.loc[self.data['is_decider'] == 1.0]
            self.data.reset_index(drop=True, inplace=True)

        if inference:
            self.patches_dir = config['inference']['dataset']['patches_dir']
        else:
            self.patches_dir = config['dataset']['patches_dir']
        if self.patches_dir is None:
            self.patches_dir = ''

        self.use_h5_dataset = False
        try:
            self.use_h5_dataset = config['dataset']['h5_dataset'] is not None
        except KeyError:
            pass

        if remove_incomplete_samples:
            slide_index = 0
            complete_data_only = []
            if not self.use_h5_dataset:
                for slide in self.data['slide_id']:
                    slide_name = slide.replace('.svs', '.pt')
                    if os.path.exists(os.path.join(self.patches_dir, slide_name)):
                        complete_data_only.append(self.data.iloc[slide_index])
                    slide_index += 1
            else:
                self.h5_dataset = config['dataset']['h5_dataset']
                self.h5_file = h5py.File(self.h5_dataset, 'r')
                for slide in self.data['slide_id']:
                    slide_name = slide.replace('.svs', '')
                    if slide_name in self.h5_file:
                        complete_data_only.append(self.data.iloc[slide_index])
                    slide_index += 1

            self.data = pd.DataFrame(complete_data_only)
            self.data.reset_index(drop=True, inplace=True)
            print(f'Remaining samples after removing incomplete: {len(self.data)}')

        self.data.dropna(axis=1, inplace=True)

        print(f'Testing gene expression: {gene}')
        self.gene_expr_value = self.data[f'{gene}_rnaseq']
        self.data = self.data.drop(f'{gene}_rnaseq', axis=1)
        n_classes = 3
        gene_expr_class, class_intervals = pd.qcut(self.gene_expr_value, q=n_classes, retbins=True, labels=False)
        self.data['gene_expr_class'] = gene_expr_class
        print('Class intervals: [')
        for i in range(0, n_classes):
            print('\t{}: [{:.2f} - {:.2f}]'.format(i, class_intervals[i], class_intervals[i + 1]))
        print(']')

        self.gene_expr_class = self.data['gene_expr_class'].values

        rnaseq_columns = [col for col in self.data.columns if col.endswith('_rnaseq')]
        if standardize:
            for col in rnaseq_columns:
                self.data[col] = (self.data[col] - self.data[col].mean()) / self.data[col].std()
        if normalize:
            for col in rnaseq_columns:
                self.data[col] = 2 * (self.data[col] - self.data[col].min()) / (self.data[col].max() - self.data[col].min()) - 1

        # RNA
        self.rnaseq = self.data.iloc[:, self.data.columns.str.endswith('_rnaseq')].astype(float)
        if top_rnaseq is not None:
            rnaseq = self.data[self.data.columns[self.data.columns.str.contains('_rnaseq')]]
            mad = stats.median_abs_deviation(rnaseq, axis=0)
            sort_idx = np.argsort(mad)[-top_rnaseq:]
            self.rnaseq = rnaseq[rnaseq.columns[sort_idx]]
        self.rnaseq_size = len(self.rnaseq.columns)
        if standardize:
            self.rnaseq = (self.rnaseq - self.rnaseq.mean()) / self.rnaseq.std()
        if normalize:
            self.rnaseq = 2 * (self.rnaseq - self.rnaseq.min()) / (self.rnaseq.max() - self.rnaseq.min()) - 1
        self.rnaseq = torch.tensor(self.rnaseq.values, dtype=torch.float32)

        # CNV
        self.cnv = self.data.iloc[:, self.data.columns.str.endswith('_cnv')].astype(float)
        self.cnv_size = len(self.cnv.columns)
        self.cnv = torch.tensor(self.cnv.values, dtype=torch.float32)

        # MUT
        self.mut = self.data.iloc[:, self.data.columns.str.endswith('_mut')].astype(float)
        self.mut_size = len(self.mut.columns)
        self.mut = torch.tensor(self.mut.values, dtype=torch.float32)

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
                self.signature_data[signature_name] = torch.tensor(pd.DataFrame(columns).values, dtype=torch.float32)
                self.signature_sizes.append(self.signature_data[signature_name].shape[1])
            print(f'Signatures size: {self.signature_sizes}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        gene_expr_class = self.gene_expr_class[index]

        if not self.use_h5_dataset:
            slide_name = self.data['slide_id'][index].replace('.svs', '.pt')
            patches_embeddings = torch.load(os.path.join(self.patches_dir, slide_name))
        else:
            slide_name = self.data['slide_id'][index].replace('.svs', '')
            patches_embeddings = torch.tensor(self.h5_file[slide_name])

        if not self.use_signatures:
            omics_data = {
                'rnaseq': self.rnaseq[index],
                'cnv': self.cnv[index],
                'mut': self.mut[index]
            }
        else:
            omics_data = []
            for signature in self.signatures:
                signature_data = self.signature_data[signature][index]
                omics_data.append(signature_data)

        return gene_expr_class, omics_data, patches_embeddings

    def split(self, train_size, test: bool = False, patient: str = ''):
        # Ensure train_size is a valid ratio
        if not 0 < train_size < 1:
            raise ValueError("train_size should be a float between 0 and 1.")

        # Get unique patients
        unique_patients = self.data['patient'].unique()

        # Shuffle patients randomly
        np.random.shuffle(unique_patients)

        # Determine the number of patients in the train split
        train_patient_count = int(len(unique_patients) * train_size)

        # Split patients into train and test sets
        train_patients = unique_patients[:train_patient_count]
        val_patients = unique_patients[train_patient_count:]

        # Filter the data into train and test sets
        test_dataset = None
        if test:
            train_data = self.data[self.data['patient'].isin(train_patients)].copy()
            train_data = train_data[train_data['patient'] != patient]
            val_data = self.data[self.data['patient'].isin(val_patients)].copy()
            val_data = val_data[val_data['patient'] != patient]
            test_data = self.data[self.data['patient'] == patient].copy()
            test_data.reset_index(drop=True, inplace=True)
            test_dataset = MultimodalGeneExprPredDataset.from_dataframe(test_data, self)
        else:
            train_data = self.data[self.data['patient'].isin(train_patients)].copy()
            val_data = self.data[self.data['patient'].isin(val_patients)].copy()

        # Reset indices for train and test datasets
        train_data.reset_index(drop=True, inplace=True)
        val_data.reset_index(drop=True, inplace=True)

        # Create new instances of MultimodalDataset with the train and test data
        train_dataset = MultimodalGeneExprPredDataset.from_dataframe(train_data, self)
        val_dataset = MultimodalGeneExprPredDataset.from_dataframe(val_data, self)

        return train_dataset, val_dataset, test_dataset

    @classmethod
    def from_dataframe(cls, df, original_instance):
        # Create a new MultimodalDataset instance from an existing DataFrame
        # while preserving the original configuration and parameters
        instance = cls.__new__(cls)  # Create a new instance without calling __init__

        # Reset indices in the DataFrame
        df = df.reset_index(drop=True)
        # Copy attributes from the original instance
        instance.data = df
        instance.patches_dir = original_instance.patches_dir
        instance.use_h5_dataset = original_instance.use_h5_dataset
        instance.use_signatures = original_instance.use_signatures
        if original_instance.use_signatures:
            instance.signatures = original_instance.signatures
            instance.signature_sizes = original_instance.signature_sizes
        instance.h5_dataset = original_instance.h5_dataset if original_instance.use_h5_dataset else None

        if original_instance.use_h5_dataset:
            instance.h5_file = h5py.File(instance.h5_dataset, 'r')

        # Copy RNA, CNV, and MUT data with the new subset of data
        instance.gene_expr_class = original_instance.gene_expr_class

        # RNA
        rnaseq = df.iloc[:, df.columns.str.endswith('_rnaseq')].astype(float)
        if original_instance.rnaseq_size == len(rnaseq.columns):
            instance.rnaseq = torch.tensor(rnaseq.values, dtype=torch.float32)
        else:
            instance.rnaseq = torch.tensor(np.zeros((len(df), original_instance.rnaseq_size)), dtype=torch.float32)

        # CNV
        cnv = df.iloc[:, df.columns.str.endswith('_cnv')].astype(float)
        if original_instance.cnv_size == len(cnv.columns):
            instance.cnv = torch.tensor(cnv.values, dtype=torch.float32)
        else:
            instance.cnv = torch.tensor(np.zeros((len(df), original_instance.cnv_size)), dtype=torch.float32)

        # MUT
        mut = df.iloc[:, df.columns.str.endswith('_mut')].astype(float)
        if original_instance.mut_size == len(mut.columns):
            instance.mut = torch.tensor(mut.values, dtype=torch.float32)
        else:
            instance.mut = torch.tensor(np.zeros((len(df), original_instance.mut_size)), dtype=torch.float32)

        # Copy signature data if use_signatures is True
        if original_instance.use_signatures:
            instance.signature_sizes = original_instance.signature_sizes
            instance.signature_data = {}
            for signature_name in original_instance.signatures:
                signature_data = original_instance.signature_data[signature_name]
                indices = df.index
                instance.signature_data[signature_name] = signature_data[indices]

        return instance

    def __del__(self):
        if self.use_h5_dataset:
            self.h5_file.close()
