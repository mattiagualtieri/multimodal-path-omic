import torch
import os
import pandas as pd
import numpy as np
import h5py

from torch.utils.data import Dataset
from scipy import stats


class MultimodalGeneExprPredDataset(Dataset):
    def __init__(self, file: str, config, gene: str):
        self.data = pd.read_csv(file)

        if config['dataset']['decider_only']:
            print('Using DECIDER data only')
            self.data = self.data.loc[self.data['is_decider'] == 1.0]
            self.data.reset_index(drop=True, inplace=True)

        self.patches_dir = config['dataset']['patches_dir']
        if self.patches_dir is None:
            self.patches_dir = ''

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        gene_expr_class = self.gene_expr_class[index]

        slide_name = self.data['slide_id'][index].replace('.svs', '.pt')
        patches_embeddings = torch.load(os.path.join(self.patches_dir, slide_name))

        return gene_expr_class, patches_embeddings

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

        instance.gene_expr_class = original_instance.gene_expr_class

        return instance
