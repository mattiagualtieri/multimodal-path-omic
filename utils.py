import os
import pandas as pd
import h5py
import numpy as np
import torch


def preprocess_clinical(input_file, output_file):
    data = pd.read_csv(input_file)
    risk_labels, q_bins = pd.qcut(data['survival_months'], q=4, retbins=True, labels=False)
    cases = 0
    with h5py.File(output_file, 'w') as hdf5_file:
        for index in data.index:
            cases += 1
            case_data = data.loc[index]
            case_group = hdf5_file.create_group('case_{:03d}'.format(index))
            labels_group = case_group.create_group('clinical')
            labels_group.create_dataset('censorship_status', data=np.array(case_data['censorship']))
            labels_group.create_dataset('survival_months', data=np.array(case_data['survival_months']))
            labels_group.create_dataset('survival_class', data=np.array(risk_labels[index]))

    print(f'Created labels datasets for {cases} cases')


def preprocess_omics(input_file, signatures_file, output_file):
    omics = pd.read_csv(input_file)
    signatures = pd.read_csv(signatures_file)
    cases = 0
    with h5py.File(output_file, 'a') as hdf5_file:
        for index in omics.index:
            cases += 1
            omics_data = omics.loc[index]
            case_group = hdf5_file['case_{:03d}'.format(index)]
            omics_group = case_group.create_group('omics')
            for omics_category in signatures.columns:
                data = []
                for gene in signatures[omics_category].dropna():
                    gene += '_rnaseq'
                    if gene in omics_data:
                        data.append(omics_data[gene])
                omics_group.create_dataset(omics_category, data=np.array(data))
    print(f'Created omics datasets for {cases} cases')


def preprocess_patch_embeddings(input_file, emb_dir, output_file):
    data = pd.read_csv(input_file)
    cases = 0
    with h5py.File(output_file, 'a') as hdf5_file:
        for index in data.index:
            cases += 1
            slide_id = data.loc[index]['slide_id']
            slide_id = slide_id.replace('.svs', '.h5')
            slide_path = os.path.join(emb_dir, slide_id)
            if os.path.exists(slide_path):
                case_group = hdf5_file['case_{:03d}'.format(index)]
                with h5py.File(slide_path, 'r') as emb_hdf5_file:
                    embeddings = torch.tensor(emb_hdf5_file['features'][()])
                    wsi_group = case_group.create_group('wsi')
                    wsi_group.create_dataset('patches', data=np.array(embeddings))
                    cases += 1
            else:
                print(f'Warning: file {slide_path} not found')
    print(f'Created WSI datasets for {cases} cases')


def remove_incomplete_cases(dataset_file):
    with h5py.File(dataset_file, 'r+') as hdf5_file:
        cases = list(hdf5_file.keys())
        removed = 0
        print(f'Total cases in dataset: {len(cases)}')
        for case_id in cases:
            to_remove = False
            case_group = hdf5_file[case_id]
            try:
                case_group['clinical']
            except KeyError:
                print(f'"clinical" group not found for case {case_id}')
                to_remove = True
            try:
                case_group['omics']
            except KeyError:
                print(f'"omics" group not found for case {case_id}')
                to_remove = True
            try:
                case_group['wsi']
            except KeyError:
                print(f'"wsi" group not found for case {case_id}')
                to_remove = True

            if to_remove:
                print(f'Deleting case {case_id}')
                del hdf5_file[case_id]
                removed += 1

        cases = list(hdf5_file.keys())
        print(f'Total complete cases in dataset: {len(cases)}')


def get_omics_sizes_from_dataset(hdf5_file):
    category_counts = {}
    with h5py.File(hdf5_file, 'r') as f:
        first_case_id = list(f.keys())[0]  # Get the first case ID
        omics_group = f[first_case_id]['omics']  # Access the 'omics' group under the first case ID
        for category in omics_group.keys():
            category_counts.setdefault(category, 0)
            category_counts[category] = len(omics_group[category])  # Count number of values in each category
    sorted_counts = [category_counts[category] for category in sorted(category_counts.keys())]
    return sorted_counts
