import torch.cuda
import yaml
import os
import time
import datetime
import wandb
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from sksurv.metrics import concordance_index_censored
from models.loss import CrossEntropySurvivalLoss
from mcat import MultimodalCoAttentionTransformer
from dataset.dataset import MultimodalDatasetV2


def inference(config, device, loader, model):
    model.eval()
    output_attention_file = config['inference']['attention_file']
    for batch_index, (survival_months, survival_class, censorship, omics_data, patches_embeddings) in enumerate(
            loader):
        survival_months = survival_months.to(device)
        survival_class = survival_class.to(device)
        survival_class = survival_class.unsqueeze(0).to(torch.int64)
        censorship = censorship.type(torch.FloatTensor).to(device)
        print(f'survival_months: {survival_months.item()}, survival_class: {survival_class.item()}, censorship: {censorship.item()}')
        patches_embeddings = patches_embeddings.to(device)
        omics_data = [omic_data.to(device) for omic_data in omics_data]
        hazards, survs, Y, attention_scores = model(wsi=patches_embeddings, omics=omics_data)
        print(f'hazards: {hazards}, survs: {survs}, Y: {Y}')

        print(f'Saving attention weights in {output_attention_file}')
        torch.save(attention_scores['coattn'], output_attention_file)


def main():
    with open('config/config.yaml') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available')
        device = 'cpu'
    elif device == 'cuda' and torch.cuda.is_available():
        print('CUDA is available!')
        print(f'Device count: {torch.cuda.device_count()}')
        for device_index in range(torch.cuda.device_count()):
            print(f'Using device: {torch.cuda.get_device_name(device_index)}')
    print(f'Running on {device.upper()}')

    # Dataset
    file_csv = config['inference']['dataset']['file']
    dataset = MultimodalDatasetV2(file_csv, config, use_signatures=True, inference=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # Model
    omics_sizes = dataset.signature_sizes
    fusion = config['model']['fusion']
    model = MultimodalCoAttentionTransformer(omic_sizes=omics_sizes, fusion=fusion)
    checkpoint_path = config['inference']['model']['load_from_checkpoint']
    if checkpoint_path is None:
        raise RuntimeError('No checkpoint specified')
    print(f'Loading model checkpoint from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device=device)

    print('Inference started...')
    inference(config, device, loader, model)


if __name__ == '__main__':
    print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M")}] MCAT inference started')
    main()
    print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M")}] MCAT inference finished')
