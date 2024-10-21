import torch
import yaml
import os
import datetime

from torch.utils.data import DataLoader
from dataset.dataset import MultimodalDataset
from models.mcat.mcat import MultimodalCoAttentionTransformer
from models.nacagat.nacagat import NarrowContextualAttentionGateTransformer


def inference(config, device, loader, model, patient):
    model.eval()
    output_dir = config['inference']['output_dir']
    model_name = config['model']['name']
    now = datetime.datetime.now().strftime('%Y%m%d%H%M')
    for batch_index, (survival_months, survival_class, censorship, omics_data, patches_embeddings) in enumerate(
            loader):
        survival_months = survival_months.to(device)
        survival_class = survival_class.to(device)
        survival_class = survival_class.unsqueeze(0).to(torch.int64)
        censorship = censorship.type(torch.FloatTensor).to(device)
        print(f'[{batch_index}] Survival months: {survival_months.item()}, Survival class: {survival_class.item()}, '
              f'Censorship: {censorship.item()}')
        patches_embeddings = patches_embeddings.to(device)
        omics_data = [omic_data.to(device) for omic_data in omics_data]

        with torch.no_grad():
            hazards, survs, Y, attention_scores = model(wsi=patches_embeddings, omics=omics_data)
            risk = -torch.sum(survs, dim=1).cpu().numpy()
            print(f'Hazards: {hazards}, Survs: {survs}, Risk: {risk}, Y: {Y}')
            print(f'Attn min: {attention_scores["coattn"].min()}, Attn max: {attention_scores["coattn"].max()}, Attn '
                  f'mean: {attention_scores["coattn"].mean()}')

            output_file = os.path.join(output_dir, f'ATTN_{model_name}_{patient}_{now}_{batch_index}.pt')
            print(f'Saving attention in {output_file}')
            torch.save(attention_scores['coattn'], output_file)


def main(config_path: str):
    with open(config_path) as config_file:
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
    file_csv = config['dataset']['file']
    normalize = config['dataset']['normalize']
    standardize = config['dataset']['standardize']
    print(f'Normalization: {normalize}, Standardization: {standardize}')
    dataset = MultimodalDataset(file_csv, config, use_signatures=True, normalize=normalize, standardize=standardize)
    patient = config['inference']['patient']
    _, __, dataset = dataset.split(0.9, test=True, patient=patient)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # Model
    model_size = config['model']['model_size']
    omics_sizes = dataset.signature_sizes
    fusion = config['model']['fusion']
    model_name = config['model']['name']
    print(f'Using model: {model_name}')
    if model_name.lower() == 'mcat':
        model = MultimodalCoAttentionTransformer(model_size=model_size, omic_sizes=omics_sizes, fusion=fusion,
                                                 device=device, inference=True)
    elif model_name.lower() == 'nacagat':
        model = NarrowContextualAttentionGateTransformer(model_size=model_size, omic_sizes=omics_sizes, fusion=fusion,
                                                         device=device)
    else:
        raise RuntimeError(f'Model {model_name} not implemented!')

    checkpoint_path = config['model']['checkpoint']
    if checkpoint_path is None:
        raise RuntimeError('No checkpoint specified')
    print(f'Loading model checkpoint from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device=device)

    print('Inference started...')
    inference(config, device, loader, model, patient)


if __name__ == '__main__':
    print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M")}] Inference started')
    main('config/config.yaml')
    print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M")}] Inference finished')


def test_inference_main():
    print('Testing inference main...')

    main('config/config_test.yaml')

    print('Test successful')
