import torch.cuda
import yaml
import datetime
import torch.nn as nn

from torch.utils.data import DataLoader
from models.mcat.mcat import MultimodalCoAttentionTransformer
from dataset.dataset import MultimodalDatasetV2


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
    file_csv = config['dataset']['file']
    use_signatures = config['model']['use_signatures']
    top_rnaseq = config['model']['top_rnaseq']
    dataset = MultimodalDatasetV2(file_csv, config, use_signatures=use_signatures, top_rnaseq=top_rnaseq)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # Model
    if use_signatures:
        omics_sizes = dataset.signature_sizes
    else:
        omics_sizes = [
            dataset.rnaseq_size,
            dataset.cnv_size,
            dataset.mut_size
        ]
    model_name = config['model']['name'].lower()
    if model_name == 'mcat':
        model = MultimodalCoAttentionTransformer(omic_sizes=omics_sizes)
    else:
        raise RuntimeError(f'Inference for model {model_name} not implemented')
    checkpoint_path = config['model']['checkpoint']
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = nn.DataParallel(model)
    model.to(device=device)

    print('Inference started...')
    model.eval()
    for batch_index, (survival_months, survival_class, censorship, omics_data, patches_embeddings) in enumerate(
            loader):
        print(f'batch {batch_index}, survival_months: {survival_months}, censorship: {censorship}')
        patches_embeddings = patches_embeddings.to(device)
        omics_data = [omic_data.to(device) for omic_data in omics_data]
        hazards, survs, Y, attention_scores = model(wsi=patches_embeddings, omics=omics_data)
        print(f'hazards: {hazards}, survs: {survs}, Y: {Y}')
        A_coattn = attention_scores['coattn']
        print(f'Co-Attention scores: {len(A_coattn)}')


if __name__ == '__main__':
    print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M")}] Inference started')
    main()
    print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M")}] Inference finished')
