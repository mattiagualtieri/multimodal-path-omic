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


def train(epoch, config, device, train_loader, model, loss_function, optimizer):
    model.train()
    grad_acc_step = config['training']['grad_acc_step']
    checkpoint_epoch = config['model']['checkpoint_epoch']
    train_loss = 0.0
    risk_scores = np.zeros((len(train_loader)))
    censorships = np.zeros((len(train_loader)))
    event_times = np.zeros((len(train_loader)))
    for batch_index, (survival_months, survival_class, censorship, omics_data, patches_embeddings) in enumerate(
            train_loader):
        survival_months = survival_months.to(device)
        survival_class = survival_class.to(device)
        survival_class = survival_class.unsqueeze(0).to(torch.int64)
        censorship = censorship.type(torch.FloatTensor).to(device)
        patches_embeddings = patches_embeddings.to(device)
        omics_data = [omic_data.to(device) for omic_data in omics_data]
        hazards, survs, Y, attention_scores = model(wsi=patches_embeddings, omics=omics_data)

        if config['training']['loss'] == 'ce':
            loss = loss_function(Y, survival_class.long())
        elif config['training']['loss'] == 'ces':
            loss = loss_function(hazards, survs, survival_class, c=censorship, alpha=0.0)
        else:
            raise RuntimeError(f'Loss "{config["training"]["loss"]}" not implemented')
        loss_value = loss.item()

        risk = -torch.sum(survs, dim=1).detach().cpu().numpy()
        risk_scores[batch_index] = risk.item()
        censorships[batch_index] = censorship.item()
        event_times[batch_index] = survival_months.item()

        train_loss += loss_value

        if (batch_index + 1) % 32 == 0:
            print('\tbatch: {}, loss: {:.4f}, label: {}, survival_months: {}, risk: {:.4f}'.format(
                batch_index, loss_value, survival_class.item(), survival_months.item(), float(risk.item())))
        loss = loss / grad_acc_step
        loss.backward()

        if (batch_index + 1) % grad_acc_step == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Calculate loss and error for epoch
    train_loss /= len(train_loader)
    c_index = concordance_index_censored((1 - censorships).astype(bool), event_times, risk_scores)[0]
    print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch + 1, train_loss, c_index))
    if checkpoint_epoch % (epoch + 1) == 0:
        now = datetime.datetime.now().strftime('%Y%m%d%H%M')
        filename = f'{config["model"]["name"]}_{epoch + 1}_{now}.pt'
        checkpoint_dir = config['model']['checkpoint_dir']
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        print(f'Saving model into {checkpoint_path}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, checkpoint_path)
    wandb_enabled = config['wandb_enabled']
    if wandb_enabled:
        wandb.log({"train_loss": train_loss, "train_c_index": c_index})


def validate(epoch, config, device, val_loader, model, loss_function):
    model.eval()
    val_loss = 0.0
    risk_scores = np.zeros((len(val_loader)))
    censorships = np.zeros((len(val_loader)))
    event_times = np.zeros((len(val_loader)))
    for batch_index, (survival_months, survival_class, censorship, omics_data, patches_embeddings) in enumerate(
            val_loader):
        survival_months = survival_months.to(device)
        survival_class = survival_class.to(device)
        survival_class = survival_class.unsqueeze(0).to(torch.int64)
        censorship = censorship.type(torch.FloatTensor).to(device)
        patches_embeddings = patches_embeddings.to(device)
        omics_data = [omic_data.to(device) for omic_data in omics_data]
        with torch.no_grad():
            hazards, survs, Y, attention_scores = model(wsi=patches_embeddings, omics=omics_data)

        if config['training']['loss'] == 'ce':
            loss = loss_function(Y, survival_class.long())
        elif config['training']['loss'] == 'ces':
            loss = loss_function(hazards, survs, survival_class, c=censorship, alpha=0.0)
        else:
            raise RuntimeError(f'Loss "{config["training"]["loss"]}" not implemented')
        loss_value = loss.item()

        risk = -torch.sum(survs, dim=1).cpu().numpy()
        risk_scores[batch_index] = risk.item()
        censorships[batch_index] = censorship.item()
        event_times[batch_index] = survival_months.item()

        val_loss += loss_value

    # calculate loss and error
    val_loss /= len(val_loader)
    c_index = concordance_index_censored((1 - censorships).astype(bool), event_times, risk_scores)[0]
    print('Epoch: {}, val_loss: {:.4f}, val_c_index: {:.4f}'.format(epoch, val_loss, c_index))
    wandb_enabled = config['wandb_enabled']
    if wandb_enabled:
        wandb.log({"val_loss": val_loss, "val_c_index": c_index})


def wandb_init(config):
    wandb.init(
        project='MCAT',
        config={
            'learning_rate': config['training']['lr'],
            'weight_decay': config['training']['weight_decay'],
            'gradient_acceleration_step': config['training']['grad_acc_step'],
            'epochs': config['training']['epochs'],
            'architecture': config['model']['name'],
            'fusion': config['model']['fusion']
        }
    )


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
    dataset = MultimodalDatasetV2(file_csv, config, use_signatures=True)
    train_size = config['training']['train_size']
    print(f'Using {int(train_size * 100)}% train, {100 - int(train_size * 100)}% validation')
    train_size = int(train_size * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    # Model
    omics_sizes = dataset.signature_sizes
    fusion = config['model']['fusion']
    model = MultimodalCoAttentionTransformer(omic_sizes=omics_sizes, fusion=fusion)
    checkpoint_path = config['model']['load_from_checkpoint']
    checkpoint = None
    if checkpoint_path is not None:
        print(f'Loading model checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    model = nn.DataParallel(model)
    model.to(device=device)
    # Loss function
    if config['training']['loss'] == 'ce':
        print('Using CrossEntropyLoss during training')
        loss_function = nn.CrossEntropyLoss()
    elif config['training']['loss'] == 'ces':
        print('Using CrossEntropySurvivalLoss during training')
        loss_function = CrossEntropySurvivalLoss()
    else:
        raise RuntimeError(f'Loss "{config["training"]["loss"]}" not implemented')
    # Optimizer
    lr = config['training']['lr']
    weight_decay = config['training']['weight_decay']
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr, weight_decay=weight_decay)
    starting_epoch = 0
    if checkpoint_path is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']

    wandb_enabled = config['wandb_enabled']
    if wandb_enabled:
        print('Setting up wandb for report')
        wandb_init(config)

    print('Training started...')
    model.train()
    epochs = config['training']['epochs']
    for epoch in range(starting_epoch, epochs):
        start_time = time.time()
        train(epoch, config, device, train_loader, model, loss_function, optimizer)
        validate(epoch, config, device, val_loader, model, loss_function)
        end_time = time.time()
        print('Time elapsed for epoch {}: {:.0f}s'.format(epoch, end_time - start_time))

    validate('final validation', config, device, val_loader, model, loss_function)
    if wandb_enabled:
        wandb.finish()


if __name__ == '__main__':
    print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M")}] MCAT main started')
    main()
    print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M")}] MCAT main finished')
