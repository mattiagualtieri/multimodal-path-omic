import torch.cuda
import yaml
import time
import datetime
import h5py
import wandb
import numpy as np

from torch.utils.data import DataLoader, random_split
from sksurv.metrics import concordance_index_censored
from models.utils import get_rnaseq_size_from_dataset, get_cnv_size_from_dataset, l1_reg_all
from models.loss import CrossEntropySurvivalLoss, NegativeLogLikelihoodSurvivalLoss, CoxSurvivalLoss
from snn import SelfNormalizingNetwork
from dataset.dataset import GenomicDataset


def train(epoch, config, device, train_loader, model, loss_function, optimizer):
    wandb_enabled = config['wandb_enabled']
    model.train()
    grad_acc_step = config['training']['grad_acc_step']
    train_loss = 0.0
    batch_size = train_loader.batch_size
    risk_scores = np.zeros((len(train_loader.dataset)))
    censorships = np.zeros((len(train_loader.dataset)))
    event_times = np.zeros((len(train_loader.dataset)))
    for batch_index, (survival_months, survival_class, censorship, rnaseq, cnv) in enumerate(
            train_loader):
        survival_months = survival_months.to(device)
        survival_class = survival_class.to(device)
        survival_class = survival_class.to(torch.int64)
        censorship = censorship.type(torch.FloatTensor).to(device)
        cnv = cnv.to(device)
        rnaseq = rnaseq.to(device)
        genomics = torch.concat((cnv, rnaseq), dim=1)
        hazards, survs, Y = model(genomics)

        if config['training']['loss'] == 'ces':
            loss = loss_function(hazards, survs, survival_class, c=censorship, alpha=0.0)
        elif config['training']['loss'] == 'nlls':
            loss = loss_function(hazards, survs, survival_class, c=censorship, alpha=0.0)
        elif config['training']['loss'] == 'coxs':
            loss = loss_function(hazards, survs, c=censorship)
        else:
            raise NotImplementedError('Loss not implemented')

        loss_value = loss.item()
        loss_reg = l1_reg_all(model) * 0.0001

        risk = -torch.sum(survs, dim=1).detach().cpu().numpy()
        start_index = batch_index * batch_size
        end_index = start_index + batch_size
        risk_scores[start_index:end_index] = risk
        censorships[start_index:end_index] = censorship
        event_times[start_index:end_index] = survival_months

        train_loss += loss_value + loss_reg

        if (batch_index + 1) % 1 == 0:
            print('\tbatch: {}, loss: {:.4f}'.format(batch_index, loss_value, ))
        loss = loss / grad_acc_step + loss_reg
        loss.backward()

        if (batch_index + 1) % grad_acc_step == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Calculate loss and error for epoch
    train_loss /= len(train_loader)
    c_index = concordance_index_censored((1 - censorships).astype(bool), event_times, risk_scores)[0]
    print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss, c_index))
    if wandb_enabled:
        wandb.log({"train_loss": train_loss, "train_c_index": c_index})


def validate(epoch, config, device, val_loader, model, loss_function):
    wandb_enabled = config['wandb_enabled']
    model.eval()
    val_loss = 0.0
    batch_size = val_loader.batch_size
    risk_scores = np.zeros((len(val_loader.dataset)))
    censorships = np.zeros((len(val_loader.dataset)))
    event_times = np.zeros((len(val_loader.dataset)))
    for batch_index, (survival_months, survival_class, censorship, rnaseq, cnv) in enumerate(
            val_loader):
        survival_months = survival_months.to(device)
        survival_class = survival_class.to(device)
        survival_class = survival_class.to(torch.int64)
        censorship = censorship.type(torch.FloatTensor).to(device)
        cnv = cnv.to(device)
        rnaseq = rnaseq.to(device)
        genomics = torch.concat((cnv, rnaseq), dim=1)
        with torch.no_grad():
            hazards, survs, Y = model(genomics)

        if config['training']['loss'] == 'ces':
            loss = loss_function(hazards, survs, survival_class, c=censorship, alpha=0.0)
        elif config['training']['loss'] == 'nlls':
            loss = loss_function(hazards, survs, survival_class, c=censorship, alpha=0.0)
        elif config['training']['loss'] == 'coxs':
            loss = loss_function(hazards, survs, c=censorship)
        else:
            raise NotImplementedError('Loss not implemented')

        loss_value = loss.item()
        loss_reg = l1_reg_all(model) * 0.0001

        risk = -torch.sum(survs, dim=1).cpu().numpy()
        start_index = batch_index * batch_size
        end_index = start_index + batch_size
        risk_scores[start_index:end_index] = risk
        censorships[start_index:end_index] = censorship
        event_times[start_index:end_index] = survival_months

        val_loss += loss_value + loss_reg

    # calculate loss and error
    val_loss /= len(val_loader)
    c_index = concordance_index_censored((1 - censorships).astype(bool), event_times, risk_scores)[0]
    print('Epoch: {}, val_loss: {:.4f}, val_c_index: {:.4f}'.format(epoch, val_loss, c_index))
    if wandb_enabled:
        wandb.log({"val_loss": val_loss, "val_c_index": c_index})


def wandb_init(config):
    wandb.init(
        project='SNN',
        config={
            'architecture': config['model']['name'],
            'loss': config['training']['loss'],
            'model_size': config['training']['model_size'],
            'learning_rate': config['training']['lr'],
            'batch_size': config['training']['batch_size'],
            'weight_decay': config['training']['weight_decay'],
            'gradient_acceleration_step': config['training']['grad_acc_step'],
            'epochs': config['training']['epochs'],
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

    dataset_file = config['dataset']['dataset_file']

    with (h5py.File(dataset_file, 'r') as hdf5_file):
        # Dataset
        dataset = GenomicDataset(hdf5_file)
        train_size = config['training']['train_size']
        print(f'Using {int(train_size * 100)}% train, {100 - int(train_size * 100)}% validation')
        train_size = int(train_size * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        batch_size = config['training']['batch_size']
        print(f'Using batch size: {batch_size}')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        # Model
        rnaseq_size = get_rnaseq_size_from_dataset(dataset_file)
        cnv_size = get_cnv_size_from_dataset(dataset_file)
        model_size = config['training']['model_size']
        model = SelfNormalizingNetwork(input_dim=(rnaseq_size + cnv_size), model_size=model_size)
        model.relocate(device)
        # Loss function
        if config['training']['loss'] == 'ces':
            print('Using CrossEntropyLoss during training')
            loss_function = CrossEntropySurvivalLoss()
        elif config['training']['loss'] == 'nlls':
            print('Using NegativeLogLikelihoodSurvivalLoss during training')
            loss_function = NegativeLogLikelihoodSurvivalLoss()
        elif config['training']['loss'] == 'coxs':
            print('Using CoxSurvivalLoss during training')
            loss_function = CoxSurvivalLoss()
        else:
            raise NotImplementedError('Loss not implemented')
        # Optimizer
        lr = config['training']['lr']
        weight_decay = config['training']['weight_decay']
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=lr, weight_decay=weight_decay)

        wandb_enabled = config['wandb_enabled']
        if wandb_enabled:
            print('Setting up wandb for report')
            wandb_init(config)

        print('Training started...')
        model.train()
        epochs = config['training']['epochs']
        for epoch in range(epochs):
            start_time = time.time()
            train(epoch, config, device, train_loader, model, loss_function, optimizer)
            validate(epoch, config, device, val_loader, model, loss_function)
            end_time = time.time()
            print('Time elapsed for epoch {}: {:.0f}s'.format(epoch, end_time - start_time))

        validate('final validation', config, device, val_loader, model, loss_function)
        if wandb_enabled:
            wandb.finish()


if __name__ == '__main__':
    print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M")}] SNN main started')
    main()
    print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M")}] SNN main finished')
