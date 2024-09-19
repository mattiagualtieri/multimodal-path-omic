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
from models.loss import CrossEntropySurvivalLoss, SurvivalClassificationTobitLoss
from nacagat import NarrowContextualAttentionGateTransformer
from dataset.dataset import MultimodalDataset


def train(epoch, config, device, train_loader, model, loss_function, optimizer):
    model.train()
    grad_acc_step = config['training']['grad_acc_step']
    checkpoint_epoch = config['model']['checkpoint_epoch']
    train_loss = 0.0
    risk_scores = torch.zeros(len(train_loader), device=device)
    censorships = torch.zeros(len(train_loader), device=device)
    event_times = torch.zeros(len(train_loader), device=device)
    start_batch_time = time.time()
    for batch_index, (survival_months, survival_class, censorship, omics_data, patches_embeddings) in enumerate(
            train_loader):

        survival_months = survival_months.to(device, non_blocking=True)
        survival_class = survival_class.to(device, non_blocking=True)
        survival_class = survival_class.unsqueeze(0).to(torch.int64)
        censorship = censorship.type(torch.FloatTensor).to(device, non_blocking=True)
        patches_embeddings = patches_embeddings.to(device)
        omics_data = [omic_data.to(device) for omic_data in omics_data]
        hazards, survs, Y, attention_scores = model(wsi=patches_embeddings, omics=omics_data)

        if config['training']['loss'] == 'ce':
            loss = loss_function(Y, survival_class.long())
        elif config['training']['loss'] == 'ces':
            loss = loss_function(hazards, survs, survival_class, c=censorship)
        elif config['training']['loss'] == 'sct':
            loss = loss_function(Y, survival_class, c=censorship)
        else:
            raise RuntimeError(f'Loss "{config["training"]["loss"]}" not implemented')
        loss_value = loss.item()

        risk = -torch.sum(survs, dim=1)
        risk_scores[batch_index] = risk
        censorships[batch_index] = censorship
        event_times[batch_index] = survival_months

        train_loss += loss_value

        if (batch_index + 1) % 50 == 0:
            print('\tbatch: {}, loss: {:.4f}, label: {}, survival_months: {:.2f}, risk: {:.4f}'.format(
                batch_index, loss_value, survival_class.item(), survival_months.item(), float(risk.item())))
            end_batch_time = time.time()
            print('\t\taverage speed: {:.2f}s per batch'.format((end_batch_time - start_batch_time) / 32))
            start_batch_time = time.time()
        loss = loss / grad_acc_step
        loss.backward()

        if (batch_index + 1) % grad_acc_step == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Calculate loss and error for epoch
    train_loss /= len(train_loader)
    risk_scores = risk_scores.detach().cpu().numpy()
    censorships = censorships.detach().cpu().numpy()
    event_times = event_times.detach().cpu().numpy()
    c_index = concordance_index_censored((1 - censorships).astype(bool), event_times, risk_scores)[0]
    print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch + 1, train_loss, c_index))
    if checkpoint_epoch > 0:
        if (epoch + 1) % checkpoint_epoch == 0 and epoch != 0:
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
            loss = loss_function(hazards, survs, survival_class, c=censorship)
        elif config['training']['loss'] == 'sct':
            loss = loss_function(Y, survival_class, c=censorship)
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
    if epoch == 'final validation':
        print('Epoch: {}, val_loss: {:.4f}, val_c_index: {:.4f}'.format(epoch, val_loss, c_index))
    else:
        print('Epoch: {}, val_loss: {:.4f}, val_c_index: {:.4f}'.format(epoch + 1, val_loss, c_index))
    wandb_enabled = config['wandb_enabled']
    if wandb_enabled:
        wandb.log({"val_loss": val_loss, "val_c_index": c_index})


def wandb_init(config):
    wandb.init(
        project='NaCAGAT',
        settings=wandb.Settings(
            init_timeout=300,
        ),
        config={
            'dataset': config['dataset']['name'],
            'optimizer': config['training']['optimizer'],
            'learning_rate': config['training']['lr'],
            'weight_decay': config['training']['weight_decay'],
            'gradient_acceleration_step': config['training']['grad_acc_step'],
            'epochs': config['training']['epochs'],
            'architecture': config['model']['name'],
            'fusion': config['model']['fusion'],
            'loss': config['training']['loss'],
            'model_size': config['model']['model_size'],
            'normalization': config['dataset']['normalize'],
            'standardization': config['dataset']['standardize']
        }
    )


def main(config_path: str):
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    wandb_enabled = config['wandb_enabled']
    if wandb_enabled:
        print('Setting up wandb for report')
        # os.environ["WANDB_SILENT"] = "true"
        os.environ['WANDB__SERVICE_WAIT'] = '300'
        wandb_init(config)

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
    train_size = config['training']['train_size']
    print(f'Using {int(train_size * 100)}% train, {100 - int(train_size * 100)}% validation')
    train_dataset, val_dataset = dataset.split(train_size)
    print(f'Samples in train: {len(train_dataset)}, Samples in validation: {len(val_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    # Model
    model_size = config['model']['model_size']
    omics_sizes = dataset.signature_sizes
    fusion = config['model']['fusion']
    model_name = config['model']['name']
    model = NarrowContextualAttentionGateTransformer(model_size=model_size, omic_sizes=omics_sizes, fusion=fusion, device=device)
    print(f'Trainable parameters of {model_name}: {model.get_trainable_parameters()}')
    checkpoint_path = config['model']['load_from_checkpoint']
    checkpoint = None
    if checkpoint_path is not None:
        print(f'Loading model checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    if device == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device=device)
    # Loss function
    if config['training']['loss'] == 'ce':
        print('Using CrossEntropyLoss during training')
        loss_function = nn.CrossEntropyLoss()
    elif config['training']['loss'] == 'ces':
        print('Using CrossEntropySurvivalLoss during training')
        loss_function = CrossEntropySurvivalLoss()
    elif config['training']['loss'] == 'sct':
        print('Using SurvivalClassificationTobitLoss during training')
        loss_function = SurvivalClassificationTobitLoss()
    else:
        raise RuntimeError(f'Loss "{config["training"]["loss"]}" not implemented')
    # Optimizer
    lr = config['training']['lr']
    weight_decay = config['training']['weight_decay']
    optimizer_name = config['training']['optimizer']
    if optimizer_name == 'adadelta':
        optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()),
                                         lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamax':
        optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()),
                                       lr=lr, weight_decay=weight_decay)
    else:
        optimizer_name = 'adam'
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=lr, weight_decay=weight_decay)
    print(f'Using optimizer: {optimizer_name}')

    starting_epoch = 0
    if checkpoint_path is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']

    print('Training started...')
    model.train()
    epochs = config['training']['epochs']
    for epoch in range(starting_epoch, epochs):
        print(f'Epoch: {epoch + 1}')
        start_time = time.time()
        train(epoch, config, device, train_loader, model, loss_function, optimizer)
        validate(epoch, config, device, val_loader, model, loss_function)
        end_time = time.time()
        print('Time elapsed for epoch {}: {:.0f}s'.format(epoch + 1, end_time - start_time))

    validate('final validation', config, device, val_loader, model, loss_function)
    if wandb_enabled:
        wandb.finish()


if __name__ == '__main__':
    print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M")}] NaCAGAT main started')
    main('config/config.yaml')
    print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M")}] NaCAGAT main finished')


def test_main():
    print('Testing NaCAGAT main...')

    main('config/config_test.yaml')

    print('Test successful')
