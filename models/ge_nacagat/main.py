import torch.cuda
import yaml
import os
import time
import datetime
import wandb
import torch.nn as nn
import torch.optim.lr_scheduler as lrs

from torch.utils.data import DataLoader
from models.utils import l1_reg
from ge_nacagat import GeneExprNarrowContextualAttentionGateTransformer
from dataset.ge_dataset import MultimodalGeneExprPredDataset


def train(epoch, config, device, train_loader, model, loss_function, optimizer, scheduler, reg_function):
    model.train()
    grad_acc_step = config['training']['grad_acc_step']
    use_scheduler = config['training']['scheduler']
    lambda_reg = config['training']['lambda']
    checkpoint_epoch = config['model']['checkpoint_epoch']
    train_loss = 0.0
    start_batch_time = time.time()
    for batch_index, (gene_expr_class, omics_data, patches_embeddings) in enumerate(
            train_loader):

        gene_expr_class = gene_expr_class.to(device, non_blocking=True)
        patches_embeddings = patches_embeddings.to(device)
        omics_data = [omic_data.to(device) for omic_data in omics_data]
        Y, attention_scores = model(wsi=patches_embeddings, omics=omics_data)

        if config['training']['loss'] == 'ce':
            loss = loss_function(Y.unsqueeze(0), gene_expr_class)
        else:
            raise RuntimeError(f'Loss "{config["training"]["loss"]}" not implemented')
        loss_value = loss.item()

        if reg_function is None:
            loss_reg = 0
        else:
            loss_reg = reg_function(model) * lambda_reg

        train_loss += loss_value + loss_reg

        if (batch_index + 1) % 50 == 0:
            print('\tbatch: {}, loss: {:.4f}, gene_expr_value: {:.4f}, prediction: {:.4f}'.format(
                batch_index, loss_value + loss_reg, gene_expr_class.item(), Y))
            end_batch_time = time.time()
            print('\t\taverage speed: {:.2f}s per batch'.format((end_batch_time - start_batch_time) / 32))
            start_batch_time = time.time()
        loss = loss / grad_acc_step + loss_reg
        loss.backward()

        if (batch_index + 1) % grad_acc_step == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Calculate loss and error for epoch
    train_loss /= len(train_loader)

    if use_scheduler:
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        print('Epoch: {}, lr: {:.8f}, train_loss: {:.4f}'.format(epoch + 1, lr, train_loss))
    else:
        print('Epoch: {}, train_loss: {:.4f}'.format(epoch + 1, train_loss))
    if checkpoint_epoch > 0:
        if (epoch + 1) % checkpoint_epoch == 0 and epoch != 0:
            now = datetime.datetime.now().strftime('%Y%m%d%H%M')
            filename = f'{config["model"]["name"]}_{config["dataset"]["name"]}_E{epoch + 1}_{now}.pt'
            checkpoint_dir = config['model']['checkpoint_dir']
            checkpoint_path = os.path.join(checkpoint_dir, filename)
            print(f'Saving model into {checkpoint_path}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)
    wandb_enabled = config['wandb']['enabled']
    if wandb_enabled:
        wandb.log({"train_loss": train_loss, "train_mse": train_loss})


def validate(epoch, config, device, val_loader, model, loss_function, reg_function):
    model.eval()
    val_loss = 0.0
    lambda_reg = config['training']['lambda']
    for batch_index, (gene_expr_class, omics_data, patches_embeddings) in enumerate(
            val_loader):

        gene_expr_class = gene_expr_class.to(device, non_blocking=True)
        gene_expr_class = gene_expr_class.unsqueeze(0).to(torch.int64)
        patches_embeddings = patches_embeddings.to(device)
        omics_data = [omic_data.to(device) for omic_data in omics_data]
        with torch.no_grad():
            Y, attention_scores = model(wsi=patches_embeddings, omics=omics_data)

        if config['training']['loss'] == 'ce':
            loss = loss_function(Y, gene_expr_class.long())
        else:
            raise RuntimeError(f'Loss "{config["training"]["loss"]}" not implemented')
        loss_value = loss.item()

        if reg_function is None:
            loss_reg = 0
        else:
            loss_reg = reg_function(model) * lambda_reg

        val_loss += loss_value + loss_reg

    # calculate loss and error
    val_loss /= len(val_loader)
    if epoch == 'final validation':
        print('Epoch: {}, val_loss: {:.4f}'.format(epoch, val_loss))
    else:
        print('Epoch: {}, val_loss: {:.4f}'.format(epoch + 1, val_loss))
    wandb_enabled = config['wandb']['enabled']
    if wandb_enabled:
        wandb.log({"val_loss": val_loss, "val_mse": val_loss})


def test(config, device, epoch, val_loader, model, patient, save=False):
    model.eval()
    output_dir = config['training']['test_output_dir']
    now = datetime.datetime.now().strftime('%Y%m%d%H%M')
    for batch_index, (gene_expr_class, omics_data, patches_embeddings) in enumerate(
            val_loader):

        gene_expr_class = gene_expr_class.to(device, non_blocking=True)
        gene_expr_class = gene_expr_class.unsqueeze(0).to(torch.int64)
        patches_embeddings = patches_embeddings.to(device)
        omics_data = [omic_data.to(device) for omic_data in omics_data]
        print(f'[{batch_index}] Gene Expression Class: {gene_expr_class.item()}')
        with torch.no_grad():
            Y, attention_scores = model(wsi=patches_embeddings, omics=omics_data)
            print(f'Prediction: {Y}')
            print(f'Attn min: {attention_scores["coattn"].min()}, Attn max: {attention_scores["coattn"].max()}, Attn '
                  f'mean: {attention_scores["coattn"].mean()}')

            if save:
                output_file = os.path.join(output_dir, f'ATTN_{patient}_{now}_E{epoch}_{batch_index}.pt')
                print(f'Saving attention in {output_file}')
                torch.save(attention_scores['coattn'], output_file)


def wandb_init(config):
    wandb.init(
        project=config['wandb']['project'],
        settings=wandb.Settings(
            init_timeout=300,
        ),
        config={
            'model': config['model']['name'],
            'dataset': config['dataset']['name'],
            'optimizer': config['training']['optimizer'],
            'learning_rate': config['training']['lr'],
            'weight_decay': config['training']['weight_decay'],
            'gradient_acceleration_step': config['training']['grad_acc_step'],
            'epochs': config['training']['epochs'],
            'architecture': config['model']['name'],
            'fusion': config['model']['fusion'],
            'loss': config['training']['loss'],
            'scheduler': config['training']['scheduler'],
            'lambda': config['training']['lambda'],
            'gamma': config['training']['gamma'],
            'model_size': config['model']['model_size'],
            'normalization': config['dataset']['normalize'],
            'standardization': config['dataset']['standardize'],
            'leave_one_out': config['training']['leave_one_out']
        }
    )


def main(config_path: str):
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    wandb_enabled = config['wandb']['enabled']
    if wandb_enabled:
        print('Setting up wandb for report')
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
    gene = config['model']['gene']
    dataset = MultimodalGeneExprPredDataset(file_csv, config, gene=gene, use_signatures=True, normalize=normalize, standardize=standardize)
    leave_one_out = config['training']['leave_one_out'] is not None
    train_size = config['training']['train_size']
    print(f'Using {int(train_size * 100)}% train, {100 - int(train_size * 100)}% validation')
    test_patient = config['training']['leave_one_out']
    train_dataset, val_dataset, test_dataset = dataset.split(train_size, test=leave_one_out, patient=test_patient)
    print(f'Samples in train: {len(train_dataset)}, Samples in validation: {len(val_dataset)}')
    if test_dataset is not None:
        print(f'Testing patient {test_patient}')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    output_attn_epoch = config['training']['output_attn_epoch']
    # Model
    model_size = config['model']['model_size']
    omics_sizes = dataset.signature_sizes
    fusion = config['model']['fusion']
    model_name = config['model']['name']
    model = GeneExprNarrowContextualAttentionGateTransformer(model_size=model_size, omic_sizes=omics_sizes, fusion=fusion, device=device)
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
    else:
        raise RuntimeError(f'Loss "{config["training"]["loss"]}" not implemented')
    # Optimizer
    lr = config['training']['lr']
    weight_decay = config['training']['weight_decay']
    optimizer_name = config['training']['optimizer']
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=lr)
    elif optimizer_name == 'adadelta':
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

    scheduler = config['training']['scheduler']
    if scheduler == 'exp':
        gamma = config['training']['gamma']
        scheduler = lrs.ExponentialLR(optimizer, gamma=gamma)
    else:
        scheduler = None

    starting_epoch = 0
    if checkpoint_path is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']

    lambda_param = config['training']['lambda']
    if lambda_param:
        reg_function = l1_reg
    else:
        reg_function = None

    print('Training started...')
    model.train()
    epochs = config['training']['epochs']
    for epoch in range(starting_epoch, epochs):
        print(f'Epoch: {epoch + 1}')
        start_time = time.time()
        train(epoch, config, device, train_loader, model, loss_function, optimizer, scheduler, reg_function)
        validate(epoch, config, device, val_loader, model, loss_function, reg_function)
        if leave_one_out:
            save = False
            if (epoch + 1) % output_attn_epoch == 0:
                save = True
            test_patient = config['training']['leave_one_out']
            test(config, device, epoch + 1, test_loader, model, test_patient, save=save)
        end_time = time.time()
        print('Time elapsed for epoch {}: {:.0f}s'.format(epoch + 1, end_time - start_time))

    validate('final validation', config, device, val_loader, model, loss_function, reg_function)

    if wandb_enabled:
        wandb.finish()


if __name__ == '__main__':
    print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M")}] GeneExpr-NaCAGAT main started')
    main('config/config.yaml')
    print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M")}] GeneExpr-NaCAGAT main finished')


def test_main():
    print('Testing GeneExpr-NaCAGAT main...')

    main('config/config_test.yaml')

    print('Test successful')
