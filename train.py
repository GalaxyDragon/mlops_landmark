import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from implementations.config import CONFIG, MODEL_PATH
from implementations.utils import set_seed
from implementations.landmark_model import LandmarkRetrievelModel
from implementations.landmark_dataset import prepare_loaders
from implementations.landmark_train import run_training


if __name__ == "__main__":
    set_seed(CONFIG['seed'])

    model = LandmarkRetrievelModel(CONFIG['model_name'])
    model.to(CONFIG['device'])
    train_loader, valid_loader = prepare_loaders()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = None

    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['T_max'], eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CONFIG['T_0'], T_mult=1, eta_min=CONFIG['min_lr'])


    model, history = run_training(model, optimizer, scheduler,
                                  device=CONFIG['device'],
                                  num_epochs=CONFIG['epochs'],
                                  train_loader=train_loader,
                                  valid_loader=valid_loader)

    torch.save(model, MODEL_PATH+'/model.pkl')
    print(history)

