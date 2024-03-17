import gc
import copy
import time
import numpy as np
import torch
from torch.cuda import amp

from tqdm import tqdm
from collections import defaultdict

from sklearn.metrics import accuracy_score
import torch.nn as nn


from config import CONFIG, c_

def _criterion(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)

@torch.no_grad()
def _valid_one_epoch(model, dataloader, device, epoch, optimizer):
    model.eval()

    dataset_size = 0
    running_loss = 0.0
    epoch_loss = None

    TARGETS = []
    PREDS = []

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (images, labels) in bar:
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)

        batch_size = images.size(0)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        loss = _criterion(outputs, labels)

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        PREDS.append(preds.view(-1).cpu().detach().numpy())
        TARGETS.append(labels.view(-1).cpu().detach().numpy())

        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])

    TARGETS = np.concatenate(TARGETS)
    PREDS = np.concatenate(PREDS)
    val_acc = accuracy_score(TARGETS, PREDS)
    gc.collect()

    return epoch_loss, val_acc


def _train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    scaler = amp.GradScaler()

    dataset_size = 0
    running_loss = 0.0
    epoch_loss = None

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (images, labels) in bar:
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)

        batch_size = images.size(0)

        with amp.autocast(enabled=True):
            outputs = model(images)
            loss = _criterion(outputs, labels)
            loss = loss / CONFIG['n_accumulate']

        scaler.scale(loss).backward()

        if (step + 1) % CONFIG['n_accumulate'] == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            for p in model.parameters():
                p.grad = None

            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()

    return epoch_loss

## обучение
def run_training(model, optimizer, scheduler, device, num_epochs, train_loader, valid_loader):
    # To automatically log gradients

    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_acc = 0
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        train_epoch_loss = _train_one_epoch(model, optimizer, scheduler,
                                           dataloader=train_loader,
                                           device=CONFIG['device'], epoch=epoch)

        val_epoch_loss, val_epoch_acc = _valid_one_epoch(model, valid_loader,
                                                        device=CONFIG['device'],
                                                        epoch=epoch,
                                                         optimizer=optimizer)

        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        history['Valid Acc'].append(val_epoch_acc)


        print(f'Valid Acc: {val_epoch_acc}')

        # deep copy the model
        if val_epoch_acc >= best_epoch_acc:
            print(f"{c_}Validation Acc Improved ({best_epoch_acc} ---> {val_epoch_acc})")
            best_epoch_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = "ACC{:.4f}_epoch{:.0f}.bin".format(best_epoch_acc, epoch)
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory

        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best ACC: {:.4f}".format(best_epoch_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history