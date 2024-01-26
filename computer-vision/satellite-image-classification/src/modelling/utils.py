import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(model, valid_dataloader: DataLoader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in valid_dataloader]
    return model.validation_epoch_end(outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_model(
    model,
    epochs: int,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    max_lr: float,
    weight_decay=0,
    optimizer=None,
    grad_clip=None,
):
    history = []
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )

    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []

        for batch in train_loader:
            loss = model.training_step(batch)
            print(loss)
            train_losses.append(loss)
#
#            if grad_clip:
#                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

        optimizer.step()
        optimizer.zero_grad()

        lrs.append(get_lr(optimizer))
        scheduler.step()

        result = evaluate(model, valid_loader)
        result['train_error'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end()
        history.append(result)

    return history
