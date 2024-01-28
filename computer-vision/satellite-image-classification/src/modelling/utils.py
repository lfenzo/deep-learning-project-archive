import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
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
    torch.cuda.empty_cache()
    history = []
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )

    model.to(model.device)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []

        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))
            scheduler.step()

        result = evaluate(model, valid_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lr'] = torch.tensor(lrs).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)

    return history


def evaluate_model(model, loader: DataLoader):
    predicted_labels = []
    correct_labels = []

    model.eval()
    model.to(model.device)

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(model.device)
            labels = labels.to(model.device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            predicted_labels = np.append(predicted_labels, predicted.cpu().numpy())
            correct_labels = np.append(correct_labels, labels.cpu().numpy())

    model.to(torch.device("cpu"))
    accuracy = np.sum(predicted_labels == correct_labels) / len(correct_labels)
    print(f"Accuracy for test set: {accuracy}")

    return predicted_labels, correct_labels
