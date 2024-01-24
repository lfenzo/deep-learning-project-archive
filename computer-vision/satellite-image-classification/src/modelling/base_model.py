import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import utils


class BaseImageClassifier(nn.Module):

    def __init__(self,):
        super().__init__()
        self.reset_parameters()

    def reset_parameters(self):
        self.weight = torch.empty(3, 4)
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')

    def training_step(self, batch) -> Tensor:
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        return loss

    def validation_step(self, batch) -> dict[str, Tensor]:
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        acc = utils.accuracy(outputs, labels)
        return {'valid_accuracy': acc, 'valid_loss': loss}

    def validation_epoch_end(self, outputs) -> dict[str, Tensor]:
        batch_losses = [x['valid_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accuracies = [x['valid_accuracy'] for x in outputs]
        epoch_accuracy = torch.stack(batch_accuracies).mean()
        return {'valid_accuracy': epoch_accuracy.item(), 'valid_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result) -> None:
        print(
            "Epoch [{}] | lr: {:.5f} | train loss: {:.4f} | valid loss: {:.4f} | accuracy: {:.4f}"
            .format(epoch), result['lr'], result['train_loss'], result['valid_loss'], result['accuracy']
        )
