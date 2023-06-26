from dataclasses import dataclass
import os
from pathlib import Path

import torch
from torch.autograd import Variable
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# from trainer.lenet import LeNet5
from architectures.resnet import ResNet34
from trainer.utils import transformer, adjust_learning_rate
from torch import nn


@dataclass
class TeacherTrainerHyperparams:
    epochs: int
    batch_size: int
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler | None = None


class TeacherTrainer:
    """Train and save trainer model"""

    def __init__(
        self, 
        model: nn.Module,
        hyperparams: TeacherTrainerHyperparams,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        model_path: Path,
    ):

        self.acc = 0

        self.model_path = model_path
        
        self.net = model.cuda()
        self.hyperparams = hyperparams

        self.train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=hyperparams.batch_size, 
            shuffle=True, 
            num_workers=0
        )
        self.eval_dataloader = DataLoader(
            dataset=eval_dataset, 
            batch_size=hyperparams.batch_size, 
            num_workers=0
        )

        self.criterion = torch.nn.CrossEntropyLoss().cuda()

    def train_step(self, epoch):
        """Train step for teacher"""

        self.net.train()
        for i, (images, labels) in enumerate(pbar := tqdm(self.train_dataloader)):
            images = images.cuda()
            labels = labels.cuda()

            self.hyperparams.optimizer.zero_grad()
            output = self.net(images)
            loss = self.criterion(output, labels)

            loss.backward()
            self.hyperparams.optimizer.step()
            if self.hyperparams.lr_scheduler:
                self.hyperparams.lr_scheduler.step()
            pbar.set_description(f"Epoch {epoch} | Loss {loss.item():.4f}")

    def eval_step(self):
        """Validation step for the teacher"""
        self.net.eval()
        total_correct = 0
        avg_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.eval_dataloader):
                images = images.cuda()
                labels = labels.cuda()
                output = self.net(images)
                avg_loss += self.criterion(output, labels).sum()
                pred = output.argmax(dim=1)
                total_correct += pred.eq(labels.data.view_as(pred)).sum()

        avg_loss /= len(self.eval_dataloader.dataset) # type: ignore
        self.acc = float(total_correct) / len(self.eval_dataloader.dataset) # type: ignore

        print('Test Avg. Loss: %f, Accuracy: %f' %
              (avg_loss.item(), self.acc))  # type: ignore

    def train(self):
        """Trainer run function"""

        for epoch in range(1, self.hyperparams.epochs+1):
            self.train_step(epoch)
            self.eval_step()
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net, self.model_path)


# class TeacherTrainer:
#     """Train and save trainer model"""

#     def __init__(
#         self, 
#         dataset: str,
#         model: nn.Module,
#         model_path: Path,
#         data_path: Path,
#         hyperparams: TeacherTrainerHyperparams | None = None,
#     ):

#         self.acc = 0
#         self.acc_best = 0

#         self.dataset = dataset
#         self.model_path = model_path

#         if not os.path.exists(data_path):
#             os.makedirs(data_path)

#         train_trans, test_trans = transformer(self.dataset)

#         if self.dataset == 'mnist':
#             data_train = MNIST(
#                 root=data_path, # type: ignore
#                 transform=train_trans,
#                 download=True
#             )
#             self.data_test = MNIST(
#                 root=data_path, # type: ignore
#                 train=False,
#                 transform=test_trans,
#                 download=True
#             )
#             batch_size = 256
#             self.epochs = 10
#             self.net = model.cuda()  # LeNet5().cuda()
#             self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

#         elif self.dataset == 'cifar10':
#             data_train = CIFAR10(
#                 data_path, transform=train_trans, download=True) # type: ignore
#             self.data_test = CIFAR10(
#                 data_path, train=False, transform=test_trans, download=True) # type: ignore
#             batch_size = 128
#             self.epochs = 200
#             self.net = ResNet34().cuda()
#             self.optimizer = torch.optim.SGD(
#                 self.net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

#         elif self.dataset == 'cifar100':
#             data_train = CIFAR100(
#                 data_path, transform=train_trans, download=True) # type: ignore
#             self.data_test = CIFAR100(
#                 data_path, train=False, transform=test_trans, download=True) # type: ignore
#             batch_size = 128
#             self.epochs = 200
#             self.net = ResNet34(num_classes=100).cuda()
#             self.optimizer = torch.optim.SGD(
#                 self.net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
#         else:
#             raise ValueError("dataset not configured")

#         self.data_train_loader = DataLoader(
#             data_train, batch_size=batch_size, shuffle=True, num_workers=0)
#         self.data_test_loader = DataLoader(
#             self.data_test, batch_size=batch_size, num_workers=0)

#         self.criterion = torch.nn.CrossEntropyLoss().cuda()

#     def train(self, epoch):
#         """Train step for teacher"""
#         if self.dataset != 'mnist':
#             adjust_learning_rate(self.optimizer, epoch)
#         self.net.train()
#         for i, (images, labels) in enumerate(self.data_train_loader):
#             images, labels = Variable(images).cuda(), Variable(labels).cuda()

#             self.optimizer.zero_grad()
#             output = self.net(images)
#             loss = self.criterion(output, labels)

#             if i == 1:
#                 print('Train - Epoch %d, Batch: %d, Loss: %f' %
#                       (epoch, i, loss.data.item()))
#             loss.backward()
#             self.optimizer.step()

#     def test(self):
#         """Validation step for the teacher"""
#         self.net.eval()
#         total_correct = 0
#         avg_loss = 0.0
#         with torch.no_grad():
#             for i, (images, labels) in enumerate(self.data_test_loader):
#                 images, labels = Variable(
#                     images).cuda(), Variable(labels).cuda()
#                 output = self.net(images)
#                 avg_loss += self.criterion(output, labels).sum()
#                 pred = output.data.max(1)[1]
#                 total_correct += pred.eq(labels.data.view_as(pred)).sum()

#         avg_loss /= len(self.data_test)
#         self.acc = float(total_correct) / len(self.data_test)
#         if self.acc_best < self.acc:
#             self.acc_best = self.acc

#         print('Test Avg. Loss: %f, Accuracy: %f' %
#               (avg_loss.data.item(), self.acc))  # type: ignore

#     def build(self):
#         """Trainer run function"""

#         print('-'*30+' Train teacher start '+'-'*30)
#         for epoch in range(1, self.epochs+1):
#             self.train(epoch)
#             self.test()
#         self.model_path.parent.mkdir(parents=True, exist_ok=True)
#         torch.save(self.net, self.model_path)
#         print('-'*30+' Train teacher end '+'-'*30)
