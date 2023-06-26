from dataclasses import dataclass
from pathlib import Path

import torchvision
import torch
from torch.autograd import Variable
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm 

from trainer.utils import transformer


@dataclass
class StudentTrainerHyperparams:
    epochs: int
    batch_size: int
    teacher_temperature: float
    optimizer: torch.optim.Optimizer


class StudentTrainer:

    def __init__(
            self,
            teacher: torch.nn.Module,
            student: torch.nn.Module,
            model_save_path: Path,
            train_dataset: Dataset,
            test_dataset: Dataset,
            hyperparams: StudentTrainerHyperparams
        ):
        self.teacher = teacher.cuda().eval()
        self.student = student.cuda()
        self.save_path = model_save_path
        model_save_path.parent.mkdir(parents=True, exist_ok=True)

        self.train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=hyperparams.batch_size, 
            shuffle=True, 
            num_workers=0
        )
        self.test_dataloader = DataLoader(
            dataset=test_dataset, 
            batch_size=hyperparams.batch_size, 
            num_workers=0
        )
        self.hyperparams = hyperparams
        self.criterion_train = torch.nn.BCELoss().cuda()
        self.criterion_test = torch.nn.CrossEntropyLoss().cuda()
        self.acc = 0.0

    def train_step(self, epoch):
        self.student.train()
        for i, (images, labels) in enumerate(pbar := tqdm(self.train_dataloader)):
            images = images.cuda() 
            labels = labels.cuda()
            teacher_output = F.softmax(self.teacher(images) / self.hyperparams.teacher_temperature, dim=1)
            student_output = F.softmax(self.student(images), dim=1)

            self.hyperparams.optimizer.zero_grad()
            loss = self.criterion_train(student_output, teacher_output.detach())
            loss.backward()
            self.hyperparams.optimizer.step()
            pbar.set_description(f'Epoch {epoch} | Loss {loss.item()}')

    def eval_step(self):
        self.student.eval()
        total_correct = 0
        avg_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_dataloader):
                images = images.cuda()
                labels = labels.cuda()

                outputs = self.student(images)
                outputs_s = torch.nn.Softmax(dim=1)(outputs)

                avg_loss += self.criterion_test(outputs_s, labels).sum()
                pred = outputs.argmax(dim=1)
                total_correct += pred.eq(labels.data.view_as(pred)).sum()

        avg_loss /= len(self.test_dataloader.dataset) # type: ignore
        self.acc = float(total_correct) / len(self.test_dataloader.dataset) # type: ignore
        print('Test Avg. Loss: %f, Accuracy: %f' %
              (avg_loss.data.item(), self.acc)) # type: ignore

    def train(self):
        for epoch in range(1, self.hyperparams.epochs):
            self.train_step(epoch)
            self.eval_step()
        torch.save(self.student, self.save_path)


# class StudentTrainer:

#     def __init__(
#             self,
#             dataset: str,
#             data_path: str,
#             teacher: torch.nn.Module,
#             student: torch.nn.Module,
#             save_path: Path,
#             test_path: Path = Path('./data/real/')
#         ):
#         self.dataset = dataset

#         self.data_path = data_path
#         self.teacher = teacher
#         self.student = student.cuda()

#         self.save_path = save_path
#         save_path.parent.mkdir(parents=True, exist_ok=True)

#         _, test_trans = transformer(self.dataset)

#         self.data_train = torchvision.datasets.ImageFolder(
#             root=self.data_path, 
#             transform=test_trans
#         )

#         batch_size = 256

#         if self.dataset == 'mnist':
#             self.data_test = MNIST(
#                 root=test_path, 
#                 train=False, 
#                 transform=test_trans, 
#                 download=True
#             )
#             batch_size = 256
#             self.epochs = 20

#             self.optimizer = torch.optim.Adam(
#                 self.student.parameters(), lr=0.01, weight_decay=1e-4)

#         elif self.dataset == 'cifar10':
#             self.data_test = CIFAR10(
#                 test_path, 
#                 train=False, 
#                 transform=test_trans, 
#                 download=True
#             )
#             batch_size = 128
#             self.epochs = 200

#             self.optimizer = torch.optim.SGD(
#                 self.student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

#         elif self.dataset == 'cifar100':
#             self.data_test = CIFAR100(
#                 test_path, train=False, transform=test_trans, download=True)
#             batch_size = 128
#             self.epochs = 200
#             self.optimizer = torch.optim.SGD(
#                 self.student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

#         self.trainset_loader = DataLoader(
#             dataset=self.data_train, 
#             batch_size=batch_size, 
#             shuffle=True, 
#             num_workers=0
#         )
#         self.testset_loader = DataLoader(
#             dataset=self.data_test, 
#             batch_size=batch_size, 
#             num_workers=0
#         )

#         self.criterion_train = torch.nn.BCELoss().cuda()
#         self.criterion_test = torch.nn.CrossEntropyLoss().cuda()

#         self.acc = 0.0
#         self.acc_best = 0.0

#     def train(self, epoch):
#         self.student.train()
#         for i, (images, labels) in enumerate(pbar := tqdm(self.trainset_loader)):
#             images, labels = Variable(images).cuda(), Variable(labels).cuda()

#             outputs_T_s = F.softmax(self.teacher(images) / 20.0, dim=1)
#             outputs_S_s = F.softmax(self.student(images), dim=1)

#             self.optimizer.zero_grad()
#             loss = self.criterion_train(outputs_S_s, outputs_T_s.detach())
#             loss.backward()
#             self.optimizer.step()

#             pbar.set_description(f'Epoch {epoch} | Loss {loss.item()}')

#     def test(self):
#         self.student.eval()
#         total_correct = 0
#         avg_loss = 0.0
#         with torch.no_grad():
#             for i, (images, labels) in enumerate(self.testset_loader):
#                 images, labels = Variable(images).cuda(), Variable(labels).cuda()

#                 outputs = self.student(images)
#                 outputs_s = torch.nn.Softmax(dim=1)(outputs)

#                 avg_loss += self.criterion_test(outputs_s, labels).sum()
#                 pred = outputs.data.max(1)[1]
#                 total_correct += pred.eq(labels.data.view_as(pred)).sum()

#         avg_loss /= len(self.data_test)
#         self.acc = float(total_correct) / len(self.data_test)
#         if self.acc_best < self.acc:
#             self.acc_best = self.acc
#         print('Test Avg. Loss: %f, Accuracy: %f' %
#               (avg_loss.data.item(), self.acc)) # type: ignore

#     def build(self):
#         print('-'*30+' Train student start '+'-'*30)
#         for epoch in range(1, self.epochs):
#             self.train(epoch)
#             self.test()
#         torch.save(self.student, self.save_path)
#         print('-'*30+' Train student end '+'-'*30)
