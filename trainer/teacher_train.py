import os

import torch
from torch.autograd import Variable
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import  CIFAR10, CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 

from trainer.lenet import LeNet5
import trainer.resnet 
from trainer.utils import transformer

class trainer():
    def __init__(self, dataset, data_path='./trainer/dataset/',model_path='./trainer/models/'):

        self.acc = 0
        self.acc_best = 0
    
        self.dataset=dataset
        self.model_path=model_path
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        
        train_trans, test_trans=transformer(self.dataset)

        if self.dataset == 'mnist':
            data_train = MNIST(data_path, transform=train_trans, download=True)
            self.data_test = MNIST(data_path, train=False, transform=test_trans, download=True)
            batch_size=256
            self.epochs=10
            self.net = LeNet5().cuda()
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
            
        elif self.dataset == 'cifar10':
            data_train = CIFAR10(data_path, transform=train_trans, download=True)
            self.data_test = CIFAR10(data_path, train=False, transform=test_trans, download=True)
            batch_size=128
            self.epochs=200
            self.net = resnet.ResNet34().cuda()
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

        elif self.dataset == 'cifar100':
            data_train = CIFAR100(data_path, transform=transform_train, download=True)
            self.data_test = CIFAR100(data_path, train=False, transform=transform_test, download=True)
            batch_size=128
            self.epochs=200
            self.net = resnet.ResNet34(num_classes=100).cuda()
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

        self.data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=0)
        self.data_test_loader = DataLoader(self.data_test, batch_size=batch_size, num_workers=0)

        self.criterion = torch.nn.CrossEntropyLoss().cuda()

            
    def train(self,epoch):
        if self.dataset != 'mnist':
            adjust_learning_rate(optimizer, epoch)
        self.net.train()
        for i, (images, labels) in enumerate(self.data_train_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
    
            self.optimizer.zero_grad()
            output = self.net(images)
            loss = self.criterion(output, labels)
    
            if i == 1:
                print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))
            loss.backward()
            self.optimizer.step()
    
    
    def test(self):
        self.net.eval()
        total_correct = 0
        avg_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.data_test_loader):
                images, labels = Variable(images).cuda(), Variable(labels).cuda()
                output = self.net(images)
                avg_loss += self.criterion(output, labels).sum()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
    
        avg_loss /= len(self.data_test)
        self.acc = float(total_correct) / len(self.data_test)
        if self.acc_best < self.acc:
            self.acc_best = self.acc
        print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), self.acc))
    
    

    def build(self):
        print('-'*30+' Train teacher start '+'-'*30)
        for epoch in range(1, self.epochs):
            self.train(epoch)
            self.test()
        torch.save(self.net,self.model_path + 'teacher_'+self.dataset+'.pt')
        print('-'*30+' Train teacher end '+'-'*30)
    
