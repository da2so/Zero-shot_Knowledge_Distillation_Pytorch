import os

from torchvision import transforms
import torchvision
import torch 
from torch.autograd import Variable
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import  CIFAR10, CIFAR100
from torch.utils.data import Dataset, DataLoader

from trainer.utils import transformer

class s_trainer():
    def __init__(self, dataset,data_path, teacher, student,save_path,test_path='./trainer/dataset/'):
        self.dataset=dataset

        self.data_path=data_path
        self.teacher=teacher
        self.student=student.cuda()

        self.save_path=save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        train_trans, test_trans=transformer(self.dataset)
        
        self.data_train= torchvision.datasets.ImageFolder(root = self.data_path,transform = test_trans)
        
        if self.dataset == 'mnist':
            self.data_test = MNIST(test_path, train=False, transform=test_trans, download=True)
            batch_size=256
            self.epochs=20
            
            self.optimizer = torch.optim.Adam(self.student.parameters(), lr=0.01, weight_decay=1e-4)

        elif self.dataset == 'cifar10':
            self.data_test = CIFAR10(test_path, train=False, transform=test_trans, download=True)
            batch_size=128
            self.epochs=200

            self.optimizer = torch.optim.SGD(self.student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

        elif self.dataset == 'cifar100':
            self.data_test = CIFAR100(test_path, train=False, transform=transform_test, download=True)
            batch_size=128
            self.epochs=200

            self.optimizer = torch.optim.SGD(self.student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

        self.trainset_loader=DataLoader(self.data_train, batch_size = batch_size, shuffle = True,num_workers = 0 )
        self.testset_loader=DataLoader(self.data_test, batch_size=batch_size, num_workers=0)
        
        self.criterion_train = torch.nn.BCELoss().cuda()
        self.criterion_test= torch.nn.CrossEntropyLoss().cuda()

        self.acc=0.0
        self.acc_best=0.0
    def train(self,epoch):
        self.student.train()
        for i, (images, labels) in enumerate(self.trainset_loader):
                images, labels = Variable(images).cuda(), Variable(labels).cuda()
                if self.dataset =='mnist':
                    images= images[:,0,:,:]
                    images=images.unsqueeze(dim=1)
                
                outputs_T = self.teacher(images) /20.0
                outputs_T_s= torch.nn.Softmax(dim=1)(outputs_T)

                outputs_S = self.student(images)
                outputs_S_s=torch.nn.Softmax(dim=1)(outputs_S)

                self.optimizer.zero_grad()
                loss = self.criterion_train(outputs_S_s, outputs_T_s.detach())

                if i == 1:
                    print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))
                loss.backward()
                self.optimizer.step()


                
    def test(self):
        self.student.eval()
        total_correct = 0
        avg_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.testset_loader):
                images, labels = Variable(images).cuda(), Variable(labels).cuda()
                if self.dataset =='mnist':
                    images= images[:,0,:,:]
                    images=images.unsqueeze(dim=1)

                outputs = self.student(images)
                outputs_s= torch.nn.Softmax(dim=1)(outputs)

                avg_loss += self.criterion_test(outputs_s, labels).sum()
                pred = outputs.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
    
        avg_loss /= len(self.data_test)
        self.acc = float(total_correct) / len(self.data_test)
        if self.acc_best < self.acc:
            self.acc_best = self.acc
        print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), self.acc))
    
        
    def build(self):
        print('-'*30+' Train student start '+'-'*30)
        for epoch in range(1, self.epochs):
            self.train(epoch)
            self.test()
        torch.save(self.student,self.save_path + 'student_'+self.dataset+'.pt')
        print('-'*30+' Train student end '+'-'*30)
