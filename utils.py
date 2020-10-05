from PIL import Image
import numpy as np

import torch

from trainer.resnet import ResNet34, ResNet18
from trainer.lenet import LeNet5, LeNet5Half

def cuda_available():
    use_cuda = torch.cuda.is_available()
    return use_cuda

def set_gpu_device(num):
    torch.cuda.set_device(num)


def load_model(dataset, model_path):
    assert ('.pt' or '.pth') in model_path
    if torch.typename(torch.load(model_path)) == 'OrderedDict':
        if dataset=='mnist':
            model = LeNet5()
        elif dataset== 'cifar10' or dataset=='cifar100':
            model = ResNet34()
        model.load_state_dict(torch.load(model_path))

    else:
        model = torch.load(model_path)

    model.eval()
    if cuda_available():
        model.cuda()

    return model


def data_info(dataset):
    if dataset=='mnist':
        cwh= [1,32,32]
        num_classes=10
        student=LeNet5Half()
    elif dataset == 'cifar10':
        cwh=[3,32,32]
        num_classes=10
        student=ResNet18()

    elif dataset =='cifar100':
        cwh=[3,32,32]
        num_classes=100
        student=ResNet18()

    return cwh, num_classes, student

def save_img(image,save_path):

    np_image=image.data.clone().cpu().numpy()
    print(np.shape(np_image))
    np_image=np.squeeze(np_image)
    PIL_image = Image.fromarray(np_image,'L')

    PIL_image.save(save_path)
