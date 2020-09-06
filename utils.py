import torch

from resnet import ResNet34, ResNet18


def cuda_available():
    use_cuda = torch.cuda.is_available()
    return use_cuda


def set_gpu_device(num):
    torch.cuda.set_device(num)


def load_model(model_path):
    assert ('.pt' or '.pth') in model_path
    if torch.typename(torch.load(model_path)) == 'OrderedDict':

        model = ResNet34()
        model.load_state_dict(torch.load(model_path))

    else:
        model = torch.load(model_path)

    model.eval()
    if cuda_available():
        model.cuda()

    return model

