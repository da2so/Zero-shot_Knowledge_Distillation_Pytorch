import argparse

from torchvision import models

from utils import *
from ZSKD import ZSKD
from trainer.teacher_train import trainer

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='mnist', help='[mnist, cifar10, cifar100]')

    parser.add_argument('--teacher_model_path', type=str, default='trainer/models/teacher_mnist.pt',
                        help='Choose the teacher network')
    parser.add_argument('--t_train', type=bool, default=False, help='Train teacher network??')
    parser.add_argument('--n', type=int, default=2, help='Number of DIs crafted per category')
    parser.add_argument('--beta', type=list, default=[0.1, 1.], help='Beta  scaling vectors')
    parser.add_argument('--t', type=int, default=20, help='Temperature for distillation')
    parser.add_argument('--batch_size', type=int, default=10, help='batch_size')
    parser.add_argument('--lr', type=int, default=0.05, help='learning rate')
    parser.add_argument('--iters', type=int, default=1500, help='iteration number')

    args = parser.parse_args()

    if args.t_train == True:
        t_trainer=trainer(args.dataset)
        t_trainer.build()
    teacher = load_model(args.dataset, args.teacher_model_path)
        
    #Perform Zero-shot Knowledge distillation
    zskd = ZSKD(args.dataset, teacher, args.n, args.beta, args.t, args.batch_size, args.lr, args.iters)
    zskd.build()


if __name__ == "__main__":
    set_gpu_device(0)
    main()