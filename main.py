import argparse

from utils import *
from ZSKD import ZSKD
from torchvision import models


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--teacher_model_path', type=str, default='models/teacher_cifar10.pt',
                        help='Choose the teacher network')
    parser.add_argument('--t_train', type=bool, default=True, help='Train teacher network??')
    parser.add_argument('--n', type=int, default=2, help='Number of DIs crafted per category')
    parser.add_argument('--beta', type=list, default=[0.1, 1.], help='Beta  scaling vectors')
    parser.add_argument('--t', type=int, default=20, help='Temperature for distillation')
    parser.add_argument('--batch_size', type=int, default=10, help='batch_size')
    parser.add_argument('--lr', type=int, default=0.05, help='learning rate')
    parser.add_argument('--iters', type=int, default=1500, help='iteration number')

    args = parser.parse_args()

    teacher = load_model(args.teacher_model_path)

    #Perform Zero-shot Knowledge distillation
    zskd = ZSKD(teacher, args.n, args.beta, args.t, args.batch_size, args.lr, args.iters)
    zskd()


if __name__ == "__main__":
    set_gpu_device(0)
    main()