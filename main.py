import argparse

from torchvision import models

from utils import *
from ZSKD import ZSKD
from trainer.teacher_train import t_trainer
from trainer.student_train import s_trainer

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cifar10', help='[mnist, cifar10, cifar100]')
    parser.add_argument('--t_train', type=str2bool, default='False', help='Train teacher network??')
    parser.add_argument('--num_sample', type=int, default=24000, help='Number of DIs crafted per category')
    parser.add_argument('--beta', type=list, default=[0.1, 1.], help='Beta  scaling vectors')
    parser.add_argument('--t', type=int, default=20, help='Temperature for distillation')
    parser.add_argument('--batch_size', type=int, default=100, help='batch_size')
    parser.add_argument('--lr', type=int, default=0.01, help='learning rate')
    parser.add_argument('--iters', type=int, default=1500, help='iteration number')

    parser.add_argument('--s_save_path', type=str, default='./saved_model/', help='save path for student network')
    parser.add_argument('--do_genimgs', type=str2bool, default='True', help='generate synthesized images from ZSKD??')

    args = parser.parse_args()
    t_model_path='./trainer/models/teacher_'+args.dataset+'.pt'
    #load teacher network
    if args.t_train == True:
        T_trainer=t_trainer(args.dataset)
        T_trainer.build()
    teacher = load_model(args.dataset, t_model_path)
        
    #perform Zero-shot Knowledge distillation
    if args.do_genimgs==True:
        zskd = ZSKD(args.dataset, teacher, args.num_sample, args.beta, args.t, args.batch_size, args.lr, args.iters)
        student, save_root= zskd.build()
    else:
        _,_ ,student = data_info(args.dataset)
        save_root= './saved_img/' +args.dataset+'/'

    #train student network 
    S_trainer=s_trainer(args.dataset,save_root, teacher, student,args.s_save_path)
    S_trainer.build()
    

if __name__ == "__main__":
    set_gpu_device(0)
    main()