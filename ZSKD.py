import os
import tqdm
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.autograd import Variable


class ZSKD():
    def __init__(self, teacher, n, beta, t, batch_size, lr, iters):
        self.teacher = teacher
        self.n = n
        self.beta = beta
        self.t = t
        self.batch_size = batch_size
        self.lr = lr
        self.iters = iters

        self.gen_num=1
    def __call__(self):

        # lim_0, lim_1 = 2, 2
        file_num=np.zeros((10),dtype=int)

        def get_class_similarity():

            # Find last layer
            t_layer = list(self.teacher.children())[-1]
            while 'Sequential' in str(t_layer):
                t_layer = list(t_layer.children())[-1]

            t_weights = list(t_layer.parameters())[0].cuda()  # size(#class number, #weights in final-layer )

            # Compute concentration parameter
            t_weights_norm = F.normalize(t_weights, p=2, dim=1)
            cls_sim = torch.matmul(t_weights_norm, t_weights_norm.T)
            cls_sim_norm = torch.div(cls_sim - torch.min(cls_sim, dim=1).values,
                                     torch.max(cls_sim, dim=1).values - torch.min(cls_sim, dim=1).values)
            return cls_sim_norm

        cls_sim_norm = get_class_similarity()
        cls_num = cls_sim_norm.shape[0]
        print('\n ----------    ZSKD start    ---------- ')
        for k in range(cls_num):

            inputs = torch.randn((self.batch_size, 3, 32, 32), requires_grad=True, device='cuda')

            optimizer = torch.optim.Adam([inputs], self.lr)
            loss = torch.nn.BCELoss()
            for b in self.beta:
                for n in range(self.n // len(self.beta)):

                    # sampling target label from Dirichlet distribution
                    dir_dist = torch.distributions.dirichlet.Dirichlet(b * cls_sim_norm[k])
                    y=Variable(dir_dist.rsample((self.batch_size,)),requires_grad=False)

                    # optimization for images
                    for iter in range(self.iters):
                        logit = self.teacher(inputs)/20.0
                        output= torch.nn.Softmax(dim=1)(logit)
                        l = loss(output,y)
                        optimizer.zero_grad()
                        l.backward()
                        optimizer.step()


                    # save the synthesized images
                    t_cls = torch.argmax(y, dim=1).detach().cpu().numpy()
                    for m in range(self.batch_size):
                        save_dir = 'result/' +str(t_cls[m])+'_'+str(b)+'/'
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        vutils.save_image(inputs[m, :, :, :].data.clone(), save_dir + str(file_num[t_cls[m]]) + '.png', normalize=True)
                        file_num[t_cls[m]]+=1
                    print('Generate {} synthesized images [{}/{}]'.format(\
                        self.batch_size,self.batch_size*self.gen_num,self.batch_size *cls_num * self.n ))

                    self.gen_num+=1

        print('\n ----------ZSKD end---------- \n')






