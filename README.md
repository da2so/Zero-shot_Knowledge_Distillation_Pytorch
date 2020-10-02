# Zero-Shot Knowledge Distillation in Deep Networks

ZSKD with PyTorch 



## Requirements

- Pytorch 1.14 
- Python 3.6


## Running the code

For mnist dataset,

```shell
python main.py --dataset=mnist --t_train=False --teahcer_model_path=models/teacher_mnist.pt --n=10
```

For cifar10 dataset,

```shell
python main.py --dataset=cifar10 --t_train=False --teahcer_model_path=models/teacher_cifar10.pt --n=10
```


Arguments:

- `dataset` - available dataset: ['mnist', 'cifar10', 'cifar100']
- `teahcer_model_path` - teacher model path (.pt) 
- `t_train` - Train teacher network?? 
	- if True, train teacher network
	- elif False, load trained teacher network
- `n` - Number of DIs crafted per category
- `beta` - Beta scaling vectors
- `batch_size` - batch size
- `lr` - learning rate
- `iters` - iteration number



## Understanding this method(algorithm)

Check my blog!!
[Here](https://da2so.github.io/2020-08-12-Zero_Shot_Knowledge_Distillation_in_Deep_Networks/)