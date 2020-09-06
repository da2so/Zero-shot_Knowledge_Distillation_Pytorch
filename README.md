# Zero-Shot Knowledge Distillation in Deep Networks

ZSKD with PyTorch 



## Requirements

- Pytorch 1.14 
- Python 3.6
- CUDA 10.1 


## Running the code

```shell
python main.py --teahcer_model_path=models/teacher_cifar10.pt --n=10
```

Arguments:

- `teahcer_model_path` - teacher model path (.pt) 
- `t_train` - Train teacher network?? 
	- Please, use True (not yet implement the code that train a teacher network)
- `n` - Number of DIs crafted per category
- `beta` - Beta scaling vectors
- `batch_size` - batch size
- `lr` - learning rate
- `iters` - iteration number



## Understanding this method(algorithm)

Check my blog!!
[Here](https://da2so.github.io/2020-08-12-Zero_Shot_Knowledge_Distillation_in_Deep_Networks/)