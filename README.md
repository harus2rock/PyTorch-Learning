# PyTorch-Learning
SHOEISHA PyTorch introductory development

## Environmental construction 
```
$ pip install pandas jupyter matplotlib scipy scikit-learn pillow tqdm cython seaborn
$ pip install torch==0.4.1 torchvision==0.2.1 
```

## Check GPU on Python
```
$ python
>>> import torch
>>> torch.cuda.is_available()
True
```

## Chech GPU on Terminal
```
$ watch -d -n 0.5 nvidia-smi
```

## Use GPU
```
data = data.to('cuda')
target = target.to('cuda')
model = model.to('cuda')
```

## Check the number of CPU cores in Linux
### the number of CPU
```
$ cat /proc/cpuinfo | grep "physical id" | uniq
physical id	: 0
```

### the number of processer
```
$ cat /proc/cpuinfo | grep "processor"
processor	: 0
processor	: 1
processor	: 2

   :

processor	: 18
processor	: 19
```

### the number of CPU cores
```
$ cat /proc/cpuinfo | grep "cpu cores" | uniq
cpu cores	: 10
```