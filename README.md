# PyTorch-Learning
SHOEISHA PyTorch introductory development

## Environmental construction 
```
$ pip install pandas jupyter matplotlib scipy scikit-learn pillow tqdm cython
$ pip install torch==0.4.1 torchvision==0.2.1 
```

## Check GPU on Python
```
$ python
>>> import torch
>>> torch.cuda.is_available()
True
```