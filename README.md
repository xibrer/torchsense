
# TorchSense
<div>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

Torchsense is a library for sensor data processing with PyTorch. It provides I/O, signal and data processing functions, datasets, model implementations and application components.

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/xibrer/torchsense.git

# pip install

pip install torchsense
```

## Project Structure

The directory structure of new project looks like this:

```
data/
    ├── class_x
    │   ├── xxx.ext
    │   ├── xxy.ext
    │   └── ...
    │       └── xxz.ext
    └── class_y
        ├── 123.ext
        ├── nsdf3.ext
        └── ...
        └── asd932_.ext
```


## How to run

Train model with default configuration

```python 
# ensure you already install torch and lightning
from torchsense.datasets.folder import ImageFolder
from torch.utils.data import DataLoader

# input keys you want input
list1 = [key1,key2,key3]# example ["acc", "mix_mic", "mic"]
data = ImageFolder(root="data", params=list1)
train_set, test_set = data.train_test_split(0.5)
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

for i, batch in enumerate(train_loader):
    acc, mix, mic = batch[0]
    # ... train your self
```

now model only support .mat file
