

# TorchSense

<div>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>

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

## Dataset Structure

The directory structure of your dataset folder looks like this:

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

### Train model with default configuration

<details>
<summary>Show details</summary>

you can only use our data loader

- the only you need to input are `params([input_key1,...],[target_key]) `and `data_path`

```python
from torchsense.trainer import Trainer
from torchsense.datasets.custom import SensorFolder
from torch.utils.data import DataLoader
from torchsense.models.gan_g import Generator
from torchsense import transforms as T
from torchaudio.transforms import Spectrogram

def train():
    # data part
    data_path = "data1"
    transform1 = T.Compose([
        T.ToTensor(),
        T.Normalize(-1, 1),
        Spectrogram(n_fft=512, hop_length=160, win_length=256, power=1),
    ])
    transform2 = T.Compose([
        T.ToTensor(),
        T.Interpolate(5000),
        Spectrogram(n_fft=100, hop_length=10, win_length=100, power=1),
    ])
    data = SensorFolder(root=data_path,
                        params=(["acc[2]", "mix_mic"], ["mic"]),
                        transform=[transform2, transform1],
                        target_transform=transform1)
    train_set, test_set = data.train_test_split(0.5)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              drop_last=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                            drop_last=True, num_workers=0)

    # model part
    model = Generator()

    # training part
    trainer = Trainer(model, max_epochs=5, task="m")
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    train()

```

</details>



### Train model with custom dataset

<details>
<summary>Show details</summary>

you can only use our trainer or model

- the only you need to input are `model`or`dataset`
```python
import torch
from torchsense.trainer import Trainer
from torchsense.models.cnn4 import CNN4


def train():
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor

    train_set = MNIST(root="./tmp/data1/MNIST", train=True, transform=ToTensor(), download=True)
    val_set = MNIST(root="./tmp/data1/MNIST", train=False, transform=ToTensor(), download=False)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=64, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=0
    )

    model = CNN4()
    trainer = Trainer(model, max_epochs=5)
    trainer.fit(train_loader, val_loader)
```

</details>

## Thanks

u-net parts reference [milesial](https://github.com/milesial/Pytorch-UNet/tree/master)
