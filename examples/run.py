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


if __name__ == "__main__":
    train()
