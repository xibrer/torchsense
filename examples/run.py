import torch
from torchsense.trainer import Trainer
from torchsense.models.cnn4 import CNN4
import torchvision.transforms as T


def train():
    from torchvision.datasets import MNIST
    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train_set = MNIST(root="./tmp/data1/MNIST", train=True, transform=transform, download=True)
    val_set = MNIST(root="./tmp/data1/MNIST", train=False, transform=transform, download=False)

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
