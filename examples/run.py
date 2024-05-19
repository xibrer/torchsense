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
