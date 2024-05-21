from torchsense.trainer import Trainer
from torchsense.datasets.custom import SensorFolder
from torch.utils.data import DataLoader
from torchsense.models import Generator, UNet
from torchsense import transforms as T
from torchaudio.transforms import Spectrogram,GriffinLim
from torchsense.utils import load_from_ckpt
import torchaudio


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
    target_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(-1, 1),
        Spectrogram(n_fft=512, hop_length=160, win_length=256, power=1),
    ])
    stage_transform = T.Compose([
        # T.Normalize(-1, 1),
        GriffinLim(n_fft=512, hop_length=160, win_length=256, power=1),
    ])
    stage1_model = Generator()
    pre_model = load_from_ckpt(stage1_model, "epoch=19-val_loss=0.80.ckpt")
    data = SensorFolder(root=data_path,
                        params=(["acc[2]", "mix_mic"], ["mic"]),
                        # pre_model=pre_model,
                        # stage_transform=stage_transform,
                        transform=[transform2, transform1],
                        target_transform=target_transform)
    train_set, test_set = data.train_test_split(0.5)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              drop_last=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                            drop_last=True, num_workers=0)

    # model part

    model = Generator()
    # training part
    trainer = Trainer(model,  max_epochs=5, task="r", )
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    train()
