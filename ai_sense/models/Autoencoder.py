import torch.nn as nn
from utils import show_parameter


class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_dims: list,
                 latent_dim: int,
                 map_size: int,
                 length: int):
        super(Encoder, self).__init__()
        modules = []
        for h_dim in hidden_dims:
            length = length / 2
            modules.append(nn.Sequential(nn.Conv1d(in_channels, out_channels=h_dim,
                                                   kernel_size=3, stride=2, padding=1),
                                         nn.LayerNorm(int(length)),
                                         nn.LeakyReLU()))
            in_channels = h_dim
        self.net = nn.Sequential(*modules)
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(hidden_dims[-1]
                                          * map_size, latent_dim),
                                nn.LeakyReLU())

    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_dims: list,
                 latent_dim: int,
                 map_size: int,
                 length: int):
        super(Decoder, self).__init__()
        modules = []
        self.map_size = map_size
        hidden_dims.reverse()
        self.hidden_dims = hidden_dims
        length = map_size
        for i in range(len(hidden_dims) - 1):
            length = 2 * length
            modules.append(
                nn.Sequential(nn.ConvTranspose1d(hidden_dims[i], hidden_dims[i + 1],
                                                 kernel_size=3, stride=2, padding=1,
                                                 output_padding=1),
                              nn.LayerNorm(int(length)),
                              nn.LeakyReLU())
            )
        self.net = nn.Sequential(*modules)
        self.fc = nn.Sequential(nn.Linear(latent_dim, hidden_dims[0] * map_size),
                                nn.LeakyReLU())

    def forward(self, x):
        x = self.fc(x)
        # print(x)
        x = x.reshape(-1, self.hidden_dims[0], self.map_size)
        x = self.net(x)
        return x


class Autoencoder(nn.Module):
    def __init__(
            self,
            num_input_channels: int,
            latent_dim: int,
            hidden_dims: list = None,
            length: int = 3520,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        feature_map_size = int(length / pow(2, len(hidden_dims)))

        # Creating encoder and decoder
        self.encoder = Encoder(
            num_input_channels, hidden_dims, latent_dim, feature_map_size, length)
        self.decoder = Decoder(
            num_input_channels, hidden_dims, latent_dim, feature_map_size, length)
        self.final_layer = nn.Sequential(
            nn.Conv1d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_dims[-1], out_channels=1,
                      kernel_size=3, padding=1),
            nn.Linear(880,440),
            nn.Dropout(0.5),
            nn.Linear(440,2)
        )

    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        x_hat = self.final_layer(x_hat)
        return x_hat

    def get_reconstruction_loss(self, *output, labels):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)"""
        x_hat = output[0]
        _loss = nn.SmoothL1Loss()
        loss = _loss(x_hat, labels)
        return loss


if __name__ == "__main__":
    model = Autoencoder(1, 128).cuda()
    show_parameter(model)
