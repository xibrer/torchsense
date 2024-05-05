import torch
from torch import nn
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self,length:int,):
        super(Encoder, self).__init__()

        modules = []
        hidden_dims = [32, 64, 128, 256, 512]
        map_size = int(length / pow(2, len(hidden_dims)))
        in_channels = 1
        for h_dim in hidden_dims:
            length = length / 2
            modules.append(nn.Sequential(nn.Conv1d(in_channels, out_channels=h_dim,
                                                   kernel_size=3, stride=2, padding=1),
                                         nn.LayerNorm(int(length)),
                                         nn.LeakyReLU()))
            in_channels = h_dim
        self.net = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(nn.Conv1d(hidden_dims[-1],hidden_dims[-2],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1),
                            nn.BatchNorm1d(hidden_dims[-2]),
                            nn.LeakyReLU(),
                            nn.Conv1d(hidden_dims[-2], out_channels= 1,
                                      kernel_size= 3, padding= 1),
                            nn.Dropout(0.2),
                            nn.Linear(40,2),
                           )
    def forward(self, x):
        x = self.net(x)
        # x = self.fc(x)
        x =self.final_layer(x)
        return x
class Encoder2D(nn.Module):
    def __init__(self,height:int,width:int):
        super(Encoder2D, self).__init__()

        modules = []
        hidden_dims = [32, 64, 128, 256, 512]
        
        in_channels = 1
        for h_dim in hidden_dims:
            height = int(height / 2)
            width = int(width /2)
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim,
                                                   kernel_size=3, stride=2, padding=1),
                                         nn.LayerNorm([int(height), int(width)]),
                                         nn.LeakyReLU()))
            in_channels = h_dim
        self.net = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(nn.Conv2d(hidden_dims[-1],hidden_dims[-2],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1),
                            nn.BatchNorm2d(hidden_dims[-2]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-2], out_channels= 1,
                                      kernel_size= 3, padding= 1),
                            # nn.Dropout(0.2),
                            # nn.Linear(40,2),
                           )
    def forward(self, x):
        x = self.net(x)
        # x = self.fc(x)
        x =self.final_layer(x)
        return x
class lstm(nn.Module):
    def __init__(self):
        super(lstm, self).__init__()
        self.hidden_size = 128
        self.time_step = 32
        self.length = 3520
        self.input_size = int(self.length / self.time_step)
        # 输入的维度（input size）指的是每个时间步输入的特征向量的维度
        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
        )
        self.encoder = Encoder(self.hidden_size*self.time_step)
        self.encoder2d = Encoder2D(self.time_step,self.hidden_size)
    def forward(self, x):
        # print('0',x.shape,len(x))
        # (batch_size, sequence_length, input_size) 这里序列长度含义和时间步含义重合
        x = x.view(-1, self.time_step, self.input_size)
        # print('1',x.shape )
        r_out, (h_n, h_c) = self.rnn(x, None)
        # batch_size ,time step,hidden_size
        b, l, h = r_out.shape
        # print(r_out.shape )
        r_out = r_out.reshape(b,-1,l,h)
        # print(r_out.shape )
        out = self.encoder2d(r_out)
        # print(out.shape)
        return out


class gru(nn.Module):
    def __init__(self):
        super(gru, self).__init__()
        self.hidden_size = 128
        self.time_step = 10
        self.rnn = nn.GRU(
            input_size=352,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
        )

        self.out = nn.Sequential(nn.Tanh(),
                                 nn.Linear(self.hidden_size, 1024),
                                 nn.Dropout(0.5),
                                 nn.Linear(1024, 2))  # 最后时刻的hidden映射

    def forward(self, x):
        x = x.view(len(x), self.time_step, -1)

        r_out, h_n = self.rnn(x, None)
        b, l, h = r_out.shape
        # print(r_out.shape )
        r_out = r_out.reshape(l * b, h)
        # print(r_out.shape )
        out = self.out(r_out)
        # print('2',b,out.shape )
        out = out.unsqueeze(1)
        out = out.reshape(b, 1, -1)
        return out


if __name__ == "__main__":
    # gradient check
    batch_size = 32
    model = lstm().cuda()
    loss_fn = torch.nn.MSELoss()
    input = Variable(torch.randn(batch_size, 1, 3520)).cuda()
    target = Variable(torch.randn(batch_size, 1, 3520)).cuda()

    output = model(input)
    print(output.shape)
    # output = output[0][0].double()
    # res = torch.autograd.gradcheck(loss_fn, (output, target.squeeze()), eps=1e-6, raise_exception=True)
    print(model)
    print(output.size())
